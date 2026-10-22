"""Run every method on one image and print what each one produced.

    python -m backend.cli.run_methods data/bri/archive/Testing/glioma/Te-gl_0010.jpg
    python -m backend.cli.run_methods scan.png --methods 2,4 --open
    python -m backend.cli.run_methods scan.png --spect data/spect_mri/SPECT/10007.png

For each method the terminal gets one block: what that method is, what it
predicted and how confidently, which files it wrote, and the caveats that come
with the number. Every image a method produced — the preprocessed input, the
segmentation mask, the Grad-CAM heatmap, the fused image — is copied into a run
folder so the paths stay valid after the prediction cache is cleaned.

Nothing here computes a metric. The accuracies printed are read from each
method's recorded held-out evaluation; the prediction is whatever the engine
returned, including "no prediction" when a method has no trained classifier.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from backend import config
from backend.cli import render
from backend.methods.common.metrics_store import metrics_for
from backend.methods.registry import METHOD_IDS, get_method, list_methods, load_engine

# --------------------------------------------------------------------------- #
# Terminal styling: colour only when the output is a terminal that wants it.
# --------------------------------------------------------------------------- #
_COLOUR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOUR else text


def bold(t: str) -> str:
    return _c(t, "1")


def dim(t: str) -> str:
    return _c(t, "2")


def cyan(t: str) -> str:
    return _c(t, "36")


def green(t: str) -> str:
    return _c(t, "32")


def yellow(t: str) -> str:
    return _c(t, "33")


def red(t: str) -> str:
    return _c(t, "31")


LABEL_WIDTH = 14


def field(label: str, *lines: str) -> None:
    """One labelled field, continuation lines aligned under the first."""
    pad = " " * (LABEL_WIDTH + 2)
    for i, line in enumerate(lines):
        prefix = f"  {dim(label.ljust(LABEL_WIDTH))}" if i == 0 else pad
        print(f"{prefix}{line}")


def bar(fraction: float, width: int = 24) -> str:
    filled = max(0, min(width, round(fraction * width)))
    return "█" * filled + dim("·" * (width - filled))


def slug(name: str) -> str:
    """A folder-safe short name, so a run folder reads as what it holds."""
    keep = [c.lower() if c.isalnum() else "-" for c in name]
    return "-".join(part for part in "".join(keep).split("-") if part)


@dataclass
class MethodRun:
    method_id: str
    display_number: int
    short_name: str
    prediction: Optional[str]
    confidence: Optional[float]
    probabilities: dict
    seconds: float
    model_version: str
    images: dict
    warnings: list
    details: dict
    directory: Path

    def to_dict(self) -> dict:
        return {
            "method_id": self.method_id, "display_number": self.display_number,
            "short_name": self.short_name, "prediction": self.prediction,
            "confidence": self.confidence, "probabilities": self.probabilities,
            "seconds": round(self.seconds, 3), "model_version": self.model_version,
            "images": {k: str(v) for k, v in self.images.items()},
            "warnings": self.warnings, "details": self.details,
        }


# --------------------------------------------------------------------------- #
# What each produced image actually is, for this method
# --------------------------------------------------------------------------- #
def image_notes(spec, details: dict) -> dict[str, str]:
    """A sentence per output image, specific to the method that made it."""
    gradcam_stage = next((s.description for s in spec.pipeline if s.id == "gradcam"), "")
    notes = {
        "input": "the preprocessed slice the model actually read, not the file you passed",
        "gradcam": details.get("gradcam_note") or gradcam_stage or
                   "pixels that moved the predicted class, warm = more",
        "mask": details.get("segmentation_note") or "segmentation produced by this method",
        "fused": "this method's fusion stage output; it did not produce the class above",
    }
    if details.get("segmentation_in_method") is False:
        notes["mask"] = ("borrowed from the U-Net + ConvLSTM method and shown for context — "
                         "it is not part of this method and did not affect the class")
    return notes


def head_line(details: dict) -> Optional[str]:
    """One line naming the thing that actually produced the class, when it is
    not simply the network's own softmax layer."""
    selection = details.get("feature_selection")
    if selection:
        svm = selection.get("svm") or {}
        return (f"SVM on {selection.get('features_selected')} of "
                f"{selection.get('features_available')} features selected by "
                f"{selection.get('algorithm')} ({svm.get('kernel')} kernel)")
    if details.get("result_source") == "ai_assessment":
        return "an AI model's reading — this method has no trained classifier"
    return None


def describe(spec, metrics: dict) -> list[str]:
    """Two or three lines on what this method is and what it measured."""
    lines = [spec.summary]
    if metrics.get("accuracy") is not None:
        lines.append(
            f"Recorded held-out test accuracy {metrics['accuracy'] * 100:.2f}% "
            f"(macro F1 {metrics['f1']:.4f}) — measured once, not on this image."
        )
    else:
        lines.append("No held-out evaluation recorded for this method.")
    volumetric = metrics.get("volumetric_segmentation")
    if volumetric:
        lines.append(
            f"A 3D U-Net was also trained (volumetric Dice {volumetric.get('dice')}); it is not "
            f"what segments this image — an upload is a single slice."
        )
    return lines


# --------------------------------------------------------------------------- #
def wrap(text: str, width: int = 86) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width) or [""]


def draw(run: "MethodRun", mode: str) -> None:
    """Draw this method's images in the terminal, side by side and captioned."""
    if mode == "off" or not run.images:
        return
    order = [n for n in ("input", "mask", "gradcam", "fused") if n in run.images]
    render.show([run.images[n] for n in order], order,
                mode="inline" if mode == "inline" else "blocks")


def run_method(method_id: str, image_bytes: bytes, spect_bytes: Optional[bytes],
               run_dir: Path, source: Path) -> MethodRun:
    spec = get_method(method_id)
    engine = load_engine(method_id)
    started = time.perf_counter()
    if spect_bytes and getattr(engine, "accepts_secondary_image", False):
        result = engine.predict(image_bytes, spect_bytes)
    else:
        result = engine.predict(image_bytes)
    seconds = time.perf_counter() - started

    directory = run_dir / f"method{spec.display_number}_{slug(spec.short_name)}"
    directory.mkdir(parents=True, exist_ok=True)
    images: dict[str, Path] = {}
    for name, path in (("input", result.original_path),
                       ("mask", result.mask_path),
                       ("gradcam", result.overlay_path),
                       ("fused", getattr(result, "fused_path", None))):
        if path and Path(path).exists():
            target = directory / f"{name}.png"
            shutil.copy(Path(path), target)
            images[name] = target
    shutil.copy(source, directory / f"source{source.suffix}")

    return MethodRun(
        method_id=method_id, display_number=spec.display_number, short_name=spec.short_name,
        prediction=result.prediction, confidence=result.confidence,
        probabilities=dict(result.probabilities), seconds=seconds,
        model_version=result.model_version, images=images,
        warnings=list(result.warnings), details=dict(result.details), directory=directory,
    )


def print_run(run: MethodRun, source: Path, images: str) -> None:
    spec = get_method(run.method_id)
    metrics = metrics_for(run.method_id)
    notes = image_notes(spec, run.details)

    print()
    print(cyan("━" * 92))
    print(f" {bold(f'METHOD {run.display_number}')}  {bold(spec.short_name)}  "
          f"{dim('· ' + spec.modality)}")
    print(cyan("━" * 92))

    for i, line in enumerate(describe(spec, metrics)):
        for j, part in enumerate(wrap(line)):
            field("What it is" if i == 0 and j == 0 else "", part)

    print()
    field("Input", str(source))
    if run.prediction:
        confidence = run.confidence or 0.0
        colour = green if confidence >= 80 else yellow
        field("Prediction", f"{bold(colour(run.prediction.upper()))}  "
                            f"{colour(f'{confidence:.2f}%')}  "
                            f"{dim(f'in {run.seconds:.2f}s · model {run.model_version}')}")
    else:
        field("Prediction", red("none — this method produced no class"),
              dim(f"model {run.model_version}"))

    head = head_line(run.details)
    if head:
        field("Decided by", *wrap(head))

    if run.probabilities:
        rows = sorted(run.probabilities.items(), key=lambda kv: -kv[1])
        field("Probabilities", *[f"{name:<14}{bar(value / 100)} {value:>6.2f}%"
                                 for name, value in rows])

    if run.images:
        print()
        draw(run, images)
        print()
        lines = []
        for name in run.images:
            parts = wrap(notes.get(name, ""), width=70)
            lines.append(f"{name:<9}{parts[0]}")
            lines.extend(f"{'':<9}{dim(part)}" for part in parts[1:])
        field("What you see", *lines)
        field("Saved to", str(run.directory))

    if run.warnings:
        shown = [w for w in run.warnings]
        lines = []
        for warning in shown:
            parts = wrap(warning, width=78)
            lines.append(yellow("• ") + parts[0])
            lines.extend("  " + p for p in parts[1:])
        field("Caveats", *lines)



def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run every method on one image and show what each produced.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Examples:\n"
                "  python -m backend.cli.run_methods scan.jpg\n"
                "  python -m backend.cli.run_methods scan.jpg --methods 2,4 --open\n"
                "  python -m backend.cli.run_methods mri.png --spect spect.png --inline\n"),
    )
    ap.add_argument("image", nargs="?", help="MRI slice (png/jpg/tif)")
    ap.add_argument("--methods", default="all",
                    help="displayed numbers or ids, e.g. '1,3' or 'method2' (default: all)")
    ap.add_argument("--spect", default=None,
                    help="registered SPECT counterpart, for the method that fuses two modalities")
    ap.add_argument("--outdir", default=None,
                    help="where to write the run folder (default: outputs/)")
    ap.add_argument("--open", action="store_true", help="open the produced images afterwards")
    ap.add_argument("--images", choices=["auto", "blocks", "inline", "off"], default="auto",
                    help=("how to draw the images: 'blocks' (coloured half-blocks, works in any "
                          "colour terminal), 'inline' (iTerm2 protocol, true resolution), "
                          "'auto' picks inline when the terminal supports it, 'off' prints paths "
                          "only"))
    ap.add_argument("--inline", action="store_true", help="shorthand for --images inline")
    ap.add_argument("--json", action="store_true", help="also print the run as JSON")
    ap.add_argument("--list", action="store_true", help="list the methods and exit")
    args = ap.parse_args()

    if args.list:
        for spec in list_methods():
            status = metrics_for(spec.method_id)
            accuracy = (f"{status['accuracy'] * 100:.2f}%" if status.get("accuracy") is not None
                        else "not evaluated")
            print(f"  {spec.display_number}  {spec.short_name:<36} {spec.method_id:<9} {accuracy}")
        return 0

    if not args.image:
        ap.error("an image is required (or pass --list)")
    source = Path(args.image).expanduser()
    if not source.exists():
        print(red(f"ERROR: no such file: {source}"), file=sys.stderr)
        return 1

    wanted = METHOD_IDS
    if args.methods.strip().lower() != "all":
        by_number = {str(s.display_number): s.method_id for s in list_methods()}
        chosen = []
        for token in args.methods.split(","):
            token = token.strip()
            if token in by_number:
                chosen.append(by_number[token])
            elif token in METHOD_IDS:
                chosen.append(token)
            else:
                print(red(f"ERROR: unknown method {token!r}. Use 1-4 or {', '.join(METHOD_IDS)}."),
                      file=sys.stderr)
                return 1
        wanted = tuple(dict.fromkeys(chosen))

    image_mode = "inline" if args.inline else args.images
    if image_mode == "auto":
        image_mode = "inline" if render.supports_inline() else "blocks"

    image_bytes = source.read_bytes()
    spect_bytes = None
    if args.spect:
        spect_path = Path(args.spect).expanduser()
        if not spect_path.exists():
            print(red(f"ERROR: no such file: {spect_path}"), file=sys.stderr)
            return 1
        spect_bytes = spect_path.read_bytes()

    root = Path(args.outdir).expanduser() if args.outdir else config.ROOT_DIR / "outputs"
    run_dir = root / f"{time.strftime('%Y%m%d-%H%M%S')}_{source.stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(bold(f"Brain tumour analysis · {source.name}"))
    print(dim(f"{len(wanted)} method(s) · device {config.DEVICE} · output {run_dir}"))

    runs: list[MethodRun] = []
    for method_id in sorted(wanted, key=lambda m: get_method(m).display_number):
        try:
            run = run_method(method_id, image_bytes, spect_bytes, run_dir, source)
        except Exception as exc:  # one method failing must not hide the others
            print()
            print(red(f"METHOD {get_method(method_id).display_number} "
                      f"({get_method(method_id).short_name}) failed: {exc}"))
            continue
        runs.append(run)
        print_run(run, source, image_mode)

    print()
    print(cyan("━" * 92))
    print(f" {bold('SUMMARY')}")
    print(cyan("━" * 92))
    for run in runs:
        verdict = (f"{run.prediction} ({run.confidence:.1f}%)" if run.prediction
                   else "no prediction")
        print(f"  {run.display_number}. {run.short_name:<34} {verdict:<28} "
              f"{dim(f'{run.seconds:.2f}s')}")
    if len({r.prediction for r in runs if r.prediction}) > 1:
        print()
        for line in wrap("The methods disagree. They are different architectures trained "
                         "separately; where they disagree, no vote among them is evidence — "
                         "consult each one's recorded test accuracy above."):
            print(f"  {yellow(line)}")

    payload = {"image": str(source), "spect": args.spect, "run_dir": str(run_dir),
               "device": str(config.DEVICE), "methods": [r.to_dict() for r in runs]}
    (run_dir / "run.json").write_text(json.dumps(payload, indent=2, default=str))
    print()
    print(f"  {dim('files')}  {run_dir}")
    if args.json:
        print(json.dumps(payload, indent=2, default=str))

    if args.open and runs:
        opener = {"darwin": "open", "win32": "start"}.get(sys.platform, "xdg-open")
        for run in runs:
            for path in run.images.values():
                subprocess.run([opener, str(path)], check=False)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
