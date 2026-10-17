"""Evaluate the MRI–SPECT fusion stage on every paired slice.

    python -m backend.methods.method2.training.evaluate_fusion
    python -m backend.methods.method2.training.evaluate_fusion --samples 8

This is an evaluation, not a training run: the fusion is a deterministic
multi-scale transform with nothing to fit, so there is no split to protect and
no checkpoint to write. Every pair in the dataset is used.

The dataset carries no diagnosis labels, so **no classification claim is made
here and none can be**. What is measured is whether the fused image kept what
its two sources carried, using reference-free fusion metrics, against the
trivial averaging baseline so the numbers have something to be compared with.

Both the fused and the baseline numbers are written, per pair and aggregated,
along with a handful of MRI | SPECT | fused panels for the gallery.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import cv2

from backend import config
from backend.methods.common.metrics_store import update_metrics
from backend.methods.common.runcard import RunCard, write_run_card
from backend.methods.method2 import fusion as F
from backend.methods.registry import METHOD2

TAG = "m2-fusion"
LOGS_DIR = config.LOGS_DIR / "method2"
SAMPLES_DIR = LOGS_DIR / "fusion_samples"
RESULT_PATH = LOGS_DIR / "fusion_metrics.json"
METRIC_KEYS = ("entropy", "std", "spatial_frequency", "mutual_information_mri",
               "mutual_information_spect", "ssim_mri", "ssim_spect", "qabf")


def case_of(name: str) -> str:
    """The case id a filename belongs to: 10007.png -> case 10, slice 007."""
    m = re.match(r"(\d+?)(\d{3})\.\w+$", name)
    return m.group(1) if m else name.split(".")[0]


def discover_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Files present in both modality folders, matched by name."""
    mri_dir, spect_dir = root / "MRI", root / "SPECT"
    if not (mri_dir.is_dir() and spect_dir.is_dir()):
        return []
    spect = {p.name: p for p in spect_dir.iterdir() if p.is_file() and not p.name.startswith(".")}
    pairs = [(p, spect[p.name]) for p in sorted(mri_dir.iterdir())
             if p.is_file() and not p.name.startswith(".") and p.name in spect]
    return pairs


def aggregate(rows: list[dict]) -> dict:
    out = {}
    for key in METRIC_KEYS:
        values = [r[key] for r in rows]
        out[key] = {
            "mean": round(statistics.fmean(values), 4),
            "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure the MRI–SPECT fusion stage")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--levels", type=int, default=F.DEFAULT_LEVELS)
    ap.add_argument("--samples", type=int, default=6, help="MRI|SPECT|fused panels to save")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser() if args.data_root else config.PAIRED_MRI_SPECT_PATH
    pairs = discover_pairs(root)
    if not pairs:
        print(f"ERROR: no MRI/SPECT pairs under {root}.\n"
              f"Expected two folders, MRI/ and SPECT/, holding identically named images.\n"
              f"Set PAIRED_MRI_SPECT_PATH if the dataset lives elsewhere.")
        return 1

    cases = sorted({case_of(p.name) for p, _ in pairs}, key=lambda c: (len(c), c))
    print(f"[{TAG}] {len(pairs)} paired slices from {len(cases)} cases in {root}")

    started = time.perf_counter()
    fused_rows, baseline_rows, per_pair, notes = [], [], [], set()
    for mri_path, spect_path in pairs:
        mri = cv2.imread(str(mri_path), cv2.IMREAD_GRAYSCALE)
        spect = cv2.imread(str(spect_path), cv2.IMREAD_COLOR)
        if mri is None or spect is None:
            print(f"[{TAG}] unreadable pair skipped: {mri_path.name}")
            continue
        pair = F.fuse(mri, spect, levels=args.levels)
        notes.update(pair.notes)
        fused_m = F.metrics(pair.mri, pair.spect_luminance, pair.fused_luminance)
        base_m = F.metrics(pair.mri, pair.spect_luminance,
                           F.average_fuse(pair.mri, pair.spect_luminance))
        fused_rows.append(fused_m)
        baseline_rows.append(base_m)
        per_pair.append({"file": mri_path.name, "case": case_of(mri_path.name),
                         "laplacian": fused_m, "average_baseline": base_m})

    elapsed = time.perf_counter() - started
    fused_agg, base_agg = aggregate(fused_rows), aggregate(baseline_rows)
    print(f"[{TAG}] fused {len(fused_rows)} pairs in {elapsed:.1f}s "
          f"({elapsed / max(1, len(fused_rows)) * 1000:.1f} ms/pair)\n")
    print(f"[{TAG}] {'metric':<26} {'Laplacian fusion':>18} {'averaging baseline':>20}")
    for key in METRIC_KEYS:
        print(f"[{TAG}] {key:<26} {fused_agg[key]['mean']:>10.4f} ± {fused_agg[key]['std']:<5.3f} "
              f"{base_agg[key]['mean']:>12.4f} ± {base_agg[key]['std']:<5.3f}")

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for old in SAMPLES_DIR.glob("*.png"):
        old.unlink()
    step = max(1, len(pairs) // max(1, args.samples))
    saved = []
    for mri_path, spect_path in pairs[::step][: args.samples]:
        mri = cv2.imread(str(mri_path), cv2.IMREAD_GRAYSCALE)
        spect = cv2.imread(str(spect_path), cv2.IMREAD_COLOR)
        pair = F.fuse(mri, spect, levels=args.levels)
        out = SAMPLES_DIR / f"case{case_of(mri_path.name)}_{mri_path.stem}.png"
        cv2.imwrite(str(out), F.side_by_side(pair))
        saved.append(out.name)
    print(f"\n[{TAG}] {len(saved)} sample panels -> {SAMPLES_DIR}")

    warnings = [
        "This dataset carries no diagnosis labels, so the fusion stage is measured by "
        "reference-free fusion quality only. No classification accuracy is claimed for it, and "
        "the method's four-class prediction still comes from its trained MRI branch.",
        "The pairs are supplied pre-registered. This stage fuses them; it does not register "
        "them, and a misregistered pair would fuse into a misaligned image.",
        f"{len(cases)} cases only ({len(pairs)} slices). These numbers describe this dataset, "
        f"not MRI–SPECT fusion in general.",
        "Mutual information with the SPECT luminance is *lower* than plain averaging "
        f"({fused_agg['mutual_information_spect']['mean']:.3f} vs "
        f"{base_agg['mutual_information_spect']['mean']:.3f}) and that is the intended "
        "trade-off, not a defect: the detail rule hands the luminance to whichever source "
        "has real structure, which is usually the MRI. The SPECT measurement itself is its "
        "colour, and the chrominance channels are carried through untouched.",
        *sorted(notes),
    ]

    section = {
        "stage": "fusion",
        "method": "Laplacian pyramid fusion of MRI with the SPECT luminance; SPECT chrominance "
                  "carried through unchanged",
        "trained": False,
        "deterministic": True,
        "dataset": str(root),
        "pairs": len(fused_rows),
        "cases": len(cases),
        "pyramid_levels": args.levels,
        "metrics": fused_agg,
        "average_baseline": base_agg,
        "qabf_gain_over_baseline": round(fused_agg["qabf"]["mean"] - base_agg["qabf"]["mean"], 4),
        "evaluation_time_s": round(elapsed, 2),
        "ms_per_pair": round(elapsed / max(1, len(fused_rows)) * 1000, 2),
        "warnings": warnings,
    }
    update_metrics(METHOD2.method_id, "fusion", section)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps({
        **section,
        "metric_definitions": {
            "entropy": "information content of the fused image (bits)",
            "std": "contrast, as the standard deviation of fused intensity",
            "spatial_frequency": "detail: row/column gradient energy",
            "mutual_information_mri": "information the fused image shares with the MRI",
            "mutual_information_spect": "information the fused image shares with the SPECT luminance",
            "ssim_mri": "structural similarity to the MRI",
            "ssim_spect": "structural similarity to the SPECT luminance",
            "qabf": "Xydeas & Petrovic edge preservation, 0-1: how much of both sources' "
                    "gradient information survived",
        },
        "cases": cases,
        "samples": saved,
        "per_pair": per_pair,
    }, indent=2, default=str))
    print(f"[{TAG}] metrics -> {RESULT_PATH}")

    write_run_card(
        config.LOGS_DIR / METHOD2.run_card_filenames["fusion"],
        RunCard(method_id=METHOD2.method_id, stage="fusion",
                dataset={"root": str(root), "name": "Paired MRI–SPECT slices",
                         "pairs": len(fused_rows), "cases": len(cases),
                         "labels": "none — this dataset has no diagnosis labels"},
                split_strategy={"type": "none — a deterministic transform is not fitted, so "
                                        "every pair is evaluated"},
                random_seed=config.SEED,
                preprocessing={"spect_colour_space": "YCrCb; Cr/Cb carried through unchanged"},
                image_size=256,
                architecture="Laplacian pyramid fusion, local-energy coefficient selection",
                model_config={"levels": args.levels, "base_rule": "weighted average",
                              "detail_rule": "larger local energy wins",
                              "energy_window": F._ENERGY_WINDOW},
                optimizer={"name": "none — deterministic transform"},
                epochs=0, best_epoch=0, metrics=section, checkpoint_path="",
                inference_time_s=round(elapsed / max(1, len(fused_rows)), 6),
                train_duration_s=0.0, model_version="deterministic",
                device="cpu", warnings=warnings),
        root=config.ROOT_DIR,
    )
    print(f"[{TAG}] done. Q_abf {fused_agg['qabf']['mean']:.4f} vs "
          f"{base_agg['qabf']['mean']:.4f} for plain averaging.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
