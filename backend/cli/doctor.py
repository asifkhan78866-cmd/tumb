"""Check whether this copy of the project can actually run.

    python -m backend.cli.doctor

Written for the case where someone cloned the repository onto a machine that
has never seen the project: it reports what is present, what is missing, and
the one command that fixes each missing thing — rather than letting the first
prediction fail with an import error or a silent "not trained".

What it is careful about: a clone contains the trained checkpoints, but not the
datasets, and not the ImageNet backbones Method 1 downloads at first use. Those
absences are normal and are reported as such; only the ones that stop a
prediction are errors.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

OK, WARN, BAD = "ok", "warn", "missing"
_SYMBOL = {OK: "✓", WARN: "·", BAD: "✗"}
_COLOUR = {OK: "32", WARN: "33", BAD: "31"}
_TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def line(state: str, label: str, detail: str = "", fix: str = "") -> bool:
    mark = _SYMBOL[state]
    if _TTY:
        mark = f"\033[{_COLOUR[state]}m{mark}\033[0m"
    print(f"  {mark} {label:<46}{detail}")
    if fix:
        print(f"      {'→':<2} {fix}")
    return state != BAD


def check_python() -> bool:
    version = "%d.%d.%d" % sys.version_info[:3]
    in_venv = sys.prefix != sys.base_prefix
    good = sys.version_info[:2] >= (3, 10)
    return line(OK if good else BAD, "Python", f"{version}"
                f"{'  (virtual environment)' if in_venv else '  (not a virtualenv)'}",
                "" if good else "scripts/setup.sh")


def check_imports() -> bool:
    required = {"torch": "deep learning", "cv2": "image handling", "sklearn": "the SVM heads",
                "joblib": "loading the SVM bundles", "fastapi": "the API",
                "matplotlib": "plots", "scipy": "volume resampling"}
    missing = []
    for module, purpose in required.items():
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(f"{module} ({purpose})")
    if missing:
        return line(BAD, "Python packages", f"{len(missing)} missing: {', '.join(missing)}",
                    ".venv/bin/pip install -r backend/requirements.txt")
    import torch

    return line(OK, "Python packages", f"all present · torch {torch.__version__}")


def check_device() -> bool:
    from backend import config

    device = str(config.DEVICE)
    note = "GPU" if device.startswith(("cuda", "mps")) else "CPU — slower, identical results"
    return line(OK, "Compute device", f"{device}  ({note})")


def check_methods() -> bool:
    from backend.methods.registry import list_methods, runtime_status

    every_one = True
    for spec in sorted(list_methods(), key=lambda s: s.display_number):
        status = runtime_status(spec)
        missing = [role for role, w in status["weights"].items() if not w["present"]]
        if not missing:
            detail = "checkpoints present"
            state = OK
        else:
            detail = f"no {', '.join(missing)} checkpoint"
            state = BAD if "classification" in missing else WARN
        # Name the command that rebuilds the missing piece, not whatever the
        # method happens to list first — for Method 2 that is a Red Fox search,
        # which does not produce a classifier.
        fix = ""
        if missing:
            wanted = ("train_segmentation",) if missing == ["segmentation"] else \
                     ("train_classifier", "train_fusion")
            fix = next((e for e in spec.training_entrypoints if any(w in e for w in wanted)),
                       spec.training_entrypoints[0])
        every_one &= line(state, f"Method {spec.display_number} · {spec.short_name}", detail, fix)
    return every_one


def check_metrics() -> bool:
    from backend.methods.common.metrics_store import metrics_for
    from backend.methods.registry import METHOD_IDS

    unevaluated = [m for m in METHOD_IDS if metrics_for(m).get("accuracy") is None]
    if unevaluated:
        return line(WARN, "Recorded metrics", f"{len(unevaluated)} method(s) have none",
                    "they will show 'not evaluated' rather than a number")
    return line(OK, "Recorded metrics", "all four carry a held-out evaluation")


def check_backbones() -> bool:
    """Method 1 rebuilds four ImageNet backbones from torchvision at load time.

    The repository ships its trained head, not those backbones: they are ~800 MB
    of third-party weights that torchvision downloads once and caches. Until
    that happens, Method 1 needs an internet connection the first time it runs.
    """
    import torch

    cache = Path(torch.hub.get_dir()) / "checkpoints"
    wanted = {"googlenet": "googlenet", "vgg16": "vgg16", "resnet50": "resnet50",
              "alexnet": "alexnet"}
    have = {k for k, stem in wanted.items()
            if cache.is_dir() and any(p.name.startswith(stem) for p in cache.glob("*.pth"))}
    if len(have) == len(wanted):
        return line(OK, "ImageNet backbones", f"cached in {cache}")
    return line(WARN, "ImageNet backbones",
                f"{len(have)}/{len(wanted)} cached — Method 1 downloads the rest (~800 MB) on first use",
                "needs an internet connection once; every other method works offline")


# The images the documented commands name. They are committed, so a clone has
# them even with no dataset — which is the point of checking for these rather
# than counting whatever happens to be in data/.
SAMPLES = (
    "data/bri/archive/Testing/meningioma/Te-me_0014.jpg",
    "data/bri/archive/Testing/glioma/Te-gl_0012.jpg",
    "data/spect_mri/SPECT/10017.png",
)


def check_samples() -> bool:
    from backend import config

    present = [s for s in SAMPLES if (config.ROOT_DIR / s).exists()]
    if not present:
        return line(WARN, "Sample images", "none of the documented examples are here",
                    "git pull, or pass your own image to run_methods")
    state = OK if len(present) == len(SAMPLES) else WARN
    return line(state, "Sample images", f"{len(present)}/{len(SAMPLES)} documented examples present")


def check_dataset() -> bool:
    from backend import config

    root = config.BRI_DATASET_PATH
    counts = {part: sum(1 for _ in (root / part).glob("*/*.jpg"))
              for part in ("Training", "Testing") if (root / part).is_dir()}
    total = sum(counts.values())
    if total > 1000:
        return line(OK, "Full dataset", f"{total} images at {root}")
    return line(WARN, "Full dataset", "not downloaded (predictions work; training does not)",
                "scripts/download_dataset.sh")


def check_frontend() -> bool:
    from backend import config

    if not (config.ROOT_DIR / "frontend/node_modules").is_dir():
        return line(WARN, "Frontend packages", "not installed",
                    "cd frontend && npm install")
    node = shutil.which("node")
    return line(OK, "Frontend packages", f"installed{f' · node {node}' if node else ''}")


def main() -> int:
    print()
    print("  NeuroSeg AI — what this copy can run")
    print("  " + "─" * 84)

    essentials = [check_python(), check_imports()]
    if not all(essentials):
        print()
        print("  Fix the above first; nothing else can be checked until then.")
        print()
        return 1

    essentials.append(check_device())
    print()
    essentials.append(check_methods())
    essentials.append(check_metrics())
    essentials.append(check_backbones())
    print()
    check_samples()
    check_dataset()
    check_frontend()

    print()
    if all(essentials):
        print("  Ready. Try:")
        print("    .venv/bin/python -m backend.cli.run_methods "
              "data/bri/archive/Testing/meningioma/Te-me_0014.jpg")
    else:
        print("  Something a prediction needs is missing — see the arrows above.")
    print()
    return 0 if all(essentials) else 1


if __name__ == "__main__":
    raise SystemExit(main())
