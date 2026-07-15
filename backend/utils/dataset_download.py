"""Automatic dataset acquisition via the Kaggle API.

Order of preference:
  1. BraTS 2023/2020 mirror  (config.KAGGLE_PRIMARY)
  2. Brain MRI Images for Brain Tumor Detection (config.KAGGLE_FALLBACK_1)
  3. Brain Tumor MRI Dataset (config.KAGGLE_FALLBACK_2)

Requires Kaggle credentials in ``~/.kaggle/kaggle.json`` (or the KAGGLE_USERNAME /
KAGGLE_KEY environment variables). Running this module directly downloads,
extracts, verifies and summarises the dataset.

Usage:
    python -m backend.utils.dataset_download
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

from backend import config


def _ensure_kaggle():
    """Import and authenticate the Kaggle API, giving a helpful error otherwise."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The 'kaggle' package is required. Install with `pip install kaggle` and "
            "place your kaggle.json in ~/.kaggle/."
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise RuntimeError(
            "Kaggle authentication failed. Set KAGGLE_USERNAME/KAGGLE_KEY or put "
            "kaggle.json in ~/.kaggle/ (chmod 600)."
        ) from exc
    return api


def _download(api, slug: str, dest: Path) -> bool:
    """Download and unzip a Kaggle dataset. Returns True on success."""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        print(f"[dataset] Attempting download: {slug}")
        api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
        print(f"[dataset] Downloaded and extracted to {dest}")
        return True
    except Exception as exc:
        print(f"[dataset] Failed to download {slug}: {exc}")
        return False


def _download_group(api, target: Path, label: str, slugs: list[str], force: bool) -> bool:
    """Download the first slug in ``slugs`` that succeeds for a given task group."""
    marker = target / f".downloaded_{label}"
    if marker.exists() and not force:
        print(f"[dataset] {label} data already present (use --force to re-download).")
        return True
    for slug in slugs:
        sub = target / slug.split("/")[-1]
        if _download(api, slug, sub):
            marker.write_text(slug)
            return True
    return False


def download_dataset(kind: str = "both", force: bool = False) -> Path:
    """Download datasets for the requested task(s).

    Parameters
    ----------
    kind : "classification", "segmentation", or "both" (default).
        * classification -> Brain Tumor MRI Dataset (4 labelled classes),
          falling back to the Brain MRI detection dataset.
        * segmentation   -> BraTS (ground-truth masks).
    force : re-download even if a marker file exists.

    Returns the dataset directory. Both groups extract into ``backend/dataset``
    so the training scripts can discover whichever data they need.
    """
    target = config.DATASET_DIR
    api = _ensure_kaggle()

    plan: list[tuple[str, list[str]]] = []
    if kind in ("classification", "both"):
        plan.append(("classification",
                     [config.KAGGLE_CLASSIFICATION, config.KAGGLE_CLASSIFICATION_FALLBACK]))
    if kind in ("segmentation", "both"):
        plan.append(("segmentation", [config.KAGGLE_SEGMENTATION]))
    if not plan:
        raise ValueError(f"Unknown kind={kind!r}. Use classification|segmentation|both.")

    results = {label: _download_group(api, target, label, slugs, force) for label, slugs in plan}

    if not any(results.values()):
        raise RuntimeError(
            "All dataset downloads failed. Check your Kaggle credentials "
            "(backend/.env or ~/.kaggle/kaggle.json) and your network connection."
        )
    for label, ok in results.items():
        if not ok:
            print(f"[dataset] WARNING: {label} dataset could not be downloaded.")

    summary = summarize_dataset(target)
    (target / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[dataset] Summary:")
    print(json.dumps(summary, indent=2))
    return target


def _count(patterns, root: Path) -> int:
    total = 0
    for pat in patterns:
        total += len(list(root.rglob(pat)))
    return total


def summarize_dataset(root: Path | None = None) -> dict:
    """Walk the dataset directory and produce an integrity summary."""
    root = Path(root or config.DATASET_DIR)
    image_exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    volume_exts = ("*.nii", "*.nii.gz")

    # Class folders: any directory whose name matches a known class label.
    class_dirs = {}
    for d in root.rglob("*"):
        if d.is_dir():
            name = d.name.lower()
            for cls in config.CLASS_NAMES:
                if cls in name.replace("-", "").replace("_", ""):
                    class_dirs.setdefault(cls, 0)
                    class_dirs[cls] += _count(image_exts, d)

    n_images = _count(image_exts, root)
    n_masks = len([p for p in root.rglob("*") if "seg" in p.name.lower() or "mask" in p.name.lower()])
    n_volumes = _count(volume_exts, root)

    # "Patients" for BraTS-style volume datasets = number of subject folders.
    patient_dirs = {p.parent for p in root.rglob("*_flair.nii*")}
    n_patients = len(patient_dirs) if patient_dirs else len([d for d in root.iterdir() if d.is_dir()])

    return {
        "root": str(root),
        "num_patients": n_patients,
        "classes": {config.CLASS_LABELS.get(k, k): v for k, v in class_dirs.items()} or "n/a (segmentation-only dataset)",
        "image_count": n_images,
        "mask_count": n_masks,
        "volume_count": n_volumes,
        "integrity_ok": (n_images > 0) or (n_volumes > 0),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Auto-download brain-tumor datasets from Kaggle")
    ap.add_argument("--kind", choices=["classification", "segmentation", "both"], default="both")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()
    try:
        download_dataset(kind=args.kind, force=args.force)
    except (RuntimeError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
