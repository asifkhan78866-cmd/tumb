"""Safe dataset acquisition.

Design rules, all of which the previous version broke in at least one place:

* **Nothing downloads implicitly.** Training scripts no longer trigger a
  multi-gigabyte download as a side effect of being run; you ask for a dataset
  explicitly or you get a readable error telling you how.
* **Nothing is overwritten by surprise.** An existing, non-empty destination is
  left alone unless ``--force`` is passed, and ``--force`` prints exactly what it
  is about to replace.
* **Credentials are never logged.** Only whether a credential is present is
  reported, never any part of its value, and the Kaggle client is configured
  through the environment rather than by writing a token to disk.
* **Downloads are verified.** After extraction the destination is checked for
  the files the dataset is supposed to contain, and a download that produced
  nothing usable is reported as a failure rather than a success.
* **The destination is always printed**, before and after.

Usage:
    python -m backend.utils.dataset_download --list
    python -m backend.utils.dataset_download --kind classification
    python -m backend.utils.dataset_download --kind segmentation --force
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from backend import config


@dataclass(frozen=True)
class DatasetSource:
    key: str
    slug: str
    dest: Path
    description: str
    verify: Callable[[Path], tuple[bool, str]]


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
CLASS_DIRS = {"glioma", "meningioma", "notumor", "pituitary"}


def _verify_classification(root: Path) -> tuple[bool, str]:
    found = {d.name.lower() for d in root.rglob("*") if d.is_dir()} & CLASS_DIRS
    images = sum(1 for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not found:
        return False, f"no class folders found (expected {sorted(CLASS_DIRS)})"
    missing = sorted(CLASS_DIRS - found)
    if missing:
        return False, f"missing class folders: {missing}"
    if images < 100:
        return False, f"only {images} images found — the download looks truncated"
    # Every MRI method relies on the dataset's own split being directly under the root.
    for split in ("Training", "Testing"):
        split_dir = root / split
        if not split_dir.is_dir():
            return False, f"{split}/ not found directly under {root}"
        absent = sorted(c for c in CLASS_DIRS if not (split_dir / c).is_dir())
        if absent:
            return False, f"{split}/ is missing class folders {absent}"
    return True, f"{images} images across {sorted(found)} in Training/ and Testing/"


def _flatten_single_wrapper(dest: Path) -> None:
    """If extraction produced ``dest/<one folder>/{Training,Testing}``, move them up.

    Different packagings of the same Kaggle dataset add a wrapper folder; the
    loaders expect ``Training/`` and ``Testing/`` directly under the root.
    """
    if (dest / "Training").is_dir():
        return
    children = [c for c in dest.iterdir() if not c.name.startswith(".")]
    if len(children) == 1 and children[0].is_dir() and (children[0] / "Training").is_dir():
        wrapper = children[0]
        for item in wrapper.iterdir():
            shutil.move(str(item), str(dest / item.name))
        wrapper.rmdir()
        print(f"[dataset]   flattened wrapper folder '{wrapper.name}/' so Training/ and Testing/ sit at the root")


def _verify_segmentation(root: Path) -> tuple[bool, str]:
    h5 = sum(1 for _ in root.rglob("volume_*_slice_*.h5"))
    nii = sum(1 for _ in root.rglob("*_flair.nii*"))
    if h5:
        volumes = len({p.name.split("_slice_")[0] for p in root.rglob("volume_*_slice_*.h5")})
        if volumes < 2:
            return False, f"only {volumes} volume(s) found — too few for a volume-level split"
        return True, f"{h5} slices across {volumes} volumes"
    if nii:
        return True, f"{nii} NIfTI FLAIR volumes"
    return False, "no BraTS .h5 slices or NIfTI volumes found"


def _verify_spect(root: Path) -> tuple[bool, str]:
    images = sum(
        1 for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".dcm"}
    )
    dirs = {d.name.lower() for d in root.rglob("*") if d.is_dir()}
    if images == 0:
        return False, "no image files found"
    return True, f"{images} images in folders {sorted(dirs)[:8]}"


def _sources() -> dict[str, DatasetSource]:
    return {
        "classification": DatasetSource(
            key="classification",
            slug=config.KAGGLE_CLASSIFICATION,
            dest=config.BRI_DATASET_PATH,
            description="Brain Tumor MRI Dataset — 4 labelled classes (MRI Methods 1, 2 and 3)",
            verify=_verify_classification,
        ),
        "segmentation": DatasetSource(
            key="segmentation",
            slug=config.KAGGLE_SEGMENTATION,
            dest=config.BRATS_DATASET_PATH,
            description="BraTS 2020 mirror — ground-truth tumour masks (both methods' segmentation)",
            verify=_verify_segmentation,
        ),
    }


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def credential_status() -> dict[str, bool]:
    """Report only whether each credential is present. Never its value."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return {
        "KAGGLE_USERNAME": bool(os.getenv("KAGGLE_USERNAME", "").strip()),
        "KAGGLE_KEY": bool(os.getenv("KAGGLE_KEY", "").strip()),
        "kaggle.json": kaggle_json.exists(),
        "HF_TOKEN": bool(os.getenv("HF_TOKEN", "").strip()),
    }


def _authenticate_kaggle():
    """Authenticate the Kaggle client from environment variables or ~/.kaggle."""
    status = credential_status()
    if not ((status["KAGGLE_USERNAME"] and status["KAGGLE_KEY"]) or status["kaggle.json"]):
        raise RuntimeError(
            "No Kaggle credentials found.\n"
            "  Set KAGGLE_USERNAME and KAGGLE_KEY in .env, or place kaggle.json at "
            "~/.kaggle/kaggle.json (chmod 600).\n"
            "  Get a token at https://www.kaggle.com/settings -> Create New API Token."
        )
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:
        raise RuntimeError(
            "The 'kaggle' package is required: pip install kaggle"
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        # Deliberately does not echo the exception's payload, which can contain
        # the credential that was attempted.
        raise RuntimeError(
            f"Kaggle authentication failed ({type(exc).__name__}). Check that your "
            f"username and key are correct and that the token has not been revoked."
        ) from None
    return api


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _is_populated(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def download_one(source: DatasetSource, force: bool = False, dry_run: bool = False) -> bool:
    """Download and verify one dataset. Returns True on success."""
    print(f"\n[dataset] {source.key}: {source.description}")
    print(f"[dataset]   source:      kaggle:{source.slug}")
    print(f"[dataset]   destination: {source.dest}")

    if _is_populated(source.dest):
        ok, detail = source.verify(source.dest)
        if not force:
            print(f"[dataset]   already present ({detail}). Use --force to re-download.")
            return ok
        print(f"[dataset]   --force given; REPLACING existing contents ({detail}).")
        if dry_run:
            print("[dataset]   dry run: nothing removed.")
        else:
            shutil.rmtree(source.dest)

    if dry_run:
        print("[dataset]   dry run: nothing downloaded.")
        return True

    api = _authenticate_kaggle()
    source.dest.mkdir(parents=True, exist_ok=True)
    try:
        print(f"[dataset]   downloading… (this can take a long time and several GB)")
        api.dataset_download_files(source.slug, path=str(source.dest), unzip=True, quiet=False)
    except Exception as exc:
        print(f"[dataset]   FAILED: {type(exc).__name__}: {exc}")
        return False

    if source.key == "classification":
        _flatten_single_wrapper(source.dest)
    ok, detail = source.verify(source.dest)
    if ok:
        print(f"[dataset]   OK — verified: {detail}")
        print(f"[dataset]   extracted to: {source.dest}")
    else:
        print(f"[dataset]   VERIFICATION FAILED: {detail}")
        print(f"[dataset]   the files are at {source.dest}; inspect them before training.")
    return ok


def download_dataset(kind: str = "both", force: bool = False, dry_run: bool = False) -> dict[str, bool]:
    """Download the dataset(s) for ``kind``: classification | segmentation | both."""
    sources = _sources()
    if kind == "both":
        chosen = list(sources.values())
    elif kind in sources:
        chosen = [sources[kind]]
    else:
        raise ValueError(f"Unknown kind {kind!r}. Use: {', '.join(sources)}, or both.")

    results = {s.key: download_one(s, force=force, dry_run=dry_run) for s in chosen}

    if dry_run:
        print("\n[dataset] dry run complete — nothing was downloaded, deleted or written.")
        return results

    summary = summarize_datasets()
    summary_path = config.DATASET_DIR / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[dataset] summary written to {summary_path}")
    return results


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
def summarize_datasets() -> dict:
    """Describe every configured dataset root without downloading anything."""
    roots = {
        "classification (BRI)": config.BRI_DATASET_PATH,
        "segmentation (BraTS)": config.BRATS_DATASET_PATH,
        "spect": config.SPECT_DATASET_PATH,
    }
    verifiers = {
        "classification (BRI)": _verify_classification,
        "segmentation (BraTS)": _verify_segmentation,
        "spect": _verify_spect,
    }
    out: dict[str, dict] = {}
    for name, root in roots.items():
        present = _is_populated(root)
        ok, detail = verifiers[name](root) if present else (False, "not present")
        out[name] = {"path": str(root), "present": present, "usable": ok, "detail": detail}
    return out


def _print_status() -> None:
    print("Credentials (presence only — values are never printed):")
    for name, present in credential_status().items():
        print(f"  {name:18s}: {'set' if present else 'not set'}")
    print("\nDatasets:")
    for name, info in summarize_datasets().items():
        state = "usable" if info["usable"] else ("present" if info["present"] else "missing")
        print(f"  {name:22s} [{state:7s}] {info['path']}")
        print(f"  {'':22s}  {info['detail']}")


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Download brain-tumour datasets. Nothing downloads unless you ask."
    )
    ap.add_argument("--kind", choices=["classification", "segmentation", "both"], default=None)
    ap.add_argument("--force", action="store_true",
                    help="replace an existing dataset (prints what it will remove)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would happen without downloading or deleting")
    ap.add_argument("--list", action="store_true",
                    help="show credential and dataset status, then exit")
    args = ap.parse_args(argv)

    if args.list or args.kind is None:
        _print_status()
        if args.kind is None and not args.list:
            print("\nNothing to do. Pass --kind classification|segmentation|both to download.")
        return 0

    try:
        results = download_dataset(kind=args.kind, force=args.force, dry_run=args.dry_run)
    except (RuntimeError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        return 0

    failed = [k for k, ok in results.items() if not ok]
    if failed:
        print(f"\n[dataset] NOT usable: {failed}", file=sys.stderr)
        return 1
    print("\n[dataset] all requested datasets are present and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
