"""Dataset inventory for Method 1's classification data.

Run this before training to confirm the dataset is where the loader expects it,
that all four classes are present, and that nothing is corrupt:

    python -m backend.methods.method1.inventory
    python -m backend.methods.method1.inventory --data-root ./data/bri/archive

It reads images to check they decode, so a truncated download is caught here
rather than as a confusing failure thirty minutes into a training run.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from backend import config
from backend.methods.method1 import config as m1
from backend.methods.method1.datasets import (
    DatasetError,
    class_distribution,
    discover_classification_samples,
    official_split_of,
    partition_by_official_split,
)


def check_readable(paths: list[str], sample_limit: int | None = None) -> tuple[int, list[str]]:
    """Return ``(corrupt_count, examples)``. Decodes every image unless limited."""
    import cv2

    corrupt: list[str] = []
    targets = paths if sample_limit is None else paths[:sample_limit]
    for p in targets:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None or getattr(img, "size", 0) == 0:
            corrupt.append(p)
    return len(corrupt), corrupt[:10]


def main() -> int:
    ap = argparse.ArgumentParser(description="Inventory Method 1's classification dataset")
    ap.add_argument("--data-root", default=None, help="defaults to BRI_DATASET_PATH")
    ap.add_argument("--skip-decode", action="store_true", help="skip the corrupt-image check")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    root = Path(args.data_root) if args.data_root else m1.BRI_PATH
    print(f"Dataset path: {root.resolve() if root.exists() else root}  "
          f"({'exists' if root.exists() else 'MISSING'})")
    if not root.exists():
        print(
            f"\nERROR: {root} does not exist.\n"
            f"  Place the dataset so that {root}/Training/<class>/ and "
            f"{root}/Testing/<class>/ exist,\n"
            f"  then set BRI_DATASET_PATH in .env to that directory."
        )
        return 1

    try:
        samples, meta = discover_classification_samples(root)
    except DatasetError as exc:
        print(f"\nERROR: {exc}")
        return 1

    train_pool, test = partition_by_official_split(samples)
    train_counts = class_distribution(train_pool)
    test_counts = class_distribution(test)

    print(f"\n{'CLASS':<14}{'TRAIN':>10}{'TEST':>10}{'TOTAL':>10}")
    for name in m1.CLASS_NAMES:
        tr, te = train_counts[name], test_counts[name]
        print(f"{name:<14}{tr:>10}{te:>10}{tr + te:>10}")
    print(f"{'-' * 44}")
    print(f"{'TOTAL':<14}{sum(train_counts.values()):>10}{sum(test_counts.values()):>10}"
          f"{len(samples):>10}")

    exts = Counter(Path(p).suffix.lower() for p, _ in samples)
    print(f"\nExtensions        : {dict(exts)}")
    print(f"Official split    : {'yes (Training/Testing)' if meta['has_official_split'] else 'NO'}")
    print(f"Split level       : {meta['split_level']}")

    # Imbalance, reported on the training pool where it actually affects fitting.
    values = [v for v in train_counts.values() if v]
    if values:
        ratio = max(values) / min(values)
        verdict = "balanced" if ratio < 1.5 else ("mild" if ratio < 2.5 else "SEVERE")
        print(f"Class imbalance   : {ratio:.2f}x max/min on train pool — {verdict}")
        if ratio >= 1.5:
            print("                    (inverse-frequency class weights are applied by default)")

    corrupt_n, examples = (0, [])
    if not args.skip_decode:
        print(f"\nDecoding {len(samples)} images to check for corruption…")
        corrupt_n, examples = check_readable([p for p, _ in samples])
    print(f"Corrupt/unreadable: {corrupt_n}")
    for e in examples:
        print(f"                    {e}")

    for w in meta.get("warnings", []):
        print(f"\n! {w}")

    if args.json:
        print("\n" + json.dumps(
            {
                "path": str(root.resolve()),
                "train": train_counts,
                "test": test_counts,
                "total": len(samples),
                "extensions": dict(exts),
                "has_official_split": meta["has_official_split"],
                "corrupt": corrupt_n,
                "warnings": meta.get("warnings", []),
            },
            indent=2,
        ))

    ok = corrupt_n == 0 and all(train_counts.values()) and all(test_counts.values())
    print(f"\nRESULT: {'dataset is usable' if ok else 'PROBLEMS FOUND — see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
