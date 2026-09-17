"""The BRI (Kaggle brain-tumour MRI) split shared by every MRI classifier.

The U-Net–ConvLSTM–SFLA, transfer-learning and ZFNet methods are trained and
tested on exactly the same images, so their numbers are comparable:

* the dataset's own ``Training/`` / ``Testing/`` split is kept;
* training copies byte-identical to a test image are dropped (the test set is
  never modified);
* validation is carved from ``Training/`` only, grouped by content hash so an
  image and its exact copy cannot straddle train and validation.

The split is image-level: the dataset has no patient identifiers.

Also provides a disk cache for deterministic per-image preprocessing, computed
once with a process pool instead of every epoch.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from backend import config
from backend.methods.common.splits import group_split, group_train_val_test_split, split_summary

__all__ = ["BRISplit", "prepare_bri_split", "cached_arrays", "SPLIT_DESCRIPTION"]

SPLIT_DESCRIPTION = (
    "dataset Training/Testing split; validation carved from Training only; "
    "image-level split; no patient identifiers"
)


@dataclass
class BRISplit:
    ds_meta: dict
    split_kind: str
    summary: dict
    train: list
    val: list
    test: list
    removed_test_duplicates: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def prepare_bri_split(
    data_root=None,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    ignore_official_split: bool = False,
    verbose: bool = True,
    tag: str = "data",
) -> BRISplit:
    """Discover, de-leak and split the BRI dataset. Raises DatasetError on bad data."""
    from backend.methods.method1 import config as m1
    from backend.methods.method1.datasets import (
        DatasetError,
        class_distribution,
        content_group_of,
        discover_classification_samples,
        drop_test_duplicates,
        file_digest,
        partition_by_official_split,
    )

    warnings: list[str] = []
    samples, ds_meta = discover_classification_samples(data_root or m1.BRI_PATH)
    warnings.extend(ds_meta.get("warnings", []))

    digests = {p: file_digest(p) for p, _ in samples}
    group_of = content_group_of(digests)
    removed: list[str] = []

    if ds_meta.get("has_official_split") and not ignore_official_split:
        train_pool, test_s = partition_by_official_split(samples)
        train_pool, removed = drop_test_duplicates(train_pool, test_s, digests)
        if removed:
            warnings.append(
                f"Dropped {len(removed)} Training images that are byte-identical to a "
                f"Testing image, so the test score is not inflated by memorised copies. "
                f"The Testing set itself was not modified."
            )
        train_s, val_s = group_split(train_pool, group_of, [1.0 - val_frac, val_frac], seed=config.SEED)
        # Byte-identical copies are additionally kept on one side of train/val;
        # that is not patient-level separation.
        split_kind = SPLIT_DESCRIPTION
    else:
        train_s, val_s, test_s = group_train_val_test_split(
            samples, group_of, val_frac, test_frac, seed=config.SEED
        )
        split_kind = "content-hash grouped random three-way split (no Training/Testing folders)"

    summary = split_summary({"train": train_s, "val": val_s, "test": test_s}, group_of)
    if verbose:
        print(f"[{tag}] split: {split_kind}")
        print(f"[{tag}]   counts {summary['counts']} | leak-free={summary['leak_free']}"
              f" | train copies of test images dropped={len(removed)}")
        print(f"[{tag}]   per-class train: {class_distribution(train_s)}")
        print(f"[{tag}]   per-class val  : {class_distribution(val_s)}")
        print(f"[{tag}]   per-class test : {class_distribution(test_s)}")
    if not summary["leak_free"]:
        raise DatasetError(f"split shares images between parts: {summary['group_overlap']}")
    return BRISplit(ds_meta, split_kind, summary, train_s, val_s, test_s, removed, warnings)


def cached_arrays(
    samples: Sequence[tuple[str, int]],
    fn: Callable,
    key: dict,
    name: str,
    workers: Optional[int] = None,
) -> np.ndarray:
    """Return ``np.stack([fn((path, key)) ...])`` for ``samples``, cached on disk.

    ``fn`` must be a module-level function (it is pickled into worker
    processes) taking ``(path, key)``. ``key`` fully describes the
    preprocessing and is part of the cache key, so changing it recomputes.
    """
    from concurrent.futures import ProcessPoolExecutor

    paths = [p for p, _ in samples]
    ordered = sorted(paths)
    digest = hashlib.sha1(json.dumps([key, ordered], sort_keys=True).encode()).hexdigest()[:16]
    cache = config.MODEL_CACHE_DIR / f"{name}_{digest}.npy"
    index = {p: i for i, p in enumerate(ordered)}
    if cache.exists():
        arr = np.load(cache, mmap_mode=None)
        print(f"[cache] hit: {cache.name} ({len(arr)} images)")
    else:
        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        print(f"[cache] preprocessing {len(ordered)} images with {workers} processes…")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            arr = np.stack(list(ex.map(fn, [(p, key) for p in ordered], chunksize=32)))
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, arr)
        print(f"[cache] written: {cache}")
    return arr[[index[p] for p in paths]]
