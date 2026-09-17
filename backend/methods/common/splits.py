"""Group-aware dataset splitting.

The single most damaging mistake in a slice-based medical imaging pipeline is
splitting *slices* at random: adjacent slices from one patient volume are nearly
identical, so a random split puts near-duplicates on both sides and the reported
score measures memorisation rather than generalisation.

Everything here therefore splits on a **group key** (patient / volume id). A
group never appears in more than one split. Splitting is deterministic for a
given ``seed`` and independent of the order in which samples are discovered, so
two processes that enumerate the same dataset produce the same split.

This module is deliberately dependency-free (stdlib only) so it can be unit
tested without torch, numpy or any dataset on disk.
"""
from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path
from typing import Callable, Hashable, Iterable, Sequence, TypeVar

T = TypeVar("T")

__all__ = [
    "brats_volume_id",
    "bri_patient_id",
    "group_split",
    "group_train_val_test_split",
    "split_summary",
]


# --------------------------------------------------------------------------- #
# Group-key extractors
# --------------------------------------------------------------------------- #
_BRATS_VOLUME_RE = re.compile(r"(volume_\d+)", re.IGNORECASE)
_BRATS_SUBJECT_RE = re.compile(r"(BraTS[\w-]*?_\d+)", re.IGNORECASE)


def brats_volume_id(path: str | Path) -> str:
    """Return the volume/subject id for a BraTS sample path.

    Handles both the per-slice HDF5 mirror (``volume_37_slice_102.h5``) and the
    NIfTI layout (``BraTS20_Training_037/..._flair.nii``). Falls back to the
    parent directory name, which is correct for per-subject folders.
    """
    p = Path(path)
    m = _BRATS_VOLUME_RE.search(p.name)
    if m:
        return m.group(1).lower()
    for part in (p.name, *(q.name for q in p.parents)):
        m = _BRATS_SUBJECT_RE.search(part)
        if m:
            return m.group(1).lower()
    return p.parent.name.lower()


def bri_patient_id(path: str | Path) -> str:
    """Return a grouping key for the Kaggle brain-MRI classification dataset.

    That dataset ships one file per image with no patient identifier of any
    kind, so a *true* patient-level split is not recoverable from it. Rather
    than pretend otherwise, we group by the filename stem with any trailing
    slice/augmentation counter removed (``Tr-gl_0123_2.jpg`` -> ``tr-gl_0123``),
    which at least keeps obvious near-duplicate series together, and callers are
    expected to surface :data:`BRI_GROUPING_WARNING` alongside any metric
    computed from it.
    """
    stem = Path(path).stem.lower()
    stem = re.sub(r"[_-](\d{1,3})$", "", stem)
    return stem


BRI_GROUPING_WARNING = (
    "The Kaggle brain-MRI classification dataset carries no patient identifiers, "
    "so samples can only be grouped by image identity (Method 1 training groups "
    "byte-identical copies together). Scores from this dataset may "
    "still be optimistic if the same patient appears under unrelated filenames."
)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def _stable_group_order(groups: Iterable[Hashable]) -> list[Hashable]:
    """Deterministic, discovery-order-independent ordering of group keys."""
    return sorted(set(groups), key=lambda g: hashlib.sha1(str(g).encode()).hexdigest())


def group_split(
    samples: Sequence[T],
    group_of: Callable[[T], Hashable],
    fractions: Sequence[float],
    seed: int = 42,
) -> list[list[T]]:
    """Split ``samples`` into len(``fractions``) parts without splitting a group.

    ``fractions`` are relative sizes of every part except the first; the first
    part receives the remainder. Groups are assigned largest-first to the part
    that is furthest below its target share, which keeps the realised sizes
    close to the requested ones even when group sizes are very uneven.
    """
    if not fractions:
        raise ValueError("fractions must not be empty")
    if any(f < 0 for f in fractions):
        raise ValueError("fractions must be non-negative")
    total = sum(fractions)
    if total <= 0:
        raise ValueError("fractions must sum to a positive number")
    targets = [f / total for f in fractions]

    by_group: dict[Hashable, list[T]] = {}
    for s in samples:
        by_group.setdefault(group_of(s), []).append(s)

    order = _stable_group_order(by_group)
    rng = random.Random(seed)
    rng.shuffle(order)
    # Largest groups first so they cannot overshoot a small part at the end.
    order.sort(key=lambda g: -len(by_group[g]))

    n_total = len(samples)
    parts: list[list[T]] = [[] for _ in targets]
    for g in order:
        members = by_group[g]
        # Pick the part with the largest shortfall against its target count.
        deficits = [t * n_total - len(p) for t, p in zip(targets, parts)]
        best = max(range(len(parts)), key=lambda i: deficits[i])
        parts[best].extend(members)
    return parts


def group_train_val_test_split(
    samples: Sequence[T],
    group_of: Callable[[T], Hashable],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[T], list[T], list[T]]:
    """Convenience wrapper returning ``(train, val, test)`` group-disjoint lists.

    The test part is the held-out set: it must never be touched by training,
    early stopping or hyper-parameter optimisation.
    """
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")
    train_frac = 1.0 - val_frac - test_frac
    train, val, test = group_split(
        samples, group_of, [train_frac, val_frac, test_frac], seed=seed
    )
    return train, val, test


def split_summary(
    parts: dict[str, Sequence[T]], group_of: Callable[[T], Hashable]
) -> dict:
    """Describe a split for the training run card, including a leakage check."""
    groups = {name: {group_of(s) for s in part} for name, part in parts.items()}
    names = list(groups)
    overlaps: dict[str, list[str]] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = groups[a] & groups[b]
            if shared:
                overlaps[f"{a}|{b}"] = sorted(str(g) for g in shared)[:20]
    return {
        "counts": {name: len(part) for name, part in parts.items()},
        "group_counts": {name: len(g) for name, g in groups.items()},
        "group_overlap": overlaps,
        "leak_free": not overlaps,
    }
