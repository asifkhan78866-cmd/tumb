"""Method 2 datasets — modality-explicit, label-discovering, volume-grouped.

Two rules specific to Method 2:

1. **Labels are discovered, never assumed.** The SPECT dataset's class folders
   are read off disk and checked against the configured class list. A mismatch
   is a hard error naming exactly what was found, because inventing a mapping
   would fabricate the labels the whole evaluation rests on.
2. **Modalities are never merged.** ``--modality`` selects one source. Training
   on SPECT produces a SPECT model; training on MRI produces an MRI model and
   says so in the checkpoint. There is no code path that concatenates them into
   a single training set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from backend.methods.common.splits import brats_volume_id
from backend.methods.method2 import config as m2
from backend.methods.method2.features import build_inputs
from backend.methods.method2.transforms import M2Transform, preprocess, to_grayscale
from backend.utils.preprocessing import Augmentor

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm")

_ALIASES = {
    "normal": "normal", "notumor": "normal", "no_tumor": "normal", "healthy": "normal",
    "glioma": "glioma", "glioma_tumor": "glioma",
    "meningioma": "meningioma", "meningioma_tumor": "meningioma",
    "pituitary": "pituitary", "pituitary_tumor": "pituitary",
}


class DatasetError(RuntimeError):
    """Raised when a dataset cannot support Method 2's task honestly."""


def _canonical(name: str) -> Optional[str]:
    return _ALIASES.get(name.strip().lower().replace(" ", "_").replace("-", "_"))


# --------------------------------------------------------------------------- #
# Classification (SPECT, or MRI as an explicit fallback)
# --------------------------------------------------------------------------- #
def discover_classification_samples(root: Path, modality: str) -> tuple[list[tuple[str, int]], dict]:
    """Discover ``(path, class_index)`` pairs and report exactly what was found."""
    root = Path(root)
    if not root.exists():
        raise DatasetError(
            f"{root} does not exist. Point {'SPECT_DATASET_PATH' if modality.upper() == 'SPECT' else 'BRI_DATASET_PATH'} "
            f"at a directory of class-named folders "
            f"({', '.join(m2.CLASS_NAMES)})."
        )

    samples: list[tuple[str, int]] = []
    found: dict[str, int] = {}
    unknown: dict[str, int] = {}

    for path in root.rglob("*"):
        if not (path.is_file() and path.suffix.lower() in IMAGE_EXTS):
            continue
        cls = _canonical(path.parent.name)
        if cls is None:
            unknown[path.parent.name] = unknown.get(path.parent.name, 0) + 1
            continue
        found[cls] = found.get(cls, 0) + 1
        samples.append((str(path), m2.CLASS_NAMES.index(cls)))

    if not samples:
        raise DatasetError(
            f"No class-named folders under {root}. Method 2 discovers its labels "
            f"from directory names and will not guess them.\n"
            f"  expected one of: {sorted(set(_ALIASES))}\n"
            f"  found instead:   {sorted(unknown)[:15] or '(no image files at all)'}"
        )

    missing = sorted(set(m2.CLASS_NAMES) - set(found))
    if missing:
        raise DatasetError(
            f"{root} provides only {sorted(found)}; classes {missing} are absent. "
            f"Training a {m2.NUM_CLASSES}-class DCN on a subset would produce a model "
            f"structurally incapable of predicting the missing classes. Either supply "
            f"the full class set or reduce METHOD2 class configuration deliberately."
        )

    return samples, {
        "root": str(root),
        "modality": modality,
        "num_samples": len(samples),
        "class_counts": found,
        "ignored_folders": sorted(unknown)[:15],
        "warnings": (
            []
            if modality.upper() == "SPECT"
            else [
                f"Trained on {modality} data, not SPECT. The resulting model is a "
                f"{modality} model; its numbers do not describe SPECT performance."
            ]
        ),
    }


def classification_group_of(sample: tuple[str, int]) -> str:
    """Group by the immediate parent folder when it is a study/subject directory,
    otherwise by filename stem (the same limitation Method 1 documents for BRI)."""
    p = Path(sample[0])
    if _canonical(p.parent.name) is None:
        return f"{p.parent.name}".lower()
    return p.stem.lower().rsplit("_", 1)[0]


class SpectClassificationDataset(Dataset):
    """Filtered slice + per-region uptake maps + descriptor, ready for the DCN.

    When no segmentation model is supplied the region map is all background and
    the descriptors are zero — the caller records that in the run card rather
    than the dataset silently substituting something.
    """

    def __init__(
        self,
        samples: Sequence[tuple[str, int]],
        spec: M2Transform,
        seg_model=None,
        device=None,
        augment: bool = False,
    ):
        self.samples = list(samples)
        self.spec = spec
        self.seg_model = seg_model
        self.device = device
        self.aug = Augmentor(vflip=0.0, seed=None) if augment else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            image = np.zeros((self.spec.image_size, self.spec.image_size), np.float32)
        else:
            image = preprocess(raw, self.spec)

        if self.aug is not None:
            image = self.aug((image * 255).astype(np.uint8)).astype(np.float32) / 255.0

        if self.seg_model is not None:
            with torch.no_grad():
                x = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(self.device)
                label_map = self.seg_model(x).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        else:
            label_map = np.zeros_like(image, dtype=np.uint8)

        channels, descriptor = build_inputs(image, label_map, m2.NUM_REGIONS)
        return (
            torch.from_numpy(np.ascontiguousarray(channels)).float(),
            torch.from_numpy(np.ascontiguousarray(descriptor)).float(),
            label,
        )


# --------------------------------------------------------------------------- #
# Multi-class segmentation (BraTS)
# --------------------------------------------------------------------------- #
def discover_brats_h5(root: Path) -> list[str]:
    root = Path(root)
    if not root.exists():
        return []
    data_dirs = [p for p in root.rglob("content/data") if p.is_dir()]
    if not data_dirs:
        any_h5 = list(root.rglob("volume_*_slice_*.h5"))
        if not any_h5:
            return []
        data_dirs = [any_h5[0].parent]
    return [str(p) for p in sorted(data_dirs[0].glob("volume_*_slice_*.h5"))]


def segmentation_group_of(path: str) -> str:
    return brats_volume_id(path)


class MultiClassSegmentationDataset(Dataset):
    """BraTS slices with a 4-way label map: background + 3 tumour sub-regions.

    The BraTS per-slice mirror stores ``mask`` as ``(H, W, 3)`` one-hot channels
    for necrotic core / edema / enhancing tumour. They are collapsed into a
    single integer label map in channel order, which is what
    ``METHOD2_SEG_CLASSES`` documents. Overlapping channels (which BraTS does not
    produce) would resolve to the highest index.
    """

    def __init__(self, samples: Sequence[str], spec: M2Transform, augment: bool = False):
        self.samples = list(samples)
        self.spec = spec
        self.aug = Augmentor(vflip=0.0, seed=None) if augment else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import h5py

        with h5py.File(self.samples[idx], "r") as h:
            image, mask = h["image"][:], h["mask"][:]

        img = cv2.normalize(image[:, :, 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        label_map = np.zeros(mask.shape[:2], dtype=np.uint8)
        for channel in range(min(mask.shape[2], m2.NUM_REGIONS)):
            label_map[mask[:, :, channel] > 0] = channel + 1

        size = self.spec.image_size
        img = cv2.resize(to_grayscale(img), (size, size), interpolation=cv2.INTER_LINEAR)
        label_map = cv2.resize(label_map, (size, size), interpolation=cv2.INTER_NEAREST)

        if self.aug is not None:
            img, label_map = self.aug(img, label_map)

        img = img.astype(np.float32) / 255.0
        mn, mx = float(img.min()), float(img.max())
        if mx - mn > 1e-6:
            img = (img - mn) / (mx - mn)

        return (
            torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0),
            torch.from_numpy(np.ascontiguousarray(label_map)).long(),
        )
