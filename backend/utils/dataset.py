"""PyTorch datasets built from whatever the auto-downloader produced.

Two dataset types are supported so training runs regardless of which Kaggle
source succeeded:

* ``ClassificationDataset`` — reads class-named folders (glioma / meningioma /
  notumor / pituitary, or yes/no which is mapped to tumor/notumor).
* ``SegmentationDataset`` — uses NIfTI volumes (BraTS) when available, otherwise
  derives weak tumor masks from 2D images via Otsu thresholding so the
  segmentation pipeline is still trainable end-to-end.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from backend import config
from backend.utils.preprocessing import (
    Augmentor,
    apply_clahe,
    normalize,
    remove_noise,
    resize,
    to_grayscale,
)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Map common folder-name variants to canonical class names.
_CLASS_ALIASES = {
    "glioma": "glioma", "glioma_tumor": "glioma",
    "meningioma": "meningioma", "meningioma_tumor": "meningioma",
    "pituitary": "pituitary", "pituitary_tumor": "pituitary",
    "notumor": "notumor", "no_tumor": "notumor", "no": "notumor", "healthy": "notumor",
    "yes": "glioma",  # binary "tumor present" datasets -> generic tumor class
}


def _canonical_class(folder_name: str) -> Optional[str]:
    key = folder_name.strip().lower().replace(" ", "_").replace("-", "_")
    return _CLASS_ALIASES.get(key)


def discover_classification_samples(root: Path) -> list[tuple[str, int]]:
    """Return (image_path, class_index) pairs found under ``root``."""
    samples: list[tuple[str, int]] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            cls = _canonical_class(path.parent.name)
            if cls is not None:
                samples.append((str(path), config.CLASS_NAMES.index(cls)))
    return samples


class ClassificationDataset(Dataset):
    def __init__(self, samples, size: int = config.IMAGE_SIZE, augment: bool = False):
        self.samples = samples
        self.size = size
        self.aug = Augmentor(seed=None) if augment else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.size, self.size), np.uint8)
        img = remove_noise(img)
        img = apply_clahe(img)
        img = resize(img, self.size)
        if self.aug is not None:
            img = self.aug(img)
        img = normalize(img).astype(np.float32)
        x = torch.from_numpy(img).unsqueeze(0)
        return x, label


def _otsu_mask(img_u8: np.ndarray) -> np.ndarray:
    """Weak tumor mask: brightest connected region after Otsu + morphology."""
    blur = cv2.GaussianBlur(img_u8, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    # keep only the largest bright blob (approx. tumor for demo supervision)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        th = np.where(labels == largest, 255, 0).astype(np.uint8)
    return th


def discover_nifti_pairs(root: Path) -> list[tuple[str, str, int]]:
    """Find (flair_volume, seg_volume, slice_index) triples for BraTS data."""
    triples: list[tuple[str, str, int]] = []
    try:
        import nibabel as nib
    except Exception:
        return triples
    for flair in glob.glob(str(root / "**" / "*_flair.nii*"), recursive=True):
        seg = flair.replace("_flair", "_seg")
        if not Path(seg).exists():
            continue
        vol = nib.load(flair)
        depth = vol.shape[2]
        # sample the informative central slices
        for z in range(depth // 4, 3 * depth // 4, 3):
            triples.append((flair, seg, z))
    return triples


class SegmentationDataset(Dataset):
    """Segmentation samples from NIfTI volumes or weak Otsu masks on 2D images."""

    def __init__(self, root: Path, size: int = config.IMAGE_SIZE, augment: bool = False):
        self.size = size
        self.aug = Augmentor(seed=None) if augment else None
        self.nifti = discover_nifti_pairs(root)
        self.images: list[str] = []
        if not self.nifti:
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    self.images.append(str(p))
        self._nib = None
        if self.nifti:
            import nibabel as nib  # noqa
            self._nib = nib

    def __len__(self):
        return len(self.nifti) if self.nifti else len(self.images)

    def _load_nifti(self, idx):
        flair, seg, z = self.nifti[idx]
        img = self._nib.load(flair).get_fdata()[:, :, z]
        msk = self._nib.load(seg).get_fdata()[:, :, z]
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        msk = (msk > 0).astype(np.uint8) * 255
        return img, msk

    def _load_image(self, idx):
        img = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.size, self.size), np.uint8)
        msk = _otsu_mask(img)
        return img, msk

    def __getitem__(self, idx):
        if self.nifti:
            img, msk = self._load_nifti(idx)
        else:
            img, msk = self._load_image(idx)
        img = to_grayscale(img)
        img = remove_noise(img)
        img = apply_clahe(img)
        img = resize(img, self.size)
        msk = resize(msk, self.size)
        if self.aug is not None:
            img, msk = self.aug(img, msk)
        img = normalize(img).astype(np.float32)
        msk = (msk > 127).astype(np.float32)
        x = torch.from_numpy(img).unsqueeze(0)
        y = torch.from_numpy(msk).unsqueeze(0)
        return x, y
