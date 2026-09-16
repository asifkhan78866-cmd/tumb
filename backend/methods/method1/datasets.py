"""Method 1 datasets — patient/volume-grouped, label-honest, transform-shared.

Three rules are enforced here rather than left to the caller:

1. **No silent relabelling.** A binary tumour/no-tumour dataset is not quietly
   remapped onto four classes; discovery fails with an explanation instead.
2. **No slice-level leakage.** Splits go through
   :mod:`backend.methods.common.splits`, which keeps every patient/volume in a
   single part.
3. **One geometry.** Images are built by
   :mod:`backend.methods.method1.transforms`, the same module inference calls,
   and the ROI crop uses precomputed boxes from the *trained* segmentation model
   so the classifier trains on what it will actually be served.
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from backend import config
from backend.methods.common.splits import BRI_GROUPING_WARNING, bri_patient_id, brats_volume_id
from backend.methods.method1 import config as m1
from backend.methods.method1.transforms import (
    TransformSpec,
    clahe,
    crop_to_roi,
    denoise,
    resize,
    scale_to_unit,
    to_grayscale,
)
from backend.utils.preprocessing import Augmentor

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Folder-name variants that genuinely denote one of the four target classes.
# Binary "yes"/"no" datasets are intentionally absent: see _reject_binary.
_CLASS_ALIASES = {
    "glioma": "glioma", "glioma_tumor": "glioma", "gliomatumor": "glioma",
    "meningioma": "meningioma", "meningioma_tumor": "meningioma", "meningiomatumor": "meningioma",
    "pituitary": "pituitary", "pituitary_tumor": "pituitary", "pituitarytumor": "pituitary",
    "notumor": "notumor", "no_tumor": "notumor", "healthy": "notumor", "normal": "notumor",
}
_BINARY_MARKERS = {"yes", "no", "tumor", "tumour", "brain_tumor", "brain_tumour"}


class DatasetError(RuntimeError):
    """Raised when a dataset cannot support the requested task honestly."""


def _canonical_class(folder_name: str) -> Optional[str]:
    key = folder_name.strip().lower().replace(" ", "_").replace("-", "_")
    return _CLASS_ALIASES.get(key)


# --------------------------------------------------------------------------- #
# Classification discovery
# --------------------------------------------------------------------------- #
def discover_classification_samples(
    root: Path, respect_official_split: bool = True
) -> tuple[list[tuple[str, int]], dict]:
    """Find ``(image_path, class_index)`` pairs under ``root``.

    Returns the samples plus a metadata dict describing what was found. Raises
    :class:`DatasetError` when the directory only supports a binary
    tumour/no-tumour distinction — mapping "yes" onto *glioma*, as the original
    code did, silently fabricates three quarters of the label space.
    """
    root = Path(root)
    if not root.exists():
        raise DatasetError(
            f"{root} does not exist. Download it with "
            f"`python -m backend.utils.dataset_download --kind classification` or point "
            f"BRI_DATASET_PATH at an existing copy."
        )

    samples: list[tuple[str, int]] = []
    matched_dirs: set[str] = set()
    unmatched_dirs: set[str] = set()
    official_split: dict[str, int] = {}

    for path in root.rglob("*"):
        if not (path.is_file() and path.suffix.lower() in IMAGE_EXTS):
            continue
        folder = path.parent.name
        cls = _canonical_class(folder)
        if cls is None:
            unmatched_dirs.add(folder.lower())
            continue
        matched_dirs.add(cls)
        samples.append((str(path), m1.CLASS_NAMES.index(cls)))
        for part in path.parts:
            low = part.lower()
            if low in {"training", "train", "testing", "test"}:
                official_split[low] = official_split.get(low, 0) + 1

    if not samples:
        if unmatched_dirs & _BINARY_MARKERS:
            raise DatasetError(
                f"{root} looks like a binary tumour/no-tumour dataset "
                f"(folders: {sorted(unmatched_dirs & _BINARY_MARKERS)}). Method 1 "
                f"classifies four classes and will not invent the missing labels. Use "
                f"a genuinely four-class dataset such as "
                f"masoudnickparvar/brain-tumor-mri-dataset."
            )
        raise DatasetError(
            f"No labelled classification images under {root}. Expected class-named "
            f"folders: {sorted(set(_CLASS_ALIASES.values()))}. Found instead: "
            f"{sorted(unmatched_dirs)[:12]}"
        )

    missing = sorted(set(m1.CLASS_NAMES) - matched_dirs)
    if missing:
        raise DatasetError(
            f"{root} is missing classes {missing}; only {sorted(matched_dirs)} were "
            f"found. Training a {m1.NUM_CLASSES}-class model on a subset would produce "
            f"a model that can never predict the missing classes."
        )

    meta = {
        "root": str(root),
        "num_samples": len(samples),
        "classes_found": sorted(matched_dirs),
        "official_split_counts": official_split,
        "respect_official_split": respect_official_split and bool(official_split),
        "grouping": "filename-stem (no patient ids in this dataset)",
        "warnings": [BRI_GROUPING_WARNING],
    }
    return samples, meta


def classification_group_of(sample: tuple[str, int]) -> str:
    return bri_patient_id(sample[0])


# --------------------------------------------------------------------------- #
# Segmentation discovery (BraTS)
# --------------------------------------------------------------------------- #
def discover_brats_h5(root: Path, tumor_only: bool = True) -> list[str]:
    """Find BraTS per-slice HDF5 files (``volume_X_slice_Y.h5``)."""
    root = Path(root)
    if not root.exists():
        return []
    data_dirs = [p for p in root.rglob("content/data") if p.is_dir()]
    if not data_dirs:
        any_h5 = list(root.rglob("volume_*_slice_*.h5"))
        if not any_h5:
            return []
        data_dirs = [any_h5[0].parent]
    data_dir = data_dirs[0]

    csv_paths = list(root.rglob("*Metadata*.csv"))
    if tumor_only and csv_paths:
        import csv as _csv

        keep: list[str] = []
        with open(csv_paths[0]) as f:
            for row in _csv.DictReader(f):
                counts = sum(
                    int(float(row.get(f"label{i}_pxl_cnt", 0) or 0)) for i in (0, 1, 2)
                )
                if counts <= 0:
                    continue
                fpath = data_dir / Path(row["slice_path"]).name
                if fpath.exists():
                    keep.append(str(fpath))
        if keep:
            return sorted(keep)
    return [str(p) for p in sorted(data_dir.glob("volume_*_slice_*.h5"))]


def discover_nifti_pairs(root: Path) -> list[tuple[str, str, int]]:
    """Find ``(flair_volume, seg_volume, slice_index)`` triples for NIfTI BraTS."""
    triples: list[tuple[str, str, int]] = []
    try:
        import nibabel as nib
    except Exception:
        return triples
    for flair in sorted(glob.glob(str(Path(root) / "**" / "*_flair.nii*"), recursive=True)):
        seg = flair.replace("_flair", "_seg")
        if not Path(seg).exists():
            continue
        depth = nib.load(flair).shape[2]
        for z in range(depth // 4, 3 * depth // 4, 3):
            triples.append((flair, seg, z))
    return triples


def segmentation_group_of(sample) -> str:
    """Volume id for a BraTS sample (h5 path or NIfTI triple)."""
    return brats_volume_id(sample[0] if isinstance(sample, tuple) else sample)


# --------------------------------------------------------------------------- #
# ROI box cache — keeps training geometry identical to inference geometry
# --------------------------------------------------------------------------- #
def roi_cache_path(spec: TransformSpec, model_version: str) -> Path:
    key = hashlib.sha1(f"{spec.id}|{spec.image_size}|{model_version}".encode()).hexdigest()[:12]
    return config.MODEL_CACHE_DIR / f"method1_roi_{key}.json"


@torch.no_grad()
def build_roi_cache(
    samples: Sequence[tuple[str, int]],
    seg_model,
    spec: TransformSpec,
    model_version: str,
    device=None,
    batch_size: int = 32,
) -> dict[str, Optional[list[int]]]:
    """Precompute each image's ROI bounding box with the trained U-Net.

    Cached on disk so repeated training runs (and the SFLA search, which trains
    many times) pay for it once.
    """
    path = roi_cache_path(spec, model_version)
    if path.exists():
        try:
            cached = json.loads(path.read_text())
            if len(cached) >= len(samples):
                print(f"[method1] ROI cache hit: {path}")
                return cached
        except Exception:
            pass

    device = device or config.DEVICE
    seg_model.eval()
    boxes: dict[str, Optional[list[int]]] = {}
    buf: list[tuple[str, np.ndarray]] = []

    def flush():
        if not buf:
            return
        batch = torch.from_numpy(np.stack([b[1] for b in buf])).float().unsqueeze(1).to(device)
        masks = (torch.sigmoid(seg_model(batch)) > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
        for (p, _), mask in zip(buf, masks):
            ys, xs = np.where(mask > 0)
            boxes[p] = (
                [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())]
                if len(xs) >= spec.roi_min_pixels
                else None
            )
        buf.clear()

    print(f"[method1] Building ROI cache for {len(samples)} images…")
    for i, (p, _) in enumerate(samples):
        buf.append((p, _load_prepared(p, spec)))
        if len(buf) >= batch_size:
            flush()
        if i and i % 2000 == 0:
            print(f"[method1]   {i}/{len(samples)}")
    flush()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(boxes))
    print(f"[method1] ROI cache written to {path}")
    return boxes


def _load_prepared(path: str, spec: TransformSpec) -> np.ndarray:
    """Disk → the float32 [0,1] array that ``preprocess`` would produce."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return np.zeros((spec.image_size, spec.image_size), np.float32)
    g = to_grayscale(img)
    if spec.order == "resize_then_denoise":
        g = resize(g, spec.image_size)
        if spec.denoise:
            g = denoise(g)
        if spec.clahe:
            g = clahe(g)
    else:
        if spec.denoise:
            g = denoise(g)
        if spec.clahe:
            g = clahe(g)
        g = resize(g, spec.image_size)
    return scale_to_unit(g)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class ClassificationDataset(Dataset):
    """Four-class MRI classification with the inference-identical geometry."""

    def __init__(
        self,
        samples: Sequence[tuple[str, int]],
        spec: TransformSpec,
        augment: bool = False,
        roi_boxes: Optional[dict[str, Optional[list[int]]]] = None,
    ):
        self.samples = list(samples)
        self.spec = spec
        self.roi_boxes = roi_boxes or {}
        # Vertical flips are disabled: flipping an axial brain slice top-to-bottom
        # is anatomically impossible and pituitary tumours are defined largely by
        # their location, so the label stops being true of the image.
        self.aug = Augmentor(vflip=0.0, seed=None) if augment else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = _load_prepared(path, self.spec)

        if self.spec.roi_crop:
            box = self.roi_boxes.get(path)
            if box is not None:
                mask = np.zeros_like(img, dtype=np.uint8)
                y0, x0, y1, x1 = box
                mask[y0 : y1 + 1, x0 : x1 + 1] = 1
                img, _ = crop_to_roi(img, mask, self.spec)

        if self.aug is not None:
            img = self.aug((img * 255).astype(np.uint8))
            img = img.astype(np.float32) / 255.0

        return torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0), label


def _otsu_mask(img_u8: np.ndarray) -> np.ndarray:
    """Weak tumour mask for mask-free 2D datasets (clearly flagged by callers)."""
    blur = cv2.GaussianBlur(img_u8, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        th = np.where(labels == largest, 255, 0).astype(np.uint8)
    return th


class SegmentationDataset(Dataset):
    """Binary whole-tumour segmentation from an explicit, pre-discovered sample list.

    Taking the list as a constructor argument (rather than re-walking the tree)
    is what makes the train/val/test parts provably the same objects the split
    produced — the previous code built two independent instances and relied on
    directory iteration order matching.
    """

    def __init__(self, samples: Sequence, kind: str, spec: TransformSpec, augment: bool = False):
        self.samples = list(samples)
        self.kind = kind  # "h5" | "nifti" | "image"
        self.spec = spec
        self.aug = Augmentor(vflip=0.0, seed=None) if augment else None
        self._nib = None
        if kind == "nifti":
            import nibabel as nib

            self._nib = nib

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        s = self.samples[idx]
        if self.kind == "h5":
            import h5py

            with h5py.File(s, "r") as h:
                image, mask = h["image"][:], h["mask"][:]
            img = cv2.normalize(image[:, :, 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            msk = ((mask.sum(axis=2) > 0).astype(np.uint8)) * 255
            return img, msk
        if self.kind == "nifti":
            flair, seg, z = s
            img = self._nib.load(flair).get_fdata()[:, :, z]
            msk = self._nib.load(seg).get_fdata()[:, :, z]
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return img, (msk > 0).astype(np.uint8) * 255
        img = cv2.imread(s, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.spec.image_size, self.spec.image_size), np.uint8)
        return img, _otsu_mask(img)

    def __getitem__(self, idx: int):
        img, msk = self._load(idx)
        img = to_grayscale(img)
        if self.spec.order == "resize_then_denoise":
            img = resize(img, self.spec.image_size)
            if self.spec.denoise:
                img = denoise(img)
            if self.spec.clahe:
                img = clahe(img)
        else:
            if self.spec.denoise:
                img = denoise(img)
            if self.spec.clahe:
                img = clahe(img)
            img = resize(img, self.spec.image_size)
        msk = resize(msk, self.spec.image_size)

        if self.aug is not None:
            img, msk = self.aug(img, msk)

        x = torch.from_numpy(np.ascontiguousarray(scale_to_unit(img))).float().unsqueeze(0)
        y = torch.from_numpy(np.ascontiguousarray((msk > 127).astype(np.float32))).unsqueeze(0)
        return x, y


def discover_segmentation_samples(root: Path) -> tuple[list, str, list[str]]:
    """Return ``(samples, kind, warnings)`` preferring real masks over weak ones."""
    root = Path(root)
    warnings: list[str] = []
    h5 = discover_brats_h5(root)
    if h5:
        return h5, "h5", warnings
    nifti = discover_nifti_pairs(root)
    if nifti:
        return nifti, "nifti", warnings
    images = [str(p) for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    if images:
        warnings.append(
            "No ground-truth masks found. Falling back to weak Otsu-derived masks: "
            "the resulting Dice measures agreement with a thresholding heuristic, "
            "NOT with radiologist annotation. Train on BraTS for a real score."
        )
        return images, "image", warnings
    return [], "image", ["No segmentation data found at all."]


def make_roi_provider(callable_: Callable) -> Callable:  # pragma: no cover - thin alias
    return callable_
