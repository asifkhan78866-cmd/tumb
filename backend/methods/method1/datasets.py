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
    preprocess_file,
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
def official_split_of(path: str) -> Optional[str]:
    """Return "train" / "test" when the dataset's own split folder is on the path.

    The Kaggle BRI set ships ``Training/`` and ``Testing/``. That split is the
    one every published number on this dataset is measured against, so it is
    preserved rather than merged and re-drawn.
    """
    parts = {p.lower() for p in Path(path).parts}
    if parts & {"testing", "test"}:
        return "test"
    if parts & {"training", "train"}:
        return "train"
    return None


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

    by_split: dict[str, list[tuple[str, int]]] = {"train": [], "test": [], "none": []}
    for sample in samples:
        by_split[official_split_of(sample[0]) or "none"].append(sample)
    has_official = bool(by_split["train"] and by_split["test"])

    per_class_split = {}
    for name in m1.CLASS_NAMES:
        idx = m1.CLASS_NAMES.index(name)
        per_class_split[name] = {
            "train": sum(1 for s in by_split["train"] if s[1] == idx),
            "test": sum(1 for s in by_split["test"] if s[1] == idx),
        }

    warnings = [BRI_GROUPING_WARNING]
    if has_official:
        warnings.append(
            "Using the dataset's own Training/Testing split. It is an IMAGE-level "
            "split: this dataset ships no patient identifiers, so slices from one "
            "patient may appear on both sides and the test score can be optimistic."
        )
    else:
        warnings.append(
            "No Training/Testing folders found — falling back to a grouped random "
            "three-way split."
        )

    meta = {
        "root": str(root),
        "num_samples": len(samples),
        "classes_found": sorted(matched_dirs),
        "official_split_counts": official_split,
        "has_official_split": has_official,
        "respect_official_split": respect_official_split and has_official,
        "per_class_split": per_class_split,
        "counts": {"train_pool": len(by_split["train"]), "test": len(by_split["test"]),
                   "unassigned": len(by_split["none"])},
        "grouping": "filename-stem (no patient ids in this dataset)",
        "split_level": "image-level (dataset provides no patient ids)",
        "warnings": warnings,
    }
    return samples, meta


def partition_by_official_split(
    samples: Sequence[tuple[str, int]]
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Split into ``(train_pool, test)`` using the dataset's own folders."""
    train_pool, test = [], []
    for sample in samples:
        (test if official_split_of(sample[0]) == "test" else train_pool).append(sample)
    return train_pool, test


def class_distribution(samples: Sequence[tuple[str, int]]) -> dict[str, int]:
    return {
        name: sum(1 for s in samples if s[1] == i)
        for i, name in enumerate(m1.CLASS_NAMES)
    }


def class_weights(samples: Sequence[tuple[str, int]]) -> list[float]:
    """Inverse-frequency weights, normalised to mean 1.

    Applied to the loss so a minority class is not simply ignored. Reported in
    the run card so the effect on the metrics is traceable.
    """
    counts = class_distribution(samples)
    total = sum(counts.values()) or 1
    n = len(m1.CLASS_NAMES)
    raw = [total / (n * max(1, counts[name])) for name in m1.CLASS_NAMES]
    mean = sum(raw) / len(raw)
    return [w / mean for w in raw]


def classification_group_of(sample: tuple[str, int]) -> str:
    return bri_patient_id(sample[0])


def file_digest(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def content_group_of(digests: dict[str, str]) -> Callable[[tuple[str, int]], str]:
    """Group key = image content hash.

    The Kaggle BRI set contains byte-identical copies under different filenames,
    both inside Training and across Training/Testing. Grouping by filename stem
    lets those copies land on both sides of the train/validation split, which
    inflates the validation score the early stopper and SFLA optimise.
    """
    return lambda sample: digests[sample[0]]


def drop_test_duplicates(
    train_pool: Sequence[tuple[str, int]],
    test: Sequence[tuple[str, int]],
    digests: dict[str, str],
) -> tuple[list[tuple[str, int]], list[str]]:
    """Remove training images that are byte-identical to a test image.

    The test set itself is never modified; only the leaking training copies go.
    """
    test_hashes = {digests[p] for p, _ in test}
    kept = [s for s in train_pool if digests[s[0]] not in test_hashes]
    removed = [p for p, _ in train_pool if digests[p] in test_hashes]
    return kept, removed


# --------------------------------------------------------------------------- #
# Preprocessed-array cache — deterministic preprocessing done once, not per epoch
# --------------------------------------------------------------------------- #
def load_preprocessed(
    samples: Sequence[tuple[str, int]], spec: TransformSpec, workers: Optional[int] = None
) -> np.ndarray:
    """Return an (N, H, W) float32 array of ``preprocess(image, spec)`` outputs.

    Calls the exact function the inference engine calls, so a cached array is
    identical to what serving computes for the same file. Non-local-means
    denoising dominates the cost, so the result is cached on disk keyed by the
    spec and the file list, and computed with a process pool.
    """
    from concurrent.futures import ProcessPoolExecutor
    import os

    paths = [p for p, _ in samples]
    key = hashlib.sha1(
        json.dumps([spec.to_dict(), sorted(paths)], sort_keys=True).encode()
    ).hexdigest()[:16]
    cache = config.MODEL_CACHE_DIR / f"method1_prep_{spec.id}_{key}.npz"
    index = {p: i for i, p in enumerate(sorted(paths))}
    if cache.exists():
        arr = np.load(cache)["x"]
        print(f"[method1] preprocessed cache hit: {cache.name} ({len(arr)} images)")
    else:
        ordered = sorted(paths)
        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        print(f"[method1] preprocessing {len(ordered)} images with {workers} processes "
              f"(spec {spec.id}); cached for later runs…")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            arr = np.stack(list(ex.map(preprocess_file, [(p, spec) for p in ordered],
                                       chunksize=32)))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, x=arr)
        print(f"[method1] preprocessed cache written: {cache}")
    return arr[[index[p] for p in paths]]


class CachedClassificationDataset(Dataset):
    """Whole-slice classification over preprocessed arrays.

    Equivalent to :class:`ClassificationDataset` for specs without ROI cropping,
    minus the per-epoch disk read and denoise. Augmentation (training only) is
    seeded so a run is reproducible.
    """

    def __init__(self, images: np.ndarray, labels: Sequence[int], augment: bool = False,
                 seed: int = 42):
        self.images = images
        self.labels = np.asarray(labels, dtype=np.int64)
        self.aug = Augmentor(vflip=0.0, seed=seed) if augment else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        if self.aug is not None:
            img = self.aug((img * 255).astype(np.uint8)).astype(np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0), int(self.labels[idx])


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
        self.kind = kind  # "cjdata" | "pairs" | "h5" | "nifti" | "image"
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
        if self.kind == "cjdata":
            image, mask, _, _ = read_cjdata(s[0])
            return image, mask
        if self.kind == "pairs":
            image_path, mask_path = s
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                size = self.spec.image_size
                return np.zeros((size, size), np.uint8), np.zeros((size, size), np.uint8)
            # LGG slices stack three sequences; the masks annotate the FLAIR
            # abnormality, so the FLAIR channel is the one the model should see.
            if img.ndim == 3 and img.shape[2] >= 2:
                img = img[..., 1]
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return img, ((msk > 0).astype(np.uint8) * 255)
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


def discover_mask_pairs(root: Path) -> list[tuple[str, str]]:
    """Find ``(image, mask)`` pairs written as ``<name>.tif`` / ``<name>_mask.tif``.

    This is the LGG MRI Segmentation layout (one folder per patient, one mask per
    slice). Only pairs where both files exist are returned.
    """
    pairs: list[tuple[str, str]] = []
    for mask in sorted(Path(root).rglob("*_mask.*")):
        image = mask.with_name(mask.name.replace("_mask", ""))
        if image.exists() and image.suffix.lower() in IMAGE_EXTS + (".tif", ".tiff"):
            pairs.append((str(image), str(mask)))
    return pairs


def lgg_patient_id(path: str) -> str:
    """Patient id for an LGG slice: its folder name (``TCGA_CS_4941_19960909``)."""
    return Path(path).parent.name.lower()


def read_cjdata(path: str) -> tuple[np.ndarray, np.ndarray, str, int]:
    """Read one figshare (Cheng) ``.mat`` slice: image, tumour mask, patient id, label.

    The files are MATLAB v7.3 (HDF5) and store arrays column-major, so both the
    image and the mask come back transposed. The patient id is a char array; it
    is what makes a patient-level split possible on this dataset.
    """
    import h5py

    with h5py.File(path, "r") as h:
        g = h["cjdata"]
        image = np.array(g["image"]).T.astype(np.float32)
        mask = np.array(g["tumorMask"]).T.astype(np.uint8)
        pid = "".join(chr(int(c)) for c in np.array(g["PID"]).flatten())
        label = int(np.array(g["label"]).flatten()[0])
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image, (mask > 0).astype(np.uint8) * 255, pid.strip(), label


def discover_cjdata_samples(root: Path) -> list[tuple[str, str]]:
    """Find figshare ``.mat`` slices as ``(path, patient_id)`` pairs."""
    out: list[tuple[str, str]] = []
    for path in sorted(Path(root).rglob("*.mat")):
        if path.name.lower() == "cvind.mat":   # the dataset's own fold indices
            continue
        try:
            _, _, pid, _ = read_cjdata(str(path))
        except Exception:
            continue
        out.append((str(path), pid))
    return out


def discover_segmentation_samples(root: Path) -> tuple[list, str, list[str]]:
    """Return ``(samples, kind, warnings)`` preferring real masks over weak ones."""
    root = Path(root)
    warnings: list[str] = []
    cjdata = discover_cjdata_samples(root)
    if cjdata:
        return cjdata, "cjdata", warnings
    pairs = discover_mask_pairs(root)
    if pairs:
        return pairs, "pairs", warnings
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
