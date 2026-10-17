"""The Medical Segmentation Decathlon's brain-tumour task (BraTS volumes).

``Task01_BrainTumour`` is BraTS data published without an account: 484 training
volumes, each four co-registered MRI sequences (FLAIR, T1w, T1Gd, T2w) at
240×240×155, with a voxel label map of the three tumour sub-regions. It is the
only source in this project that carries real multi-region annotation *and*
real volumes, so it feeds both of the stages that were declared but untrained:
the multi-class segmentation head, and the volumetric U-Net.

Two things this module is careful about.

**Labels.** The MSD label order is 1 = oedema, 2 = non-enhancing tumour,
3 = enhancing tumour. This project's class list is background, necrotic core,
oedema, enhancing tumour — a different order. :data:`MSD_TO_PROJECT` maps one
to the other explicitly rather than trusting the integers to line up, because
a silent mismatch would train the network to call oedema a necrotic core and
nothing downstream would notice.

**Splitting.** Every slice of a volume belongs to one patient. Splits are made
over volume ids, never over slices, so no slice of a test patient can appear in
training — the leak that makes slice-level brain-MRI results look far better
than they are.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

__all__ = [
    "MODALITIES", "MSD_TO_PROJECT", "MSD_LABEL_NAMES", "VolumePair", "discover_msd",
    "load_volume", "remap_labels", "slice_cache_dir", "materialise_slices", "SliceIndex",
    "volume_cache_dir", "materialise_volumes", "brain_bbox",
]

# Channel order inside the MSD image volume.
MODALITIES = ("FLAIR", "T1w", "T1Gd", "T2w")
MSD_LABEL_NAMES = {1: "oedema", 2: "non-enhancing tumour", 3: "enhancing tumour"}

# MSD label value -> this project's class index (see METHOD2_SEG_CLASSES:
# background, necrotic core, edema, enhancing tumour).
MSD_TO_PROJECT = {0: 0, 1: 2, 2: 1, 3: 3}


@dataclass(frozen=True)
class VolumePair:
    """One patient: the 4-sequence image volume and its label volume."""

    volume_id: str
    image: Path
    label: Path


@dataclass
class SliceIndex:
    """Where the extracted slices live and what they came from."""

    directory: Path
    files: list[str] = field(default_factory=list)
    volumes: int = 0
    modality: str = "FLAIR"
    label_map: dict = field(default_factory=lambda: dict(MSD_TO_PROJECT))
    warnings: list[str] = field(default_factory=list)

    def volume_of(self, path: str) -> str:
        """The patient a slice belongs to — the key every split must group on."""
        return Path(path).name.split("_slice_")[0]


def discover_msd(root: Path) -> list[VolumePair]:
    """Image/label pairs under an extracted ``Task01_BrainTumour`` folder.

    macOS tar archives carry ``._`` resource forks and the dataset ships hidden
    files; both are skipped rather than loaded and failed on.
    """
    root = Path(root)
    candidates = [root, *(p.parent for p in root.rglob("dataset.json"))]
    for base in candidates:
        images, labels = base / "imagesTr", base / "labelsTr"
        if not (images.is_dir() and labels.is_dir()):
            continue
        pairs = []
        for image in sorted(images.glob("*.nii*")):
            if image.name.startswith("."):
                continue
            label = labels / image.name
            if label.exists():
                pairs.append(VolumePair(image.name.split(".")[0], image, label))
        if pairs:
            return pairs
    return []


def load_volume(pair: VolumePair, modality: Optional[str] = "FLAIR"
                ) -> tuple[np.ndarray, np.ndarray]:
    """``(image, labels)`` as float32 and uint8, labels already remapped.

    ``modality=None`` keeps all four sequences as a trailing channel axis.
    """
    import nibabel as nib

    image = np.asanyarray(nib.load(str(pair.image)).dataobj).astype(np.float32)
    labels = np.asanyarray(nib.load(str(pair.label)).dataobj).astype(np.uint8)
    if modality is not None:
        if modality not in MODALITIES:
            raise ValueError(f"Unknown modality {modality!r}; expected one of {MODALITIES}.")
        image = image[..., MODALITIES.index(modality)]
    return image, remap_labels(labels)


def remap_labels(labels: np.ndarray) -> np.ndarray:
    """MSD label values -> this project's class indices."""
    out = np.zeros_like(labels, dtype=np.uint8)
    for src, dst in MSD_TO_PROJECT.items():
        if dst:
            out[labels == src] = dst
    return out


def normalise(volume: np.ndarray) -> np.ndarray:
    """Scale a volume to [0, 1] using its own non-zero statistics.

    Brain MRI intensity has no absolute meaning and the background is exactly
    zero, so percentiles of the *brain* are what makes two scans comparable.
    """
    brain = volume[volume > 0]
    if brain.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    lo, hi = np.percentile(brain, (1.0, 99.0))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def slice_cache_dir(root: Path, modality: str, stride: int, size: int) -> Path:
    from backend import config

    return config.MODEL_CACHE_DIR / f"msd_slices_{modality.lower()}_s{stride}_{size}"


def materialise_slices(pairs: Iterable[VolumePair], out_dir: Path, *, modality: str = "FLAIR",
                       stride: int = 2, size: int = 128, min_tumour_pixels: int = 64,
                       background_every: int = 8, verbose: bool = True) -> SliceIndex:
    """Write one ``.npz`` per kept slice, so training reads cheaply.

    Decoding a 240×240×155×4 volume for every slice access would dominate
    training time, so this runs once and caches. Which slices are kept is a
    documented choice, not an accident:

    * every ``stride``-th slice that holds at least ``min_tumour_pixels``
      annotated voxels — the slices the task is actually about;
    * one in ``background_every`` tumour-free slices, so the network still sees
      healthy anatomy and does not learn that something is always there.
    """
    out_dir = Path(out_dir)
    index_path = out_dir / "index.json"
    if index_path.exists():
        data = json.loads(index_path.read_text())
        if data.get("files"):
            if verbose:
                print(f"[msd] slice cache hit: {len(data['files'])} slices in {out_dir}")
            return SliceIndex(out_dir, data["files"], data.get("volumes", 0),
                              data.get("modality", modality), data.get("label_map", {}),
                              data.get("warnings", []))

    out_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    files: list[str] = []
    pairs = list(pairs)
    kept_tumour = kept_background = 0
    for n, pair in enumerate(pairs, 1):
        volume, labels = load_volume(pair, modality)
        volume = normalise(volume)
        for k in range(0, volume.shape[2], stride):
            label_slice = labels[:, :, k]
            tumour = int((label_slice > 0).sum())
            if tumour < min_tumour_pixels:
                if background_every <= 0 or (k // max(stride, 1)) % background_every:
                    continue
                kept_background += 1
            else:
                kept_tumour += 1
            image = cv2.resize(volume[:, :, k], (size, size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(label_slice, (size, size), interpolation=cv2.INTER_NEAREST)
            name = f"{pair.volume_id}_slice_{k:03d}.npz"
            np.savez_compressed(out_dir / name, image=image.astype(np.float32),
                                label=mask.astype(np.uint8))
            files.append(name)
        if verbose and (n % 25 == 0 or n == len(pairs)):
            print(f"[msd] {n}/{len(pairs)} volumes -> {len(files)} slices", flush=True)

    warnings = [
        f"Slices were sampled, not taken whole: every {stride}th slice with at least "
        f"{min_tumour_pixels} annotated voxels, plus one in {background_every} tumour-free "
        f"slices ({kept_tumour} annotated, {kept_background} tumour-free). A model trained on "
        f"this sees far more tumour than a whole scan contains.",
        f"Slices were resampled to {size}×{size} from 240×240, so fine structure is lost "
        f"before the network sees it.",
    ]
    index = SliceIndex(out_dir, files, len(pairs), modality, dict(MSD_TO_PROJECT), warnings)
    index_path.write_text(json.dumps({
        "files": files, "volumes": len(pairs), "modality": modality, "stride": stride,
        "size": size, "min_tumour_pixels": min_tumour_pixels,
        "background_every": background_every, "label_map": {str(k): v for k, v in MSD_TO_PROJECT.items()},
        "annotated_slices": kept_tumour, "tumour_free_slices": kept_background,
        "warnings": warnings,
    }, indent=2))
    if verbose:
        print(f"[msd] {len(files)} slices written to {out_dir}")
    return index


# --------------------------------------------------------------------------- #
# Whole volumes, for the 3D model
# --------------------------------------------------------------------------- #
def brain_bbox(volume: np.ndarray, pad: int = 2) -> tuple[slice, slice, slice]:
    """The box around the non-zero brain, padded.

    BraTS volumes are mostly black air; cropping to the head before resampling
    spends the resolution budget on anatomy instead of background.
    """
    nz = np.argwhere(volume > 0)
    if nz.size == 0:
        return tuple(slice(0, s) for s in volume.shape)  # type: ignore[return-value]
    lo = np.maximum(nz.min(axis=0) - pad, 0)
    hi = np.minimum(nz.max(axis=0) + 1 + pad, volume.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))  # type: ignore[return-value]


def volume_cache_dir(modality: str, size: int) -> Path:
    from backend import config

    return config.MODEL_CACHE_DIR / f"msd_volumes_{modality.lower()}_{size}"


def materialise_volumes(pairs: Iterable[VolumePair], out_dir: Path, *, modality: str = "FLAIR",
                        size: int = 128, verbose: bool = True) -> list[str]:
    """Cache each volume brain-cropped and resampled to ``size`` cubed.

    Decoding a NIfTI volume costs seconds; a training run touches every volume
    many times. The cache holds float16 images and the integer label map, which
    is small enough to keep the whole dataset on disk and fast enough to stream.
    """
    out_dir = Path(out_dir)
    index_path = out_dir / "index.json"
    if index_path.exists():
        files = json.loads(index_path.read_text()).get("files") or []
        if files:
            if verbose:
                print(f"[msd] volume cache hit: {len(files)} volumes in {out_dir}")
            return files

    out_dir.mkdir(parents=True, exist_ok=True)
    from scipy.ndimage import zoom

    files: list[str] = []
    pairs = list(pairs)
    for n, pair in enumerate(pairs, 1):
        image, labels = load_volume(pair, modality)
        box = brain_bbox(image)
        image, labels = normalise(image[box]), labels[box]
        factors = [size / s for s in image.shape]
        image = zoom(image, factors, order=1).astype(np.float16)
        # Nearest neighbour for labels: interpolating class indices would invent
        # sub-regions that were never annotated.
        labels = zoom(labels, factors, order=0).astype(np.uint8)
        name = f"{pair.volume_id}.npz"
        np.savez_compressed(out_dir / name, image=image, label=labels)
        files.append(name)
        if verbose and (n % 25 == 0 or n == len(pairs)):
            print(f"[msd] {n}/{len(pairs)} volumes cached", flush=True)

    index_path.write_text(json.dumps({"files": files, "modality": modality, "size": size,
                                      "label_map": {str(k): v for k, v in MSD_TO_PROJECT.items()}},
                                     indent=2))
    if verbose:
        print(f"[msd] {len(files)} volumes cached to {out_dir}")
    return files
