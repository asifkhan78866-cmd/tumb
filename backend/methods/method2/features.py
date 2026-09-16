"""The modality-specific feature stage that sits between segmentation and the DCN.

The Method 2 flow places a modality-specific representation between the
multi-class segmentation and the classifier. This module is that stage. It turns
a filtered slice plus a per-pixel region label map into:

``uptake_maps``
    One image channel per tumour sub-region, each the intensity of the slice
    masked to that region. These are stacked with the filtered slice and fed to
    the DCN as input channels, so the network sees *where* the activity is,
    region by region, rather than having to rediscover the segmentation.

``feature_vector``
    A fixed-length descriptor concatenated to the DCN's pooled features:
    per-region area fraction, mean / std / max / 90th-percentile uptake, the
    region-to-background uptake ratio, plus a global intensity histogram. These
    are the quantities a nuclear-medicine reading is actually based on.

TERMINOLOGY
-----------
The reference diagram for this method writes "Photon Emission Computed
Tomography (PECT)". The dataset and this implementation are **SPECT** (Single
Photon Emission Computed Tomography). They are not synonyms, so the real
modality is carried in ``config.METHOD2_MODALITY`` and the spec's wording is
only rendered where a run explicitly asks for it. Nothing here claims the two
are the same thing.

On MRI inputs the same descriptors are still computable — they are intensity
statistics — but "uptake" is not a meaningful word for an MRI signal, so
:func:`feature_names` labels them ``intensity_*`` when the modality is not a
nuclear-medicine one, and the modality is recorded in every checkpoint.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "HISTOGRAM_BINS",
    "feature_dim",
    "feature_names",
    "uptake_maps",
    "feature_vector",
    "build_inputs",
]

HISTOGRAM_BINS = 16
_PER_REGION_STATS = ("area_fraction", "mean", "std", "max", "p90", "contrast_ratio")


def feature_dim(num_regions: int) -> int:
    """Length of the descriptor for ``num_regions`` foreground regions."""
    return num_regions * len(_PER_REGION_STATS) + HISTOGRAM_BINS


def feature_names(region_names: list[str], modality: str = "SPECT") -> list[str]:
    prefix = "uptake" if modality.upper() in {"SPECT", "PECT", "PET"} else "intensity"
    names: list[str] = []
    for region in region_names:
        for stat in _PER_REGION_STATS:
            names.append(f"{region}_{prefix}_{stat}" if stat != "area_fraction" else f"{region}_area_fraction")
    names.extend(f"hist_bin_{i:02d}" for i in range(HISTOGRAM_BINS))
    return names


def uptake_maps(image: np.ndarray, label_map: np.ndarray, num_regions: int) -> np.ndarray:
    """Return ``(num_regions, H, W)`` intensity maps, one per foreground region.

    Region indices are ``1..num_regions``; label 0 is background.
    """
    maps = np.zeros((num_regions, *image.shape), dtype=np.float32)
    for r in range(1, num_regions + 1):
        maps[r - 1] = np.where(label_map == r, image, 0.0)
    return maps


def feature_vector(image: np.ndarray, label_map: np.ndarray, num_regions: int) -> np.ndarray:
    """Compute the fixed-length modality descriptor.

    Regions absent from the slice contribute zeros — an honest "no signal here"
    rather than an imputed value.
    """
    image = image.astype(np.float32)
    total = float(image.size) or 1.0
    background = image[label_map == 0]
    bg_mean = float(background.mean()) if background.size else 0.0

    feats: list[float] = []
    for r in range(1, num_regions + 1):
        values = image[label_map == r]
        if values.size == 0:
            feats.extend([0.0] * len(_PER_REGION_STATS))
            continue
        mean = float(values.mean())
        feats.extend(
            [
                float(values.size) / total,
                mean,
                float(values.std()),
                float(values.max()),
                float(np.percentile(values, 90)),
                mean / bg_mean if bg_mean > 1e-6 else 0.0,
            ]
        )

    hist, _ = np.histogram(image, bins=HISTOGRAM_BINS, range=(0.0, 1.0))
    feats.extend((hist / total).astype(np.float32).tolist())
    return np.asarray(feats, dtype=np.float32)


def build_inputs(
    image: np.ndarray, label_map: np.ndarray, num_regions: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(channels, descriptor)`` ready for the DCN.

    ``channels`` is ``(1 + num_regions, H, W)``: the filtered slice followed by
    one uptake map per region.
    """
    channels = np.concatenate(
        [image[None, ...].astype(np.float32), uptake_maps(image, label_map, num_regions)], axis=0
    )
    return channels, feature_vector(image, label_map, num_regions)
