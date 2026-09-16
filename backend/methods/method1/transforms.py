"""Method 1 preprocessing — ONE definition, imported by training and inference.

The original pipeline trained the classifier on whole preprocessed slices but
served it mask-cropped ROIs, so the reported accuracy described inputs the API
never sent. That class of bug is only fixable structurally: there is exactly one
implementation of the geometry here, both sides import it, and the spec that
produced a checkpoint is written into the checkpoint so serving can reproduce it
rather than assume it.

Two specs exist:

``M1_V1_LEGACY``
    denoise → CLAHE → resize, **no ROI crop**. This is what the classifier
    shipped in this repository was actually trained with. Serving an untagged
    legacy checkpoint uses this spec so existing weights keep their meaning.

``M1_V2`` (default for new training runs)
    resize → denoise → CLAHE, **with ROI crop** when a validated segmentation
    mask is available. Resizing first is the same filter applied to far fewer
    pixels: non-local-means cost scales with image area, and at inference the
    image is reduced to ``image_size`` immediately afterwards anyway, so the
    full-resolution denoise was work that was then thrown away.

Ordering is a *property of the spec*, never a global switch, because changing it
underneath a trained checkpoint silently shifts the input distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import cv2
import numpy as np

__all__ = [
    "TransformSpec",
    "M1_V1_LEGACY",
    "M1_V2",
    "SPECS",
    "get_spec",
    "preprocess",
    "crop_to_roi",
    "classifier_input",
    "to_tensor",
]


@dataclass(frozen=True)
class TransformSpec:
    """A complete, serialisable description of Method 1's input geometry."""

    id: str
    image_size: int = 128
    order: str = "resize_then_denoise"  # or "denoise_then_resize"
    denoise: bool = True
    clahe: bool = True
    roi_crop: bool = True
    roi_pad: int = 8
    roi_min_pixels: int = 20
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "image_size": self.image_size,
            "order": self.order,
            "denoise": self.denoise,
            "clahe": self.clahe,
            "roi_crop": self.roi_crop,
            "roi_pad": self.roi_pad,
            "roi_min_pixels": self.roi_min_pixels,
            "description": self.description,
        }

    def resized(self, image_size: int) -> "TransformSpec":
        return replace(self, image_size=image_size)


M1_V1_LEGACY = TransformSpec(
    id="m1-v1-legacy",
    image_size=128,
    order="denoise_then_resize",
    roi_crop=False,
    description=(
        "Original geometry: full-resolution non-local-means denoise, CLAHE, then "
        "resize; classifier sees the whole slice, never an ROI crop. Used for the "
        "untagged classifier checkpoint shipped with the repository."
    ),
)

M1_V2 = TransformSpec(
    id="m1-v2-roi",
    image_size=128,
    order="resize_then_denoise",
    roi_crop=True,
    description=(
        "Resize first, then denoise and CLAHE; classifier sees the padded "
        "bounding box of the segmentation mask, resampled to image_size. Training "
        "and inference share this definition."
    ),
)

SPECS: dict[str, TransformSpec] = {s.id: s for s in (M1_V1_LEGACY, M1_V2)}
DEFAULT_SPEC = M1_V2


def get_spec(transform_id: Optional[str]) -> TransformSpec:
    """Resolve a spec id from a checkpoint; unknown/absent ids fall back safely.

    An absent id means a checkpoint written before specs were recorded, which in
    this repository can only be the legacy classifier — so it maps to V1.
    """
    if not transform_id:
        return M1_V1_LEGACY
    return SPECS.get(transform_id, DEFAULT_SPEC)


# --------------------------------------------------------------------------- #
# Primitive steps
# --------------------------------------------------------------------------- #
def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Ensure a single-channel 2D uint8 image."""
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[..., 0]
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def denoise(img: np.ndarray) -> np.ndarray:
    """Edge-preserving non-local-means denoising."""
    return cv2.fastNlMeansDenoising(img, None, h=7, templateWindowSize=7, searchWindowSize=21)


def clahe(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(img)


def resize(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def scale_to_unit(img: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1].

    The previous implementation z-scored and *then* min-max scaled. Min-max of an
    affine transform equals min-max of the original — ((x-m)/s - min')/(max'-min')
    reduces to (x-min)/(max-min) — so the z-score step provably did nothing. Only
    the step that has an effect is kept.
    """
    img = img.astype(np.float32) / 255.0
    mn, mx = float(img.min()), float(img.max())
    if mx - mn > 1e-6:
        img = (img - mn) / (mx - mn)
    return img.astype(np.float32)


# --------------------------------------------------------------------------- #
# Composed pipeline
# --------------------------------------------------------------------------- #
def preprocess(img: np.ndarray, spec: TransformSpec = DEFAULT_SPEC) -> np.ndarray:
    """Deterministic preprocessing. Returns float32 HxW in [0, 1].

    Step order is taken from ``spec.order`` and is part of the checkpoint's
    identity — see the module docstring.
    """
    g = to_grayscale(img)
    if spec.order == "resize_then_denoise":
        g = resize(g, spec.image_size)
        if spec.denoise:
            g = denoise(g)
        if spec.clahe:
            g = clahe(g)
    elif spec.order == "denoise_then_resize":
        if spec.denoise:
            g = denoise(g)
        if spec.clahe:
            g = clahe(g)
        g = resize(g, spec.image_size)
    else:
        raise ValueError(
            f"Unknown transform order {spec.order!r}; expected "
            f"'resize_then_denoise' or 'denoise_then_resize'."
        )
    return scale_to_unit(g)


def crop_to_roi(
    img: np.ndarray, mask: np.ndarray, spec: TransformSpec = DEFAULT_SPEC
) -> tuple[np.ndarray, bool]:
    """Crop to the padded bounding box of ``mask`` and resample to image_size.

    Returns ``(image, cropped)``. When the mask is empty or implausibly small the
    original image is returned unchanged with ``cropped=False``, so the caller
    can record that no ROI was actually applied.
    """
    if mask is None:
        return img, False
    ys, xs = np.where(mask > 0)
    if len(xs) < spec.roi_min_pixels:
        return img, False
    pad = spec.roi_pad
    y0 = max(0, int(ys.min()) - pad)
    x0 = max(0, int(xs.min()) - pad)
    y1 = min(img.shape[0], int(ys.max()) + pad)
    x1 = min(img.shape[1], int(xs.max()) + pad)
    if y1 - y0 < 2 or x1 - x0 < 2:
        return img, False
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (spec.image_size, spec.image_size)), True


def classifier_input(
    img: np.ndarray,
    mask: Optional[np.ndarray],
    spec: TransformSpec = DEFAULT_SPEC,
) -> tuple[np.ndarray, bool]:
    """Produce exactly the array the classifier was trained on.

    Both ``ClassificationDataset`` and the inference engine call this, which is
    what keeps training and serving in step. When ``spec.roi_crop`` is False the
    mask is ignored entirely — a spec that was trained on whole slices is never
    served crops, regardless of whether a mask happens to be available.
    """
    if not spec.roi_crop or mask is None:
        return img, False
    return crop_to_roi(img, mask, spec)


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded bytes into an OpenCV array."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image bytes — is this a valid image file?")
    return img


def to_tensor(img: np.ndarray):
    """HxW float array -> (1, 1, H, W) tensor. torch is imported lazily."""
    import torch

    return torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0).unsqueeze(0)
