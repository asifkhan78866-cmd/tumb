"""Method 2 preprocessing — grayscale conversion then filtering/denoising.

Deliberately *not* shared with Method 1. The two methods are independent
research pipelines; making them share a preprocessing implementation would mean
a change made for one silently alters the other's inputs and invalidates its
checkpoints. Only the generic helpers in ``backend.utils`` are common.

Documented step order (matching the Method 2 flow diagram):

    input → grayscale → resize → median filter → bilateral filter → scale to [0,1]

Resizing precedes filtering: both filters are neighbourhood operations whose
cost scales with image area, and the image is reduced to ``image_size``
immediately afterwards either way. The order is recorded in ``MethodTransform.id``
and written into every checkpoint, so a model is always served the geometry it
was trained on.

Method 2 uses a median + bilateral pair rather than Method 1's non-local means:
SPECT reconstructions carry impulse-like noise that a median filter removes
cleanly, and the bilateral pass preserves the lesion boundaries the multi-class
segmentation head has to find.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

__all__ = ["M2Transform", "M2_V1", "SPECS", "get_spec", "preprocess", "to_tensor", "decode_image_bytes"]


@dataclass(frozen=True)
class M2Transform:
    id: str
    image_size: int = 128
    order: str = "resize_then_filter"
    median_ksize: int = 3
    bilateral_d: int = 5
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "image_size": self.image_size,
            "order": self.order,
            "median_ksize": self.median_ksize,
            "bilateral_d": self.bilateral_d,
            "bilateral_sigma_color": self.bilateral_sigma_color,
            "bilateral_sigma_space": self.bilateral_sigma_space,
            "description": self.description,
        }

    def resized(self, image_size: int) -> "M2Transform":
        return replace(self, image_size=image_size)


M2_V1 = M2Transform(
    id="m2-v1",
    image_size=128,
    order="resize_then_filter",
    description="Grayscale → resize → median(3) → bilateral(5) → min-max scale.",
)

SPECS: dict[str, M2Transform] = {M2_V1.id: M2_V1}
DEFAULT_SPEC = M2_V1


def get_spec(transform_id: str | None) -> M2Transform:
    return SPECS.get(transform_id or "", DEFAULT_SPEC)


def to_grayscale(img: np.ndarray) -> np.ndarray:
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


def filter_image(img: np.ndarray, spec: M2Transform) -> np.ndarray:
    out = cv2.medianBlur(img, spec.median_ksize)
    return cv2.bilateralFilter(
        out, spec.bilateral_d, spec.bilateral_sigma_color, spec.bilateral_sigma_space
    )


def preprocess(img: np.ndarray, spec: M2Transform = DEFAULT_SPEC) -> np.ndarray:
    """Returns float32 HxW in [0, 1]."""
    g = to_grayscale(img)
    if spec.order == "resize_then_filter":
        g = cv2.resize(g, (spec.image_size, spec.image_size), interpolation=cv2.INTER_LINEAR)
        g = filter_image(g, spec)
    elif spec.order == "filter_then_resize":
        g = filter_image(g, spec)
        g = cv2.resize(g, (spec.image_size, spec.image_size), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError(f"Unknown Method 2 transform order {spec.order!r}.")
    g = g.astype(np.float32) / 255.0
    mn, mx = float(g.min()), float(g.max())
    if mx - mn > 1e-6:
        g = (g - mn) / (mx - mn)
    return g.astype(np.float32)


def preprocess_file(args) -> np.ndarray:
    """``(path, spec_dict)`` -> ``preprocess`` output. Module-level for the process pool."""
    path, key = args
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Unreadable image: {path}")
    return preprocess(img, get_spec(key["id"]))


def decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image bytes — is this a valid image file?")
    return img


def to_tensor(img: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0).unsqueeze(0)
