"""ZFNet input geometry — one definition for training and inference.

Grayscale → resize to 224×224 (uint8, cached for training) → [0, 1] →
standardise with mean 0.5 / std 0.25, single channel.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch

MEAN, STD = 0.5, 0.25


@dataclass(frozen=True)
class M4Transform:
    id: str = "m4-gray224"
    image_size: int = 224
    channels: int = 1
    normalisation: str = "(x - 0.5) / 0.25"
    description: str = (
        "Grayscale, resize to 224×224 (area interpolation when shrinking), single channel, "
        "fixed standardisation. Training adds GPU augmentation only."
    )

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_SPEC = M4Transform()


def get_spec(transform_id=None) -> M4Transform:
    return DEFAULT_SPEC


def preprocess_file(args) -> np.ndarray:
    """``(path, key)`` -> uint8 (224, 224). Module-level for the process pool."""
    from backend.methods.common.classifier_engine import gray_resize

    path, key = args
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Unreadable image: {path}")
    return gray_resize(img, int(key["image_size"]))


def to_input(x: torch.Tensor) -> torch.Tensor:
    return (x - MEAN) / STD
