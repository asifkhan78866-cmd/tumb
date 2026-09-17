"""Transfer-learning input geometry — one definition for training and inference.

Grayscale → resize to 224×224 (uint8, cached for training) → [0, 1] →
replicate to three channels → ImageNet mean/std normalisation, which is what
the pretrained backbones expect.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class M3Transform:
    id: str = "m3-gray224-rgb-imagenet"
    image_size: int = 224
    channels: int = 3
    normalisation: str = "ImageNet mean/std"
    description: str = (
        "Grayscale, resize to 224×224 (area interpolation when shrinking), replicate to "
        "3 channels, ImageNet normalisation. Training adds GPU augmentation only."
    )

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_SPEC = M3Transform()


def get_spec(transform_id=None) -> M3Transform:
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
    """(B, 1, H, W) in [0, 1] -> (B, 3, H, W) ImageNet-normalised."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x.expand(-1, 3, -1, -1) - mean) / std
