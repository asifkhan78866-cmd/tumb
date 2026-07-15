"""MRI preprocessing and augmentation pipeline.

Implements resize, normalisation, intensity scaling, noise removal (non-local
means / median), CLAHE, and geometric augmentations (flips, rotation, random
crop). Works on single grayscale slices represented as ``float32`` arrays in the
range [0, 1] or ``uint8`` images.
"""
from __future__ import annotations

import io
from typing import Optional

import cv2
import numpy as np
import torch

from backend import config


# --------------------------------------------------------------------------- #
# Core single-image preprocessing (used at both train and inference time)
# --------------------------------------------------------------------------- #
def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Ensure a single-channel 2D uint8 image."""
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img = img[..., 0]
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def remove_noise(img: np.ndarray) -> np.ndarray:
    """Edge-preserving denoising via non-local means."""
    return cv2.fastNlMeansDenoising(img, None, h=7, templateWindowSize=7, searchWindowSize=21)


def apply_clahe(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(img)


def resize(img: np.ndarray, size: int = config.IMAGE_SIZE) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def normalize(img: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] then z-score normalise, guarding against flat images."""
    img = img.astype(np.float32) / 255.0
    std = img.std()
    if std > 1e-6:
        img = (img - img.mean()) / std
    # rescale to [0,1] for stable network inputs
    mn, mx = img.min(), img.max()
    if mx - mn > 1e-6:
        img = (img - mn) / (mx - mn)
    return img


def preprocess_image(
    img: np.ndarray,
    size: int = config.IMAGE_SIZE,
    denoise: bool = True,
    clahe: bool = True,
) -> np.ndarray:
    """Full deterministic preprocessing used for inference.

    Returns a ``float32`` HxW array in [0, 1].
    """
    g = to_grayscale(img)
    if denoise:
        g = remove_noise(g)
    if clahe:
        g = apply_clahe(g)
    g = resize(g, size)
    g = normalize(g)
    return g.astype(np.float32)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """HxW float array -> (1, 1, H, W) tensor."""
    t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    return t


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes into an OpenCV array (handles PNG/JPG/etc.)."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img


# --------------------------------------------------------------------------- #
# Augmentation (training only)
# --------------------------------------------------------------------------- #
class Augmentor:
    """Random augmentations applied jointly to an image (and optional mask)."""

    def __init__(
        self,
        hflip: float = 0.5,
        vflip: float = 0.5,
        rotate: float = 0.5,
        crop: float = 0.3,
        max_angle: float = 20.0,
        seed: Optional[int] = None,
    ):
        self.hflip = hflip
        self.vflip = vflip
        self.rotate = rotate
        self.crop = crop
        self.max_angle = max_angle
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray] = None):
        if self.rng.random() < self.hflip:
            img = np.ascontiguousarray(img[:, ::-1])
            if mask is not None:
                mask = np.ascontiguousarray(mask[:, ::-1])
        if self.rng.random() < self.vflip:
            img = np.ascontiguousarray(img[::-1, :])
            if mask is not None:
                mask = np.ascontiguousarray(mask[::-1, :])
        if self.rng.random() < self.rotate:
            angle = float(self.rng.uniform(-self.max_angle, self.max_angle))
            img, mask = self._rotate(img, mask, angle)
        if self.rng.random() < self.crop:
            img, mask = self._random_crop(img, mask)
        return (img, mask) if mask is not None else img

    def _rotate(self, img, mask, angle):
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if mask is not None:
            mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        return img, mask

    def _random_crop(self, img, mask, scale=(0.8, 1.0)):
        h, w = img.shape[:2]
        s = float(self.rng.uniform(*scale))
        ch, cw = int(h * s), int(w * s)
        y = int(self.rng.integers(0, h - ch + 1))
        x = int(self.rng.integers(0, w - cw + 1))
        img = cv2.resize(img[y : y + ch, x : x + cw], (w, h), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask[y : y + ch, x : x + cw], (w, h), interpolation=cv2.INTER_NEAREST)
        return img, mask
