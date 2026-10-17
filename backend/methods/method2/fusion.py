"""MRI–SPECT image fusion: the stage this method is named after.

The two modalities carry different information. MRI resolves anatomy — edges,
boundaries, structure — in a single intensity channel. SPECT resolves function:
a low-resolution, colour-mapped perfusion or uptake image whose *colour* is the
measurement and whose spatial detail is poor. Fusing them means keeping SPECT's
colour where it is and borrowing MRI's structure for the detail.

That is exactly what the transform here does, and it is deterministic — there
is nothing to train:

1. The SPECT image is converted to YCrCb. ``Cr``/``Cb`` hold the colour, which
   *is* the functional reading, so they are carried through untouched. Only the
   luminance ``Y`` takes part in the fusion.
2. ``Y`` and the MRI slice are decomposed into Laplacian pyramids. At every
   detail level the coefficient with the larger local energy wins, which is the
   classical multi-scale rule: whichever source has real structure at that
   scale and position contributes it, instead of the two being averaged into
   something blurrier than either.
3. The coarsest level is a weighted average, so overall brightness stays
   between the two sources rather than tracking only one of them.
4. The fused luminance is recombined with the untouched ``Cr``/``Cb``. Hue
   therefore survives the fusion; what an 8-bit RGB rendering can still shift is
   colour *magnitude*, where a brightened luminance clips against the gamut.

No labels are involved and none are needed: this stage is measured with
reference-free fusion metrics (``metrics`` below), not with classification
accuracy. The method's four-class prediction still comes from its trained MRI
branch, and the code never pretends otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

__all__ = ["FusedPair", "fuse", "average_fuse", "metrics", "to_uint8", "DEFAULT_LEVELS"]

DEFAULT_LEVELS = 4
_ENERGY_WINDOW = 5


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0 if img.dtype != np.uint8 and img.max() <= 1.0 + 1e-6 else img,
                   0, 255).astype(np.uint8)


def _as_float(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    return img / 255.0 if img.max() > 1.0 + 1e-6 else img


@dataclass
class FusedPair:
    """A fused slice plus the inputs it came from, all float32 in [0, 1]."""

    fused_rgb: np.ndarray           # (H, W, 3)
    fused_luminance: np.ndarray     # (H, W)
    mri: np.ndarray                 # (H, W)
    spect_rgb: np.ndarray           # (H, W, 3)
    spect_luminance: np.ndarray     # (H, W)
    levels: int = DEFAULT_LEVELS
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Laplacian pyramid fusion
# --------------------------------------------------------------------------- #
def _gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    pyramid = [img]
    for _ in range(levels):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def _laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    gauss = _gaussian_pyramid(img, levels)
    lap = []
    for i in range(levels):
        size = (gauss[i].shape[1], gauss[i].shape[0])
        lap.append(gauss[i] - cv2.pyrUp(gauss[i + 1], dstsize=size))
    lap.append(gauss[-1])  # the residual low-pass image
    return lap


def _local_energy(band: np.ndarray, window: int = _ENERGY_WINDOW) -> np.ndarray:
    """Energy of a detail band in a neighbourhood, not at a single pixel.

    Choosing per pixel on magnitude alone is noise-sensitive: one bright speck
    wins a coefficient its neighbourhood does not support. Summing squares over
    a window asks which source actually carries structure there.
    """
    return cv2.boxFilter(band * band, -1, (window, window), normalize=True)


def _reconstruct(pyramid: list[np.ndarray]) -> np.ndarray:
    img = pyramid[-1]
    for band in reversed(pyramid[:-1]):
        img = cv2.pyrUp(img, dstsize=(band.shape[1], band.shape[0])) + band
    return img


def fuse(mri: np.ndarray, spect: np.ndarray, levels: int = DEFAULT_LEVELS,
         base_weight: float = 0.5) -> FusedPair:
    """Fuse an MRI slice with its registered SPECT counterpart.

    ``mri`` is greyscale, ``spect`` is the colour SPECT image. Either may be
    uint8 or float in [0, 1]; the SPECT image is resized to the MRI's geometry
    when the two differ, which is recorded in the result's notes.
    """
    mri = _as_float(mri)
    if mri.ndim == 3:
        mri = cv2.cvtColor(to_uint8(mri), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    spect_u8 = to_uint8(_as_float(spect))
    if spect_u8.ndim == 2:
        spect_u8 = cv2.cvtColor(spect_u8, cv2.COLOR_GRAY2BGR)

    notes: list[str] = []
    if spect_u8.shape[:2] != mri.shape[:2]:
        notes.append(
            f"The SPECT image was {spect_u8.shape[1]}×{spect_u8.shape[0]} and the MRI "
            f"{mri.shape[1]}×{mri.shape[0]}; the SPECT image was resampled to the MRI's "
            f"geometry. The pair is assumed to be registered — this does not register them."
        )
        spect_u8 = cv2.resize(spect_u8, (mri.shape[1], mri.shape[0]), interpolation=cv2.INTER_LINEAR)

    ycrcb = cv2.cvtColor(spect_u8, cv2.COLOR_BGR2YCrCb)
    spect_y = ycrcb[..., 0].astype(np.float32) / 255.0

    # A pyramid level halves the size, so the number of levels is capped by the image.
    usable = int(min(levels, max(1, np.floor(np.log2(min(mri.shape[:2]))) - 2)))
    if usable != levels:
        notes.append(f"Pyramid depth reduced from {levels} to {usable} for a "
                     f"{mri.shape[1]}×{mri.shape[0]} image.")

    lap_mri = _laplacian_pyramid(mri, usable)
    lap_spect = _laplacian_pyramid(spect_y, usable)

    fused_pyramid = []
    for i, (a, b) in enumerate(zip(lap_mri, lap_spect)):
        if i == len(lap_mri) - 1:  # residual: keep overall brightness between the sources
            fused_pyramid.append(base_weight * a + (1.0 - base_weight) * b)
        else:
            take_mri = _local_energy(a) >= _local_energy(b)
            fused_pyramid.append(np.where(take_mri, a, b))

    fused_y = np.clip(_reconstruct(fused_pyramid), 0.0, 1.0)
    out = ycrcb.copy()
    out[..., 0] = to_uint8(fused_y)
    fused_bgr = cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)
    return FusedPair(
        fused_rgb=fused_bgr.astype(np.float32) / 255.0,
        fused_luminance=fused_y,
        mri=mri,
        spect_rgb=spect_u8.astype(np.float32) / 255.0,
        spect_luminance=spect_y,
        levels=usable,
        notes=notes,
    )


def average_fuse(mri: np.ndarray, spect_luminance: np.ndarray) -> np.ndarray:
    """The trivial baseline: the mean of the two luminances.

    Reported next to the real fusion so its metrics mean something — a number
    with nothing to compare against says little about whether fusion worked.
    """
    return np.clip(0.5 * _as_float(mri) + 0.5 * _as_float(spect_luminance), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Reference-free fusion quality
# --------------------------------------------------------------------------- #
def _entropy(img: np.ndarray) -> float:
    hist = np.bincount(to_uint8(img).ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    joint, _, _ = np.histogram2d(to_uint8(a).ravel(), to_uint8(b).ravel(), bins=bins,
                                 range=[[0, 255], [0, 255]])
    joint = joint / max(joint.sum(), 1.0)
    px, py = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (px @ py)[nz])).sum())


def _spatial_frequency(img: np.ndarray) -> float:
    """How much detail the image carries (row and column gradient energy)."""
    x = _as_float(img)
    rf = np.sqrt(np.mean(np.diff(x, axis=1) ** 2))
    cf = np.sqrt(np.mean(np.diff(x, axis=0) ** 2))
    return float(np.sqrt(rf ** 2 + cf ** 2))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _as_float(a), _as_float(b)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    saa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    sbb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    sab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    ssim_map = (((2 * mu_a * mu_b + c1) * (2 * sab + c2)) /
                ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2)))
    return float(ssim_map.mean())


def _gradient(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _as_float(img)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2), np.arctan2(gy, gx)


def _qabf(a: np.ndarray, b: np.ndarray, fused: np.ndarray) -> float:
    """Xydeas & Petrovic edge-preservation measure, in [0, 1].

    How much of each source's gradient information survives into the fused
    image, weighted by how strong that gradient was. This is the standard
    measure for whether a fusion kept detail or washed it out.
    """
    ga, aa = _gradient(a)
    gb, ab = _gradient(b)
    gf, af = _gradient(fused)
    eps = 1e-8

    def q(gs, as_, ):
        g_ratio = np.where(gs > gf, gf / (gs + eps), gs / (gf + eps))
        g_ratio = np.clip(g_ratio, 0.0, 1.0)
        a_ratio = 1.0 - np.abs(as_ - af) / (np.pi / 2)
        a_ratio = np.clip(a_ratio, 0.0, 1.0)
        qg = 0.9994 / (1 + np.exp(-15 * (g_ratio - 0.5)))
        qa = 0.9879 / (1 + np.exp(-22 * (a_ratio - 0.8)))
        return qg * qa

    qa_, qb_ = q(ga, aa), q(gb, ab)
    denom = (ga + gb).sum() + eps
    return float((qa_ * ga + qb_ * gb).sum() / denom)


def metrics(mri: np.ndarray, spect_luminance: np.ndarray, fused: np.ndarray) -> dict:
    """Reference-free fusion quality for one fused slice.

    There is no ground-truth "correct fusion" to compare against, so every
    number here describes the fused image's relationship to its two sources.
    """
    return {
        "entropy": round(_entropy(fused), 4),
        "std": round(float(_as_float(fused).std()), 4),
        "spatial_frequency": round(_spatial_frequency(fused), 4),
        "mutual_information_mri": round(_mutual_information(fused, mri), 4),
        "mutual_information_spect": round(_mutual_information(fused, spect_luminance), 4),
        "ssim_mri": round(_ssim(fused, mri), 4),
        "ssim_spect": round(_ssim(fused, spect_luminance), 4),
        "qabf": round(_qabf(mri, spect_luminance, fused), 4),
    }


def side_by_side(pair: FusedPair, labels: tuple[str, str, str] = ("MRI", "SPECT", "Fused"),
                 scale: int = 1) -> np.ndarray:
    """MRI | SPECT | fused as one labelled BGR image, for the gallery."""
    panels = [cv2.cvtColor(to_uint8(pair.mri), cv2.COLOR_GRAY2BGR),
              to_uint8(pair.spect_rgb), to_uint8(pair.fused_rgb)]
    if scale != 1:
        panels = [cv2.resize(p, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                  for p in panels]
    out = np.hstack(panels)
    for i, label in enumerate(labels):
        cv2.putText(out, label, (10 + i * panels[0].shape[1], 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def fuse_bytes(mri_bytes: bytes, spect_bytes: bytes,
               levels: int = DEFAULT_LEVELS) -> Optional[FusedPair]:
    """Decode two uploaded images and fuse them; ``None`` if either fails to decode."""
    mri = cv2.imdecode(np.frombuffer(mri_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    spect = cv2.imdecode(np.frombuffer(spect_bytes, np.uint8), cv2.IMREAD_COLOR)
    if mri is None or spect is None:
        return None
    return fuse(mri, spect, levels=levels)
