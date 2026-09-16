"""Multi-class segmentation head for Method 2.

Where Method 1 predicts one binary whole-tumour mask, Method 2 predicts a
``num_classes``-way softmax over background and the individual tumour
sub-regions, because the downstream SPECT feature stage needs per-region uptake
statistics and cannot compute them from a merged mask.

The encoder/decoder body is the same well-understood U-Net topology (imported
from ``backend.models.unet`` rather than copy-pasted), configured with
``out_channels = len(seg_classes)``. It is registered under its own architecture
name so a Method 2 checkpoint can never be loaded into Method 1's binary U-Net.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from backend.models.unet import UNet

__all__ = ["MultiClassUNet"]


class MultiClassUNet(nn.Module):
    """U-Net body with a multi-class (softmax) output head."""

    architecture = "MultiClassUNet"

    def __init__(self, in_channels: int = 1, num_classes: int = 4, base: int = 32, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.hparams = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "base": base,
            "dropout": dropout,
        }
        self.unet = UNet(in_channels=in_channels, out_channels=num_classes, base=base, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)  # raw logits, (B, num_classes, H, W)

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Argmax label map, (B, H, W) int64."""
        return self.forward(x).argmax(dim=1)
