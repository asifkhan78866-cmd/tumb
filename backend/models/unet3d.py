"""3D U-Net for volumetric brain-tumour segmentation (Çiçek et al., 2016).

The 2D U-Net beside this one segments a slice at a time, which is what the API
can serve because an upload is one image. This one segments a *volume*: every
convolution and pooling operation is three-dimensional, so a voxel's label is
decided with the slices above and below it in view, which is the whole reason
the volumetric variant exists — tumour boundaries are surfaces, and a 2D model
has to rediscover continuity it was never shown.

It is deliberately narrower than the paper's (``base=16`` by default). 3D
feature maps cost memory cubically, and the point here is a trained, measured
volumetric model on one machine rather than a reproduction of a cluster run.
``base`` is recorded in the checkpoint so the model is rebuilt as trained.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UNet3D"]


class DoubleConv3d(nn.Module):
    """(Conv3d -> InstanceNorm -> ReLU) x 2.

    InstanceNorm rather than BatchNorm: a 3D batch is one or two volumes, and
    BatchNorm's statistics are unusable at that size.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout3d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    """Volumetric U-Net: three downsamplings, three upsamplings, skip connections."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2, base: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.hparams = {"in_channels": in_channels, "num_classes": num_classes,
                        "base": base, "dropout": dropout}
        b = base
        self.enc1 = DoubleConv3d(in_channels, b)
        self.enc2 = DoubleConv3d(b, b * 2)
        self.enc3 = DoubleConv3d(b * 2, b * 4)
        self.bottleneck = DoubleConv3d(b * 4, b * 8, dropout)
        self.pool = nn.MaxPool3d(2)

        self.up3 = nn.ConvTranspose3d(b * 8, b * 4, 2, stride=2)
        self.dec3 = DoubleConv3d(b * 8, b * 4)
        self.up2 = nn.ConvTranspose3d(b * 4, b * 2, 2, stride=2)
        self.dec2 = DoubleConv3d(b * 4, b * 2)
        self.up1 = nn.ConvTranspose3d(b * 2, b, 2, stride=2)
        self.dec1 = DoubleConv3d(b * 2, b)
        self.head = nn.Conv3d(b, num_classes, 1)

    @staticmethod
    def _join(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Concatenate a decoder map with its skip, padding an odd dimension.

        A volume whose size is not a multiple of 8 loses a voxel at each
        pooling step; padding here keeps such volumes usable instead of
        requiring every input to be pre-cropped.
        """
        if up.shape[2:] != skip.shape[2:]:
            diff = [s - u for s, u in zip(skip.shape[2:], up.shape[2:])]
            up = F.pad(up, [p for d in reversed(diff) for p in (d // 2, d - d // 2)])
        return torch.cat([up, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(self._join(self.up3(b), e3))
        d2 = self.dec2(self._join(self.up2(d3), e2))
        d1 = self.dec1(self._join(self.up1(d2), e1))
        return self.head(d1)
