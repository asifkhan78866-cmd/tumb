"""2D U-Net for binary brain-tumor segmentation.

Encoder / decoder with skip connections, batch normalisation and dropout — the
standard Ronneberger et al. architecture, parameterised so it works on CPU or GPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x 2 with dropout on the second block."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscale with maxpool then double-conv."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Upscale (transposed conv), concatenate skip connection, then double-conv."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if odd input sizes caused a mismatch
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = nn.functional.pad(
            x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Binary segmentation U-Net.

    Parameters
    ----------
    in_channels : number of input image channels (1 for grayscale MRI slice)
    out_channels : number of output mask channels (1 for binary tumor mask)
    base : number of filters in the first encoder block
    dropout : dropout probability used throughout
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.inc = DoubleConv(in_channels, base, dropout)
        self.down1 = Down(base, base * 2, dropout)
        self.down2 = Down(base * 2, base * 4, dropout)
        self.down3 = Down(base * 4, base * 8, dropout)
        self.down4 = Down(base * 8, base * 16, dropout)

        self.up1 = Up(base * 16, base * 8, dropout)
        self.up2 = Up(base * 8, base * 4, dropout)
        self.up3 = Up(base * 4, base * 2, dropout)
        self.up4 = Up(base * 2, base, dropout)
        self.outc = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)  # raw logits; apply sigmoid downstream
