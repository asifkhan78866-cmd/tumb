"""ConvLSTM-based tumor classifier.

The segmented tumor region is fed through convolutional feature extractors, the
resulting feature maps are treated as a short spatial "sequence" and processed by
a ConvLSTM, followed by dense + softmax layers for the four-way classification:
Glioma / Meningioma / No-Tumor / Pituitary.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """A single Convolutional LSTM cell (Shi et al., 2015)."""

    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_ch = hidden_ch
        # Gates: input, forget, output, cell candidate -> 4 * hidden channels
        self.conv = nn.Conv2d(
            in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=padding, bias=True
        )

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_ch, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch: int, spatial: tuple[int, int], device):
        h = torch.zeros(batch, self.hidden_ch, *spatial, device=device)
        c = torch.zeros(batch, self.hidden_ch, *spatial, device=device)
        return h, c


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> MaxPool feature extractor block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class ConvLSTMClassifier(nn.Module):
    """Conv feature extractor + ConvLSTM + Dense/Softmax classifier.

    Input: (B, C, H, W) segmented tumor image.
    Output: (B, num_classes) logits.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base: int = 32,
        lstm_hidden: int = 64,
        lstm_steps: int = 3,
    ):
        super().__init__()
        self.lstm_steps = lstm_steps

        self.features = nn.Sequential(
            ConvBlock(in_channels, base),        # H/2
            ConvBlock(base, base * 2),           # H/4
            ConvBlock(base * 2, base * 4),       # H/8
        )
        self.convlstm = ConvLSTMCell(base * 4, lstm_hidden)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)                       # (B, C, h, w)
        b, _, h, w = feats.shape
        state = self.convlstm.init_state(b, (h, w), feats.device)
        # Recurrently refine the same feature map for `lstm_steps` steps,
        # letting the ConvLSTM accumulate spatial context.
        for _ in range(self.lstm_steps):
            state = self.convlstm(feats, state)
        h_final, _ = state
        return self.head(h_final)                      # raw logits
