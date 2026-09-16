"""Dense Convolutional Network classifier for Method 2.

**What this is, stated plainly:** ``DenseConvNetClassifier`` is a DenseNet-BC
(Huang et al., 2017, *Densely Connected Convolutional Networks*) implemented
from scratch and adapted to this task. It is not a novel architecture, and this
docstring exists so nothing downstream implies that it is. The adaptations are:

* single-channel (or few-channel) input rather than 3-channel RGB;
* a compact stem sized for 128x128 inputs instead of ImageNet's 224x224;
* configurable growth rate and block depths so the model fits a small dataset;
* an optional **modality feature vector** concatenated to the pooled
  convolutional features before the classifier head — this is where Method 2's
  SPECT feature stage enters the network.

The dense connectivity itself is the standard formulation: inside a dense block
every layer receives the concatenated feature maps of all preceding layers,
``x_l = H_l([x_0, x_1, ..., x_{l-1}])``, with BN-ReLU-Conv1x1 bottlenecks and
compression at the transition layers (hence "BC").

It shares no code with Method 1's ConvLSTM classifier by design.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["DenseConvNetClassifier", "DenseBlock", "TransitionLayer"]


class _BottleneckLayer(nn.Module):
    """BN-ReLU-Conv1x1 (4k channels) -> BN-ReLU-Conv3x3 (k channels)."""

    def __init__(self, in_ch: int, growth: int, bn_size: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = bn_size * growth
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, inner, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(inner)
        self.conv2 = nn.Conv2d(inner, growth, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.norm1(x)))
        out = self.conv2(self.relu(self.norm2(out)))
        return self.dropout(out)


class DenseBlock(nn.Module):
    """``num_layers`` bottleneck layers with dense (concatenative) connectivity."""

    def __init__(self, num_layers: int, in_ch: int, growth: int, bn_size: int = 4, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [_BottleneckLayer(in_ch + i * growth, growth, bn_size, dropout) for i in range(num_layers)]
        )
        self.out_channels = in_ch + num_layers * growth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    """BN-ReLU-Conv1x1 compression followed by 2x average pooling."""

    def __init__(self, in_ch: int, compression: float = 0.5):
        super().__init__()
        out_ch = max(1, int(in_ch * compression))
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DenseConvNetClassifier(nn.Module):
    """DenseNet-BC style classifier with an optional modality-feature branch.

    Parameters
    ----------
    in_channels : input image channels. Method 2 feeds the filtered slice plus
        one uptake map per segmented tumour sub-region.
    num_classes : number of output classes.
    growth_rate : ``k`` — feature maps added by each dense layer.
    block_config : layers per dense block.
    num_init_features : channels produced by the stem.
    feature_dim : length of the SPECT feature vector concatenated before the
        head; 0 disables the branch entirely.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        growth_rate: int = 12,
        block_config: tuple[int, ...] = (6, 12, 16),
        num_init_features: int = 24,
        bn_size: int = 4,
        compression: float = 0.5,
        dropout: float = 0.2,
        feature_dim: int = 0,
    ):
        super().__init__()
        self.hparams = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "growth_rate": growth_rate,
            "block_config": list(block_config),
            "num_init_features": num_init_features,
            "bn_size": bn_size,
            "compression": compression,
            "dropout": dropout,
            "feature_dim": feature_dim,
        }
        self.feature_dim = feature_dim

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        blocks: list[nn.Module] = []
        channels = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, channels, growth_rate, bn_size, dropout)
            blocks.append(block)
            channels = block.out_channels
            if i != len(block_config) - 1:
                trans = TransitionLayer(channels, compression)
                blocks.append(trans)
                channels = trans.out_channels
        self.blocks = nn.Sequential(*blocks)

        self.final_norm = nn.BatchNorm2d(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)

        head_in = channels + feature_dim
        self.feature_norm = nn.LayerNorm(feature_dim) if feature_dim > 0 else None
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        out = self.stem(x)
        out = self.blocks(out)
        out = torch.relu(self.final_norm(out))
        pooled = torch.flatten(self.pool(out), 1)

        if self.feature_dim > 0:
            if features is None:
                features = pooled.new_zeros((pooled.size(0), self.feature_dim))
            if features.shape[1] != self.feature_dim:
                raise ValueError(
                    f"Expected a modality feature vector of length {self.feature_dim}, "
                    f"got {features.shape[1]}."
                )
            pooled = torch.cat([pooled, self.feature_norm(features.float())], dim=1)

        return self.classifier(pooled)  # raw logits
