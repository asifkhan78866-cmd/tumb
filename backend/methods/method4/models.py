"""ZFNet (Zeiler & Fergus, 2014), adapted for single-channel brain MRI.

The layer layout follows the paper: a 7×7 stride-2 stem with 96 filters, a
5×5 stride-2 layer with 256, three 3×3 layers (384, 384, 256), overlapping
3×3/2 max pooling, and two fully connected layers before the classifier.

Documented adaptations: one input channel instead of three; BatchNorm after
each convolution in place of local response normalisation (which the
literature has since superseded and which trains poorly from scratch on a
small dataset); and a configurable fully connected width and dropout, which
Red Fox Optimization searches over.
"""
from __future__ import annotations

from torch import nn

__all__ = ["ZFNet"]


def _conv(cin: int, cout: int, k: int, s: int, p: int) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class ZFNet(nn.Module):
    def __init__(self, num_classes: int = 4, in_channels: int = 1, fc_units: int = 4096,
                 dropout: float = 0.5):
        super().__init__()
        self.hparams = {"num_classes": num_classes, "in_channels": in_channels,
                        "fc_units": fc_units, "dropout": dropout}
        self.features = nn.Sequential(
            *_conv(in_channels, 96, 7, 2, 1),    # 224 -> 110
            nn.MaxPool2d(3, 2, 1),               # -> 55
            *_conv(96, 256, 5, 2, 0),            # -> 26
            nn.MaxPool2d(3, 2, 1),               # -> 13
            *_conv(256, 384, 3, 1, 1),
            *_conv(384, 384, 3, 1, 1),
            *_conv(384, 256, 3, 1, 1),           # conv5 (index 14): Grad-CAM target
            nn.MaxPool2d(3, 2, 0),               # -> 6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_units, fc_units),
            nn.ReLU(inplace=True),
            nn.Linear(fc_units, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    @property
    def gradcam_target(self) -> nn.Module:
        # conv5's own output: BatchNorm consumes it out-of-place, so the gradient
        # hook is not disturbed by the in-place ReLU that follows.
        return self.features[14]
