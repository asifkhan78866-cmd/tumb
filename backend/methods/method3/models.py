"""Pretrained backbones with a new four-class head.

``build(name, num_classes, pretrained)`` returns the model; ``head_parameters``
and ``gradcam_layer`` let training freeze the backbone for warm-up and let
inference draw a heatmap without knowing the architecture.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models as tvm

__all__ = ["BACKBONES", "build", "head_parameters", "gradcam_layer"]

BACKBONES = {
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "densenet121": (tvm.densenet121, tvm.DenseNet121_Weights.IMAGENET1K_V1),
}


def build(name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    ctor, weights = BACKBONES[name]
    model = ctor(weights=weights if pretrained else None)
    if name == "resnet50":
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif name == "efficientnet_b0":
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(model.classifier[1].in_features, num_classes)
        )
    else:  # densenet121
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(model.classifier.in_features, num_classes)
        )
    return model


def head_parameters(model: nn.Module, name: str):
    return (model.fc if name == "resnet50" else model.classifier).parameters()


def gradcam_layer(model: nn.Module, name: str) -> nn.Module:
    if name == "resnet50":
        return model.layer4
    if name == "efficientnet_b0":
        return model.features[-1]
    return model.features.denseblock4


def set_backbone_trainable(model: nn.Module, name: str, trainable: bool) -> None:
    head = set(id(p) for p in head_parameters(model, name))
    for p in model.parameters():
        if id(p) not in head:
            p.requires_grad_(trainable)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------- #
# Feature-fusion ensemble
# --------------------------------------------------------------------------- #
# Four ImageNet backbones read the same image; each is reduced to a short
# branch vector, the four are concatenated, and a meta-learner decides how to
# weigh them. The backbones stay frozen — only the branch heads and the
# meta-learner are trained, which is what makes caching their output valid.
FUSION_BACKBONES = ("googlenet", "vgg16", "resnet50", "alexnet")

_FUSION_SPECS = {
    "googlenet": (tvm.googlenet, tvm.GoogLeNet_Weights.IMAGENET1K_V1, 1024),
    "vgg16": (tvm.vgg16, tvm.VGG16_Weights.IMAGENET1K_V1, 512),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2, 2048),
    "alexnet": (tvm.alexnet, tvm.AlexNet_Weights.IMAGENET1K_V1, 256),
}


def feature_dim(name: str) -> int:
    return _FUSION_SPECS[name][2]


class FrozenBackbone(nn.Module):
    """One ImageNet backbone reduced to a pooled feature vector.

    ``pretrained=False`` builds the same shape with random weights, which is
    what a checkpoint load needs — the stored weights replace them.
    """

    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        if name not in _FUSION_SPECS:
            raise ValueError(f"Unknown fusion backbone {name!r}; choose from {sorted(_FUSION_SPECS)}")
        ctor, weights, self.out_dim = _FUSION_SPECS[name]
        self.name = name
        kwargs = {"weights": weights} if pretrained else {"weights": None}
        if name == "googlenet":
            # Without this, torchvision builds the auxiliary classifiers that only
            # exist for training and warns on every load.
            kwargs["aux_logits"] = bool(pretrained)
        model = ctor(**kwargs)
        if name in ("googlenet", "resnet50"):
            model.fc = nn.Identity()
            if name == "googlenet":
                model.aux1 = model.aux2 = None
            self.body, self.pool = model, None
        else:  # vgg16, alexnet: convolutional stack + our own pooling
            self.body, self.pool = model.features, nn.AdaptiveAvgPool2d(1)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # frozen: never leaves eval
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        if self.pool is not None:
            out = self.pool(out)
        return torch.flatten(out, 1)

    @property
    def last_conv(self) -> nn.Module:
        """Layer a Grad-CAM heatmap is drawn over."""
        if self.name == "resnet50":
            return self.body.layer4
        if self.name == "googlenet":
            return self.body.inception5b
        return self.body[-3]  # last conv block of vgg16 / alexnet


def _branch_head(in_dim: int, dropout: float = 0.5) -> nn.Sequential:
    """Per-backbone head: in_dim -> 256 -> 128 -> 64."""
    return nn.Sequential(
        nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
        nn.Linear(256, 128), nn.ReLU(inplace=True),
        nn.Linear(128, 64), nn.ReLU(inplace=True),
    )


class FusionHead(nn.Module):
    """Branch heads + meta-learner over the concatenated branch vectors.

    Trained on cached backbone features, so it can also run standalone.
    """

    def __init__(self, backbones: tuple[str, ...] = FUSION_BACKBONES, num_classes: int = 4,
                 dropout: float = 0.5):
        super().__init__()
        self.backbone_names = tuple(backbones)
        self.hparams = {"backbones": list(self.backbone_names), "num_classes": num_classes,
                        "dropout": dropout}
        self.branches = nn.ModuleDict(
            {name: _branch_head(feature_dim(name), dropout) for name in self.backbone_names}
        )
        self.meta = nn.Sequential(
            nn.Linear(64 * len(self.backbone_names), 256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        fused = torch.cat([self.branches[n](features[n]) for n in self.backbone_names], dim=1)
        return self.meta(fused)


class FusionEnsemble(nn.Module):
    """Serving model: the frozen backbones plus the trained fusion head."""

    def __init__(self, backbones: tuple[str, ...] = FUSION_BACKBONES, num_classes: int = 4,
                 dropout: float = 0.5, pretrained: bool = True):
        super().__init__()
        self.backbone_names = tuple(backbones)
        self.backbones = nn.ModuleDict(
            {name: FrozenBackbone(name, pretrained=pretrained) for name in self.backbone_names}
        )
        self.head = FusionHead(self.backbone_names, num_classes, dropout)
        self.hparams = self.head.hparams

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The backbones are frozen by their parameters, not by no_grad: Grad-CAM
        # needs the graph to reach a branch's last convolution. Callers that only
        # want a prediction wrap this in torch.no_grad() themselves.
        features = {n: self.backbones[n](x) for n in self.backbone_names}
        return self.head(features)

    @property
    def gradcam_backbone(self) -> FrozenBackbone:
        """ResNet-50 when present: the deepest branch, and the clearest heatmaps."""
        name = "resnet50" if "resnet50" in self.backbones else self.backbone_names[0]
        return self.backbones[name]
