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
