"""Loss functions and segmentation metrics."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation (expects raw logits)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross-Entropy loss."""

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice


@torch.no_grad()
def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, smooth: float = 1.0) -> float:
    probs = (torch.sigmoid(logits) > thr).float()
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + smooth) / (union + smooth)
    return dice.mean().item()


@torch.no_grad()
def iou_score(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, smooth: float = 1.0) -> float:
    probs = (torch.sigmoid(logits) > thr).float()
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1) - inter
    iou = (inter + smooth) / (union + smooth)
    return iou.mean().item()
