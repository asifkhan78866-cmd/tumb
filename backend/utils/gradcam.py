"""Grad-CAM explainability for the ConvLSTM classifier.

Registers hooks on a target convolutional layer, back-propagates the score of the
predicted class, and produces a heatmap that is overlaid on the input MRI.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward()

        # Global-average-pool the gradients to obtain channel weights.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 1e-8:
            cam /= cam.max()
        return cam


def overlay_heatmap(gray_img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a Grad-CAM heatmap (in [0,1]) on a grayscale image (in [0,1])."""
    base = (np.clip(gray_img, 0, 1) * 255).astype(np.uint8)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(heat, alpha, base, 1 - alpha, 0)
