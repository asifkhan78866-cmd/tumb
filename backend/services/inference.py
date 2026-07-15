"""End-to-end inference pipeline.

Upload -> preprocess -> U-Net segmentation -> crop tumor -> ConvLSTM
classification -> Grad-CAM overlay. Models are loaded lazily and cached; if
trained weights are missing the models fall back to randomly-initialised weights
so the API still works for demos (flagged via ``weights_loaded``).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from backend import config
from backend.models import ConvLSTMClassifier, UNet
from backend.utils.gradcam import GradCAM, overlay_heatmap
from backend.utils.preprocessing import decode_image_bytes, preprocess_image, to_tensor


@dataclass
class InferenceResult:
    prediction: str
    class_key: str
    confidence: float
    inference_time_s: float
    original_path: Path
    mask_path: Path
    overlay_path: Path
    probabilities: dict = field(default_factory=dict)
    prediction_id: str = ""


class InferenceEngine:
    """Singleton-style engine holding the loaded models."""

    def __init__(self):
        self.device = config.DEVICE
        self.seg_model: Optional[UNet] = None
        self.cls_model: Optional[ConvLSTMClassifier] = None
        self.seg_weights_loaded = False
        self.cls_weights_loaded = False

    # ------------------------------------------------------------------ #
    def load(self):
        if self.seg_model is None:
            self.seg_model = UNet(
                config.SEG_IN_CHANNELS, config.SEG_OUT_CHANNELS, config.BASE_FILTERS
            ).to(self.device)
            self.seg_weights_loaded = self._maybe_load(self.seg_model, config.SEG_WEIGHTS_PATH)
            self.seg_model.eval()

        if self.cls_model is None:
            self.cls_model = ConvLSTMClassifier(
                in_channels=1, num_classes=config.NUM_CLASSES
            ).to(self.device)
            self.cls_weights_loaded = self._maybe_load(self.cls_model, config.CLS_WEIGHTS_PATH)
            self.cls_model.eval()

    def _maybe_load(self, model: torch.nn.Module, path: Path) -> bool:
        if path.exists():
            ckpt = torch.load(path, map_location=self.device)
            state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state)
            print(f"[inference] Loaded weights from {path}")
            return True
        print(f"[inference] WARNING: {path} not found — using random weights.")
        return False

    # ------------------------------------------------------------------ #
    def predict(self, image_bytes: bytes) -> InferenceResult:
        self.load()
        pid = uuid.uuid4().hex[:12]
        start = time.perf_counter()

        raw = decode_image_bytes(image_bytes)
        proc = preprocess_image(raw)  # HxW float [0,1]
        x = to_tensor(proc).to(self.device)

        # --- Segmentation ---
        with torch.no_grad():
            seg_logits = self.seg_model(x)
            seg_prob = torch.sigmoid(seg_logits)
        mask = (seg_prob.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        # --- Crop tumor region (fallback to full image if empty mask) ---
        cropped = self._crop_to_mask(proc, mask)
        cx = to_tensor(cropped).to(self.device)

        # --- Classification ---
        with torch.no_grad():
            cls_logits = self.cls_model(cx)
            probs = F.softmax(cls_logits, dim=1).squeeze().cpu().numpy()
        class_idx = int(np.argmax(probs))
        class_key = config.CLASS_NAMES[class_idx]
        confidence = float(probs[class_idx] * 100.0)

        # --- Grad-CAM overlay ---
        overlay = self._gradcam_overlay(cx, class_idx, cropped)

        elapsed = time.perf_counter() - start

        # --- Persist images ---
        original_path = config.PREDICTIONS_DIR / f"{pid}_original.png"
        mask_path = config.PREDICTIONS_DIR / f"{pid}_mask.png"
        overlay_path = config.PREDICTIONS_DIR / f"{pid}_overlay.png"
        cv2.imwrite(str(original_path), (proc * 255).astype(np.uint8))
        cv2.imwrite(str(mask_path), self._mask_overlay(proc, mask))
        cv2.imwrite(str(overlay_path), overlay)

        return InferenceResult(
            prediction=config.CLASS_LABELS[class_key],
            class_key=class_key,
            confidence=round(confidence, 2),
            inference_time_s=round(elapsed, 4),
            original_path=original_path,
            mask_path=mask_path,
            overlay_path=overlay_path,
            probabilities={
                config.CLASS_LABELS[config.CLASS_NAMES[i]]: round(float(p) * 100, 2)
                for i, p in enumerate(probs)
            },
            prediction_id=pid,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _crop_to_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(mask > 0)
        if len(xs) < 20:  # empty / tiny mask -> use whole image
            return img
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        pad = 8
        y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
        y1, x1 = min(img.shape[0], y1 + pad), min(img.shape[1], x1 + pad)
        crop = img[y0:y1, x0:x1]
        return cv2.resize(crop, (config.IMAGE_SIZE, config.IMAGE_SIZE))

    @staticmethod
    def _mask_overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        base = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        colored = np.zeros_like(base)
        colored[mask > 0] = (0, 0, 255)  # red tumor overlay (BGR)
        out = cv2.addWeighted(colored, 0.5, base, 1.0, 0)
        # draw contour for clarity
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 255), 1)
        return out

    def _gradcam_overlay(self, cx, class_idx, cropped) -> np.ndarray:
        try:
            target = self.cls_model.features[-1].block[0]  # last conv layer
            cam_engine = GradCAM(self.cls_model, target)
            cx_grad = cx.clone().requires_grad_(True)
            cam = cam_engine(cx_grad, class_idx)
            cam_engine.remove()
            return overlay_heatmap(cropped, cam)
        except Exception as exc:  # pragma: no cover
            print(f"[inference] Grad-CAM failed: {exc}")
            base = cv2.cvtColor((cropped * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            return base


engine = InferenceEngine()
