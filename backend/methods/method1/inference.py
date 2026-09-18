"""Method 1 inference — U-Net segmentation → ROI crop → ConvLSTM → Grad-CAM.

Two behaviours here are deliberate and load-bearing:

**An untrained network never produces a reported result.** The previous engine
fell back to randomly initialised weights and returned the resulting noise as a
"segmentation mask" — a random mask rendered next to a confidence percentage in
a medical UI. Now a missing or invalid checkpoint sets ``segmentation_available``
to False, attaches a warning, and returns no mask at all. A visual placeholder is
produced only when ``APP_ENV=development``, and it is drawn with the word
PLACEHOLDER burned into the image so it cannot be mistaken for output.

**Geometry comes from the checkpoint.** The transform spec recorded at training
time decides whether the classifier sees a whole slice or an ROI crop, so a model
trained on whole slices is never served crops. Untagged legacy checkpoints map to
``M1_V1_LEGACY``, which is what the shipped classifier was actually trained with.
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
from backend.methods.common.checkpoint import CheckpointError, load_checkpoint, peek_meta
from backend.methods.method1 import config as m1
from backend.methods.method1.models import ConvLSTMClassifier, UNet
from backend.methods.method1.transforms import (
    TransformSpec,
    classifier_input,
    decode_image_bytes,
    get_spec,
    preprocess,
    to_tensor,
)
from backend.methods.registry import METHOD1, dataset_context
from backend.utils.gradcam import GradCAM, overlay_heatmap

__all__ = ["Method1Result", "Method1Engine", "engine"]


@dataclass
class Method1Result:
    prediction: Optional[str] = None
    prediction_key: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: dict = field(default_factory=dict)
    segmentation_available: bool = False
    original_path: Optional[Path] = None
    mask_path: Optional[Path] = None
    overlay_path: Optional[Path] = None
    inference_time_s: float = 0.0
    model_version: str = "untrained"
    warnings: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    prediction_id: str = ""


class Method1Engine:
    """Lazily-loaded, thread-safe-enough singleton holding Method 1's models."""

    method_id = METHOD1.method_id

    def __init__(self) -> None:
        self.device = config.DEVICE
        self.seg_model: Optional[UNet] = None
        self.cls_model: Optional[ConvLSTMClassifier] = None
        self.seg_weights_loaded = False
        self.cls_weights_loaded = False
        self.seg_meta: dict = {}
        self.cls_meta: dict = {}
        self.load_warnings: list[str] = []
        self.transform: TransformSpec = m1.LEGACY_TRANSFORM
        self._loaded = False

    # ------------------------------------------------------------------ #
    def load(self, force: bool = False) -> None:
        if self._loaded and not force:
            return
        self.load_warnings = []
        self._load_segmentation()
        self._load_classifier()
        self._loaded = True

    def _load_segmentation(self) -> None:
        # Seeded so that if a fallback model is ever constructed it is at least
        # reproducible across restarts. Its output is still never reported.
        torch.manual_seed(config.SEED)
        model = UNet(m1.SEG_IN_CHANNELS, m1.SEG_OUT_CHANNELS, m1.BASE_FILTERS).to(self.device)
        self.seg_weights_loaded = False
        self.seg_meta = {}
        self.seg_mask_source = None
        try:
            meta, warns = load_checkpoint(
                model,
                m1.SEG_WEIGHTS_PATH,
                expected_method=self.method_id,
                expected_architecture=m1.SEG_ARCHITECTURE,
                expected_role="segmentation",
            )
            self.seg_meta = meta
            self.seg_weights_loaded = True
            self.load_warnings.extend(warns)
            extra = meta.get("extra") or {}
            self.seg_mask_source = str(extra.get("mask_source") or "unknown")
            # Trained on one sequence; on others it often outlines nothing at all,
            # which is a silent failure unless it is said out loud.
            trained_sequence = ("FLAIR" if "FLAIR" in self.seg_mask_source
                                else "contrast-enhanced T1" if "contrast-enhanced T1" in self.seg_mask_source
                                else None)
            if trained_sequence:
                other = "contrast-enhanced T1" if trained_sequence == "FLAIR" else "FLAIR"
                self.load_warnings.append(
                    f"The segmentation model was trained on {self.seg_mask_source}. On a "
                    f"different sequence ({other}, for example) it may outline nothing or "
                    f"outline the wrong region."
                )
            if not extra.get("ground_truth_masks", False):
                self.load_warnings.append(
                    f"The segmentation model was trained on {self.seg_mask_source} masks, "
                    f"not radiologist annotation: it learned to reproduce a brightness "
                    f"heuristic, so the outline marks bright regions and is not a validated "
                    f"tumour boundary. Train on BraTS for ground-truth segmentation."
                )
        except CheckpointError as exc:
            self.load_warnings.append(
                f"Segmentation unavailable: {exc} Train it with "
                f"`python -m backend.methods.method1.training.train_segmentation`."
            )
        model.eval()
        self.seg_model = model

    def _load_classifier(self) -> None:
        meta_peek = peek_meta(m1.CLS_WEIGHTS_PATH) or {}
        self.transform = get_spec(meta_peek.get("transform_id"))
        hp = dict(meta_peek.get("extra", {}).get("hyperparameters", {}) or {})

        torch.manual_seed(config.SEED)
        model = ConvLSTMClassifier(
            in_channels=1,
            num_classes=m1.NUM_CLASSES,
            base=int(hp.get("base", 32)),
            lstm_hidden=int(hp.get("lstm_hidden", 64)),
            lstm_steps=int(hp.get("lstm_steps", 3)),
            dropout=float(hp.get("dropout", 0.4)),
        ).to(self.device)

        self.cls_weights_loaded = False
        self.cls_meta = {}
        try:
            meta, warns = load_checkpoint(
                model,
                m1.CLS_WEIGHTS_PATH,
                expected_method=self.method_id,
                expected_architecture=m1.CLS_ARCHITECTURE,
                expected_role="classification",
                expected_class_names=m1.CLASS_NAMES,
            )
            self.cls_meta = meta
            self.cls_weights_loaded = True
            self.load_warnings.extend(warns)
        except CheckpointError as exc:
            self.load_warnings.append(
                f"Classification unavailable: {exc} Train it with "
                f"`python -m backend.methods.method1.training.train_classifier`."
            )
        model.eval()
        self.cls_model = model

    # ------------------------------------------------------------------ #
    @property
    def model_version(self) -> str:
        return str(self.cls_meta.get("model_version") or "untrained")

    def status(self) -> dict:
        self.load()
        return {
            "segmentation_available": self.seg_weights_loaded,
            "classifier_available": self.cls_weights_loaded,
            "transform": self.transform.to_dict(),
            "model_version": self.model_version,
            "warnings": list(self.load_warnings),
        }

    # ------------------------------------------------------------------ #
    def predict(self, image_bytes: bytes) -> Method1Result:
        """Run the full pipeline. Blocking and CPU-bound — call from a worker thread."""
        self.load()
        pid = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        warnings = list(self.load_warnings)

        raw = decode_image_bytes(image_bytes)
        proc = preprocess(raw, self.transform)

        # --- Segmentation (only when a real checkpoint is loaded) --------- #
        mask: Optional[np.ndarray] = None
        if self.seg_weights_loaded:
            with torch.no_grad():
                seg_prob = torch.sigmoid(self.seg_model(to_tensor(proc).to(self.device)))
            mask = (seg_prob.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        # --- Classifier input: exactly the training geometry -------------- #
        cls_img, roi_applied = classifier_input(proc, mask, self.transform)

        prediction = prediction_key = None
        confidence = None
        probabilities: dict[str, float] = {}
        overlay: Optional[np.ndarray] = None

        if self.cls_weights_loaded:
            cx = to_tensor(cls_img).to(self.device)
            with torch.no_grad():
                probs = F.softmax(self.cls_model(cx), dim=1).squeeze(0).cpu().numpy()
            idx = int(np.argmax(probs))
            prediction_key = m1.CLASS_NAMES[idx]
            prediction = m1.CLASS_LABELS[prediction_key]
            confidence = round(float(probs[idx] * 100.0), 2)
            probabilities = {
                m1.CLASS_LABELS[m1.CLASS_NAMES[i]]: round(float(p) * 100, 2)
                for i, p in enumerate(probs)
            }
            overlay = self._gradcam(cx, idx, cls_img)
        else:
            warnings.append("No classification result: the classifier is not trained.")

        elapsed = time.perf_counter() - started

        # --- Persist images ------------------------------------------------ #
        original_path = config.PREDICTIONS_DIR / f"{pid}_original.png"
        cv2.imwrite(str(original_path), (proc * 255).astype(np.uint8))

        mask_path = None
        if mask is not None:
            mask_path = config.PREDICTIONS_DIR / f"{pid}_mask.png"
            cv2.imwrite(str(mask_path), self._mask_overlay(proc, mask))
        elif config.APP_ENV == "development":
            mask_path = config.PREDICTIONS_DIR / f"{pid}_mask_placeholder.png"
            cv2.imwrite(str(mask_path), self._placeholder(proc))
            warnings.append(
                "The segmentation panel shows a development PLACEHOLDER, not a model "
                "output. No segmentation weights are loaded."
            )

        overlay_path = None
        if overlay is not None:
            overlay_path = config.PREDICTIONS_DIR / f"{pid}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)

        return Method1Result(
            prediction=prediction,
            prediction_key=prediction_key,
            confidence=confidence,
            probabilities=probabilities,
            segmentation_available=mask is not None,
            original_path=original_path,
            mask_path=mask_path,
            overlay_path=overlay_path,
            inference_time_s=round(elapsed, 4),
            model_version=self.model_version,
            warnings=warnings,
            details={
                "transform": self.transform.to_dict(),
                "roi_crop_applied": roi_applied,
                "segmentation_mask_source": self.seg_mask_source,
                "segmentation_is_ground_truth_trained": bool(
                    (self.seg_meta.get("extra") or {}).get("ground_truth_masks", False)
                ),
                "tumor_pixel_fraction": (
                    round(float((mask > 0).mean()), 4) if mask is not None else None
                ),
                "dataset_context": dataset_context(METHOD1),
                "segmentation_model": METHOD1.segmentation_model,
                "classifier_model": METHOD1.classifier_model,
                "optimization": METHOD1.optimization,
            },
            prediction_id=pid,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _mask_overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        base = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        colored = np.zeros_like(base)
        colored[mask > 0] = (0, 0, 255)
        out = cv2.addWeighted(colored, 0.5, base, 1.0, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 255), 1)
        return out

    @staticmethod
    def _placeholder(img: np.ndarray) -> np.ndarray:
        """Development-only stand-in, labelled so it cannot be mistaken for output."""
        base = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        base = (base * 0.35).astype(np.uint8)
        h, w = base.shape[:2]
        cv2.putText(base, "PLACEHOLDER", (int(w * 0.04), int(h * 0.48)),
                    cv2.FONT_HERSHEY_SIMPLEX, w / 340.0, (60, 60, 240), 1, cv2.LINE_AA)
        cv2.putText(base, "not a model output", (int(w * 0.04), int(h * 0.60)),
                    cv2.FONT_HERSHEY_SIMPLEX, w / 620.0, (200, 200, 200), 1, cv2.LINE_AA)
        return base

    def _gradcam(self, cx, class_idx: int, image: np.ndarray) -> Optional[np.ndarray]:
        try:
            target = self.cls_model.features[-1].block[0]
            cam_engine = GradCAM(self.cls_model, target)
            try:
                cam = cam_engine(cx.clone().requires_grad_(True), class_idx)
            finally:
                cam_engine.remove()
                self.cls_model.zero_grad(set_to_none=True)
            return overlay_heatmap(image, cam)
        except Exception as exc:  # pragma: no cover - explainability is best-effort
            print(f"[method1] Grad-CAM failed: {exc}")
            return None


engine = Method1Engine()
