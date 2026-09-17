"""Inference engine for whole-image CNN classifiers (no segmentation stage).

Shared by the transfer-learning method and the Red Fox optimised ZFNet. The
contract matches the other engines: a missing or foreign checkpoint means no
prediction (never a randomly initialised guess), and the geometry applied to
an upload is the one recorded in the checkpoint.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from backend import config
from backend.methods.common.checkpoint import CheckpointError, load_checkpoint, peek_meta
from backend.methods.registry import MethodSpec, dataset_context
from backend.utils.gradcam import GradCAM, overlay_heatmap

__all__ = ["ClassifierResult", "WholeImageClassifierEngine", "decode_to_gray224"]


def decode_to_gray224(data: bytes, size: int = 224) -> np.ndarray:
    """Upload bytes -> uint8 (size, size) grayscale, exactly as training cached it."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image bytes — is this a valid image file?")
    return gray_resize(img, size)


def gray_resize(img: np.ndarray, size: int) -> np.ndarray:
    if img.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code) if img.shape[2] in (3, 4) else img[..., 0]
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    interp = cv2.INTER_AREA if max(img.shape[:2]) > size else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


@dataclass
class ClassifierResult:
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


class WholeImageClassifierEngine:
    """Subclasses provide ``build_model(meta)``, ``to_input`` and ``gradcam_layer``."""

    spec: MethodSpec
    weights_path: Path
    architecture: str
    class_names: list[str]
    class_labels: dict[str, str]
    image_size: int = 224
    train_command: str = ""

    def __init__(self) -> None:
        self.device = config.DEVICE
        self.model = None
        self.meta: dict = {}
        self.loaded = False
        self.load_warnings: list[str] = []
        self._ready = False

    @property
    def method_id(self) -> str:
        return self.spec.method_id

    # -- subclass hooks ---------------------------------------------------- #
    def build_model(self, meta: dict) -> torch.nn.Module:  # pragma: no cover - abstract
        raise NotImplementedError

    def to_input(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def gradcam_layer(self, model: torch.nn.Module) -> torch.nn.Module:  # pragma: no cover
        raise NotImplementedError

    def extra_details(self) -> dict:
        return {}

    # -- loading ----------------------------------------------------------- #
    def load(self, force: bool = False) -> None:
        # Once loaded, stay loaded. While untrained, re-check on every request so a
        # checkpoint written by a training run is picked up without a server restart.
        if self._ready and not force and (self.loaded or not self.weights_path.exists()):
            return
        self.load_warnings, self.loaded, self.meta, self.model = [], False, {}, None
        meta = peek_meta(self.weights_path)
        if meta is None:
            self.load_warnings.append(
                f"Classification unavailable: {self.weights_path}: checkpoint not found. "
                f"Train it with `{self.train_command}`."
            )
            self._ready = True
            return
        try:
            model = self.build_model(meta).to(self.device)
            meta, warns = load_checkpoint(
                model, self.weights_path,
                expected_method=self.method_id,
                expected_architecture=self.architecture,
                expected_role="classification",
                expected_class_names=self.class_names,
            )
            model.eval()
            self.model, self.meta, self.loaded = model, meta, True
            self.load_warnings.extend(warns)
        except (CheckpointError, KeyError, ValueError, RuntimeError) as exc:
            self.load_warnings.append(f"Classification unavailable: {exc}")
        self._ready = True

    @property
    def model_version(self) -> str:
        return str(self.meta.get("model_version") or "untrained")

    def status(self) -> dict:
        self.load()
        return {
            "segmentation_available": False,
            "classifier_available": self.loaded,
            "model_version": self.model_version,
            "warnings": list(self.load_warnings),
        }

    # -- prediction -------------------------------------------------------- #
    def predict(self, image_bytes: bytes) -> ClassifierResult:
        self.load()
        pid = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        warnings = list(self.load_warnings)

        gray = decode_to_gray224(image_bytes, self.image_size)
        original_path = config.PREDICTIONS_DIR / f"{pid}_{self.method_id}_original.png"
        cv2.imwrite(str(original_path), gray)

        prediction = prediction_key = confidence = None
        probabilities: dict[str, float] = {}
        overlay_path = None
        result_source = "trained_model" if self.loaded else "none"
        model_version = self.model_version
        ai_details = None
        if self.loaded:
            x = self.to_input(torch.from_numpy(gray).float().div(255.0)[None, None].to(self.device))
            with torch.no_grad():
                probs = F.softmax(self.model(x).float(), dim=1).squeeze(0).cpu().numpy()
            idx = int(np.argmax(probs))
            prediction_key = self.class_names[idx]
            prediction = self.class_labels[prediction_key]
            confidence = round(float(probs[idx]) * 100, 2)
            probabilities = {
                self.class_labels[c]: round(float(p) * 100, 2) for c, p in zip(self.class_names, probs)
            }
            overlay = self._gradcam(x, idx, gray)
            if overlay is not None:
                overlay_path = config.PREDICTIONS_DIR / f"{pid}_{self.method_id}_overlay.png"
                cv2.imwrite(str(overlay_path), overlay)
        else:
            warnings.append("No classification result: this method's classifier is not trained.")
            if config.method2_ai_available():
                ai_details, ai_warnings = self._ai_assessment(image_bytes)
                warnings[:0] = ai_warnings  # disclaimer first, so truncated views keep it
                if ai_details:
                    result_source = "ai_assessment"
                    model_version = f"ai:{ai_details['model']}"
                    key = {"normal": "notumor"}.get(ai_details["predicted_class"],
                                                    ai_details["predicted_class"])
                    if key in self.class_names:
                        prediction_key, prediction = key, self.class_labels[key]
                        probabilities = {
                            self.class_labels[{"normal": "notumor"}.get(k, k)]: v
                            for k, v in ai_details["likelihoods"].items()
                        }
                        confidence = probabilities[prediction]
                    else:
                        warnings.append("The AI assessment was indeterminate, so no class is reported.")

        return ClassifierResult(
            prediction=prediction,
            prediction_key=prediction_key,
            confidence=confidence,
            probabilities=probabilities,
            segmentation_available=False,
            original_path=original_path,
            overlay_path=overlay_path,
            inference_time_s=round(time.perf_counter() - started, 4),
            model_version=model_version,
            warnings=warnings,
            details={
                "result_source": result_source,
                "ai_assessment": ai_details,
                "segmentation_in_method": False,
                "segmentation_note": "Not part of this method (whole-image classification)",
                "input_size": self.image_size,
                "dataset_context": dataset_context(self.spec),
                "classifier_model": self.spec.classifier_model,
                "optimization": self.spec.optimization,
                **self.extra_details(),
            },
            prediction_id=pid,
        )

    def _ai_assessment(self, image_bytes: bytes) -> tuple[Optional[dict], list[str]]:
        from backend.methods.method2.ai_assessment import run_ai_assessment

        return run_ai_assessment(
            image_bytes,
            not_from=f"this method's {self.architecture} model (not trained yet)",
            reason=(
                f"This method's {self.spec.short_name} model has not finished training, so "
                f"the image was read by an AI model instead. Results switch to the trained "
                f"model automatically once its checkpoint exists."
            ),
        )

    def _gradcam(self, x: torch.Tensor, class_idx: int, gray: np.ndarray) -> Optional[np.ndarray]:
        try:
            cam_engine = GradCAM(self.model, self.gradcam_layer(self.model))
            try:
                cam = cam_engine(x.clone().requires_grad_(True), class_idx)
            finally:
                cam_engine.remove()
                self.model.zero_grad(set_to_none=True)
            return overlay_heatmap(gray.astype(np.float32) / 255.0, cam)
        except Exception as exc:  # pragma: no cover - explainability is best-effort
            print(f"[{self.method_id}] Grad-CAM failed: {exc}")
            return None
