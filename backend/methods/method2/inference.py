"""Method 2 inference — filter → multi-class segmentation → SPECT stage → DCN.

Method 2 ships **untrained**: no checkpoint for either stage is included in this
repository. That is a deliberate, visible state, not a silent one. With no
weights the engine returns ``segmentation_available=False`` and
``prediction=None`` with warnings naming the exact training command, and it
never constructs a result from an untrained network.

Every model built here is seeded from ``config.SEED`` so that if a fallback
model is ever instantiated its behaviour is at least reproducible — but its
output is still never reported as a result.
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
from backend.methods.method2 import config as m2
from backend.methods.method2.features import build_inputs, feature_names
from backend.methods.method2.models import DenseConvNetClassifier, MultiClassUNet
from backend.methods.method2.transforms import decode_image_bytes, get_spec, preprocess
from backend.methods.registry import METHOD2, dataset_context

__all__ = ["Method2Result", "Method2Engine", "engine"]

# Distinct colour per tumour sub-region for the rendered mask (BGR).
_REGION_COLOURS = [(0, 0, 255), (0, 200, 255), (0, 255, 80)]


@dataclass
class Method2Result:
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


class Method2Engine:
    method_id = METHOD2.method_id

    def __init__(self) -> None:
        self.device = config.DEVICE
        self.seg_model: Optional[MultiClassUNet] = None
        self.cls_model: Optional[DenseConvNetClassifier] = None
        self.seg_weights_loaded = False
        self.cls_weights_loaded = False
        self.seg_meta: dict = {}
        self.cls_meta: dict = {}
        self.load_warnings: list[str] = []
        self.transform = m2.TRANSFORM
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
        torch.manual_seed(config.SEED)
        model = MultiClassUNet(in_channels=1, num_classes=m2.NUM_SEG_CLASSES).to(self.device)
        self.seg_weights_loaded = False
        self.seg_meta = {}
        try:
            meta, warns = load_checkpoint(
                model, m2.SEG_WEIGHTS_PATH,
                expected_method=self.method_id,
                expected_architecture=m2.SEG_ARCHITECTURE,
                expected_role="segmentation",
            )
            self.seg_meta = meta
            self.seg_weights_loaded = True
            self.transform = get_spec(meta.get("transform_id"))
            self.load_warnings.extend(warns)
        except CheckpointError as exc:
            self.load_warnings.append(
                f"Method 2 segmentation unavailable: {exc} Train it with "
                f"`python -m backend.methods.method2.training.train_segmentation`."
            )
        model.eval()
        self.seg_model = model

    def _load_classifier(self) -> None:
        meta_peek = peek_meta(m2.DCN_WEIGHTS_PATH) or {}
        hp = {**m2.DCN_DEFAULTS, **(meta_peek.get("extra", {}).get("hyperparameters", {}) or {})}

        torch.manual_seed(config.SEED)
        model = DenseConvNetClassifier(
            in_channels=m2.DCN_IN_CHANNELS,
            num_classes=m2.NUM_CLASSES,
            growth_rate=int(hp["growth_rate"]),
            block_config=tuple(hp["block_config"]),
            num_init_features=int(hp["num_init_features"]),
            compression=float(hp["compression"]),
            dropout=float(hp["dropout"]),
            feature_dim=m2.DCN_FEATURE_DIM,
        ).to(self.device)

        self.cls_weights_loaded = False
        self.cls_meta = {}
        self.trained_branch = None
        try:
            meta, warns = load_checkpoint(
                model, m2.DCN_WEIGHTS_PATH,
                expected_method=self.method_id,
                expected_architecture=m2.CLS_ARCHITECTURE,
                expected_role="classification",
                expected_class_names=m2.CLASS_NAMES,
            )
            self.cls_meta = meta
            self.cls_weights_loaded = True
            self.load_warnings.extend(warns)
            trained_modality = str(meta.get("extra", {}).get("modality", m2.MODALITY))
            self.trained_branch = trained_modality.upper()
            if self.trained_branch == "MRI":
                self.load_warnings.append(
                    "This result comes from the trained MRI branch of the MRI–SPECT fusion "
                    "model. The SPECT branch and the fusion step are not trained (no paired "
                    "MRI–SPECT data), so no fusion is performed."
                )
            elif self.trained_branch != m2.MODALITY.upper():
                self.load_warnings.append(
                    f"The DCN checkpoint was trained on {trained_modality} data but "
                    f"METHOD2_MODALITY is {m2.MODALITY}. Predictions are being made "
                    f"across modalities."
                )
        except CheckpointError as exc:
            self.load_warnings.append(
                f"Method 2 classification unavailable: {exc} Train it with "
                f"`python -m backend.methods.method2.training.train_classifier`."
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
            "modality": m2.MODALITY,
            "warnings": list(self.load_warnings),
        }

    # ------------------------------------------------------------------ #
    def predict(self, image_bytes: bytes) -> Method2Result:
        """Run the Method 2 pipeline. Blocking and CPU-bound — call from a worker."""
        self.load()
        pid = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        warnings = list(self.load_warnings)

        raw = decode_image_bytes(image_bytes)
        image = preprocess(raw, self.transform)

        # --- Multi-class segmentation ------------------------------------ #
        label_map: Optional[np.ndarray] = None
        if self.seg_weights_loaded:
            x = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                label_map = self.seg_model(x).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # --- Modality feature stage -------------------------------------- #
        # With no segmentation there are no regions, so every region descriptor
        # is zero. That is reported, not hidden.
        effective_labels = label_map if label_map is not None else np.zeros_like(image, dtype=np.uint8)
        channels, descriptor = build_inputs(image, effective_labels, m2.NUM_REGIONS)
        if label_map is None:
            warnings.append(
                f"The {m2.MODALITY} feature stage ran with an all-background region "
                f"map because no segmentation weights are loaded; every per-region "
                f"descriptor is zero."
            )

        # --- DCN classification ------------------------------------------ #
        prediction = prediction_key = None
        confidence = None
        probabilities: dict[str, float] = {}

        if self.cls_weights_loaded:
            xc = torch.from_numpy(channels).float().unsqueeze(0).to(self.device)
            fv = torch.from_numpy(descriptor).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = F.softmax(self.cls_model(xc, fv), dim=1).squeeze(0).cpu().numpy()
            idx = int(np.argmax(probs))
            prediction_key = m2.CLASS_NAMES[idx]
            prediction = m2.CLASS_LABELS[prediction_key]
            confidence = round(float(probs[idx] * 100.0), 2)
            probabilities = {
                m2.CLASS_LABELS[m2.CLASS_NAMES[i]]: round(float(p) * 100, 2)
                for i, p in enumerate(probs)
            }
        else:
            warnings.append("No classification result: Method 2's DCN is not trained.")

        # --- AI assessment (only while the DCN is untrained) --------------- #
        result_source = "dcn" if self.cls_weights_loaded else "none"
        model_version = self.model_version
        ai_details = None
        if not self.cls_weights_loaded and config.method2_ai_available():
            ai_details, ai_warnings = self._ai_assessment(image_bytes)
            # First, so truncated renderings (the PDF shows six) keep the disclaimer.
            warnings[:0] = ai_warnings
            if ai_details and ai_details["predicted_class"] in m2.CLASS_NAMES:
                result_source = "ai_assessment"
                model_version = f"ai:{ai_details['model']}"
                prediction_key = ai_details["predicted_class"]
                prediction = m2.CLASS_LABELS[prediction_key]
                probabilities = {
                    m2.CLASS_LABELS[k]: v for k, v in ai_details["likelihoods"].items()
                }
                confidence = probabilities[prediction]
            elif ai_details:
                result_source = "ai_assessment"
                model_version = f"ai:{ai_details['model']}"
                warnings.append(
                    "The AI assessment was indeterminate, so no class is reported. "
                    "See its findings for the reason."
                )

        elapsed = time.perf_counter() - started

        original_path = config.PREDICTIONS_DIR / f"{pid}_m2_original.png"
        cv2.imwrite(str(original_path), (image * 255).astype(np.uint8))

        mask_path = None
        if label_map is not None:
            mask_path = config.PREDICTIONS_DIR / f"{pid}_m2_mask.png"
            cv2.imwrite(str(mask_path), self._render_mask(image, label_map))

        region_areas = None
        if label_map is not None:
            region_areas = {
                m2.SEG_CLASS_NAMES[r]: round(float((label_map == r).mean()), 4)
                for r in range(m2.NUM_SEG_CLASSES)
            }

        return Method2Result(
            prediction=prediction,
            prediction_key=prediction_key,
            confidence=confidence,
            probabilities=probabilities,
            segmentation_available=label_map is not None,
            original_path=original_path,
            mask_path=mask_path,
            overlay_path=None,  # Method 2 has no Grad-CAM stage in its specification
            inference_time_s=round(elapsed, 4),
            model_version=model_version,
            warnings=warnings,
            details={
                "result_source": result_source,
                "trained_branch": self.trained_branch if self.cls_weights_loaded else None,
                "fusion_performed": False,
                "ai_assessment": ai_details,
                "transform": self.transform.to_dict(),
                "modality": m2.MODALITY,
                "spec_modality_label": m2.SPEC_MODALITY_LABEL,
                "segmentation_classes": m2.SEG_CLASS_NAMES,
                "region_area_fractions": region_areas,
                "feature_stage": {
                    "dimension": int(descriptor.shape[0]),
                    "names": feature_names(m2.SEG_CLASS_NAMES[1:], m2.MODALITY),
                    "values": [round(float(v), 5) for v in descriptor.tolist()],
                    "computed_from_segmentation": label_map is not None,
                },
                "dataset_context": dataset_context(METHOD2),
                "segmentation_model": METHOD2.segmentation_model,
                "classifier_model": METHOD2.classifier_model,
                "explainability": "Grad-CAM is not part of the Method 2 specification.",
            },
            prediction_id=pid,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _ai_assessment(image_bytes: bytes) -> tuple[Optional[dict], list[str]]:
        from backend.methods.method2.ai_assessment import run_ai_assessment

        details, warnings = run_ai_assessment(
            image_bytes,
            not_from="this method's trained MRI–SPECT fusion network (which is not trained)",
            reason=(
                "This method's MRI–SPECT fusion network is not trained (no paired MRI and "
                "SPECT scans exist to train it), so this single image was read by an AI model. "
                "No fusion was performed."
            ),
            allowed_modalities=("MRI", "SPECT"),
        )
        if details is not None:
            warnings.append(
                "No MRI–SPECT fusion was performed: a single image was assessed. Fusion needs "
                "paired MRI and SPECT scans of the same patient and a trained fusion network."
            )
        return details, warnings

    @staticmethod
    def _render_mask(image: np.ndarray, label_map: np.ndarray) -> np.ndarray:
        base = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        overlay = np.zeros_like(base)
        for r in range(1, m2.NUM_SEG_CLASSES):
            overlay[label_map == r] = _REGION_COLOURS[(r - 1) % len(_REGION_COLOURS)]
        out = cv2.addWeighted(overlay, 0.5, base, 1.0, 0)
        for r in range(1, m2.NUM_SEG_CLASSES):
            contours, _ = cv2.findContours(
                (label_map == r).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, _REGION_COLOURS[(r - 1) % len(_REGION_COLOURS)], 1)
        return out


engine = Method2Engine()
