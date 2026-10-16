"""Red Fox optimised ZFNet inference.

The network is always the feature extractor. Which head produces the class
depends on what training left on disk: if ``rfo_svm.joblib`` exists, was built
from *this* ZFNet checkpoint and won on validation macro-F1, the SVM on the Red
Fox–selected features is served; otherwise the network's own softmax layer is.
A bundle built from a different checkpoint is ignored with a warning — its
feature indices would point into a feature vector that no longer exists.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from backend import config
from backend.methods.common.classifier_engine import WholeImageClassifierEngine
from backend.methods.method4.models import ZFNet
from backend.methods.method4.transforms import DEFAULT_SPEC, to_input
from backend.methods.registry import METHOD4

__all__ = ["Method4Engine", "engine", "SVM_PATH"]

SVM_PATH = config.M4_WEIGHTS_PATH.with_name("rfo_svm.joblib")


class Method4Engine(WholeImageClassifierEngine):
    spec = METHOD4
    weights_path = config.M4_WEIGHTS_PATH
    architecture = "ZFNet"
    class_names = list(METHOD4.class_names)
    class_labels = dict(METHOD4.class_labels)
    image_size = DEFAULT_SPEC.image_size
    train_command = "python -m backend.methods.method4.training.train_classifier"

    def __init__(self) -> None:
        super().__init__()
        self.svm: Optional[dict] = None

    def build_model(self, meta: dict):
        hp = (meta.get("extra") or {}).get("hyperparameters") or {}
        return ZFNet(num_classes=len(self.class_names), in_channels=1,
                     fc_units=int(hp.get("fc_units", 4096)), dropout=float(hp.get("dropout", 0.5)))

    def to_input(self, x):
        return to_input(x)

    def gradcam_layer(self, model):
        return model.gradcam_target

    # -- the SVM head ------------------------------------------------------- #
    def load_extras(self) -> None:
        self.svm = None
        if not SVM_PATH.exists():
            return
        try:
            import joblib

            bundle = joblib.load(SVM_PATH)
            version = str(self.meta.get("model_version") or "")
            if str(bundle.get("zfnet_model_version")) != version:
                self.load_warnings.append(
                    f"The Red Fox feature-selection SVM in {SVM_PATH.name} was built from ZFNet "
                    f"{bundle.get('zfnet_model_version')}, but the loaded checkpoint is {version}; "
                    f"it is ignored and the network's own softmax layer is used. Rebuild it with "
                    f"`python -m backend.methods.method4.training.train_feature_svm`."
                )
                return
            if list(bundle.get("class_names") or []) != self.class_names:
                self.load_warnings.append(
                    f"{SVM_PATH.name} was built for different classes; it is ignored."
                )
                return
            if not bundle.get("serve"):
                return  # built, but the softmax head scored better on validation
            self.svm = bundle
        except Exception as exc:  # pragma: no cover - a broken artefact must not break serving
            self.load_warnings.append(f"The feature-selection SVM could not be loaded: {exc}")

    @property
    def served_head(self) -> str:
        return "rfo_svm" if self.svm else "zfnet_softmax"

    @property
    def model_version(self) -> str:
        if self.svm:
            return str(self.svm.get("model_version") or "untrained")
        return super().model_version

    def class_probabilities(self, x: torch.Tensor) -> np.ndarray:
        if not self.svm:
            return super().class_probabilities(x)
        with torch.no_grad():
            features = self.model.penultimate(x).float().cpu().numpy()
        # Exactly the training transform: standardise with the training-split scaler,
        # then keep the columns Red Fox selected.
        selected = self.svm["scaler"].transform(features)[:, self.svm["kept_columns"]]
        return np.asarray(self.svm["svm"].predict_proba(selected), dtype=np.float64).squeeze(0)

    def extra_details(self) -> dict:
        hp = (self.meta.get("extra") or {}).get("hyperparameters") or None
        details = {
            "red_fox_optimized_hyperparameters": hp,
            "transform": DEFAULT_SPEC.to_dict(),
            "classifier_head": self.served_head,
        }
        if self.svm:
            details["feature_selection"] = {
                "algorithm": "Red Fox Optimization over the ZFNet feature vector",
                "features_available": self.svm.get("features_available"),
                "features_selected": self.svm.get("features_selected"),
                "svm": self.svm.get("svm_hyperparameters"),
                "feature_extractor_version": self.svm.get("zfnet_model_version"),
                "probability_estimates": "Platt scaling cross-fitted on the training split",
            }
            details["gradcam_note"] = (
                "The heatmap explains the ZFNet layer that produced the features, not the SVM's "
                "decision boundary."
            )
        return details


engine = Method4Engine()
