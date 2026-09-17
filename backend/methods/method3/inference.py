"""Transfer-learning inference: rebuild the selected backbone from its checkpoint."""
from __future__ import annotations

from backend import config
from backend.methods.common.classifier_engine import WholeImageClassifierEngine
from backend.methods.method3 import models as m3models
from backend.methods.method3.transforms import DEFAULT_SPEC, to_input
from backend.methods.registry import METHOD3

__all__ = ["Method3Engine", "engine"]


class Method3Engine(WholeImageClassifierEngine):
    spec = METHOD3
    weights_path = config.M3_WEIGHTS_PATH
    architecture = "TransferLearningCNN"
    class_names = list(METHOD3.class_names)
    class_labels = dict(METHOD3.class_labels)
    image_size = DEFAULT_SPEC.image_size
    train_command = "python -m backend.methods.method3.training.train_classifier"

    def _backbone(self, meta: dict) -> str:
        return str((meta.get("extra") or {}).get("backbone", ""))

    def build_model(self, meta: dict):
        hp = (meta.get("extra") or {}).get("hyperparameters") or {}
        # pretrained=False: the checkpoint supplies every weight, so no download at serve time.
        return m3models.build(self._backbone(meta), len(self.class_names), pretrained=False,
                              dropout=float(hp.get("dropout", 0.3)))

    def to_input(self, x):
        return to_input(x)

    def gradcam_layer(self, model):
        return m3models.gradcam_layer(model, self._backbone(self.meta))

    def extra_details(self) -> dict:
        return {"backbone": self._backbone(self.meta) or None, "transform": DEFAULT_SPEC.to_dict()}


engine = Method3Engine()
