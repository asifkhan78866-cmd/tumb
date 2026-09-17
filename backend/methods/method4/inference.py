"""Red Fox optimised ZFNet inference."""
from __future__ import annotations

from backend import config
from backend.methods.common.classifier_engine import WholeImageClassifierEngine
from backend.methods.method4.models import ZFNet
from backend.methods.method4.transforms import DEFAULT_SPEC, to_input
from backend.methods.registry import METHOD4

__all__ = ["Method4Engine", "engine"]


class Method4Engine(WholeImageClassifierEngine):
    spec = METHOD4
    weights_path = config.M4_WEIGHTS_PATH
    architecture = "ZFNet"
    class_names = list(METHOD4.class_names)
    class_labels = dict(METHOD4.class_labels)
    image_size = DEFAULT_SPEC.image_size
    train_command = "python -m backend.methods.method4.training.train_classifier"

    def build_model(self, meta: dict):
        hp = (meta.get("extra") or {}).get("hyperparameters") or {}
        return ZFNet(num_classes=len(self.class_names), in_channels=1,
                     fc_units=int(hp.get("fc_units", 4096)), dropout=float(hp.get("dropout", 0.5)))

    def to_input(self, x):
        return to_input(x)

    def gradcam_layer(self, model):
        return model.gradcam_target

    def extra_details(self) -> dict:
        hp = (self.meta.get("extra") or {}).get("hyperparameters") or None
        return {"red_fox_optimized_hyperparameters": hp, "transform": DEFAULT_SPEC.to_dict()}


engine = Method4Engine()
