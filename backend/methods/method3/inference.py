"""Transfer-learning inference: rebuild the selected backbone from its checkpoint."""
from __future__ import annotations

from backend import config
from backend.methods.common.classifier_engine import WholeImageClassifierEngine
from backend.methods.method3 import models as m3models
from backend.methods.method3.models import FusionEnsemble
from backend.methods.method3.transforms import DEFAULT_SPEC, to_input
from backend.methods.registry import METHOD3

__all__ = ["Method3Engine", "engine"]


class Method3Engine(WholeImageClassifierEngine):
    spec = METHOD3
    weights_path = config.M3_WEIGHTS_PATH
    architecture = "TransferLearningCNN"
    accepted_architectures = ("TransferLearningCNN", "FusionEnsembleCNN")
    class_names = list(METHOD3.class_names)
    class_labels = dict(METHOD3.class_labels)
    image_size = DEFAULT_SPEC.image_size
    train_command = "python -m backend.methods.method3.training.train_classifier"

    def _backbone(self, meta: dict) -> str:
        return str((meta.get("extra") or {}).get("backbone", ""))

    @staticmethod
    def _is_fusion(meta: dict) -> bool:
        return str((meta or {}).get("architecture") or "") == "FusionEnsembleCNN"

    def build_model(self, meta: dict):
        extra = (meta.get("extra") or {})
        hp = extra.get("hyperparameters") or {}
        if self._is_fusion(meta):
            # The checkpoint holds the trained head only; the frozen ImageNet
            # backbones are rebuilt here (torchvision caches them after the
            # first download) — see load_target.
            return FusionEnsemble(
                tuple(extra.get("backbones") or m3models.FUSION_BACKBONES),
                num_classes=len(self.class_names), dropout=float(hp.get("dropout", 0.5)),
                pretrained=True,
            )
        # pretrained=False: the checkpoint supplies every weight, so no download at serve time.
        return m3models.build(self._backbone(meta), len(self.class_names), pretrained=False,
                              dropout=float(hp.get("dropout", 0.3)))

    def load_target(self, model):
        return model.head if isinstance(model, FusionEnsemble) else model

    def to_input(self, x):
        return to_input(x)

    def gradcam_layer(self, model):
        if isinstance(model, FusionEnsemble):
            # One heatmap per image: drawn over the ResNet branch, the deepest of the four.
            return model.gradcam_backbone.last_conv
        return m3models.gradcam_layer(model, self._backbone(self.meta))

    def extra_details(self) -> dict:
        extra = (self.meta.get("extra") or {})
        fusion = self._is_fusion(self.meta)
        return {
            "architecture": self.architecture_in_use,
            "backbone": None if fusion else (self._backbone(self.meta) or None),
            "fused_backbones": list(extra.get("backbones") or []) if fusion else None,
            "gradcam_branch": "resnet50" if fusion else self._backbone(self.meta) or None,
            "transform": DEFAULT_SPEC.to_dict(),
        }


engine = Method3Engine()
