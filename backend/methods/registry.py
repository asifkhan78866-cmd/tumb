"""The method registry — the single source of truth about what methods exist.

Each :class:`MethodSpec` is a *static description*: display name, pipeline
stages, datasets, model names, class list, weight roles and entry points. It
holds no torch objects and imports nothing heavy, so ``GET /api/methods`` and
the unit tests are cheap and the registry can be read without a working CUDA or
torch install.

Anything that depends on the machine — does a checkpoint exist, is a dataset on
disk, what did the last run score — is resolved on demand by
:func:`runtime_status`, which imports :mod:`backend.config` lazily.

Adding a third method means adding one ``MethodSpec`` and one package under
``backend/methods/``. No dispatch table anywhere else in the codebase needs to
change; the API, the frontend and the report all iterate this registry.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "PipelineStage",
    "DatasetSpec",
    "WeightSpec",
    "MethodSpec",
    "METHODS",
    "METHOD_IDS",
    "get_method",
    "list_methods",
    "load_engine",
    "runtime_status",
]


@dataclass(frozen=True)
class PipelineStage:
    id: str
    label: str
    kind: str  # input | preprocess | segmentation | feature | classifier | optimization | output | eval
    description: str = ""


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    role: str  # segmentation | classification | modality
    source: str = ""
    path_attr: str = ""  # attribute on backend.config holding the root path
    compatible: bool = True
    notes: str = ""


@dataclass(frozen=True)
class WeightSpec:
    role: str  # segmentation | classification
    path_attr: str  # attribute on backend.config
    architecture: str
    env_var: str = ""


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    display_name: str
    short_name: str
    summary: str
    modality: str
    class_names: tuple[str, ...]
    class_labels: dict[str, str]
    pipeline: tuple[PipelineStage, ...]
    datasets: tuple[DatasetSpec, ...]
    weights: tuple[WeightSpec, ...]
    segmentation_model: str
    classifier_model: str
    optimization: Optional[str]
    engine_module: str  # "backend.methods.method1.inference"
    transforms_module: str
    training_entrypoints: tuple[str, ...]
    metrics_filename: str
    run_card_filenames: dict[str, str]
    optimization_result_filename: Optional[str] = None
    notes: tuple[str, ...] = ()

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def labels(self) -> list[str]:
        return [self.class_labels[c] for c in self.class_names]

    def pipeline_labels(self) -> list[str]:
        return [s.label for s in self.pipeline]


# --------------------------------------------------------------------------- #
# Method 1 — the pipeline this repository already had, now formalised
# --------------------------------------------------------------------------- #
# NOTE ON CLASS ORDER: the classifier checkpoint that ships in this repository
# was trained with exactly this index order. Reordering it silently relabels
# every prediction, so the order is frozen here and asserted at load time by
# backend.methods.common.checkpoint.validate_meta.
METHOD1_CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
METHOD1_CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    # Displayed as "No Tumor" (not "Normal") because the shipped prediction
    # history and the existing UI compare against this exact string. The class
    # is the same "normal / no tumour" class the specification calls Normal.
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}

METHOD1 = MethodSpec(
    method_id="method1",
    display_name="Method 1 — 3D/2D U-Net Segmentation + ConvLSTM Classification + SFLA Optimization",
    short_name="U-Net + ConvLSTM + SFLA",
    summary=(
        "Binary tumour segmentation with a 2D U-Net, mask-driven ROI extraction, "
        "then four-class typing with a ConvLSTM classifier whose hyper-parameters "
        "are chosen by a Shuffled Frog Leaping Algorithm search."
    ),
    modality="MRI",
    class_names=METHOD1_CLASS_NAMES,
    class_labels=METHOD1_CLASS_LABELS,
    pipeline=(
        PipelineStage("input", "Input MRI", "input", "Single MRI slice (PNG/JPG/TIFF)."),
        PipelineStage("collect", "Data collection", "input", "BraTS 2020/2023 volumes and the Kaggle 4-class MRI set."),
        PipelineStage("preprocess", "Preprocessing", "preprocess", "Grayscale, denoise, CLAHE, resize, min-max normalise."),
        PipelineStage("segment", "U-Net segmentation", "segmentation", "2D U-Net encoder/decoder with skip connections; binary whole-tumour mask."),
        PipelineStage("roi", "ROI crop", "preprocess", "Bounding box of the mask, padded, resampled to the model input size."),
        PipelineStage("classify", "ConvLSTM classification", "classifier", "Conv feature stack + ConvLSTM cell + dense head."),
        PipelineStage("sfla", "SFLA optimization", "optimization", "Shuffled Frog Leaping search over classifier hyper-parameters (train/val only)."),
        PipelineStage("predict", "Prediction", "output", "Normal (No Tumor) / Glioma / Meningioma / Pituitary."),
        PipelineStage("gradcam", "Grad-CAM", "output", "Class-discriminative heatmap over the classified ROI."),
        PipelineStage("evaluate", "Performance evaluation", "eval", "Dice/IoU for segmentation; accuracy/precision/recall/F1/specificity for classification."),
    ),
    datasets=(
        DatasetSpec(
            key="brats",
            name="BraTS 2023 (2020 mirror)",
            role="segmentation",
            source="kaggle:awsaf49/brats2020-training-data",
            path_attr="BRATS_DATASET_PATH",
            notes="Ground-truth multi-region masks; volume ids give a true patient-level split.",
        ),
        DatasetSpec(
            key="bri",
            name="Brain Tumor MRI Dataset (BRI)",
            role="classification",
            source="kaggle:masoudnickparvar/brain-tumor-mri-dataset",
            path_attr="BRI_DATASET_PATH",
            notes=(
                "Genuinely four-class (glioma/meningioma/notumor/pituitary). Binary "
                "tumour/no-tumour datasets are rejected rather than remapped."
            ),
        ),
    ),
    weights=(
        WeightSpec("segmentation", "SEG_WEIGHTS_PATH", "UNet", "METHOD1_UNET_WEIGHTS"),
        WeightSpec("classification", "CLS_WEIGHTS_PATH", "ConvLSTMClassifier", "METHOD1_CONVLSTM_WEIGHTS"),
    ),
    segmentation_model="2D U-Net (encoder/decoder, skip connections, BatchNorm, dropout)",
    classifier_model="ConvLSTM (Conv+BN feature stack → ConvLSTM cell → dense → softmax)",
    optimization="SFLA (Shuffled Frog Leaping Algorithm) over classifier hyper-parameters",
    engine_module="backend.methods.method1.inference",
    transforms_module="backend.methods.method1.transforms",
    training_entrypoints=(
        "python -m backend.methods.method1.training.train_segmentation",
        "python -m backend.methods.method1.training.train_classifier",
        "python -m backend.methods.method1.optimization.run_sfla",
    ),
    metrics_filename="method1_metrics.json",
    run_card_filenames={
        "segmentation": "method1_segmentation_runcard.json",
        "classification": "method1_classification_runcard.json",
        "sfla": "method1_sfla_runcard.json",
    },
    optimization_result_filename="method1_sfla_result.json",
    notes=(
        "The classifier is trained on the same ROI geometry it is served — see "
        "backend/methods/method1/transforms.py for the single shared definition.",
    ),
)


# --------------------------------------------------------------------------- #
# Method 2 — independent multi-class segmentation + SPECT stage + DCN
# --------------------------------------------------------------------------- #
METHOD2_CLASS_NAMES = ("normal", "glioma", "meningioma", "pituitary")
METHOD2_CLASS_LABELS = {
    "normal": "Normal",
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary",
}

# Multi-class segmentation labels. BraTS annotates three tumour sub-regions plus
# background; that is the only multi-class mask source in this project with real
# ground truth, so Method 2's segmentation head is defined against it.
METHOD2_SEG_CLASSES = ("background", "necrotic_core", "edema", "enhancing_tumor")

METHOD2 = MethodSpec(
    method_id="method2",
    display_name="Method 2 — Multi-class Segmentation + SPECT Feature Stage + Dense Convolutional Network (DCN)",
    short_name="Multi-class Seg + SPECT + DCN",
    summary=(
        "Grayscale conversion and filtering, multi-class tumour segmentation into "
        "background/necrotic core/edema/enhancing tumour, a modality-specific "
        "SPECT feature representation, then classification with a densely "
        "connected convolutional network."
    ),
    modality="SPECT",
    class_names=METHOD2_CLASS_NAMES,
    class_labels=METHOD2_CLASS_LABELS,
    pipeline=(
        PipelineStage("input", "Input images", "input", "Single 2D image."),
        PipelineStage("gather", "Data gathering", "input", "SPECT study set; BraTS used only for the segmentation head."),
        PipelineStage("grayscale", "Grayscale conversion", "preprocess", "Single-channel conversion before any filtering."),
        PipelineStage("filter", "Filtering / denoising", "preprocess", "Resize first, then median + bilateral edge-preserving filtering."),
        PipelineStage("segment", "Multi-class segmentation", "segmentation", "4-way softmax mask: background / necrotic core / edema / enhancing tumour."),
        PipelineStage("features", "SPECT feature stage", "feature", "Modality-specific descriptors: per-region uptake statistics and intensity histogram."),
        PipelineStage("dcn", "Dense Convolutional Network", "classifier", "DenseNet-BC style densely connected blocks with transition layers."),
        PipelineStage("predict", "Multi-class classification", "output", "Normal / Glioma / Meningioma / Pituitary."),
        PipelineStage("evaluate", "Performance evaluation", "eval", "Per-class Dice/IoU for segmentation; accuracy/precision/recall/F1 for classification."),
    ),
    datasets=(
        DatasetSpec(
            key="spect",
            name="SPECT Image Dataset",
            role="modality",
            source="configure SPECT_DATASET_PATH",
            path_attr="SPECT_DATASET_PATH",
            notes=(
                "Drives the modality-specific branch and the DCN classifier. Class "
                "folders are discovered at training time and must match the "
                "configured class list — they are never assumed."
            ),
        ),
        DatasetSpec(
            key="brats",
            name="BraTS 2023 (2020 mirror)",
            role="segmentation",
            source="kaggle:awsaf49/brats2020-training-data",
            path_attr="BRATS_DATASET_PATH",
            notes="Used for the multi-class segmentation head only — it is the sole source of real multi-region masks.",
        ),
        DatasetSpec(
            key="bri",
            name="Brain Tumor MRI Dataset (BRI)",
            role="classification",
            source="kaggle:masoudnickparvar/brain-tumor-mri-dataset",
            path_attr="BRI_DATASET_PATH",
            compatible=False,
            notes=(
                "MRI, not SPECT. Available as an explicit --modality mri fallback for "
                "the DCN so the architecture can be exercised, but a model trained on "
                "it is an MRI model and is labelled as such — the two modalities are "
                "never merged into one training set."
            ),
        ),
    ),
    weights=(
        WeightSpec("segmentation", "M2_SEG_WEIGHTS_PATH", "MultiClassUNet", "METHOD2_SEGMENTATION_WEIGHTS"),
        WeightSpec("classification", "M2_DCN_WEIGHTS_PATH", "DenseConvNetClassifier", "METHOD2_DCN_WEIGHTS"),
    ),
    segmentation_model="Multi-class 2D U-Net (4-way softmax over background + 3 tumour sub-regions)",
    classifier_model="DenseConvNetClassifier — DenseNet-BC style dense blocks (documented adaptation, not a novel architecture)",
    optimization=None,
    engine_module="backend.methods.method2.inference",
    transforms_module="backend.methods.method2.transforms",
    training_entrypoints=(
        "python -m backend.methods.method2.training.train_segmentation",
        "python -m backend.methods.method2.training.train_classifier",
    ),
    metrics_filename="method2_metrics.json",
    run_card_filenames={
        "segmentation": "method2_segmentation_runcard.json",
        "classification": "method2_classification_runcard.json",
    },
    notes=(
        "Ships untrained. No checkpoint and no metrics are included, and the API "
        "reports this explicitly rather than emitting placeholder numbers.",
        "The reference diagram writes 'PECT'; the dataset and this implementation "
        "are SPECT. See METHOD2_MODALITY in .env.example.",
    ),
)


METHODS: dict[str, MethodSpec] = {m.method_id: m for m in (METHOD1, METHOD2)}
METHOD_IDS: tuple[str, ...] = tuple(METHODS)


def get_method(method_id: str) -> MethodSpec:
    """Look up a method, raising a readable ``KeyError`` for an unknown id."""
    try:
        return METHODS[method_id]
    except KeyError:
        raise KeyError(
            f"Unknown method '{method_id}'. Known methods: {', '.join(METHOD_IDS)}."
        ) from None


def list_methods() -> list[MethodSpec]:
    return list(METHODS.values())


def load_engine(method_id: str):
    """Import and return the inference engine for ``method_id`` (imports torch)."""
    spec = get_method(method_id)
    module = importlib.import_module(spec.engine_module)
    engine = getattr(module, "engine", None)
    if engine is None:
        raise RuntimeError(f"{spec.engine_module} does not expose an 'engine' object.")
    return engine


def runtime_status(spec: MethodSpec) -> dict[str, Any]:
    """Resolve machine-dependent facts: which weights and datasets are present.

    Imports :mod:`backend.config` lazily so the static registry stays importable
    without torch.
    """
    from backend import config

    weights: dict[str, Any] = {}
    for w in spec.weights:
        path = getattr(config, w.path_attr, None)
        weights[w.role] = {
            "architecture": w.architecture,
            "env_var": w.env_var,
            "path": str(path) if path else "",
            "present": bool(path and path.exists()),
        }

    datasets = []
    for d in spec.datasets:
        path = getattr(config, d.path_attr, None) if d.path_attr else None
        present = bool(path and path.exists() and any(path.iterdir())) if path else False
        datasets.append(
            {
                "key": d.key,
                "name": d.name,
                "role": d.role,
                "source": d.source,
                "path": str(path) if path else "",
                "present": present,
                "compatible": d.compatible,
                "notes": d.notes,
            }
        )

    seg_ok = weights.get("segmentation", {}).get("present", False)
    cls_ok = weights.get("classification", {}).get("present", False)
    warnings: list[str] = []
    if not seg_ok:
        warnings.append(
            f"{spec.short_name}: no segmentation checkpoint at "
            f"{weights.get('segmentation', {}).get('path', '?')} — segmentation is unavailable."
        )
    if not cls_ok:
        warnings.append(
            f"{spec.short_name}: no classifier checkpoint at "
            f"{weights.get('classification', {}).get('path', '?')} — classification is unavailable."
        )

    return {
        "weights": weights,
        "datasets": datasets,
        "segmentation_available": seg_ok,
        "classifier_available": cls_ok,
        "trained": seg_ok and cls_ok,
        "warnings": warnings,
    }


def dataset_context(spec: MethodSpec, roles: tuple[str, ...] = ()) -> str:
    """One-line human description of the datasets behind a method."""
    chosen = [d for d in spec.datasets if d.compatible and (not roles or d.role in roles)]
    return "; ".join(f"{d.name} ({d.role})" for d in chosen) or "no dataset configured"
