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
    # Position in the project's numbered method list (1-4). Independent of
    # method_id, which is baked into checkpoints and must never change.
    display_number: int = 0

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
    display_name="3D U-Net–ConvLSTM–SFLA-Based Anomaly Segmentation & Classification",
    short_name="U-Net + ConvLSTM + SFLA",
    display_number=3,
    summary=(
        "Binary tumour segmentation with a 2D U-Net, mask-driven ROI extraction, "
        "then four-class typing with a ConvLSTM classifier whose hyper-parameters "
        "are chosen by a Shuffled Frog Leaping Algorithm search. A second SFLA search "
        "selects which of the ConvLSTM's features an SVM classifies; the head that "
        "scores higher on validation is the one served."
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
        PipelineStage("features", "Feature extraction", "feature", "The ConvLSTM's penultimate dense layer becomes a fixed feature vector per image."),
        PipelineStage("sfla_features", "SFLA feature selection", "optimization", "A second SFLA search picks the feature subset and the SVM's C, gamma and kernel, scored by validation macro-F1 (train/val only)."),
        PipelineStage("svm", "SVM classifier", "classifier", "A support vector machine on the selected features. Served only if it beats the ConvLSTM's own softmax layer on validation."),
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
    segmentation_model=(
        "U-Net (encoder/decoder, skip connections, BatchNorm, dropout). Implemented in 2D on "
        "slices; a volumetric 3D U-Net needs BraTS volumes and is not trained."
    ),
    classifier_model=(
        "ConvLSTM (Conv+BN feature stack → ConvLSTM cell → dense → softmax), or an SVM on "
        "SFLA-selected ConvLSTM features — whichever wins on validation macro-F1"
    ),
    optimization=(
        "SFLA (Shuffled Frog Leaping Algorithm), twice: over the classifier's hyper-parameters, "
        "and over the feature subset plus SVM hyper-parameters"
    ),
    engine_module="backend.methods.method1.inference",
    transforms_module="backend.methods.method1.transforms",
    training_entrypoints=(
        "python -m backend.methods.method1.training.train_segmentation",
        "python -m backend.methods.method1.training.train_segmentation_3d",
        "python -m backend.methods.method1.training.train_classifier",
        "python -m backend.methods.method1.optimization.run_sfla",
        "python -m backend.methods.method1.training.train_feature_svm",
    ),
    metrics_filename="method1_metrics.json",
    run_card_filenames={
        "segmentation": "method1_segmentation_runcard.json",
        "classification": "method1_classification_runcard.json",
        "sfla": "method1_sfla_runcard.json",
        "feature_selection": "method1_feature_selection_runcard.json",
        "segmentation_3d": "method1_segmentation3d_runcard.json",
    },
    optimization_result_filename="method1/sfla_results.json",
    notes=(
        "The classifier is served the same geometry it was trained on, recorded in its "
        "checkpoint. Both geometries were trained to convergence and compared on validation "
        "macro-F1: whole slices 0.9775, U-Net ROI crops 0.9595, so whole slices are served "
        "even though a U-Net checkpoint exists — the crop depends on a Dice 0.656 mask. See "
        "backend/logs/method1/roi_geometry_comparison.json for the comparison and "
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
    display_name="MRI–SPECT Multimodal Fusion-Based Brain Tumor Classification",
    short_name="MRI–SPECT Fusion",
    display_number=4,
    summary=(
        "Multimodal design: an MRI branch (grayscale, filtering, multi-class tumour "
        "segmentation) and a SPECT branch (uptake features) fused before a densely "
        "connected classifier. Image-level MRI–SPECT fusion is implemented and measured on "
        "357 registered pairs. The MRI branch's dense network is trained on the shared MRI "
        "split and is what produces the four-class prediction; the paired set carries no "
        "diagnosis labels, so no classifier is trained on fused images."
    ),
    modality="MRI + SPECT",
    class_names=METHOD2_CLASS_NAMES,
    class_labels=METHOD2_CLASS_LABELS,
    pipeline=(
        PipelineStage("input", "Input images", "input", "Single 2D image."),
        PipelineStage("gather", "Data gathering", "input", "SPECT study set; BraTS used only for the segmentation head."),
        PipelineStage("grayscale", "Grayscale conversion", "preprocess", "Single-channel conversion before any filtering."),
        PipelineStage("filter", "Filtering / denoising", "preprocess", "Resize first, then median + bilateral edge-preserving filtering."),
        PipelineStage("segment", "Multi-class segmentation", "segmentation", "4-way softmax mask: background / necrotic core / edema / enhancing tumour."),
        PipelineStage("features", "SPECT feature stage", "feature", "Modality-specific descriptors: per-region uptake statistics and intensity histogram."),
        PipelineStage("fusion", "MRI–SPECT fusion", "feature", "Laplacian pyramid fusion of the MRI with the SPECT luminance, local-energy coefficient selection; SPECT chrominance carried through unchanged. Deterministic — nothing is trained."),
        PipelineStage("dcn", "Dense Convolutional Network", "classifier", "DenseNet-BC style densely connected blocks with transition layers."),
        PipelineStage("predict", "Multi-class classification", "output", "Normal / Glioma / Meningioma / Pituitary."),
        PipelineStage("gradcam", "Grad-CAM", "output", "Heatmap over the DCN's last dense block."),
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
            key="paired_mri_spect",
            name="Paired MRI–SPECT slices",
            role="fusion",
            source="configure PAIRED_MRI_SPECT_PATH",
            path_attr="PAIRED_MRI_SPECT_PATH",
            notes=(
                "357 registered MRI/SPECT pairs over 15 cases, matched by filename. Drives "
                "the fusion stage and its quality metrics. It carries no diagnosis labels, so "
                "nothing is trained or classified on it."
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
        "python -m backend.methods.method2.training.evaluate_fusion",
    ),
    metrics_filename="method2_metrics.json",
    run_card_filenames={
        "segmentation": "method2_segmentation_runcard.json",
        "classification": "method2_classification_runcard.json",
        "fusion": "method2_fusion_runcard.json",
    },
    notes=(
        "Image-level MRI–SPECT fusion is implemented and measured on 357 registered pairs; "
        "it is a deterministic multi-scale transform, so there is nothing in it to train. "
        "What is still untrained is a classifier on fused images: that paired set carries no "
        "diagnosis labels. The four-class accuracy reported for this method is therefore the "
        "trained MRI branch's, on MRI, and is labelled as such everywhere it appears.",
        "The reference diagram writes 'PECT'; the dataset and this implementation "
        "are SPECT. See METHOD2_MODALITY in .env.example.",
    ),
)


# --------------------------------------------------------------------------- #
# Method 3 id — displayed #1: pre-trained CNNs + transfer learning
# --------------------------------------------------------------------------- #
_BRI_CLASSIFICATION = DatasetSpec(
    key="bri",
    name="Brain Tumor MRI Dataset (BRI)",
    role="classification",
    source="kaggle:masoudnickparvar/brain-tumor-mri-dataset",
    path_attr="BRI_DATASET_PATH",
    notes=(
        "Same split as every MRI method: the dataset's Training/Testing folders, validation "
        "from Training only, exact duplicates of test images removed from training. "
        "Image-level; no patient identifiers."
    ),
)

METHOD3 = MethodSpec(
    method_id="method3",
    display_name="Deep Learning Pre-trained Models + Transfer Learning-Based Brain Tumor Classification",
    short_name="Pre-trained CNNs + Transfer Learning",
    display_number=1,
    summary=(
        "ImageNet-pretrained CNN backbones are transferred to brain MRI. Two routes are "
        "implemented: fine-tuning a single backbone (EfficientNet-B0 or ResNet-50, chosen on "
        "validation macro-F1), and a feature-fusion ensemble in which frozen GoogleNet, VGG16, "
        "ResNet-50 and AlexNet branches are concatenated and weighed by a trained meta-learner. "
        "Whichever checkpoint is on disk is the one served, and it was evaluated once on the "
        "held-out test set."
    ),
    modality="MRI",
    class_names=METHOD1_CLASS_NAMES,
    class_labels=METHOD1_CLASS_LABELS,
    pipeline=(
        PipelineStage("input", "Input MRI", "input", "Single MRI slice (PNG/JPG/TIFF)."),
        PipelineStage("preprocess", "Preprocessing", "preprocess", "Grayscale, resize to 224×224, replicate to 3 channels, ImageNet normalisation."),
        PipelineStage("pretrained", "Pre-trained backbones", "feature", "GoogleNet, VGG16, ResNet-50 and AlexNet with ImageNet weights (EfficientNet-B0 on the single-backbone route)."),
        PipelineStage("transfer", "Transfer learning", "classifier", "Fusion route: backbones frozen, a 64-d head per branch. Single-backbone route: head warm-up, then full fine-tuning at a lower learning rate."),
        PipelineStage("fusion", "Feature fusion", "feature", "The four branch vectors are concatenated and weighed by a meta-learner (256 → 128 → 64 → classes)."),
        PipelineStage("select", "Model selection", "optimization", "Backbone and route chosen by validation macro-F1 (test set untouched)."),
        PipelineStage("predict", "Prediction", "output", "Glioma / Meningioma / No Tumor / Pituitary."),
        PipelineStage("gradcam", "Grad-CAM", "output", "Heatmap over the last convolutional stage."),
        PipelineStage("evaluate", "Performance evaluation", "eval", "Accuracy, precision, recall, specificity, F1, AUC on the held-out test set."),
    ),
    datasets=(_BRI_CLASSIFICATION,),
    weights=(
        WeightSpec("classification", "M3_WEIGHTS_PATH", "FusionEnsembleCNN", "METHOD3_WEIGHTS"),
    ),
    segmentation_model="Not part of this method (whole-image classification)",
    classifier_model=(
        "Feature-fusion ensemble of frozen GoogleNet + VGG16 + ResNet-50 + AlexNet branches with a "
        "trained meta-learner, or a single fine-tuned backbone — whichever checkpoint is present"
    ),
    optimization="Backbone and route selection by validation macro-F1; branch weighting learned by the meta-learner",
    engine_module="backend.methods.method3.inference",
    transforms_module="backend.methods.method3.transforms",
    training_entrypoints=(
        "python -m backend.methods.method3.training.train_fusion",
        "python -m backend.methods.method3.training.train_classifier",
    ),
    metrics_filename="method3_metrics.json",
    run_card_filenames={"classification": "method3_classification_runcard.json"},
)

# --------------------------------------------------------------------------- #
# Method 4 id — displayed #2: Red Fox Optimization + ZFNet
# --------------------------------------------------------------------------- #
METHOD4 = MethodSpec(
    method_id="method4",
    display_name="Red Fox Optimized ZFNet-Based Brain Tumor Classification",
    short_name="Red Fox Optimized ZFNet",
    display_number=2,
    summary=(
        "A ZFNet convolutional network trained from scratch on brain MRI, with its "
        "hyper-parameters chosen by Red Fox Optimization (RFO), a metaheuristic that "
        "searches using validation macro-F1 as fitness. RFO is then applied a second time, to "
        "select which of ZFNet's learned features an SVM should classify; the head that scores "
        "higher on validation is the one served."
    ),
    modality="MRI",
    class_names=METHOD1_CLASS_NAMES,
    class_labels=METHOD1_CLASS_LABELS,
    pipeline=(
        PipelineStage("input", "Input MRI", "input", "Single MRI slice (PNG/JPG/TIFF)."),
        PipelineStage("preprocess", "Preprocessing", "preprocess", "Grayscale, resize to 224×224, normalise."),
        PipelineStage("zfnet", "ZFNet", "classifier", "7×7/2 conv stem, 5×5/2 conv, three 3×3 convs, two fully connected layers."),
        PipelineStage("rfo", "Red Fox Optimization", "optimization", "Global search, local search and habitat reproduction over learning rate, weight decay, dropout, FC width and batch size (train/val only)."),
        PipelineStage("features", "Feature extraction", "feature", "The penultimate fully connected layer's activations become a fixed feature vector per image."),
        PipelineStage("rfo_features", "Red Fox feature selection", "optimization", "A second RFO search picks the feature subset and the SVM's C, gamma and kernel, scored by validation macro-F1 (train/val only)."),
        PipelineStage("svm", "SVM classifier", "classifier", "A support vector machine on the selected features. Served only if it beats the network's own softmax layer on validation."),
        PipelineStage("predict", "Prediction", "output", "Glioma / Meningioma / No Tumor / Pituitary."),
        PipelineStage("gradcam", "Grad-CAM", "output", "Heatmap over the last convolutional layer."),
        PipelineStage("evaluate", "Performance evaluation", "eval", "Accuracy, precision, recall, specificity, F1, AUC on the held-out test set."),
    ),
    datasets=(_BRI_CLASSIFICATION,),
    weights=(WeightSpec("classification", "M4_WEIGHTS_PATH", "ZFNet", "METHOD4_WEIGHTS"),),
    segmentation_model="Not part of this method (whole-image classification)",
    classifier_model=(
        "ZFNet (Zeiler & Fergus 2014) with BatchNorm, single-channel input; its softmax layer or an "
        "SVM on Red Fox–selected ZFNet features, whichever wins on validation macro-F1"
    ),
    optimization=(
        "Red Fox Optimization (Połap & Woźniak 2021), twice: over ZFNet's hyper-parameters, and "
        "over the feature subset plus SVM hyper-parameters"
    ),
    engine_module="backend.methods.method4.inference",
    transforms_module="backend.methods.method4.transforms",
    training_entrypoints=(
        "python -m backend.methods.method4.optimization.run_rfo",
        "python -m backend.methods.method4.training.train_classifier",
        "python -m backend.methods.method4.training.train_feature_svm",
    ),
    metrics_filename="method4_metrics.json",
    run_card_filenames={
        "classification": "method4_classification_runcard.json",
        "rfo": "method4_rfo_runcard.json",
        "feature_selection": "method4_feature_selection_runcard.json",
    },
    optimization_result_filename="method4/rfo_results.json",
)


METHODS: dict[str, MethodSpec] = {
    m.method_id: m for m in sorted((METHOD1, METHOD2, METHOD3, METHOD4), key=lambda m: m.display_number)
}
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

    from backend.methods.common.checkpoint import peek_meta

    weights: dict[str, Any] = {}
    for w in spec.weights:
        path = getattr(config, w.path_attr, None)
        present = bool(path and path.exists())
        meta = (peek_meta(path) or {}) if present else {}
        weights[w.role] = {
            "architecture": w.architecture,
            "env_var": w.env_var,
            "path": str(path) if path else "",
            "filename": path.name if path else "",
            "present": present,
            "model_version": meta.get("model_version") or ("legacy-untagged" if present else None),
            "transform_id": meta.get("transform_id") or None,
            "created_at": meta.get("created_at") or None,
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

    roles = {w.role for w in spec.weights}
    has_seg = "segmentation" in roles
    # A second image is offered only by a method that both declares two modalities
    # and has a paired dataset behind the stage that would use it.
    fuses = any(d.role == "fusion" for d in spec.datasets)
    secondary = spec.modality.split("+")[-1].strip() if fuses and "+" in spec.modality else ""
    seg_ok = weights.get("segmentation", {}).get("present", False)
    cls_ok = weights.get("classification", {}).get("present", False)
    warnings: list[str] = []
    if has_seg and not seg_ok:
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
        "trained": cls_ok and (seg_ok or not has_seg),
        "has_segmentation_stage": has_seg,
        # Only a method with two modality branches can read a second image; the
        # upload page asks this before offering the field.
        "accepts_secondary_image": bool(secondary),
        "secondary_image_label": secondary,
        # An AI model assesses uploads for methods whose classifier is not trained
        # (the fusion method, and the new methods until their training finishes).
        "ai_assessment_available": (
            spec.method_id in ("method2", "method3", "method4")
            and not cls_ok and config.method2_ai_available()
        ),
        "warnings": warnings,
    }


def dataset_context(spec: MethodSpec, roles: tuple[str, ...] = ()) -> str:
    """One-line human description of the datasets behind a method."""
    chosen = [d for d in spec.datasets if d.compatible and (not roles or d.role in roles)]
    return "; ".join(f"{d.name} ({d.role})" for d in chosen) or "no dataset configured"
