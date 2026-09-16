"""Method 2 configuration.

Independent of Method 1: its own weights, its own class list, its own transform,
its own metrics file. The only things shared are process-wide settings (device,
seed, upload limits) that belong to the application rather than to a method.
"""
from __future__ import annotations

from backend import config as _global
from backend.methods.method2.features import feature_dim
from backend.methods.method2.transforms import DEFAULT_SPEC, M2Transform
from backend.methods.registry import METHOD2, METHOD2_SEG_CLASSES

METHOD_ID = METHOD2.method_id
SPEC = METHOD2

CLASS_NAMES = list(METHOD2.class_names)
CLASS_LABELS = dict(METHOD2.class_labels)
NUM_CLASSES = METHOD2.num_classes

# Multi-class segmentation labels: background + three BraTS tumour sub-regions.
SEG_CLASS_NAMES = list(METHOD2_SEG_CLASSES)
NUM_SEG_CLASSES = len(SEG_CLASS_NAMES)
NUM_REGIONS = NUM_SEG_CLASSES - 1  # foreground regions only

IMAGE_SIZE = _global.IMAGE_SIZE
TRANSFORM: M2Transform = DEFAULT_SPEC.resized(IMAGE_SIZE)

MODALITY = _global.METHOD2_MODALITY
SPEC_MODALITY_LABEL = _global.METHOD2_SPEC_MODALITY_LABEL

SEG_WEIGHTS_PATH = _global.M2_SEG_WEIGHTS_PATH
DCN_WEIGHTS_PATH = _global.M2_DCN_WEIGHTS_PATH

SEG_ARCHITECTURE = "MultiClassUNet"
CLS_ARCHITECTURE = "DenseConvNetClassifier"

# DCN input channels: the filtered slice plus one uptake map per region.
DCN_IN_CHANNELS = 1 + NUM_REGIONS
DCN_FEATURE_DIM = feature_dim(NUM_REGIONS)

DCN_DEFAULTS = {
    "growth_rate": 12,
    "block_config": (6, 12, 16),
    "num_init_features": 24,
    "compression": 0.5,
    "dropout": 0.2,
}

SPECT_PATH = _global.SPECT_DATASET_PATH
BRATS_PATH = _global.BRATS_DATASET_PATH
BRI_PATH = _global.BRI_DATASET_PATH

METRICS_PATH = _global.LOGS_DIR / METHOD2.metrics_filename


def run_card_path(stage: str):
    return _global.LOGS_DIR / METHOD2.run_card_filenames[stage]


def modality_label(use_spec_wording: bool = False) -> str:
    """The modality string to display.

    Defaults to the real modality (SPECT). ``use_spec_wording=True`` renders the
    research specification's "PECT" instead, and callers that do so are expected
    to show both.
    """
    return SPEC_MODALITY_LABEL if use_spec_wording else MODALITY
