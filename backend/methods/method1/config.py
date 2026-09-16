"""Method 1 configuration — resolved from :mod:`backend.config` and the registry.

Keeps Method 1's knobs in one place without duplicating the global settings that
both methods share (device, seeds, upload limits, dataset roots).
"""
from __future__ import annotations

from backend import config as _global
from backend.methods.method1.transforms import DEFAULT_SPEC, M1_V1_LEGACY, TransformSpec
from backend.methods.registry import METHOD1

METHOD_ID = METHOD1.method_id
SPEC = METHOD1

CLASS_NAMES = list(METHOD1.class_names)
CLASS_LABELS = dict(METHOD1.class_labels)
NUM_CLASSES = METHOD1.num_classes

IMAGE_SIZE = _global.IMAGE_SIZE
SEG_IN_CHANNELS = _global.SEG_IN_CHANNELS
SEG_OUT_CHANNELS = _global.SEG_OUT_CHANNELS
BASE_FILTERS = _global.BASE_FILTERS

SEG_WEIGHTS_PATH = _global.SEG_WEIGHTS_PATH
CLS_WEIGHTS_PATH = _global.CLS_WEIGHTS_PATH

SEG_ARCHITECTURE = "UNet"
CLS_ARCHITECTURE = "ConvLSTMClassifier"

# Geometry used by new training runs. Serving reads the spec recorded in the
# checkpoint instead, falling back to the legacy spec for untagged checkpoints.
TRAIN_TRANSFORM: TransformSpec = DEFAULT_SPEC.resized(IMAGE_SIZE)
LEGACY_TRANSFORM: TransformSpec = M1_V1_LEGACY.resized(IMAGE_SIZE)

BRATS_PATH = _global.BRATS_DATASET_PATH
BRI_PATH = _global.BRI_DATASET_PATH

METRICS_PATH = _global.LOGS_DIR / METHOD1.metrics_filename
SFLA_RESULT_PATH = _global.LOGS_DIR / (METHOD1.optimization_result_filename or "method1_sfla_result.json")


def run_card_path(stage: str):
    return _global.LOGS_DIR / METHOD1.run_card_filenames[stage]


def classifier_hyperparams() -> dict:
    """Classifier hyper-parameters, overridden by the SFLA result when present.

    The optimiser writes its best parameter set to ``SFLA_RESULT_PATH``; training
    picks it up automatically when ``SFLA_ENABLED`` is set, so an optimisation run
    actually changes what gets trained rather than only being displayed.
    """
    import json

    defaults = {
        "lr": _global.LEARNING_RATE,
        "weight_decay": 1e-4,
        "dropout": 0.4,
        "lstm_hidden": 64,
        "lstm_steps": 3,
        "base": 32,
        "batch_size": _global.BATCH_SIZE,
    }
    if not _global.SFLA_ENABLED or not SFLA_RESULT_PATH.exists():
        return defaults
    try:
        best = json.loads(SFLA_RESULT_PATH.read_text()).get("best_parameters", {})
    except Exception:
        return defaults
    defaults.update({k: v for k, v in best.items() if k in defaults})
    return defaults
