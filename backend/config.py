"""Central configuration for the Brain Tumor Segmentation & Classification system.

All paths are resolved relative to the backend package so the code runs the same
whether invoked from the repo root, from ``backend/``, or inside Docker.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent

# Load environment variables from backend/.env (Kaggle keys, hyper-params, etc.)
# before any os.getenv() calls below. Silently no-ops if python-dotenv isn't
# installed or the file is absent.
try:
    from dotenv import load_dotenv

    load_dotenv(BACKEND_DIR / ".env")
except Exception:  # pragma: no cover
    pass

WEIGHTS_DIR = BACKEND_DIR / "weights"
DATASET_DIR = BACKEND_DIR / "dataset"
UPLOADS_DIR = BACKEND_DIR / "uploads"
PREDICTIONS_DIR = BACKEND_DIR / "predictions"
LOGS_DIR = BACKEND_DIR / "logs"

for _p in (WEIGHTS_DIR, DATASET_DIR, UPLOADS_DIR, PREDICTIONS_DIR, LOGS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

SEG_WEIGHTS_PATH = WEIGHTS_DIR / "best_unet.pth"
CLS_WEIGHTS_PATH = WEIGHTS_DIR / "best_classifier.pth"

# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()  # mixed precision only makes sense on CUDA

# --------------------------------------------------------------------------- #
# Data / model hyper-parameters
# --------------------------------------------------------------------------- #
IMAGE_SIZE = 128            # square resize target for 2D slices
SEG_IN_CHANNELS = 1
SEG_OUT_CHANNELS = 1        # binary tumor / background mask
BASE_FILTERS = 32

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}
NUM_CLASSES = len(CLASS_NAMES)

# Training defaults (overridable via CLI in the training scripts)
SEG_EPOCHS = int(os.getenv("SEG_EPOCHS", 50))
CLS_EPOCHS = int(os.getenv("CLS_EPOCHS", 40))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-3))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 2))
EARLY_STOP_PATIENCE = int(os.getenv("EARLY_STOP_PATIENCE", 10))
SEED = 42

# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
API_TITLE = "Brain Tumor Segmentation & Classification API"
API_VERSION = "1.0.0"
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

# Kaggle dataset identifiers used by the auto-downloader (see utils/dataset_download.py).
#
# Classification (4 labelled classes: glioma/meningioma/pituitary/notumor) and
# segmentation (BraTS, with ground-truth masks) come from DIFFERENT datasets, so
# for accurate results on both tasks we download both.
KAGGLE_CLASSIFICATION = os.getenv(
    "KAGGLE_CLASSIFICATION", "masoudnickparvar/brain-tumor-mri-dataset"
)
KAGGLE_CLASSIFICATION_FALLBACK = os.getenv(
    "KAGGLE_CLASSIFICATION_FALLBACK", "navoneel/brain-mri-images-for-brain-tumor-detection"
)
KAGGLE_SEGMENTATION = os.getenv("KAGGLE_SEGMENTATION", "awsaf49/brats2020-training-data")

# Backwards-compatible aliases
KAGGLE_PRIMARY = KAGGLE_SEGMENTATION
KAGGLE_FALLBACK_1 = KAGGLE_CLASSIFICATION_FALLBACK
KAGGLE_FALLBACK_2 = KAGGLE_CLASSIFICATION


def summary() -> dict:
    """Return a JSON-serialisable snapshot of the runtime configuration."""
    return {
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "mixed_precision": USE_AMP,
        "image_size": IMAGE_SIZE,
        "classes": [CLASS_LABELS[c] for c in CLASS_NAMES],
        "num_classes": NUM_CLASSES,
        "seg_weights_present": SEG_WEIGHTS_PATH.exists(),
        "cls_weights_present": CLS_WEIGHTS_PATH.exists(),
    }
