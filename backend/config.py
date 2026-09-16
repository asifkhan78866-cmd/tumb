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
# Both the repo-root ``.env`` and the legacy ``backend/.env`` are read, root
# first, so existing installs keep working. ``load_dotenv`` never overrides a
# variable that is already set in the real environment.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
    load_dotenv(BACKEND_DIR / ".env")
except Exception:  # pragma: no cover
    pass


def _env_path(name: str, default: Path) -> Path:
    """Read a path from the environment, falling back to ``default``."""
    raw = os.getenv(name, "").strip()
    return Path(raw).expanduser() if raw else default


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


WEIGHTS_DIR = BACKEND_DIR / "weights"
DATASET_DIR = _env_path("DATA_ROOT", BACKEND_DIR / "dataset")
UPLOADS_DIR = BACKEND_DIR / "uploads"
PREDICTIONS_DIR = BACKEND_DIR / "predictions"
LOGS_DIR = BACKEND_DIR / "logs"
MODEL_CACHE_DIR = _env_path("MODEL_CACHE_DIR", BACKEND_DIR / ".model_cache")

for _p in (WEIGHTS_DIR, DATASET_DIR, UPLOADS_DIR, PREDICTIONS_DIR, LOGS_DIR, MODEL_CACHE_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# Per-dataset roots. Each defaults to a sub-directory of DATA_ROOT so a plain
# checkout keeps working, but any of them can point somewhere else entirely.
BRI_DATASET_PATH = _env_path("BRI_DATASET_PATH", DATASET_DIR / "brain-tumor-mri-dataset")
BRATS_DATASET_PATH = _env_path("BRATS_DATASET_PATH", DATASET_DIR / "brats2020-training-data")
SPECT_DATASET_PATH = _env_path("SPECT_DATASET_PATH", DATASET_DIR / "spect")

# Method 1 weights (legacy names kept: the shipped classifier lives here).
SEG_WEIGHTS_PATH = _env_path("METHOD1_UNET_WEIGHTS", WEIGHTS_DIR / "best_unet.pth")
CLS_WEIGHTS_PATH = _env_path("METHOD1_CONVLSTM_WEIGHTS", WEIGHTS_DIR / "best_classifier.pth")
# Method 2 weights.
M2_SEG_WEIGHTS_PATH = _env_path("METHOD2_SEGMENTATION_WEIGHTS", WEIGHTS_DIR / "method2_segmentation.pth")
M2_DCN_WEIGHTS_PATH = _env_path("METHOD2_DCN_WEIGHTS", WEIGHTS_DIR / "method2_dcn.pth")

# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
def _resolve_device() -> "torch.device":
    """Honour ``DEVICE=auto|cpu|cuda``; fall back to CPU if CUDA is unavailable."""
    want = os.getenv("DEVICE", "auto").strip().lower() or "auto"
    if want in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if want.startswith("cuda") and not torch.cuda.is_available():
        print("[config] DEVICE=cuda requested but CUDA is unavailable — using CPU.")
        return torch.device("cpu")
    return torch.device(want)


DEVICE = _resolve_device()
USE_AMP = DEVICE.type == "cuda"  # mixed precision only makes sense on CUDA

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
SEED = int(os.getenv("RANDOM_SEED", 42))

# --------------------------------------------------------------------------- #
# Method 2 modality
# --------------------------------------------------------------------------- #
# The reference diagram for Method 2 labels this stage "Photon Emission Computed
# Tomography (PECT)" while the dataset that is actually used is SPECT (Single
# Photon Emission Computed Tomography). They are NOT interchangeable terms, so
# the implementation carries the real modality and only renders the
# specification's wording where a run explicitly asks for it.
METHOD2_MODALITY = os.getenv("METHOD2_MODALITY", "SPECT").strip().upper() or "SPECT"
METHOD2_SPEC_MODALITY_LABEL = os.getenv("METHOD2_SPEC_MODALITY_LABEL", "PECT").strip()

# --------------------------------------------------------------------------- #
# SFLA (Shuffled Frog Leaping Algorithm) — Method 1 hyper-parameter search
# --------------------------------------------------------------------------- #
SFLA_ENABLED = _env_bool("SFLA_ENABLED", False)
SFLA_SEED = int(os.getenv("SFLA_SEED", SEED))
SFLA_POPULATION = int(os.getenv("SFLA_POPULATION", 20))
SFLA_MEMEPLEXES = int(os.getenv("SFLA_MEMEPLEXES", 4))
SFLA_LOCAL_ITERATIONS = int(os.getenv("SFLA_LOCAL_ITERATIONS", 5))
SFLA_ITERATIONS = int(os.getenv("SFLA_ITERATIONS", 10))

# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
API_TITLE = "Brain Tumor Segmentation & Classification API"
API_VERSION = "2.0.0"
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", 8000))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS", f"{FRONTEND_ORIGIN},http://127.0.0.1:3000"
    ).split(",")
    if o.strip()
]

# Uploads larger than this are rejected before any decoding happens.
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", 20))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

# Launching a training run from an HTTP request is off unless explicitly enabled.
TRAINING_API_ENABLED = _env_bool("TRAINING_API_ENABLED", False)

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
        "app_env": APP_ENV,
        "max_upload_mb": MAX_UPLOAD_MB,
        "method2_modality": METHOD2_MODALITY,
        "sfla_enabled": SFLA_ENABLED,
        "training_api_enabled": TRAINING_API_ENABLED,
    }
