"""Shared fixtures. Torch-dependent tests skip cleanly when torch is absent."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# The suite must never call a paid external API, whatever .env says. load_dotenv
# does not override variables already set, so this wins over the .env value.
os.environ["METHOD2_AI_ENABLED"] = "false"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="torch not installed") if False else None


def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


HAS_TORCH = _has("torch")
HAS_CV2 = _has("cv2")
HAS_FASTAPI = _has("fastapi")

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
requires_cv2 = pytest.mark.skipif(not HAS_CV2, reason="opencv not installed")
requires_api = pytest.mark.skipif(
    not (HAS_TORCH and HAS_FASTAPI and HAS_CV2), reason="API stack not installed"
)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from backend.main import app

    return TestClient(app)


@pytest.fixture
def png_bytes():
    """A small synthetic grayscale MRI-like PNG."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(7)
    img = rng.normal(110, 20, (128, 128)).clip(0, 255).astype(np.uint8)
    cv2.circle(img, (80, 60), 15, 235, -1)
    return cv2.imencode(".png", img)[1].tobytes()
