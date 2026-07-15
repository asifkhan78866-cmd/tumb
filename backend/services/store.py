"""Lightweight JSON-backed stores for prediction history, metrics and training status.

Kept intentionally simple (no external DB) so the system runs out of the box. All
state lives under ``backend/predictions`` and ``backend/logs``.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from backend import config

_LOCK = threading.Lock()
HISTORY_PATH = config.PREDICTIONS_DIR / "history.json"
TRAIN_STATUS_PATH = config.LOGS_DIR / "train_status.json"
METRICS_PATH = config.LOGS_DIR / "metrics.json"


def _read(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _write(path: Path, data):
    path.write_text(json.dumps(data, indent=2))


# --------------------------------------------------------------------------- #
# Prediction history
# --------------------------------------------------------------------------- #
def add_prediction(record: dict) -> None:
    with _LOCK:
        history = _read(HISTORY_PATH, [])
        record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **record}
        history.insert(0, record)
        _write(HISTORY_PATH, history[:200])  # cap history


def get_history(limit: int = 50) -> list:
    return _read(HISTORY_PATH, [])[:limit]


# --------------------------------------------------------------------------- #
# Training status (written by the training scripts, read by the API)
# --------------------------------------------------------------------------- #
def set_train_status(status: dict) -> None:
    with _LOCK:
        current = _read(TRAIN_STATUS_PATH, {})
        current.update(status)
        current["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _write(TRAIN_STATUS_PATH, current)


def get_train_status() -> dict:
    return _read(TRAIN_STATUS_PATH, {"state": "idle", "message": "No training has been run yet."})


# --------------------------------------------------------------------------- #
# Aggregate model metrics (written by evaluate.py)
# --------------------------------------------------------------------------- #
def set_metrics(metrics: dict) -> None:
    with _LOCK:
        _write(METRICS_PATH, metrics)


def get_metrics() -> dict:
    return _read(
        METRICS_PATH,
        {
            "note": "Placeholder metrics — run evaluate.py after training to populate real values.",
            "accuracy": 0.0,
            "dice": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "avg_inference_time_s": 0.0,
        },
    )
