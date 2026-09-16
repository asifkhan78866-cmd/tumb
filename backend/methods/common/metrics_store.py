"""Per-method metrics storage that merges instead of clobbering.

The original ``evaluate.py`` rewrote one shared ``metrics.json`` wholesale, so
evaluating segmentation on a machine with no classification data silently
replaced a real 84% accuracy with ``0.0``. Here each method owns a file and each
stage owns a section inside it; writing one stage leaves the other untouched.

Absent metrics are represented by the section being missing — never by a zero.
``None`` reaches the API as ``null`` and the UI renders "N/A — not evaluated".
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_LOCK = threading.Lock()

__all__ = ["read_metrics", "update_metrics", "metrics_for"]


def _path_for(method_id: str) -> Path:
    from backend import config
    from backend.methods.registry import get_method

    return config.LOGS_DIR / get_method(method_id).metrics_filename


def read_metrics(method_id: str) -> dict:
    path = _path_for(method_id)
    if not path.exists():
        return {"method_id": method_id}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {"method_id": method_id}
    except Exception:
        return {"method_id": method_id}


def update_metrics(method_id: str, stage: str, section: Optional[dict]) -> Path:
    """Replace one stage's section, preserving every other stage."""
    path = _path_for(method_id)
    with _LOCK:
        data = read_metrics(method_id)
        data["method_id"] = method_id
        if section is None:
            data.pop(stage, None)
        else:
            data[stage] = section
        data["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))
    print(f"[metrics] {method_id}/{stage} -> {path}")
    return path


def metrics_for(method_id: str) -> dict[str, Any]:
    """Flatten the stored sections into the shared metrics response shape.

    Every field is ``None`` unless a stage actually computed it.
    """
    from backend.methods.registry import get_method

    spec = get_method(method_id)
    data = read_metrics(method_id)
    seg = data.get("segmentation") or {}
    cls = data.get("classification") or {}
    evaluated = bool(seg or cls)

    warnings: list[str] = []
    for section in (seg, cls):
        warnings.extend(section.get("warnings", []) or [])
    if not evaluated:
        warnings.append(
            f"{spec.short_name} has not been evaluated: no metrics file has been "
            f"written yet. Train and evaluate it before quoting any number."
        )

    splits = {s.get("split") for s in (seg, cls) if s.get("split")}
    return {
        "method_id": method_id,
        "method_name": spec.display_name,
        "evaluated": evaluated,
        "split": ", ".join(sorted(splits)) if splits else "not evaluated",
        "dice": seg.get("dice"),
        "iou": seg.get("iou"),
        "accuracy": cls.get("accuracy"),
        "precision": cls.get("precision"),
        "recall": cls.get("recall"),
        "sensitivity": cls.get("sensitivity"),
        "specificity": cls.get("specificity"),
        "f1": cls.get("f1"),
        "auc": cls.get("auc"),
        "avg_inference_time_s": cls.get("avg_inference_time_s") or seg.get("avg_inference_time_s"),
        "confusion_matrix": cls.get("confusion_matrix"),
        "class_labels": spec.labels,
        "per_class": cls.get("per_class"),
        "model_version": cls.get("model_version") or seg.get("model_version"),
        "dataset_context": ", ".join(
            filter(None, [seg.get("dataset", ""), cls.get("dataset", "")])
        ),
        "warnings": warnings,
    }
