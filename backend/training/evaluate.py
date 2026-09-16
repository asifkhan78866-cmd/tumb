"""DEPRECATED — evaluation is now part of each method's training run.

Both training entry points score the held-out test split once, at the end, and
write the result through ``backend.methods.common.metrics_store``, which merges
per-stage sections instead of rewriting one shared file. The old standalone
evaluator overwrote every metric on every run, so evaluating segmentation on a
machine without classification data silently replaced a real accuracy with 0.0.

This module now reports the stored metrics for every method instead of
recomputing (and potentially destroying) them.

    python -m backend.training.evaluate
"""
from __future__ import annotations

import json

from backend.methods.common.metrics_store import metrics_for
from backend.methods.registry import METHOD_IDS, get_method


def main() -> int:
    for method_id in METHOD_IDS:
        spec = get_method(method_id)
        metrics = metrics_for(method_id)
        print(f"\n=== {spec.display_name} ===")
        if not metrics["evaluated"]:
            print("  not evaluated — no metrics recorded.")
            print(f"  train it with: {spec.training_entrypoints[0]}")
            continue
        print(f"  split: {metrics['split']}")
        for key in ("dice", "iou", "accuracy", "precision", "recall",
                    "sensitivity", "specificity", "f1", "auc"):
            value = metrics[key]
            print(f"  {key:14s}: {'not evaluated' if value is None else f'{value:.4f}'}")
        for warning in metrics["warnings"]:
            print(f"  ! {warning}")

    print("\nFull payloads are served at GET /api/metrics/{method_id}.")
    print(json.dumps({m: metrics_for(m)["evaluated"] for m in METHOD_IDS}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
