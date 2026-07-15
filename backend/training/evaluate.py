"""Evaluate trained models on a held-out validation split and persist metrics.

Writes aggregate metrics to backend/logs/metrics.json (consumed by the /metrics
API and the dashboard) plus confusion-matrix / ROC plots.

Run:
    python -m backend.training.evaluate
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from backend import config
from backend.models import ConvLSTMClassifier, UNet
from backend.services import store
from backend.training.common import seed_everything, train_val_split
from backend.utils.dataset import (
    ClassificationDataset,
    SegmentationDataset,
    discover_classification_samples,
)
from backend.utils.losses import dice_coefficient
from backend.utils.metrics import (
    classification_metrics,
    save_confusion_matrix,
    save_roc_curves,
)


@torch.no_grad()
def eval_segmentation() -> dict:
    if not config.SEG_WEIGHTS_PATH.exists():
        return {"dice": 0.0, "note": "best_unet.pth not found"}
    ds = SegmentationDataset(config.DATASET_DIR, augment=False)
    if len(ds) == 0:
        return {"dice": 0.0, "note": "no segmentation data"}
    _, val_idx = train_val_split(list(range(len(ds))), 0.2, config.SEED)
    loader = DataLoader(Subset(ds, val_idx), batch_size=config.BATCH_SIZE)
    model = UNet(config.SEG_IN_CHANNELS, config.SEG_OUT_CHANNELS, config.BASE_FILTERS).to(config.DEVICE)
    model.load_state_dict(torch.load(config.SEG_WEIGHTS_PATH, map_location=config.DEVICE)["model_state"])
    model.eval()
    dices, times = [], []
    for x, y in loader:
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        t0 = time.perf_counter()
        logits = model(x)
        times.append((time.perf_counter() - t0) / x.size(0))
        dices.append(dice_coefficient(logits, y))
    return {"dice": float(np.mean(dices)), "avg_inference_time_s": float(np.mean(times))}


@torch.no_grad()
def eval_classification() -> dict:
    if not config.CLS_WEIGHTS_PATH.exists():
        return {"accuracy": 0.0, "note": "best_classifier.pth not found"}
    samples = discover_classification_samples(config.DATASET_DIR)
    if not samples:
        return {"accuracy": 0.0, "note": "no classification data"}
    _, val_s = train_val_split(samples, 0.2, config.SEED)
    loader = DataLoader(ClassificationDataset(val_s, augment=False), batch_size=config.BATCH_SIZE)
    model = ConvLSTMClassifier(1, config.NUM_CLASSES).to(config.DEVICE)
    model.load_state_dict(torch.load(config.CLS_WEIGHTS_PATH, map_location=config.DEVICE)["model_state"])
    model.eval()

    y_true, y_pred, y_prob = [], [], []
    for x, y in loader:
        x = x.to(config.DEVICE)
        probs = F.softmax(model(x), dim=1).cpu().numpy()
        y_prob.extend(probs.tolist())
        y_pred.extend(probs.argmax(1).tolist())
        y_true.extend(y.numpy().tolist())

    m = classification_metrics(y_true, y_pred, config.NUM_CLASSES)
    labels = [config.CLASS_LABELS[c] for c in config.CLASS_NAMES]
    save_confusion_matrix(m["confusion_matrix"], labels, config.LOGS_DIR / "eval_confusion_matrix.png")
    save_roc_curves(np.array(y_true), np.array(y_prob), labels, config.LOGS_DIR / "eval_roc.png")
    return m


def main():
    argparse.ArgumentParser(description="Evaluate trained models").parse_args()
    seed_everything(config.SEED)

    seg = eval_segmentation()
    cls = eval_classification()

    metrics = {
        "accuracy": round(cls.get("accuracy", 0.0), 4),
        "dice": round(seg.get("dice", 0.0), 4),
        "sensitivity": round(cls.get("sensitivity", 0.0), 4),
        "specificity": round(cls.get("specificity", 0.0), 4),
        "precision": round(cls.get("precision", 0.0), 4),
        "recall": round(cls.get("recall", 0.0), 4),
        "f1": round(cls.get("f1", 0.0), 4),
        "avg_inference_time_s": round(seg.get("avg_inference_time_s", 0.0), 4),
        "confusion_matrix": cls.get("confusion_matrix"),
        "per_class": cls.get("per_class"),
    }
    store.set_metrics(metrics)
    print("[evaluate] Metrics written to", store.METRICS_PATH)
    for k, v in metrics.items():
        if k not in ("confusion_matrix", "per_class"):
            print(f"  {k:22s}: {v}")


if __name__ == "__main__":
    main()
