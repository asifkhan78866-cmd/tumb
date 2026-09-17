"""Training/evaluation helpers shared by the whole-image CNN classifiers.

Used by the transfer-learning method (``method3``) and the Red Fox optimised
ZFNet (``method4``). Images live as a uint8 ``(N, H, W)`` tensor in memory; each
batch is moved to the device, augmented there (training only) and normalised
by a per-method ``to_input`` function, which is exactly what inference applies
to a single upload minus the augmentation.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from backend import config
from backend.utils.metrics import classification_metrics, macro_auc

__all__ = [
    "sync", "gpu_augment", "iterate_batches", "train_epoch", "predict_probs",
    "metrics_section", "print_section", "FitResult", "fit",
]


def sync(device) -> None:
    """Wait for queued device work so wall-clock timings are real."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def gpu_augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Batch augmentation for (B, 1, H, W) images in [0, 1], on their device.

    Horizontal flip, rotation ±15°, scale 0.9–1.1, translation ±6 %, and a mild
    brightness/contrast change. No vertical flips: an upside-down axial slice is
    not a plausible scan, and the pituitary class is defined by location.
    """
    b = x.size(0)
    r = torch.rand(b, 6, generator=gen).to(x.device)
    angle = (r[:, 0] * 2 - 1) * math.radians(15)
    scale = 0.9 + r[:, 1] * 0.2
    flip = torch.where(r[:, 2] < 0.5, -1.0, 1.0)
    tx, ty = (r[:, 3] * 2 - 1) * 0.06, (r[:, 4] * 2 - 1) * 0.06
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack(
        [torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1
    )
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    contrast = (0.9 + r[:, 5] * 0.2).view(b, 1, 1, 1)
    brightness = ((torch.rand(b, 1, 1, 1, generator=gen) * 2 - 1) * 0.05).to(x.device)
    return (x * contrast + brightness).clamp(0, 1)


def iterate_batches(n: int, batch_size: int, shuffle: bool, gen: Optional[torch.Generator],
                    drop_last: bool = False):
    order = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    stop = n - (n % batch_size) if drop_last else n
    for i in range(0, stop, batch_size):
        yield order[i : i + batch_size]


def train_epoch(model, images: torch.Tensor, labels: torch.Tensor, batch_size: int, optimizer,
                criterion, device, to_input: Callable, gen: torch.Generator) -> tuple[float, float]:
    """One pass over the training set. Returns (mean loss, accuracy)."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for idx in iterate_batches(len(labels), batch_size, True, gen, drop_last=True):
        x = images[idx].to(device).float().unsqueeze(1) / 255.0
        x = to_input(gpu_augment(x, gen))
        y = labels[idx].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)
        correct += (logits.argmax(1) == y).sum().item()
        seen += len(idx)
    return total_loss / max(1, seen), correct / max(1, seen)


@torch.no_grad()
def predict_probs(model, images: torch.Tensor, batch_size: int, device,
                  to_input: Callable) -> tuple[np.ndarray, float, float]:
    """Softmax probabilities for every image. Returns (probs, s/image, total s)."""
    model.eval()
    out, model_time = [], 0.0
    started = time.perf_counter()
    for idx in iterate_batches(len(images), batch_size, False, None):
        x = to_input(images[idx].to(device).float().unsqueeze(1) / 255.0)
        sync(device)
        t0 = time.perf_counter()
        probs = F.softmax(model(x).float(), dim=1)
        sync(device)
        model_time += time.perf_counter() - t0
        out.append(probs.cpu().numpy())
    total = time.perf_counter() - started
    return np.concatenate(out), model_time / max(1, len(images)), total


def metrics_section(y_true, probs: np.ndarray, class_names: list[str],
                    avg_time: float, total_time: float) -> dict:
    """Macro, weighted and per-class metrics plus the confusion matrix."""
    y_true = np.asarray(y_true)
    m = classification_metrics(y_true.tolist(), probs.argmax(1).tolist(), len(class_names))
    auc = macro_auc(y_true, probs, len(class_names))
    per_class = {
        name: {
            "precision": round(m["per_class"]["precision"][i], 4),
            "recall": round(m["per_class"]["recall"][i], 4),
            "sensitivity": round(m["per_class"]["recall"][i], 4),
            "specificity": round(m["per_class"]["specificity"][i], 4),
            "f1": round(m["per_class"]["f1"][i], 4),
            "support": int(sum(m["confusion_matrix"][i])),
        }
        for i, name in enumerate(class_names)
    }
    support = np.array([per_class[c]["support"] for c in class_names], dtype=float)
    w = support / support.sum() if support.sum() else support

    def weighted(key: str) -> float:
        return round(float(np.dot(w, m["per_class"][key])), 4)

    return {
        "accuracy": round(m["accuracy"], 4),
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "sensitivity": round(m["sensitivity"], 4),
        "specificity": round(m["specificity"], 4),
        "f1": round(m["f1"], 4),
        "weighted": {k: weighted(k) for k in ("precision", "recall", "specificity", "f1")},
        "auc": round(auc, 4) if auc is not None else None,
        "avg_inference_time_s": round(avg_time, 6),
        "total_eval_time_s": round(total_time, 3),
        "confusion_matrix": m["confusion_matrix"],
        "class_order": list(class_names),
        "per_class": per_class,
    }


def print_section(tag: str, title: str, s: dict) -> None:
    print(f"\n[{tag}] ===== {title} =====")
    print(f"[{tag}] accuracy {s['accuracy']:.4f} | macro P {s['precision']:.4f} R/sens "
          f"{s['recall']:.4f} spec {s['specificity']:.4f} F1 {s['f1']:.4f} | AUC {s['auc']}")
    print(f"[{tag}] weighted P {s['weighted']['precision']:.4f} R {s['weighted']['recall']:.4f} "
          f"F1 {s['weighted']['f1']:.4f}")
    print(f"[{tag}] {'class':12s} {'prec':>7s} {'recall':>7s} {'spec':>7s} {'f1':>7s} {'n':>5s}")
    for name, c in s["per_class"].items():
        print(f"[{tag}] {name:12s} {c['precision']:7.4f} {c['recall']:7.4f} "
              f"{c['specificity']:7.4f} {c['f1']:7.4f} {c['support']:5d}")
    print(f"[{tag}] confusion matrix (rows=true, cols=pred, order={s['class_order']}):")
    for row in s["confusion_matrix"]:
        print(f"[{tag}]   {row}")
    print(f"[{tag}] avg inference {s['avg_inference_time_s'] * 1000:.3f} ms/image | "
          f"total {s['total_eval_time_s']:.2f}s", flush=True)


@dataclass
class FitResult:
    best_state: Optional[dict]
    best_epoch: int
    best_val_f1: float
    epochs_run: int
    train_time_s: float
    history: dict = field(default_factory=dict)


def fit(model, data: dict, *, batch_size: int, optimizer, criterion, device, to_input: Callable,
        epochs: int, patience: int, class_names: list[str], tag: str, scheduler=None,
        on_epoch_start: Optional[Callable[[int], None]] = None, seed: Optional[int] = None) -> FitResult:
    """Train with early stopping on validation macro-F1; keep the best weights.

    ``data`` holds uint8 tensors ``x_train``/``y_train``/``x_val``/``y_val``. The
    test set is never passed in.
    """
    assert "x_test" not in data, "fit() must never see the test set"
    gen = torch.Generator().manual_seed(config.SEED if seed is None else seed)
    y_val = data["y_val"].numpy()
    hist = {"train_loss": [], "train_acc": [], "val_acc": [], "val_f1": [], "lr": []}
    best_state, best_epoch, best_f1, stale = None, 0, -1.0, 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        if on_epoch_start:
            on_epoch_start(epoch)
        t0 = time.perf_counter()
        loss, acc = train_epoch(model, data["x_train"], data["y_train"], batch_size, optimizer,
                                criterion, device, to_input, gen)
        probs, _, _ = predict_probs(model, data["x_val"], max(64, batch_size), device, to_input)
        vm = classification_metrics(y_val.tolist(), probs.argmax(1).tolist(), len(class_names))
        if scheduler is not None:
            scheduler.step(vm["f1"]) if isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        for k, v in (("train_loss", loss), ("train_acc", acc), ("val_acc", vm["accuracy"]),
                     ("val_f1", vm["f1"]), ("lr", lr)):
            hist[k].append(v)
        print(f"[{tag}] epoch {epoch:03d}/{epochs} | loss {loss:.4f} | acc {acc:.4f}/{vm['accuracy']:.4f} "
              f"| val F1 {vm['f1']:.4f} | lr {lr:.2e} | {time.perf_counter() - t0:.1f}s", flush=True)
        if vm["f1"] > best_f1 + 1e-4:
            best_f1, best_epoch, stale = vm["f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[{tag}]   new best (val macro-F1 {best_f1:.4f})")
        else:
            stale += 1
            if stale >= patience:
                print(f"[{tag}] early stop at epoch {epoch} (best val macro-F1 {best_f1:.4f})")
                break

    return FitResult(best_state, best_epoch, best_f1, len(hist["train_loss"]),
                     round(time.perf_counter() - started, 1), hist)
