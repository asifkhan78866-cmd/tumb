"""Fine-tune ImageNet-pretrained backbones and keep the best one.

    python -m backend.methods.method3.training.train_classifier
    python -m backend.methods.method3.training.train_classifier --backbones efficientnet_b0 --skip-test

For each backbone: a head warm-up with the backbone frozen, then full
fine-tuning with a lower backbone learning rate and cosine decay, early-stopped
on validation macro-F1. The backbone with the best validation macro-F1 is the
model; only that one is evaluated on the test set, once, at the end.

Uses the same split as every MRI method (``backend.methods.common.bri_data``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from backend import config
from backend.methods.common.bri_data import cached_arrays, prepare_bri_split
from backend.methods.common.checkpoint import build_meta, save_checkpoint
from backend.methods.common.cnn_training import (
    FitResult,
    fit,
    metrics_section,
    predict_probs,
    print_section,
)
from backend.methods.common.metrics_store import update_metrics
from backend.methods.common.runcard import RunCard, write_run_card
from backend.methods.method1.datasets import DatasetError, class_distribution, class_weights
from backend.methods.method3 import models as m3models
from backend.methods.method3.transforms import DEFAULT_SPEC, preprocess_file, to_input
from backend.methods.registry import METHOD3
from backend.training.common import seed_everything
from backend.utils.metrics import save_confusion_matrix, save_curve, save_roc_curves

TAG = "m3"
CLASS_NAMES = list(METHOD3.class_names)
LOGS_DIR = config.LOGS_DIR / "method3"


def load_split_tensors(args, load_test: bool):
    split = prepare_bri_split(args.data_root, args.val_frac, 0.15, False, tag=TAG)
    key = DEFAULT_SPEC.to_dict()
    tensors = {}
    for part in ("train", "val") + (("test",) if load_test else ()):
        samples = getattr(split, part)
        arr = cached_arrays(samples, preprocess_file, key, name=f"method3_{part}")
        tensors[f"x_{part}"] = torch.from_numpy(arr)
        tensors[f"y_{part}"] = torch.tensor([y for _, y in samples], dtype=torch.long)
    return split, tensors


def candidate_path(name: str, split, args) -> "Path":
    """Where a finished backbone's best weights are kept, keyed by everything that shaped them."""
    key = {
        "backbone": name, "train": sorted(p for p, _ in split.train), "val": sorted(p for p, _ in split.val),
        "hp": {k: getattr(args, k) for k in ("epochs", "warmup_epochs", "patience", "batch_size", "lr",
                                             "head_lr", "weight_decay", "dropout", "label_smoothing")},
        "seed": config.SEED, "transform": DEFAULT_SPEC.to_dict(),
    }
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return config.MODEL_CACHE_DIR / f"method3_candidate_{name}_{digest}.pt"


def train_or_reuse_backbone(name: str, split, tensors: dict, weights: list[float], args, device) -> dict:
    """Train one backbone, or reload it if an identical run already finished.

    Each finished candidate is saved immediately, so interrupting a multi-backbone
    run never throws away backbones that were already complete.
    """
    path = candidate_path(name, split, args)
    if path.exists() and not args.retrain:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model = m3models.build(name, len(CLASS_NAMES), pretrained=False, dropout=args.dropout).to(device)
        model.load_state_dict(saved["state"])
        print(f"[{TAG}] reusing finished {name} candidate {path.name} "
              f"(val macro-F1 {saved['val']['f1']:.4f}); pass --retrain to train it again")
        return {"name": name, "model": model, "fit": FitResult(None, **saved["fit"]), "val": saved["val"]}
    cand = train_backbone(name, tensors, weights, args, device)
    fitres = cand["fit"]
    torch.save({"state": {k: v.cpu() for k, v in cand["model"].state_dict().items()}, "val": cand["val"],
                "fit": {"best_epoch": fitres.best_epoch, "best_val_f1": fitres.best_val_f1,
                        "epochs_run": fitres.epochs_run, "train_time_s": fitres.train_time_s,
                        "history": fitres.history}}, path)
    print(f"[{TAG}] saved finished candidate -> {path.name}")
    return cand


def train_backbone(name: str, tensors: dict, weights: list[float], args, device) -> dict:
    seed_everything(config.SEED)
    model = m3models.build(name, len(CLASS_NAMES), pretrained=True, dropout=args.dropout).to(device)
    head_ids = {id(p) for p in m3models.head_parameters(model, name)}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    head_params = [p for p in model.parameters() if id(p) in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": 0.0}, {"params": head_params, "lr": args.head_lr}],
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=args.label_smoothing,
    )
    finetune_epochs = max(1, args.epochs - args.warmup_epochs)

    def schedule(epoch: int) -> None:
        if epoch <= args.warmup_epochs:
            m3models.set_backbone_trainable(model, name, False)
            optimizer.param_groups[0]["lr"] = 0.0
            optimizer.param_groups[1]["lr"] = args.head_lr
            return
        m3models.set_backbone_trainable(model, name, True)
        t = (epoch - args.warmup_epochs - 1) / finetune_epochs
        cosine = 0.5 * (1 + math.cos(math.pi * t))
        optimizer.param_groups[0]["lr"] = args.lr * cosine
        optimizer.param_groups[1]["lr"] = args.lr * 10 * cosine

    data = {k: v for k, v in tensors.items() if not k.endswith("_test")}
    print(f"\n[{TAG}] ===== {name}: {m3models.count_parameters(model):,} parameters =====", flush=True)
    result = fit(
        model, data, batch_size=args.batch_size, optimizer=optimizer, criterion=criterion,
        device=device, to_input=to_input, epochs=args.epochs,
        patience=args.patience, class_names=CLASS_NAMES, tag=f"{TAG}:{name}",
        on_epoch_start=schedule,
    )
    model.load_state_dict(result.best_state)
    probs, avg, total = predict_probs(model, tensors["x_val"], 64, device, to_input)
    val = metrics_section(tensors["y_val"].numpy(), probs, CLASS_NAMES, avg, total)
    print_section(TAG, f"{name} VALIDATION (best epoch {result.best_epoch})", val)
    return {"name": name, "model": model, "fit": result, "val": val}


def main() -> int:
    ap = argparse.ArgumentParser(description="Transfer-learning brain tumour classifier")
    ap.add_argument("--backbones", default="efficientnet_b0,resnet50",
                    help="comma-separated; any of resnet50, efficientnet_b0, densenet121")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone learning rate (head gets 10x)")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="head learning rate during warm-up")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--skip-test", action="store_true", help="never load the test set")
    ap.add_argument("--retrain", action="store_true",
                    help="retrain backbones even if an identical finished candidate is saved")
    args = ap.parse_args()

    device = config.DEVICE
    print(f"[{TAG}] device report: {config.device_report()}")
    started = time.perf_counter()
    try:
        split, tensors = load_split_tensors(args, load_test=not args.skip_test)
    except DatasetError as exc:
        print(f"\nERROR: {exc}")
        return 1
    warnings = list(split.warnings)
    weights = class_weights(split.train)

    candidates = []
    for name in [b.strip() for b in args.backbones.split(",") if b.strip()]:
        cand = train_or_reuse_backbone(name, split, tensors, weights, args, device)
        candidates.append(cand)
        if len(candidates) > 1:
            # Keep only the best model in memory.
            candidates.sort(key=lambda c: -c["val"]["f1"])
            for loser in candidates[1:]:
                loser["model"] = None
            if device.type == "mps":
                torch.mps.empty_cache()

    candidates.sort(key=lambda c: -c["val"]["f1"])
    best = candidates[0]
    model, name, fitres = best["model"], best["name"], best["fit"]
    print(f"\n[{TAG}] selected backbone by validation macro-F1: {name} ({best['val']['f1']:.4f})")
    for c in candidates:
        print(f"[{TAG}]   {c['name']:16s} val macro-F1 {c['val']['f1']:.4f} acc {c['val']['accuracy']:.4f} "
              f"best epoch {c['fit'].best_epoch} ({c['fit'].train_time_s:.0f}s)")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    save_curve(fitres.history["train_loss"], "Training loss", LOGS_DIR / "train_loss.png")
    save_curve(fitres.history["train_acc"], "Accuracy", LOGS_DIR / "accuracy.png",
               second=fitres.history["val_acc"])
    hp = {"backbone": name, "epochs": args.epochs, "warmup_epochs": args.warmup_epochs,
          "batch_size": args.batch_size, "lr": args.lr, "head_lr": args.head_lr,
          "weight_decay": args.weight_decay, "dropout": args.dropout,
          "label_smoothing": args.label_smoothing}
    candidate_summary = [
        {"backbone": c["name"], "val_macro_f1": c["val"]["f1"], "val_accuracy": c["val"]["accuracy"],
         "best_epoch": c["fit"].best_epoch, "epochs_run": c["fit"].epochs_run,
         "train_time_s": c["fit"].train_time_s}
        for c in candidates
    ]

    ckpt = config.M3_WEIGHTS_PATH if not args.skip_test else config.M3_WEIGHTS_PATH.with_name("baseline_classifier.pth")
    meta = build_meta(
        method_id=METHOD3.method_id, architecture="TransferLearningCNN", role="classification",
        class_names=CLASS_NAMES, image_size=DEFAULT_SPEC.image_size, transform_id=DEFAULT_SPEC.id,
        hyperparameters=hp, backbone=name,
    )
    save_checkpoint(ckpt, model.state_dict(), meta, epoch=fitres.best_epoch, val_f1=best["val"]["f1"])
    print(f"[{TAG}] checkpoint -> {ckpt}")

    metadata = {
        "method_id": METHOD3.method_id,
        "method_name": METHOD3.display_name,
        "architecture": f"TransferLearningCNN ({name}, ImageNet-pretrained)",
        "class_order": CLASS_NAMES,
        "dataset_path": str(split.ds_meta["root"]),
        "dataset_counts": {"train": class_distribution(split.train),
                           "validation": class_distribution(split.val),
                           "test": class_distribution(split.test),
                           "train_copies_of_test_images_dropped": len(split.removed_test_duplicates)},
        "split_strategy": split.split_kind,
        "preprocessing": DEFAULT_SPEC.to_dict(),
        "augmentation": "train only, on GPU: hflip, rotate ±15°, scale 0.9–1.1, translate ±6%, brightness/contrast; no vflip",
        "loss": f"CrossEntropyLoss (inverse-frequency class weights, label smoothing {args.label_smoothing})",
        "optimizer": "AdamW; head warm-up then cosine fine-tuning",
        "hyperparameters": hp,
        "model_selection": {"metric": "validation macro-F1", "candidates": candidate_summary},
        "best_epoch": fitres.best_epoch,
        "epochs_run": fitres.epochs_run,
        "training_time_s": round(time.perf_counter() - started, 1),
        "random_seed": config.SEED,
        "device": str(device),
        "model_version": meta.model_version,
        "checkpoint_path": str(ckpt),
        "training_timestamp": meta.created_at,
        "validation_metrics": best["val"],
        "test_metrics": None,
        "warnings": warnings,
    }

    if args.skip_test:
        out = LOGS_DIR / "baseline_validation_metrics.json"
        out.write_text(json.dumps(metadata, indent=2, default=str))
        print(f"[{TAG}] validation-only run; test set not loaded. report -> {out}")
        return 0

    # --- The held-out test set is opened here, once, for the selected model. --- #
    probs, avg, total = predict_probs(model, tensors["x_test"], 64, device, to_input)
    y_test = tensors["y_test"].numpy()
    test = metrics_section(y_test, probs, CLASS_NAMES, avg, total)
    print_section(TAG, "FINAL TEST (Testing/ folder, evaluated once)", test)
    labels = [METHOD3.class_labels[c] for c in CLASS_NAMES]
    save_confusion_matrix(test["confusion_matrix"], labels, config.LOGS_DIR / "method3_cls_confusion_matrix.png")
    save_roc_curves(y_test, probs, labels, config.LOGS_DIR / "method3_cls_roc.png")

    section = {
        "split": split.split_kind, "split_level": split.ds_meta.get("split_level"),
        "dataset": metadata["dataset_path"], **test,
        "per_class": {k: [test["per_class"][c][k] for c in CLASS_NAMES]
                      for k in ("precision", "recall", "f1", "specificity")},
        "per_class_detail": test["per_class"], "test_size": len(split.test),
        "backbone": name, "model_version": meta.model_version, "warnings": warnings,
    }
    update_metrics(METHOD3.method_id, "classification", section)
    metadata["test_metrics"] = test
    meta_path = config.M3_WEIGHTS_PATH.parent / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    write_run_card(
        config.LOGS_DIR / METHOD3.run_card_filenames["classification"],
        RunCard(
            method_id=METHOD3.method_id, stage="classification",
            dataset={**split.ds_meta, "name": "Brain Tumor MRI Dataset (BRI)"},
            split_strategy={"type": split.split_kind, **split.summary},
            random_seed=config.SEED, preprocessing=DEFAULT_SPEC.to_dict(),
            image_size=DEFAULT_SPEC.image_size, architecture=metadata["architecture"],
            model_config={**hp, "candidates": candidate_summary}, optimizer={"name": "AdamW"},
            epochs=fitres.epochs_run, best_epoch=fitres.best_epoch, metrics=section,
            checkpoint_path=str(ckpt), inference_time_s=round(avg, 6),
            train_duration_s=metadata["training_time_s"], model_version=meta.model_version,
            device=str(device), warnings=warnings,
        ),
        root=config.ROOT_DIR,
    )
    print(f"[{TAG}] metadata -> {meta_path}")
    print(f"[{TAG}] done. Held-out test accuracy: {test['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
