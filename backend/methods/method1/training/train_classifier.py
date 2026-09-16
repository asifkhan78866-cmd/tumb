"""Train Method 1's ConvLSTM classifier.

    python -m backend.methods.method1.training.train_classifier --epochs 40

What this script guarantees, and the previous one did not:

* **Patient-grouped three-way split.** Train / validation / *held-out test*, with
  no group appearing in two parts. Early stopping watches validation only; the
  test part is opened exactly once, after training finishes.
* **Training geometry == serving geometry.** The transform spec is chosen here,
  used to build the dataset, and written into the checkpoint so
  ``inference.py`` reproduces it instead of guessing. If ROI cropping is
  requested but no segmentation checkpoint exists to produce ROIs, the run
  downgrades to the whole-slice spec and says so — it never trains on one
  geometry and serves another.
* **Honest labels.** Discovery refuses binary tumour/no-tumour datasets.
* **Provenance.** A run card records dataset, split, seed, preprocessing,
  architecture, optimiser, epochs, best epoch, metrics, checkpoint and warnings.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend import config
from backend.methods.common.checkpoint import build_meta, save_checkpoint
from backend.methods.common.metrics_store import update_metrics
from backend.methods.common.runcard import RunCard, write_run_card
from backend.methods.common.splits import (
    group_split,
    group_train_val_test_split,
    split_summary,
)
from backend.methods.method1 import config as m1
from backend.methods.method1.datasets import (
    ClassificationDataset,
    DatasetError,
    build_roi_cache,
    class_distribution,
    class_weights,
    classification_group_of,
    discover_classification_samples,
    partition_by_official_split,
)
from backend.methods.method1.models import ConvLSTMClassifier, UNet
from backend.methods.method1.transforms import M1_V1_LEGACY, M1_V2, TransformSpec
from backend.training.common import EarlyStopping, seed_everything
from backend.utils.metrics import (
    classification_metrics,
    macro_auc,
    save_confusion_matrix,
    save_curve,
    save_roc_curves,
)


def resolve_spec(requested: str, warnings: list[str]) -> tuple[TransformSpec, dict]:
    """Choose the transform spec, downgrading if ROIs cannot actually be produced."""
    spec = (M1_V2 if requested == "v2" else M1_V1_LEGACY).resized(m1.IMAGE_SIZE)
    roi_boxes: dict = {}
    if not spec.roi_crop:
        return spec, roi_boxes
    if not m1.SEG_WEIGHTS_PATH.exists():
        warnings.append(
            f"Requested ROI-cropped geometry ({spec.id}) but no segmentation "
            f"checkpoint exists at {m1.SEG_WEIGHTS_PATH}. Falling back to the "
            f"whole-slice geometry ({M1_V1_LEGACY.id}) so that training and "
            f"inference stay identical. Train the U-Net first to use ROI cropping."
        )
        return M1_V1_LEGACY.resized(m1.IMAGE_SIZE), roi_boxes
    return spec, roi_boxes


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict, np.ndarray, np.ndarray, float]:
    model.eval()
    y_true, y_pred, y_prob, times = [], [], [], []
    for x, y in loader:
        x = x.to(device)
        t0 = time.perf_counter()
        probs = F.softmax(model(x), dim=1).cpu().numpy()
        times.append((time.perf_counter() - t0) / max(1, x.size(0)))
        y_prob.extend(probs.tolist())
        y_pred.extend(probs.argmax(1).tolist())
        y_true.extend(y.numpy().tolist())
    metrics = classification_metrics(y_true, y_pred, m1.NUM_CLASSES)
    return metrics, np.array(y_true), np.array(y_prob), float(np.mean(times)) if times else 0.0


def run_epoch(model, loader, criterion, optimizer, scaler, train: bool, device):
    model.train(train)
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=config.USE_AMP):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                if config.USE_AMP:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
        y_pred.extend(logits.detach().float().argmax(1).cpu().tolist())
        y_true.extend(y.cpu().tolist())
    return total_loss / max(1, n), classification_metrics(y_true, y_pred, m1.NUM_CLASSES)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Method 1's ConvLSTM classifier")
    ap.add_argument("--epochs", type=int, default=config.CLS_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    ap.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    ap.add_argument("--transform", choices=["v1", "v2"], default="v2",
                    help="v2 = resize-then-denoise with ROI crop (default); v1 = legacy whole-slice")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ignore-official-split", action="store_true",
                    help="ignore the dataset's Training/Testing folders and split randomly")
    ap.add_argument("--no-class-weights", action="store_true",
                    help="disable inverse-frequency class weighting in the loss")
    args = ap.parse_args()

    print("[cls] device report:")
    for k, v in config.device_report().items():
        print(f"[cls]   {k:18s}= {v}")

    seed_everything(config.SEED)
    device = config.DEVICE
    warnings: list[str] = []
    started = time.perf_counter()

    root = args.data_root or m1.BRI_PATH
    try:
        samples, ds_meta = discover_classification_samples(root)
    except DatasetError as exc:
        print(f"\nERROR: {exc}")
        return 1
    warnings.extend(ds_meta.get("warnings", []))

    spec, _ = resolve_spec(args.transform, warnings)
    hp = m1.classifier_hyperparams()
    batch_size = args.batch_size or int(hp["batch_size"])
    lr = args.lr if args.lr is not None else float(hp["lr"])
    if config.SFLA_ENABLED and m1.SFLA_RESULT_PATH.exists():
        print(f"[cls] Using SFLA-optimised hyper-parameters from {m1.SFLA_RESULT_PATH}")

    # Prefer the dataset's own split. Validation is carved out of the TRAINING
    # pool only; the Testing folder is never touched until the final evaluation.
    use_official = ds_meta.get("has_official_split") and not args.ignore_official_split
    if use_official:
        train_pool, test_s = partition_by_official_split(samples)
        train_s, val_s = group_split(
            train_pool, classification_group_of,
            [1.0 - args.val_frac, args.val_frac], seed=config.SEED,
        )
        split_kind = "dataset Training/Testing split; validation carved from Training only"
    else:
        train_s, val_s, test_s = group_train_val_test_split(
            samples, classification_group_of, args.val_frac, args.test_frac, seed=config.SEED
        )
        split_kind = "grouped random three-way split (dataset provided no Training/Testing)"

    summary = split_summary(
        {"train": train_s, "val": val_s, "test": test_s}, classification_group_of
    )
    print(f"[cls] split: {split_kind}")
    print(f"[cls]   counts {summary['counts']} | train/val leak-free={not summary['group_overlap'].get('train|val')}")
    print(f"[cls]   per-class train: {class_distribution(train_s)}")
    print(f"[cls]   per-class val  : {class_distribution(val_s)}")
    print(f"[cls]   per-class test : {class_distribution(test_s)}")
    if summary["group_overlap"].get("train|val"):
        print("ERROR: train and validation share groups; refusing to train.")
        return 1

    roi_boxes: dict = {}
    if spec.roi_crop:
        seg = UNet(m1.SEG_IN_CHANNELS, m1.SEG_OUT_CHANNELS, m1.BASE_FILTERS).to(device)
        from backend.methods.common.checkpoint import load_checkpoint

        meta, warns = load_checkpoint(
            seg, m1.SEG_WEIGHTS_PATH,
            expected_method=m1.METHOD_ID, expected_architecture=m1.SEG_ARCHITECTURE,
            expected_role="segmentation",
        )
        warnings.extend(warns)
        roi_boxes = build_roi_cache(
            samples, seg, spec, str(meta.get("model_version", "unknown")), device
        )

    loader_kwargs: dict = {
        "num_workers": args.workers,
        "pin_memory": config.PIN_MEMORY,
    }
    if args.workers > 0:
        # Respawning workers every epoch dominates runtime on a small dataset.
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loaders = {
        name: DataLoader(
            ClassificationDataset(part, spec, augment=(name == "train"), roi_boxes=roi_boxes),
            batch_size=batch_size,
            shuffle=(name == "train"),
            drop_last=(name == "train"),
            **loader_kwargs,
        )
        for name, part in (("train", train_s), ("val", val_s), ("test", test_s))
    }

    model = ConvLSTMClassifier(
        in_channels=1,
        num_classes=m1.NUM_CLASSES,
        base=int(hp["base"]),
        lstm_hidden=int(hp["lstm_hidden"]),
        lstm_steps=int(hp["lstm_steps"]),
        dropout=float(hp["dropout"]),
    ).to(device)
    weights = class_weights(train_s) if not args.no_class_weights else None
    if weights:
        print(f"[cls] class weights (inverse frequency): "
              f"{dict(zip(m1.CLASS_NAMES, [round(w, 3) for w in weights]))}")
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device)
        )
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(hp["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP)
    stopper = EarlyStopping(patience=args.patience, mode="max")

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_epoch, best_state = 0, None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_m = run_epoch(model, loaders["train"], criterion, optimizer, scaler, True, device)
        va_loss, va_m = run_epoch(model, loaders["val"], criterion, optimizer, scaler, False, device)
        scheduler.step()

        hist["train_loss"].append(tr_loss); hist["val_loss"].append(va_loss)
        hist["train_acc"].append(tr_m["accuracy"]); hist["val_acc"].append(va_m["accuracy"])
        print(f"[cls] epoch {epoch:03d}/{args.epochs} | loss {tr_loss:.4f}/{va_loss:.4f} | "
              f"acc {tr_m['accuracy']:.4f}/{va_m['accuracy']:.4f} | F1 {va_m['f1']:.3f}")

        if stopper.step(va_m["accuracy"]):
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[cls]   new best (val acc {va_m['accuracy']:.4f})")
        if stopper.should_stop:
            print(f"[cls] early stop at epoch {epoch} (best val acc {stopper.best:.4f})")
            break

    if best_state is None:
        print("ERROR: no epoch improved; nothing to save.")
        return 1
    model.load_state_dict(best_state)

    # --- The held-out test set is opened here, once. --------------------- #
    test_m, y_true, y_prob, avg_time = evaluate(model, loaders["test"], device)
    labels = m1.SPEC.labels
    save_curve(hist["train_loss"], "Classifier Loss", config.LOGS_DIR / "method1_cls_loss.png",
               second=hist["val_loss"])
    save_curve(hist["train_acc"], "Accuracy", config.LOGS_DIR / "method1_cls_acc.png",
               second=hist["val_acc"])
    save_confusion_matrix(test_m["confusion_matrix"], labels,
                          config.LOGS_DIR / "method1_cls_confusion_matrix.png")
    save_roc_curves(y_true, y_prob, labels, config.LOGS_DIR / "method1_cls_roc.png")
    auc = macro_auc(y_true, y_prob, m1.NUM_CLASSES)

    meta = build_meta(
        method_id=m1.METHOD_ID,
        architecture=m1.CLS_ARCHITECTURE,
        role="classification",
        class_names=m1.CLASS_NAMES,
        image_size=spec.image_size,
        transform_id=spec.id,
        hyperparameters=hp,
        test_accuracy=test_m["accuracy"],
    )
    # New runs always write into weights/method1/, regardless of where the
    # legacy checkpoint lived, so the two never shadow each other confusingly.
    ckpt_path = config.METHOD1_WEIGHTS_DIR / "best_classifier.pth"
    save_checkpoint(ckpt_path, model.state_dict(), meta,
                    epoch=best_epoch, val_acc=stopper.best)
    print(f"[cls] checkpoint -> {ckpt_path}")

    metadata = {
        "method_id": m1.METHOD_ID,
        "method_name": m1.SPEC.display_name,
        "architecture": m1.CLS_ARCHITECTURE,
        "class_order": list(m1.CLASS_NAMES),
        "class_labels": [m1.CLASS_LABELS[c] for c in m1.CLASS_NAMES],
        "dataset_path": ds_meta["root"],
        "dataset_counts": {
            "train": class_distribution(train_s),
            "validation": class_distribution(val_s),
            "test": class_distribution(test_s),
            "total": len(samples),
        },
        "split_strategy": split_kind,
        "split_level": ds_meta.get("split_level"),
        "preprocessing": spec.to_dict(),
        "input_size": spec.image_size,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "learning_rate": lr,
        "weight_decay": hp["weight_decay"],
        "scheduler": "CosineAnnealingLR",
        "class_weights": dict(zip(m1.CLASS_NAMES, weights)) if weights else None,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "random_seed": config.SEED,
        "device": str(device),
        "device_report": config.device_report(),
        "sfla": {
            "enabled": config.SFLA_ENABLED,
            "applied_parameters": hp,
            "result_file": str(m1.SFLA_RESULT_PATH) if m1.SFLA_RESULT_PATH.exists() else None,
        },
        "model_version": meta.model_version,
        "checkpoint_path": str(ckpt_path),
        "training_timestamp": meta.created_at,
        "test_metrics": None,  # filled in below, after the held-out evaluation
        "warnings": warnings,
    }

    section = {
        # Describe the split that was actually used. Calling an image-level
        # split "patient-grouped" would overstate how independent the test set is.
        "split": split_kind,
        "split_level": ds_meta.get("split_level", "unknown"),
        "dataset": ds_meta["root"],
        "accuracy": round(test_m["accuracy"], 4),
        "precision": round(test_m["precision"], 4),
        "recall": round(test_m["recall"], 4),
        "sensitivity": round(test_m["sensitivity"], 4),
        "specificity": round(test_m["specificity"], 4),
        "f1": round(test_m["f1"], 4),
        "auc": round(auc, 4) if auc is not None else None,
        "avg_inference_time_s": round(avg_time, 5),
        "confusion_matrix": test_m["confusion_matrix"],
        "per_class": test_m["per_class"],
        "model_version": meta.model_version,
        "warnings": warnings,
    }
    update_metrics(m1.METHOD_ID, "classification", section)

    metadata["test_metrics"] = {
        k: section[k] for k in
        ("accuracy", "precision", "recall", "sensitivity", "specificity", "f1", "auc",
         "avg_inference_time_s", "confusion_matrix", "per_class")
    }
    m1.METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    m1.METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"[cls] metadata   -> {m1.METADATA_PATH}")

    write_run_card(
        m1.run_card_path("classification"),
        RunCard(
            method_id=m1.METHOD_ID,
            stage="classification",
            dataset={**ds_meta, "name": "Brain Tumor MRI Dataset (BRI)"},
            split_strategy={
                "type": split_kind,
                "level": ds_meta.get("split_level", "unknown"),
                "group_key": "filename stem (dataset has no patient ids)",
                "val_frac": args.val_frac,
                "test_frac": args.test_frac,
                **summary,
            },
            random_seed=config.SEED,
            preprocessing=spec.to_dict(),
            image_size=spec.image_size,
            architecture=m1.CLS_ARCHITECTURE,
            model_config=hp,
            optimizer={"name": "Adam", "lr": lr, "weight_decay": hp["weight_decay"],
                       "scheduler": "CosineAnnealingLR", "batch_size": batch_size},
            epochs=args.epochs,
            best_epoch=best_epoch,
            metrics=section,
            checkpoint_path=str(ckpt_path),
            inference_time_s=round(avg_time, 5),
            train_duration_s=round(time.perf_counter() - started, 1),
            model_version=meta.model_version,
            device=str(device),
            warnings=warnings,
        ),
        root=config.ROOT_DIR,
    )
    print(f"[cls] done. Held-out test accuracy: {test_m['accuracy']:.4f}")
    for w in warnings:
        print(f"[cls] WARNING: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
