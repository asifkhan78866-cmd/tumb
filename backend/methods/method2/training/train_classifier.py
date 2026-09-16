"""Train Method 2's Dense Convolutional Network classifier.

    python -m backend.methods.method2.training.train_classifier --modality spect

``--modality`` selects **one** source and is recorded in the checkpoint:

``spect`` (default)
    ``SPECT_DATASET_PATH``. This is the modality Method 2 is specified for.

``mri``
    ``BRI_DATASET_PATH``, as an explicit fallback so the architecture can be
    exercised where no SPECT study set is available. The resulting model is an
    MRI model; the checkpoint says so and every metric it produces is labelled
    with it. The two modalities are never combined into one training set.

When a Method 2 segmentation checkpoint exists it is loaded and used to produce
the region maps the feature stage needs — the same way inference does it. When
it does not, training proceeds with all-background region maps (zero region
descriptors) and records that prominently, because a DCN trained that way has
never seen the feature stage the specification puts in front of it.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend import config
from backend.methods.common.checkpoint import CheckpointError, build_meta, load_checkpoint, save_checkpoint
from backend.methods.common.metrics_store import update_metrics
from backend.methods.common.runcard import RunCard, write_run_card
from backend.methods.common.splits import group_train_val_test_split, split_summary
from backend.methods.method2 import config as m2
from backend.methods.method2.datasets import (
    DatasetError,
    SpectClassificationDataset,
    classification_group_of,
    discover_classification_samples,
)
from backend.methods.method2.models import DenseConvNetClassifier, MultiClassUNet
from backend.training.common import EarlyStopping, seed_everything
from backend.utils.metrics import (
    classification_metrics,
    macro_auc,
    save_confusion_matrix,
    save_curve,
    save_roc_curves,
)


def load_segmentation(device, warnings: list[str]):
    """Return the trained multi-class segmenter, or None (recorded as a warning)."""
    model = MultiClassUNet(in_channels=1, num_classes=m2.NUM_SEG_CLASSES).to(device)
    try:
        load_checkpoint(
            model, m2.SEG_WEIGHTS_PATH,
            expected_method=m2.METHOD_ID, expected_architecture=m2.SEG_ARCHITECTURE,
            expected_role="segmentation",
        )
        model.eval()
        print(f"[m2-cls] region maps from {m2.SEG_WEIGHTS_PATH}")
        return model
    except CheckpointError as exc:
        warnings.append(
            f"No Method 2 segmentation checkpoint ({exc}). The DCN is being trained "
            f"with all-background region maps, so every per-region "
            f"{m2.MODALITY} descriptor is zero and the feature stage contributes "
            f"nothing. Train the segmentation head first for the full pipeline."
        )
        print(f"[m2-cls] WARNING: {warnings[-1]}")
        return None


def run_epoch(model, loader, criterion, optimizer, scaler, train: bool, device):
    model.train(train)
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []
    for x, feats, y in loader:
        x, feats, y = x.to(device), feats.to(device), y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=config.USE_AMP):
                logits = model(x, feats)
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
    return total_loss / max(1, n), classification_metrics(y_true, y_pred, m2.NUM_CLASSES)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob, times = [], [], [], []
    for x, feats, y in loader:
        x, feats = x.to(device), feats.to(device)
        t0 = time.perf_counter()
        probs = F.softmax(model(x, feats), dim=1).cpu().numpy()
        times.append((time.perf_counter() - t0) / max(1, x.size(0)))
        y_prob.extend(probs.tolist())
        y_pred.extend(probs.argmax(1).tolist())
        y_true.extend(y.numpy().tolist())
    return (
        classification_metrics(y_true, y_pred, m2.NUM_CLASSES),
        np.array(y_true),
        np.array(y_prob),
        float(np.mean(times)) if times else 0.0,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Method 2's DCN classifier")
    ap.add_argument("--modality", choices=["spect", "mri"], default="spect")
    ap.add_argument("--epochs", type=int, default=config.CLS_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--workers", type=int, default=0,
                    help="keep 0 when a segmentation model generates region maps in-loader")
    ap.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    seed_everything(config.SEED)
    device = config.DEVICE
    started = time.perf_counter()
    warnings: list[str] = []

    modality = args.modality.upper()
    root = args.data_root or (m2.SPECT_PATH if modality == "SPECT" else m2.BRI_PATH)
    try:
        samples, ds_meta = discover_classification_samples(root, modality)
    except DatasetError as exc:
        print(f"\nERROR: {exc}")
        return 1
    warnings.extend(ds_meta.get("warnings", []))
    print(f"[m2-cls] {len(samples)} {modality} samples: {ds_meta['class_counts']}")

    seg_model = load_segmentation(device, warnings)
    spec = m2.TRANSFORM

    train_s, val_s, test_s = group_train_val_test_split(
        samples, classification_group_of, args.val_frac, args.test_frac, seed=config.SEED
    )
    summary = split_summary(
        {"train": train_s, "val": val_s, "test": test_s}, classification_group_of
    )
    print(f"[m2-cls] split: {summary['counts']} | leak-free={summary['leak_free']}")
    if not summary["leak_free"]:
        print("ERROR: split produced overlapping groups; refusing to train.")
        return 1

    loaders = {
        name: DataLoader(
            SpectClassificationDataset(part, spec, seg_model, device, augment=(name == "train")),
            batch_size=args.batch_size, shuffle=(name == "train"),
            num_workers=args.workers, drop_last=(name == "train"),
        )
        for name, part in (("train", train_s), ("val", val_s), ("test", test_s))
    }

    hp = dict(m2.DCN_DEFAULTS)
    model = DenseConvNetClassifier(
        in_channels=m2.DCN_IN_CHANNELS, num_classes=m2.NUM_CLASSES,
        growth_rate=int(hp["growth_rate"]), block_config=tuple(hp["block_config"]),
        num_init_features=int(hp["num_init_features"]), compression=float(hp["compression"]),
        dropout=float(hp["dropout"]), feature_dim=m2.DCN_FEATURE_DIM,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
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
        print(f"[m2-cls] epoch {epoch:03d}/{args.epochs} | loss {tr_loss:.4f}/{va_loss:.4f} | "
              f"acc {tr_m['accuracy']:.4f}/{va_m['accuracy']:.4f} | F1 {va_m['f1']:.3f}")

        if stopper.step(va_m["accuracy"]):
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[m2-cls]   new best (val acc {va_m['accuracy']:.4f})")
        if stopper.should_stop:
            print(f"[m2-cls] early stop at epoch {epoch} (best {stopper.best:.4f})")
            break

    if best_state is None:
        print("ERROR: no epoch improved; nothing to save.")
        return 1
    model.load_state_dict(best_state)

    test_m, y_true, y_prob, avg_time = evaluate(model, loaders["test"], device)
    labels = m2.SPEC.labels
    save_curve(hist["train_loss"], "M2 DCN Loss", config.LOGS_DIR / "method2_cls_loss.png",
               second=hist["val_loss"])
    save_curve(hist["train_acc"], "M2 Accuracy", config.LOGS_DIR / "method2_cls_acc.png",
               second=hist["val_acc"])
    save_confusion_matrix(test_m["confusion_matrix"], labels,
                          config.LOGS_DIR / "method2_cls_confusion_matrix.png")
    save_roc_curves(y_true, y_prob, labels, config.LOGS_DIR / "method2_cls_roc.png")
    auc = macro_auc(y_true, y_prob, m2.NUM_CLASSES)

    meta = build_meta(
        method_id=m2.METHOD_ID, architecture=m2.CLS_ARCHITECTURE, role="classification",
        class_names=m2.CLASS_NAMES, image_size=spec.image_size, transform_id=spec.id,
        hyperparameters=hp, modality=modality,
        feature_stage_active=seg_model is not None,
    )
    save_checkpoint(m2.DCN_WEIGHTS_PATH, model.state_dict(), meta,
                    epoch=best_epoch, val_acc=stopper.best)
    print(f"[m2-cls] checkpoint -> {m2.DCN_WEIGHTS_PATH}")

    section = {
        "split": "group-level held-out test",
        "dataset": f"{ds_meta['root']} ({modality})",
        "modality": modality,
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
        "feature_stage_active": seg_model is not None,
        "warnings": warnings,
    }
    update_metrics(m2.METHOD_ID, "classification", section)

    write_run_card(
        m2.run_card_path("classification"),
        RunCard(
            method_id=m2.METHOD_ID, stage="classification",
            dataset=ds_meta,
            split_strategy={"type": "group-level three-way",
                            "group_key": "study folder or filename stem",
                            "val_frac": args.val_frac, "test_frac": args.test_frac, **summary},
            random_seed=config.SEED, preprocessing=spec.to_dict(), image_size=spec.image_size,
            architecture=m2.CLS_ARCHITECTURE,
            model_config={**hp, "in_channels": m2.DCN_IN_CHANNELS,
                          "feature_dim": m2.DCN_FEATURE_DIM,
                          "note": "DenseNet-BC style; see models/dcn.py docstring"},
            optimizer={"name": "Adam", "lr": args.lr, "weight_decay": 1e-4,
                       "scheduler": "CosineAnnealingLR", "batch_size": args.batch_size},
            epochs=args.epochs, best_epoch=best_epoch, metrics=section,
            checkpoint_path=str(m2.DCN_WEIGHTS_PATH), inference_time_s=round(avg_time, 5),
            train_duration_s=round(time.perf_counter() - started, 1),
            model_version=meta.model_version, device=str(device), warnings=warnings,
        ),
        root=config.ROOT_DIR,
    )
    print(f"[m2-cls] done. Held-out test accuracy ({modality}): {test_m['accuracy']:.4f}")
    for w in warnings:
        print(f"[m2-cls] WARNING: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
