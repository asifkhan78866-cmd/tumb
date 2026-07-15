"""Train the ConvLSTM tumor classifier on segmented tumor regions.

The classifier is trained on preprocessed (and, where a segmentation model is
available, tumor-cropped) images. Features: CrossEntropy loss, Adam, cosine LR
schedule, mixed precision, checkpointing (best_classifier.pth), TensorBoard,
early stopping, and metric/plot saving.

Run:
    python -m backend.training.train_classifier --epochs 40 --batch-size 16
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from backend import config
from backend.models import ConvLSTMClassifier
from backend.services import store
from backend.training.common import EarlyStopping, seed_everything, train_val_split
from backend.utils.dataset import ClassificationDataset, discover_classification_samples
from backend.utils.metrics import (
    classification_metrics,
    save_confusion_matrix,
    save_curve,
    save_roc_curves,
)


def build_loaders(args):
    root = config.DATASET_DIR
    if not any(root.rglob("*")):
        print("[train-cls] Dataset empty — attempting auto-download...")
        from backend.utils.dataset_download import download_dataset

        download_dataset(kind="classification")

    samples = discover_classification_samples(root)
    if len(samples) == 0:
        raise RuntimeError(
            "No labelled classification images found. The downloaded dataset must "
            "contain class-named folders (glioma/meningioma/notumor/pituitary)."
        )
    train_s, val_s = train_val_split(samples, val_frac=0.2, seed=config.SEED)
    train_loader = DataLoader(
        ClassificationDataset(train_s, augment=True), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=config.USE_AMP, drop_last=True,
    )
    val_loader = DataLoader(
        ClassificationDataset(val_s, augment=False), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=config.USE_AMP,
    )
    print(f"[train-cls] train={len(train_s)} val={len(val_s)} samples")
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, scaler, train: bool):
    model.train(train)
    total_loss = 0.0
    n = 0
    all_true, all_pred, all_prob = [], [], []
    for x, y in loader:
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=config.DEVICE.type, enabled=config.USE_AMP):
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
        probs = F.softmax(logits.detach(), dim=1).cpu().numpy()
        all_prob.extend(probs.tolist())
        all_pred.extend(probs.argmax(1).tolist())
        all_true.extend(y.cpu().numpy().tolist())
    metrics = classification_metrics(all_true, all_pred, config.NUM_CLASSES)
    return total_loss / n, metrics, np.array(all_true), np.array(all_prob)


def main():
    ap = argparse.ArgumentParser(description="Train ConvLSTM classifier")
    ap.add_argument("--epochs", type=int, default=config.CLS_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    ap.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    args = ap.parse_args()

    seed_everything(config.SEED)
    store.set_train_status({"state": "running", "message": "Classifier training started",
                            "cls": {"epoch": 0, "epochs": args.epochs}})

    train_loader, val_loader = build_loaders(args)

    model = ConvLSTMClassifier(in_channels=1, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)
    stopper = EarlyStopping(patience=args.patience, mode="max")
    writer = SummaryWriter(log_dir=str(config.LOGS_DIR / "cls_tb"))

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_true = best_val_prob = None
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_m, _, _ = run_epoch(model, train_loader, criterion, optimizer, scaler, True)
        va_loss, va_m, va_true, va_prob = run_epoch(model, val_loader, criterion, optimizer, scaler, False)
        scheduler.step()

        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", va_loss, epoch)
        writer.add_scalar("Acc/train", tr_m["accuracy"], epoch)
        writer.add_scalar("Acc/val", va_m["accuracy"], epoch)
        writer.add_scalar("F1/val", va_m["f1"], epoch)

        hist["train_loss"].append(tr_loss); hist["val_loss"].append(va_loss)
        hist["train_acc"].append(tr_m["accuracy"]); hist["val_acc"].append(va_m["accuracy"])

        print(f"[cls] Epoch {epoch:03d}/{args.epochs} | loss {tr_loss:.4f}/{va_loss:.4f} | "
              f"acc {tr_m['accuracy']:.4f}/{va_m['accuracy']:.4f} | "
              f"P {va_m['precision']:.3f} R {va_m['recall']:.3f} F1 {va_m['f1']:.3f} "
              f"spec {va_m['specificity']:.3f}")

        store.set_train_status({"state": "running", "cls": {
            "epoch": epoch, "epochs": args.epochs, "train_loss": round(tr_loss, 4),
            "val_loss": round(va_loss, 4), "val_acc": round(va_m["accuracy"], 4),
            "val_f1": round(va_m["f1"], 4)}})

        is_best = stopper.step(va_m["accuracy"])
        if is_best:
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_acc": va_m["accuracy"], "metrics": va_m},
                       config.CLS_WEIGHTS_PATH)
            best_val_true, best_val_prob, best_metrics = va_true, va_prob, va_m
            print(f"[cls]   ✔ saved new best -> {config.CLS_WEIGHTS_PATH} (acc={va_m['accuracy']:.4f})")
        if stopper.should_stop:
            print(f"[cls] Early stopping at epoch {epoch} (best acc={stopper.best:.4f})")
            break

    # Plots
    class_labels = [config.CLASS_LABELS[c] for c in config.CLASS_NAMES]
    save_curve(hist["train_loss"], "Classifier Loss", config.LOGS_DIR / "cls_loss.png",
               second=hist["val_loss"])
    save_curve(hist["train_acc"], "Accuracy", config.LOGS_DIR / "cls_acc.png",
               second=hist["val_acc"])
    if best_metrics is not None:
        save_confusion_matrix(best_metrics["confusion_matrix"], class_labels,
                              config.LOGS_DIR / "cls_confusion_matrix.png")
        save_roc_curves(best_val_true, best_val_prob, class_labels,
                        config.LOGS_DIR / "cls_roc.png")

    writer.close()
    store.set_train_status({"state": "done", "message": "Classifier training complete",
                            "cls": {"best_val_acc": round(stopper.best, 4)}})
    print("[cls] Training finished. Best accuracy:", round(stopper.best, 4))


if __name__ == "__main__":
    main()
