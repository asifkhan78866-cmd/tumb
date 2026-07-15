"""Train the U-Net segmentation model.

Features: Dice+BCE loss, Adam, ReduceLROnPlateau scheduler, mixed precision (CUDA),
checkpointing (best_unet.pth), TensorBoard logging, early stopping, and automatic
plot saving. Auto-downloads the dataset if it is missing.

Run:
    python -m backend.training.train_segmentation --epochs 50 --batch-size 16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from backend import config
from backend.models import UNet
from backend.services import store
from backend.training.common import EarlyStopping, seed_everything, train_val_split
from backend.utils.dataset import SegmentationDataset
from backend.utils.losses import DiceBCELoss, dice_coefficient, iou_score
from backend.utils.metrics import save_curve


def build_loaders(args):
    root = config.DATASET_DIR
    if not any(root.rglob("*")):
        print("[train-seg] Dataset empty — attempting auto-download...")
        from backend.utils.dataset_download import download_dataset

        download_dataset(kind="segmentation")

    full = SegmentationDataset(root, augment=False)
    if len(full) == 0:
        raise RuntimeError(
            "No segmentation samples found. Ensure the dataset downloaded into "
            f"{root} (NIfTI volumes or 2D images)."
        )
    train_idx, val_idx = train_val_split(list(range(len(full))), val_frac=0.2, seed=config.SEED)

    train_ds = SegmentationDataset(root, augment=True)
    train_loader = DataLoader(
        Subset(train_ds, train_idx), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=config.USE_AMP, drop_last=True,
    )
    val_loader = DataLoader(
        Subset(full, val_idx), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=config.USE_AMP,
    )
    print(f"[train-seg] train={len(train_idx)} val={len(val_idx)} samples")
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, scaler, train: bool):
    model.train(train)
    total_loss = total_dice = total_iou = 0.0
    n = 0
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
        total_dice += dice_coefficient(logits, y) * bs
        total_iou += iou_score(logits, y) * bs
        n += bs
    return total_loss / n, total_dice / n, total_iou / n


def main():
    ap = argparse.ArgumentParser(description="Train U-Net segmentation model")
    ap.add_argument("--epochs", type=int, default=config.SEG_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    ap.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    args = ap.parse_args()

    seed_everything(config.SEED)
    store.set_train_status({"state": "running", "message": "Segmentation training started",
                            "seg": {"epoch": 0, "epochs": args.epochs}})

    train_loader, val_loader = build_loaders(args)

    model = UNet(config.SEG_IN_CHANNELS, config.SEG_OUT_CHANNELS, config.BASE_FILTERS).to(config.DEVICE)
    criterion = DiceBCELoss(bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)
    stopper = EarlyStopping(patience=args.patience, mode="max")
    writer = SummaryWriter(log_dir=str(config.LOGS_DIR / "seg_tb"))

    hist = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice, tr_iou = run_epoch(model, train_loader, criterion, optimizer, scaler, True)
        va_loss, va_dice, va_iou = run_epoch(model, val_loader, criterion, optimizer, scaler, False)
        scheduler.step(va_dice)

        for k, v in [("Loss/train", tr_loss), ("Loss/val", va_loss),
                     ("Dice/train", tr_dice), ("Dice/val", va_dice),
                     ("IoU/val", va_iou), ("LR", optimizer.param_groups[0]["lr"])]:
            writer.add_scalar(k, v, epoch)
        hist["train_loss"].append(tr_loss); hist["val_loss"].append(va_loss)
        hist["train_dice"].append(tr_dice); hist["val_dice"].append(va_dice)

        print(f"[seg] Epoch {epoch:03d}/{args.epochs} | "
              f"loss {tr_loss:.4f}/{va_loss:.4f} | dice {tr_dice:.4f}/{va_dice:.4f} | iou {va_iou:.4f}")

        store.set_train_status({"state": "running", "seg": {
            "epoch": epoch, "epochs": args.epochs, "train_loss": round(tr_loss, 4),
            "val_loss": round(va_loss, 4), "val_dice": round(va_dice, 4), "val_iou": round(va_iou, 4)}})

        is_best = stopper.step(va_dice)
        if is_best:
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_dice": va_dice},
                       config.SEG_WEIGHTS_PATH)
            print(f"[seg]   ✔ saved new best -> {config.SEG_WEIGHTS_PATH} (dice={va_dice:.4f})")
        if stopper.should_stop:
            print(f"[seg] Early stopping at epoch {epoch} (best dice={stopper.best:.4f})")
            break

    save_curve(hist["train_loss"], "Segmentation Loss", config.LOGS_DIR / "seg_loss.png",
               second=hist["val_loss"])
    save_curve(hist["train_dice"], "Dice Coefficient", config.LOGS_DIR / "seg_dice.png",
               second=hist["val_dice"])
    writer.close()
    store.set_train_status({"state": "done", "message": "Segmentation training complete",
                            "seg": {"best_val_dice": round(stopper.best, 4)}})
    print("[seg] Training finished. Best Dice:", round(stopper.best, 4))


if __name__ == "__main__":
    main()
