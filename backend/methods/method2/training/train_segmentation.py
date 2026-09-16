"""Train Method 2's multi-class segmentation head.

    python -m backend.methods.method2.training.train_segmentation --epochs 40

Predicts four classes per pixel — background, necrotic core, edema, enhancing
tumour — because the SPECT feature stage downstream needs per-region statistics
that a merged binary mask cannot provide.

BraTS is the only dataset in this project carrying real multi-region annotation,
so it is the only source accepted here. Splits are volume-level and the test part
is scored exactly once.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend import config
from backend.methods.common.checkpoint import build_meta, save_checkpoint
from backend.methods.common.metrics_store import update_metrics
from backend.methods.common.runcard import RunCard, write_run_card
from backend.methods.common.splits import group_train_val_test_split, split_summary
from backend.methods.method2 import config as m2
from backend.methods.method2.datasets import (
    MultiClassSegmentationDataset,
    discover_brats_h5,
    segmentation_group_of,
)
from backend.methods.method2.models import MultiClassUNet
from backend.training.common import EarlyStopping, seed_everything
from backend.utils.metrics import save_curve


def per_class_dice_iou(logits: torch.Tensor, target: torch.Tensor, num_classes: int):
    """Dice and IoU per class; classes absent from both prediction and target
    yield ``nan`` so they can be excluded from the mean instead of scoring 1.0."""
    pred = logits.argmax(dim=1)
    dices, ious = [], []
    for c in range(num_classes):
        p, t = (pred == c), (target == c)
        inter = (p & t).sum().item()
        p_sum, t_sum = p.sum().item(), t.sum().item()
        union = p_sum + t_sum - inter
        dices.append(float("nan") if p_sum + t_sum == 0 else 2 * inter / (p_sum + t_sum))
        ious.append(float("nan") if union == 0 else inter / union)
    return dices, ious


def run_epoch(model, loader, criterion, optimizer, scaler, train: bool, device):
    model.train(train)
    total_loss, n = 0.0, 0
    dice_acc = np.zeros(m2.NUM_SEG_CLASSES)
    dice_cnt = np.zeros(m2.NUM_SEG_CLASSES)
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
        dices, _ = per_class_dice_iou(logits.detach().float(), y, m2.NUM_SEG_CLASSES)
        for i, d in enumerate(dices):
            if not np.isnan(d):
                dice_acc[i] += d
                dice_cnt[i] += 1
    mean_dice = np.divide(dice_acc, np.maximum(dice_cnt, 1))
    # Foreground mean: background Dice is near 1.0 on every slice and would mask
    # a model that finds no tumour at all.
    return total_loss / max(1, n), float(mean_dice[1:].mean()), mean_dice.tolist()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dice_acc = np.zeros(m2.NUM_SEG_CLASSES); dice_cnt = np.zeros(m2.NUM_SEG_CLASSES)
    iou_acc = np.zeros(m2.NUM_SEG_CLASSES); iou_cnt = np.zeros(m2.NUM_SEG_CLASSES)
    times = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        t0 = time.perf_counter()
        logits = model(x)
        times.append((time.perf_counter() - t0) / max(1, x.size(0)))
        dices, ious = per_class_dice_iou(logits.float(), y, m2.NUM_SEG_CLASSES)
        for i, (d, io) in enumerate(zip(dices, ious)):
            if not np.isnan(d):
                dice_acc[i] += d; dice_cnt[i] += 1
            if not np.isnan(io):
                iou_acc[i] += io; iou_cnt[i] += 1
    dice = np.divide(dice_acc, np.maximum(dice_cnt, 1))
    iou = np.divide(iou_acc, np.maximum(iou_cnt, 1))
    return dice.tolist(), iou.tolist(), float(np.mean(times)) if times else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Method 2's multi-class segmentation head")
    ap.add_argument("--epochs", type=int, default=config.SEG_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    ap.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    seed_everything(config.SEED)
    device = config.DEVICE
    started = time.perf_counter()
    warnings: list[str] = []

    root = args.data_root or m2.BRATS_PATH
    samples = discover_brats_h5(root)
    if not samples:
        print(
            f"\nERROR: no BraTS per-slice HDF5 files under {root}. Method 2's "
            f"multi-class head needs real multi-region masks; download them with "
            f"`python -m backend.utils.dataset_download --kind segmentation` or set "
            f"BRATS_DATASET_PATH."
        )
        return 1
    print(f"[m2-seg] {len(samples)} slices from {root}")

    spec = m2.TRANSFORM
    train_s, val_s, test_s = group_train_val_test_split(
        samples, segmentation_group_of, args.val_frac, args.test_frac, seed=config.SEED
    )
    summary = split_summary(
        {"train": train_s, "val": val_s, "test": test_s}, segmentation_group_of
    )
    print(f"[m2-seg] split (volume-grouped): {summary['counts']} | "
          f"volumes {summary['group_counts']} | leak-free={summary['leak_free']}")
    if not summary["leak_free"]:
        print("ERROR: split produced overlapping volumes; refusing to train.")
        return 1

    loaders = {
        name: DataLoader(
            MultiClassSegmentationDataset(part, spec, augment=(name == "train")),
            batch_size=args.batch_size, shuffle=(name == "train"),
            num_workers=args.workers, pin_memory=config.USE_AMP, drop_last=(name == "train"),
        )
        for name, part in (("train", train_s), ("val", val_s), ("test", test_s))
    }

    model = MultiClassUNet(in_channels=1, num_classes=m2.NUM_SEG_CLASSES).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP)
    stopper = EarlyStopping(patience=args.patience, mode="max")

    hist = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    best_epoch, best_state = 0, None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice, _ = run_epoch(model, loaders["train"], criterion, optimizer, scaler, True, device)
        va_loss, va_dice, va_per = run_epoch(model, loaders["val"], criterion, optimizer, scaler, False, device)
        scheduler.step(va_dice)

        hist["train_loss"].append(tr_loss); hist["val_loss"].append(va_loss)
        hist["train_dice"].append(tr_dice); hist["val_dice"].append(va_dice)
        print(f"[m2-seg] epoch {epoch:03d}/{args.epochs} | loss {tr_loss:.4f}/{va_loss:.4f} | "
              f"fg-dice {tr_dice:.4f}/{va_dice:.4f} | per-class "
              f"{[round(d, 3) for d in va_per]}")

        if stopper.step(va_dice):
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[m2-seg]   new best (val fg-dice {va_dice:.4f})")
        if stopper.should_stop:
            print(f"[m2-seg] early stop at epoch {epoch} (best {stopper.best:.4f})")
            break

    if best_state is None:
        print("ERROR: no epoch improved; nothing to save.")
        return 1
    model.load_state_dict(best_state)

    dice, iou, avg_time = evaluate(model, loaders["test"], device)
    save_curve(hist["train_loss"], "M2 Segmentation Loss", config.LOGS_DIR / "method2_seg_loss.png",
               second=hist["val_loss"])
    save_curve(hist["train_dice"], "M2 Foreground Dice", config.LOGS_DIR / "method2_seg_dice.png",
               second=hist["val_dice"])

    meta = build_meta(
        method_id=m2.METHOD_ID, architecture=m2.SEG_ARCHITECTURE, role="segmentation",
        class_names=m2.SEG_CLASS_NAMES, image_size=spec.image_size, transform_id=spec.id,
        modality="MRI (BraTS)",
    )
    save_checkpoint(m2.SEG_WEIGHTS_PATH, model.state_dict(), meta,
                    epoch=best_epoch, val_dice=stopper.best)
    print(f"[m2-seg] checkpoint -> {m2.SEG_WEIGHTS_PATH}")

    warnings.append(
        "Method 2's segmentation head is trained on BraTS MRI because it is the only "
        "source of real multi-region masks; the classifier stage may run on SPECT. "
        "This cross-modality step is intentional and recorded, not hidden."
    )
    section = {
        "split": "volume-grouped held-out test",
        "dataset": str(root),
        "dice": round(float(np.mean(dice[1:])), 4),
        "iou": round(float(np.mean(iou[1:])), 4),
        "per_class_dice": {n: round(d, 4) for n, d in zip(m2.SEG_CLASS_NAMES, dice)},
        "per_class_iou": {n: round(i, 4) for n, i in zip(m2.SEG_CLASS_NAMES, iou)},
        "avg_inference_time_s": round(avg_time, 5) if avg_time else None,
        "model_version": meta.model_version,
        "warnings": warnings,
    }
    update_metrics(m2.METHOD_ID, "segmentation", section)

    write_run_card(
        m2.run_card_path("segmentation"),
        RunCard(
            method_id=m2.METHOD_ID, stage="segmentation",
            dataset={"name": "BraTS", "root": str(root), "num_samples": len(samples),
                     "modality": "MRI"},
            split_strategy={"type": "volume-level three-way", "group_key": "BraTS volume id",
                            "val_frac": args.val_frac, "test_frac": args.test_frac, **summary},
            random_seed=config.SEED, preprocessing=spec.to_dict(), image_size=spec.image_size,
            architecture=m2.SEG_ARCHITECTURE,
            model_config={"num_classes": m2.NUM_SEG_CLASSES, "classes": m2.SEG_CLASS_NAMES},
            optimizer={"name": "Adam", "lr": args.lr, "scheduler": "ReduceLROnPlateau",
                       "batch_size": args.batch_size},
            epochs=args.epochs, best_epoch=best_epoch, metrics=section,
            checkpoint_path=str(m2.SEG_WEIGHTS_PATH), inference_time_s=avg_time,
            train_duration_s=round(time.perf_counter() - started, 1),
            model_version=meta.model_version, device=str(device), warnings=warnings,
        ),
        root=config.ROOT_DIR,
    )
    print(f"[m2-seg] done. Held-out foreground Dice: {np.mean(dice[1:]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
