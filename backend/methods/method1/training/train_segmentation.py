"""Train Method 1's binary U-Net segmentation model.

    python -m backend.methods.method1.training.train_segmentation --epochs 50

Splits are **volume-level**: every slice of a BraTS volume lands in exactly one
of train/val/test. Splitting BraTS slices at random — as the previous script did
— puts near-identical neighbouring slices on both sides and inflates Dice
substantially, which is the single most common way a segmentation number becomes
meaningless.

The test part is scored once at the end and is never seen by early stopping.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend import config
from backend.methods.common.checkpoint import build_meta, save_checkpoint
from backend.methods.common.metrics_store import update_metrics
from backend.methods.common.runcard import RunCard, write_run_card
from backend.methods.common.splits import group_train_val_test_split, split_summary
from backend.methods.method1 import config as m1
from backend.methods.method1.datasets import (
    SegmentationDataset,
    discover_segmentation_samples,
    lgg_patient_id,
    segmentation_group_of,
)
from backend.methods.method1.models import UNet
from backend.methods.method1.transforms import M1_V2
from backend.training.common import EarlyStopping, seed_everything
from backend.utils.losses import DiceBCELoss, dice_coefficient, iou_score
from backend.utils.metrics import save_curve


def run_epoch(model, loader, criterion, optimizer, scaler, train: bool, device):
    model.train(train)
    total_loss = total_dice = total_iou = 0.0
    n = 0
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
        total_dice += dice_coefficient(logits.float(), y) * bs
        total_iou += iou_score(logits.float(), y) * bs
        n += bs
    n = max(1, n)
    return total_loss / n, total_dice / n, total_iou / n


@torch.no_grad()
def evaluate(model, loader, device):
    """Return ``(dice, iou, seconds/image, dice_on_tumour_slices, n_tumour_slices)``.

    Dice is reported twice on purpose. A slice with no tumour that is correctly
    predicted empty scores 1.0, and most slices of a brain volume contain no
    tumour, so the overall mean is dominated by easy negatives. The second
    figure — computed only over slices whose ground truth actually contains a
    tumour — is the one that describes how well the tumour is outlined.
    """
    model.eval()
    dices, ious, times, tumour_dices = [], [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        t0 = time.perf_counter()
        logits = model(x)
        times.append((time.perf_counter() - t0) / max(1, x.size(0)))
        dices.append(dice_coefficient(logits.float(), y))
        ious.append(iou_score(logits.float(), y))
        has_tumour = y.view(y.size(0), -1).sum(dim=1) > 0
        for i in torch.nonzero(has_tumour, as_tuple=False).flatten().tolist():
            tumour_dices.append(dice_coefficient(logits[i : i + 1].float(), y[i : i + 1]))
    if not dices:
        return None, None, None, None, 0
    tumour_dice = float(np.mean(tumour_dices)) if tumour_dices else None
    return float(np.mean(dices)), float(np.mean(ious)), float(np.mean(times)), tumour_dice, len(tumour_dices)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Method 1's U-Net")
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

    root = args.data_root or m1.BRATS_PATH
    samples, kind, warnings = discover_segmentation_samples(root)
    if not samples:
        print(
            f"\nERROR: no segmentation data under {root}. Download BraTS with "
            f"`python -m backend.utils.dataset_download --kind segmentation` or set "
            f"BRATS_DATASET_PATH."
        )
        return 1
    print(f"[seg] {len(samples)} samples ({kind}) from {root}")

    spec = M1_V2.resized(m1.IMAGE_SIZE)
    if kind == "cjdata":
        group_of = lambda s: s[1]  # noqa: E731 - patient id read from the .mat file
    elif kind == "pairs":
        group_of = lambda s: lgg_patient_id(s[0])  # noqa: E731 - one folder per patient
    elif kind in ("h5", "nifti"):
        group_of = segmentation_group_of
    else:
        group_of = lambda s: str(s)  # noqa: E731
    if kind == "image":
        warnings.append(
            "Samples carry no volume identifier, so the split is per-file and may "
            "not be patient-disjoint."
        )

    train_s, val_s, test_s = group_train_val_test_split(
        samples, group_of, args.val_frac, args.test_frac, seed=config.SEED
    )
    summary = split_summary({"train": train_s, "val": val_s, "test": test_s}, group_of)
    level = ("patient" if kind in ("pairs", "cjdata")
             else "volume" if kind in ("h5", "nifti") else "file")
    print(f"[seg] split ({level}-grouped): {summary['counts']} | "
          f"groups {summary['group_counts']} | leak-free={summary['leak_free']}")
    if not summary["leak_free"]:
        print("ERROR: split produced overlapping volumes; refusing to train.")
        return 1

    loaders = {
        name: DataLoader(
            SegmentationDataset(part, kind, spec, augment=(name == "train")),
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=args.workers,
            pin_memory=config.USE_AMP,
            drop_last=(name == "train"),
        )
        for name, part in (("train", train_s), ("val", val_s), ("test", test_s))
    }

    model = UNet(m1.SEG_IN_CHANNELS, m1.SEG_OUT_CHANNELS, m1.BASE_FILTERS).to(device)
    criterion = DiceBCELoss(bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP)
    stopper = EarlyStopping(patience=args.patience, mode="max")

    hist = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    best_epoch, best_state = 0, None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice, _ = run_epoch(model, loaders["train"], criterion, optimizer, scaler, True, device)
        va_loss, va_dice, va_iou = run_epoch(model, loaders["val"], criterion, optimizer, scaler, False, device)
        scheduler.step(va_dice)

        hist["train_loss"].append(tr_loss); hist["val_loss"].append(va_loss)
        hist["train_dice"].append(tr_dice); hist["val_dice"].append(va_dice)
        print(f"[seg] epoch {epoch:03d}/{args.epochs} | loss {tr_loss:.4f}/{va_loss:.4f} | "
              f"dice {tr_dice:.4f}/{va_dice:.4f} | iou {va_iou:.4f}")

        if stopper.step(va_dice):
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[seg]   new best (val dice {va_dice:.4f})")
        if stopper.should_stop:
            print(f"[seg] early stop at epoch {epoch} (best val dice {stopper.best:.4f})")
            break

    if best_state is None:
        print("ERROR: no epoch improved; nothing to save.")
        return 1
    model.load_state_dict(best_state)

    test_dice, test_iou, avg_time, tumour_dice, n_tumour = evaluate(model, loaders["test"], device)
    save_curve(hist["train_loss"], "Segmentation Loss", config.LOGS_DIR / "method1_seg_loss.png",
               second=hist["val_loss"])
    save_curve(hist["train_dice"], "Dice Coefficient", config.LOGS_DIR / "method1_seg_dice.png",
               second=hist["val_dice"])

    mask_source = {"h5": "BraTS ground truth", "nifti": "BraTS ground truth",
                   "cjdata": "figshare (Cheng) radiologist tumour masks, contrast-enhanced T1",
                   "pairs": "LGG manual FLAIR-abnormality masks",
                   "image": "weak Otsu threshold"}[kind]
    meta = build_meta(
        method_id=m1.METHOD_ID,
        architecture=m1.SEG_ARCHITECTURE,
        role="segmentation",
        image_size=spec.image_size,
        transform_id=spec.id,
        test_dice=test_dice,
        # Written into the checkpoint so serving can warn about masks learned from
        # a thresholding heuristic rather than from radiologist annotation.
        mask_source=mask_source,
        ground_truth_masks=kind in ("h5", "nifti", "pairs", "cjdata"),
        trained_on=str(root),
    )
    save_checkpoint(m1.SEG_WEIGHTS_PATH, model.state_dict(), meta,
                    epoch=best_epoch, val_dice=stopper.best)
    print(f"[seg] checkpoint -> {m1.SEG_WEIGHTS_PATH}")

    section = {
        "split": f"{level}-grouped held-out test",
        "dataset": str(root),
        "dice": round(test_dice, 4) if test_dice is not None else None,
        # The headline number for outlining quality: empty slices are excluded.
        "dice_on_tumour_slices": round(tumour_dice, 4) if tumour_dice is not None else None,
        "tumour_slices_in_test": n_tumour,
        "iou": round(test_iou, 4) if test_iou is not None else None,
        "avg_inference_time_s": round(avg_time, 5) if avg_time is not None else None,
        "model_version": meta.model_version,
        "mask_source": mask_source,
        "warnings": warnings,
    }
    update_metrics(m1.METHOD_ID, "segmentation", section)

    write_run_card(
        m1.run_card_path("segmentation"),
        RunCard(
            method_id=m1.METHOD_ID, stage="segmentation",
            dataset={"name": "BraTS", "root": str(root), "kind": kind, "num_samples": len(samples)},
            split_strategy={"type": "volume-level three-way", "group_key": "BraTS volume id",
                            "val_frac": args.val_frac, "test_frac": args.test_frac, **summary},
            random_seed=config.SEED,
            preprocessing=spec.to_dict(),
            image_size=spec.image_size,
            architecture=m1.SEG_ARCHITECTURE,
            model_config={"base_filters": m1.BASE_FILTERS, "in_channels": m1.SEG_IN_CHANNELS,
                          "out_channels": m1.SEG_OUT_CHANNELS},
            optimizer={"name": "Adam", "lr": args.lr, "weight_decay": 1e-5,
                       "scheduler": "ReduceLROnPlateau", "batch_size": args.batch_size},
            epochs=args.epochs, best_epoch=best_epoch, metrics=section,
            checkpoint_path=str(m1.SEG_WEIGHTS_PATH),
            inference_time_s=avg_time,
            train_duration_s=round(time.perf_counter() - started, 1),
            model_version=meta.model_version, device=str(device), warnings=warnings,
        ),
        root=config.ROOT_DIR,
    )
    print(f"[seg] done. Held-out test Dice: {test_dice:.4f} (all slices) | "
          f"{tumour_dice if tumour_dice is None else round(tumour_dice, 4)} on the "
          f"{n_tumour} slices that contain a tumour")
    for w in warnings:
        print(f"[seg] WARNING: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
