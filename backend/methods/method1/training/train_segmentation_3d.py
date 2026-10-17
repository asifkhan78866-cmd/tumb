"""Train the volumetric 3D U-Net on Decathlon (BraTS) volumes.

    python -m backend.methods.method1.training.train_segmentation_3d
    python -m backend.methods.method1.training.train_segmentation_3d --limit-volumes 40

This is the volumetric stage of Method 3's architecture. It is a *separate*
model from the 2D U-Net the API serves, and deliberately so: an upload is a
single slice, and a 3D U-Net cannot segment one slice — it needs the slices
above and below. So this model is trained and measured on volumes, its result
is recorded as a volumetric result, and the served path stays 2D. Claiming
otherwise would mean reporting a number produced by a model the API never runs.

Training is patch-based. A whole 128³ volume through a 3D U-Net does not fit
comfortably on one laptop GPU, and patches also give the sampler somewhere to
be useful: most of a brain volume is tumour-free, so patches are drawn centred
on annotated voxels a documented fraction of the time instead of uniformly,
which is what stops the network from learning that the answer is always
"background".

Splits are over volumes — one volume is one patient — and the test volumes are
segmented exactly once, at the end, with sliding-window inference.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from backend import config
from backend.methods.common import msd_brain
from backend.methods.common.checkpoint import build_meta, save_checkpoint
from backend.methods.common.metrics_store import update_metrics
from backend.methods.common.runcard import RunCard, write_run_card
from backend.methods.common.splits import group_train_val_test_split, split_summary
from backend.methods.method1 import config as m1
from backend.methods.method1.models import UNet3D
from backend.methods.registry import METHOD1
from backend.training.common import seed_everything
from backend.utils.metrics import save_curve

TAG = "m1-3d"
LOGS_DIR = config.LOGS_DIR / "method1"
ARCHITECTURE = "UNet3D"
WEIGHTS_PATH = m1.SEG_WEIGHTS_PATH.with_name("best_unet3d.pth")
RESULT_PATH = LOGS_DIR / "segmentation3d_metrics.json"


class VolumeStore:
    """Cached volumes, read lazily. One entry per patient."""

    def __init__(self, directory: Path, names: list[str]):
        self.directory = Path(directory)
        self.names = list(names)

    def __len__(self) -> int:
        return len(self.names)

    def load(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        with np.load(self.directory / name) as data:
            # Binary whole-tumour: every annotated sub-region is tumour. The
            # sub-region breakdown is Method 4's multi-class head, not this one.
            return data["image"].astype(np.float32), (data["label"] > 0).astype(np.uint8)


def sample_patch(image: np.ndarray, label: np.ndarray, size: int, rng: np.random.Generator,
                 tumour_prob: float) -> tuple[np.ndarray, np.ndarray]:
    """One patch, centred on tumour with probability ``tumour_prob``."""
    shape = np.array(image.shape)
    half = size // 2
    centre = None
    if rng.random() < tumour_prob:
        voxels = np.argwhere(label > 0)
        if len(voxels):
            centre = voxels[rng.integers(len(voxels))]
    if centre is None:
        centre = rng.integers(half, np.maximum(shape - half, half + 1))
    start = np.clip(centre - half, 0, np.maximum(shape - size, 0))
    sl = tuple(slice(int(s), int(s) + size) for s in start)
    patch, target = image[sl], label[sl]
    if patch.shape != (size, size, size):  # a volume smaller than the patch
        pad = [(0, size - s) for s in patch.shape]
        patch = np.pad(patch, pad)
        target = np.pad(target, pad)
    return patch, target


def dice_of(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice for one volume; 1.0 only when both are empty (a correct 'nothing here')."""
    p, t = pred.astype(bool), target.astype(bool)
    total = p.sum() + t.sum()
    if total == 0:
        return 1.0
    return float(2 * (p & t).sum() / total)


@torch.no_grad()
def segment_volume(model, image: np.ndarray, patch: int, device, overlap: int = 16) -> np.ndarray:
    """Sliding-window inference over a whole volume, averaging the overlaps."""
    model.eval()
    shape = image.shape
    logits = np.zeros((2, *shape), dtype=np.float32)
    counts = np.zeros(shape, dtype=np.float32)
    step = max(patch - overlap, 1)
    starts = [range(0, max(s - patch, 0) + 1, step) for s in shape]
    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                sl = (slice(z, z + patch), slice(y, y + patch), slice(x, x + patch))
                block = image[sl]
                pad = [(0, patch - s) for s in block.shape]
                padded = np.pad(block, pad) if any(p[1] for p in pad) else block
                t = torch.from_numpy(padded).float()[None, None].to(device)
                out = model(t).squeeze(0).cpu().numpy()
                out = out[(slice(None), *(slice(0, s) for s in block.shape))]
                logits[(slice(None), *sl)] += out
                counts[sl] += 1.0
    counts[counts == 0] = 1.0
    return (logits[1] / counts > logits[0] / counts).astype(np.uint8)


def evaluate(model, store: VolumeStore, names: list[str], patch: int, device
             ) -> tuple[dict, list[dict]]:
    """Dice per volume, and the aggregate. Tumour-free volumes are reported apart."""
    per_volume, times = [], []
    for name in names:
        image, label = store.load(name)
        t0 = time.perf_counter()
        pred = segment_volume(model, image, patch, device)
        times.append(time.perf_counter() - t0)
        per_volume.append({"volume": Path(name).stem, "dice": round(dice_of(pred, label), 4),
                           "tumour_voxels": int(label.sum()),
                           "predicted_voxels": int(pred.sum())})
    with_tumour = [v["dice"] for v in per_volume if v["tumour_voxels"] > 0]
    inter = sum(min(v["predicted_voxels"], v["tumour_voxels"]) for v in per_volume)
    summary = {
        "dice": round(float(np.mean([v["dice"] for v in per_volume])), 4),
        "dice_on_volumes_with_tumour": round(float(np.mean(with_tumour)), 4) if with_tumour else None,
        "volumes": len(per_volume),
        "volumes_with_tumour": len(with_tumour),
        "avg_inference_time_s": round(float(np.mean(times)), 3),
    }
    return summary, per_volume


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the volumetric 3D U-Net")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--volume-size", type=int, default=128)
    ap.add_argument("--patches-per-volume", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--tumour-prob", type=float, default=0.7,
                    help="fraction of training patches centred on an annotated voxel")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--limit-volumes", type=int, default=0,
                    help="use only the first N volumes (a smoke test, never a reported run)")
    ap.add_argument("--skip-test", action="store_true", help="never segment the test volumes")
    args = ap.parse_args()

    device = config.DEVICE
    started = time.perf_counter()
    seed_everything(config.SEED)
    rng = np.random.default_rng(config.SEED)

    root = Path(args.data_root).expanduser() if args.data_root else config.MSD_BRAIN_PATH
    pairs = msd_brain.discover_msd(root)
    if not pairs:
        print(f"ERROR: no Decathlon volumes under {root}. Expected imagesTr/ and labelsTr/ "
              f"(Task01_BrainTumour). Set MSD_BRAIN_PATH if it lives elsewhere.")
        return 1
    warnings: list[str] = []
    if args.limit_volumes:
        pairs = pairs[: args.limit_volumes]
        warnings.append(
            f"SMOKE TEST: only {len(pairs)} volumes were used (--limit-volumes). These numbers "
            f"describe a fraction of the dataset and must not be quoted as the method's result."
        )

    cache = msd_brain.volume_cache_dir("FLAIR", args.volume_size)
    names = msd_brain.materialise_volumes(pairs, cache, modality="FLAIR", size=args.volume_size)
    names = [n for n in names if Path(n).stem in {p.volume_id for p in pairs}]
    store = VolumeStore(cache, names)
    warnings.append(
        f"Volumes were brain-cropped and resampled to {args.volume_size}³ from 240×240×155, so "
        f"the Dice below is measured at that resolution, not at the scanner's."
    )
    warnings.append(
        "FLAIR only: the Decathlon volumes carry four co-registered sequences and this model "
        "reads one, which is the sequence whole-tumour extent is clearest in."
    )

    train_v, val_v, test_v = group_train_val_test_split(
        names, lambda n: Path(n).stem.lower(), args.val_frac, args.test_frac, seed=config.SEED
    )
    summary = split_summary({"train": train_v, "val": val_v, "test": test_v},
                            lambda n: Path(n).stem.lower())
    print(f"[{TAG}] {len(names)} volumes | split {summary['counts']} | "
          f"leak-free={summary['leak_free']}")
    if not summary["leak_free"]:
        print("ERROR: split produced overlapping volumes; refusing to train.")
        return 1

    model = UNet3D(in_channels=1, num_classes=2, base=args.base).to(device)
    print(f"[{TAG}] UNet3D on {device}: {sum(p.numel() for p in model.parameters()):,} parameters "
          f"| {args.patch}³ patches")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                           patience=max(2, args.patience // 2))
    criterion = torch.nn.CrossEntropyLoss()

    def epoch_patches(volumes: list[str], train: bool):
        """Patches for one epoch, as (image, label) batches."""
        order = list(volumes)
        if train:
            rng.shuffle(order)
        buf_x, buf_y = [], []
        for name in order:
            image, label = store.load(name)
            for _ in range(args.patches_per_volume):
                px, py = sample_patch(image, label, args.patch, rng,
                                      args.tumour_prob if train else 0.5)
                buf_x.append(px)
                buf_y.append(py)
                if len(buf_x) == args.batch_size:
                    yield (torch.from_numpy(np.stack(buf_x)).float().unsqueeze(1),
                           torch.from_numpy(np.stack(buf_y)).long())
                    buf_x, buf_y = [], []
        if buf_x:
            yield (torch.from_numpy(np.stack(buf_x)).float().unsqueeze(1),
                   torch.from_numpy(np.stack(buf_y)).long())

    def run(volumes: list[str], train: bool) -> tuple[float, float]:
        model.train(train)
        total, n, dices = 0.0, 0, []
        for x, y in epoch_patches(volumes, train):
            x, y = x.to(device), y.to(device)
            with torch.set_grad_enabled(train):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
            pred = logits.detach().argmax(1).cpu().numpy()
            target = y.cpu().numpy()
            dices.extend(dice_of(p, t) for p, t in zip(pred, target))
        return total / max(1, n), float(np.mean(dices)) if dices else 0.0

    hist = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    best_state, best_epoch, best_dice, stale = None, 0, -1.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_dice = run(train_v, True)
        va_loss, va_dice = run(val_v, False)
        scheduler.step(va_dice)
        for k, v in (("train_loss", tr_loss), ("val_loss", va_loss),
                     ("train_dice", tr_dice), ("val_dice", va_dice)):
            hist[k].append(v)
        print(f"[{TAG}] epoch {epoch:03d}/{args.epochs} | loss {tr_loss:.4f}/{va_loss:.4f} | "
              f"patch dice {tr_dice:.4f}/{va_dice:.4f} | {time.perf_counter() - t0:.0f}s",
              flush=True)
        if va_dice > best_dice + 1e-4:
            best_dice, best_epoch, stale = va_dice, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[{TAG}]   new best (val patch Dice {best_dice:.4f})")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[{TAG}] early stop at epoch {epoch} (best {best_dice:.4f})")
                break

    if best_state is None:
        print("ERROR: no epoch improved; nothing to save.")
        return 1
    model.load_state_dict(best_state)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    save_curve(hist["train_loss"], "3D U-Net loss", LOGS_DIR / "seg3d_loss.png",
               second=hist["val_loss"])
    save_curve(hist["train_dice"], "3D U-Net patch Dice", LOGS_DIR / "seg3d_dice.png",
               second=hist["val_dice"])

    # Validation is scored on whole volumes too: a patch Dice is not the number
    # anyone means by "how well does it segment a scan".
    val_summary, _ = evaluate(model, store, val_v, args.patch, device)
    print(f"[{TAG}] VALIDATION (whole volumes): Dice {val_summary['dice']:.4f} "
          f"({val_summary['volumes']} volumes)")

    meta = build_meta(
        method_id=METHOD1.method_id, architecture=ARCHITECTURE, role="segmentation",
        class_names=["background", "tumour"], image_size=args.volume_size,
        transform_id=f"msd-flair-{args.volume_size}", hyperparameters={
            "patch": args.patch, "base": args.base, "lr": args.lr,
            "batch_size": args.batch_size, "tumour_prob": args.tumour_prob,
            "patches_per_volume": args.patches_per_volume, "volume_size": args.volume_size,
        },
        volumetric=True, serves_uploads=False,
        mask_source="Medical Segmentation Decathlon Task01_BrainTumour (BraTS) voxel labels",
        ground_truth_masks=True,
    )
    save_checkpoint(WEIGHTS_PATH, model.state_dict(), meta, epoch=best_epoch, val_dice=best_dice)
    print(f"[{TAG}] checkpoint -> {WEIGHTS_PATH}")

    warnings.append(
        "This volumetric model is not what the API serves. An upload is a single slice and a 3D "
        "U-Net needs the slices around it, so uploads are segmented by the 2D U-Net; this "
        "model's Dice is a volumetric result measured on held-out volumes."
    )

    record = {
        "architecture": ARCHITECTURE, "dataset": str(root),
        "source": "Medical Segmentation Decathlon Task01_BrainTumour (BraTS volumes)",
        "modality": "FLAIR", "volume_size": args.volume_size, "patch": args.patch,
        "split": "volume-level (one volume per patient)", "split_summary": summary,
        "epochs_run": len(hist["train_loss"]), "best_epoch": best_epoch,
        "validation": val_summary, "test": None, "per_volume": None,
        "history": hist, "model_version": meta.model_version,
        "checkpoint": str(WEIGHTS_PATH), "device": str(device),
        "train_duration_s": round(time.perf_counter() - started, 1), "warnings": warnings,
    }

    if args.skip_test:
        RESULT_PATH.write_text(json.dumps(record, indent=2, default=str))
        print(f"[{TAG}] validation-only run; test volumes untouched. report -> {RESULT_PATH}")
        return 0

    # --- The held-out volumes are segmented here, once. ------------------- #
    test_summary, per_volume = evaluate(model, store, test_v, args.patch, device)
    print(f"[{TAG}] FINAL TEST (held-out volumes, segmented once): Dice {test_summary['dice']:.4f} "
          f"| on volumes with tumour {test_summary['dice_on_volumes_with_tumour']} "
          f"| {test_summary['volumes']} volumes")

    record["test"] = test_summary
    record["per_volume"] = per_volume
    RESULT_PATH.write_text(json.dumps(record, indent=2, default=str))

    section = {
        "split": "volume-level held-out test (one volume per patient)",
        "dataset": record["source"],
        "architecture": ARCHITECTURE,
        "volumetric": True,
        "serves_uploads": False,
        **test_summary,
        "validation": val_summary,
        "model_version": meta.model_version,
        "warnings": warnings,
    }
    update_metrics(METHOD1.method_id, "segmentation_3d", section)

    write_run_card(
        config.LOGS_DIR / METHOD1.run_card_filenames["segmentation_3d"],
        RunCard(method_id=METHOD1.method_id, stage="segmentation_3d",
                dataset={"name": record["source"], "root": str(root),
                         "volumes": len(names), "modality": "FLAIR"},
                split_strategy={"type": "volume-level three-way", "group_key": "volume id",
                                "val_frac": args.val_frac, "test_frac": args.test_frac, **summary},
                random_seed=config.SEED,
                preprocessing={"brain_crop": True, "resample": f"{args.volume_size}³",
                               "normalise": "1–99th percentile of brain voxels"},
                image_size=args.volume_size, architecture=ARCHITECTURE,
                model_config=dict(meta.extra.get("hyperparameters", {})),
                optimizer={"name": "AdamW", "lr": args.lr, "scheduler": "ReduceLROnPlateau"},
                epochs=record["epochs_run"], best_epoch=best_epoch, metrics=section,
                checkpoint_path=str(WEIGHTS_PATH),
                inference_time_s=test_summary["avg_inference_time_s"],
                train_duration_s=record["train_duration_s"], model_version=meta.model_version,
                device=str(device), warnings=warnings),
        root=config.ROOT_DIR,
    )
    print(f"[{TAG}] done. Held-out volumetric Dice: {test_summary['dice']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
