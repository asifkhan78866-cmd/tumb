"""Train the feature-fusion ensemble: four frozen backbones, one meta-learner.

    python -m backend.methods.method3.training.train_fusion
    python -m backend.methods.method3.training.train_fusion --skip-test

GoogleNet, VGGNet, ResNet and AlexNet each read the same image and are reduced
to a 64-d branch vector; the four are concatenated and a meta-learner learns how
to weigh them. The backbones are frozen, so their output for a given image never
changes — which is why it is computed once per image and cached, and why only
the branch heads and meta-learner are trained.

The cost of that: no augmentation. Augmented copies would each need their own
forward pass through four backbones, which is the expensive part. This is
recorded in the run card rather than left for a reader to infer.

Same split as every MRI method (``backend.methods.common.bri_data``), and the
test set is opened once, at the end, and never with ``--skip-test``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from backend import config
from backend.methods.common.bri_data import cached_arrays, prepare_bri_split
from backend.methods.common.checkpoint import build_meta, save_checkpoint
from backend.methods.common.cnn_training import metrics_section, print_section, sync
from backend.methods.common.metrics_store import update_metrics
from backend.methods.common.runcard import RunCard, write_run_card
from backend.methods.method1.datasets import DatasetError, class_distribution, class_weights
from backend.methods.method3.models import FUSION_BACKBONES, FrozenBackbone, FusionHead
from backend.methods.method3.transforms import DEFAULT_SPEC, preprocess_file, to_input
from backend.methods.registry import METHOD3
from backend.training.common import seed_everything
from backend.utils.metrics import save_confusion_matrix, save_curve, save_roc_curves

TAG = "m3-fusion"
CLASS_NAMES = list(METHOD3.class_names)
LOGS_DIR = config.LOGS_DIR / "method3"
ARCHITECTURE = "FusionEnsembleCNN"


@torch.no_grad()
def backbone_features(name: str, images: torch.Tensor, device, batch_size: int = 32,
                      paths: list[str] | None = None) -> torch.Tensor:
    """Pooled features for every image, computed once and cached on disk."""
    key = hashlib.sha1(json.dumps([name, sorted(paths or []), DEFAULT_SPEC.to_dict()],
                                  sort_keys=True).encode()).hexdigest()[:16]
    cache = config.MODEL_CACHE_DIR / f"method3_fusion_{name}_{key}.npy"
    if cache.exists():
        print(f"[{TAG}] {name}: feature cache hit ({cache.name})")
        return torch.from_numpy(np.load(cache))

    model = FrozenBackbone(name).to(device).eval()
    out = []
    started = time.perf_counter()
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(device).float().unsqueeze(1) / 255.0
        out.append(model(to_input(batch)).cpu())
    sync(device)
    features = torch.cat(out)
    np.save(cache, features.numpy())
    print(f"[{TAG}] {name}: {tuple(features.shape)} features in "
          f"{time.perf_counter() - started:.0f}s -> {cache.name}", flush=True)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return features


def extract_all(split, tensors: dict, backbones: tuple[str, ...], device) -> dict:
    """``part -> {backbone: features}`` for every part that was loaded."""
    features: dict[str, dict[str, torch.Tensor]] = {}
    for part in tensors:
        paths = [p for p, _ in getattr(split, part)]
        features[part] = {
            name: backbone_features(name, tensors[part], device, paths=paths)
            for name in backbones
        }
    return features


def run_epoch(head, features: dict, labels: torch.Tensor, batch_size: int, optimizer, criterion,
              device, train: bool, gen=None) -> tuple[float, np.ndarray]:
    head.train(train)
    names = head.backbone_names
    n = len(labels)
    order = torch.randperm(n, generator=gen) if train else torch.arange(n)
    total_loss, probs = 0.0, []
    for i in range(0, n, batch_size):
        idx = order[i : i + batch_size]
        if train and len(idx) < 2:
            continue  # BatchNorm in the meta-learner needs more than one sample
        batch = {k: features[k][idx].to(device) for k in names}
        y = labels[idx].to(device)
        with torch.set_grad_enabled(train):
            logits = head(batch)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(idx)
        probs.append(torch.softmax(logits.detach().float(), dim=1).cpu().numpy())
    return total_loss / max(1, n), np.concatenate(probs)[np.argsort(order.numpy())]


def main() -> int:
    ap = argparse.ArgumentParser(description="Feature-fusion ensemble for brain tumour MRI")
    ap.add_argument("--backbones", default=",".join(FUSION_BACKBONES))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--skip-test", action="store_true", help="never load the test set")
    args = ap.parse_args()

    backbones = tuple(b.strip() for b in args.backbones.split(",") if b.strip())
    device = config.DEVICE
    print(f"[{TAG}] device report: {config.device_report()}")
    started = time.perf_counter()
    seed_everything(config.SEED)

    try:
        split = prepare_bri_split(args.data_root, args.val_frac, 0.15, False, tag=TAG)
    except DatasetError as exc:
        print(f"\nERROR: {exc}")
        return 1
    warnings = list(split.warnings)
    warnings.append(
        "The fusion ensemble trains on cached frozen-backbone features, so no augmentation "
        "is applied: an augmented copy would need its own pass through all four backbones."
    )

    parts = ("train", "val") + (() if args.skip_test else ("test",))
    tensors, labels = {}, {}
    for part in parts:
        samples = getattr(split, part)
        tensors[part] = torch.from_numpy(
            cached_arrays(samples, preprocess_file, DEFAULT_SPEC.to_dict(), f"method3_{part}")
        )
        labels[part] = torch.tensor([y for _, y in samples], dtype=torch.long)
    features = extract_all(split, tensors, backbones, device)

    head = FusionHead(backbones, len(CLASS_NAMES), args.dropout).to(device)
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[{TAG}] fusing {list(backbones)} | meta-learner {trainable:,} trainable parameters")
    weights = class_weights(split.train)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                           patience=max(2, args.patience // 3))
    gen = torch.Generator().manual_seed(config.SEED)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_state, best_epoch, best_f1, stale = None, 0, -1.0, 0
    y_val = labels["val"].numpy()

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_probs = run_epoch(head, features["train"], labels["train"], args.batch_size,
                                      optimizer, criterion, device, True, gen)
        va_loss, va_probs = run_epoch(head, features["val"], labels["val"], max(64, args.batch_size),
                                      optimizer, criterion, device, False)
        tr_acc = float((tr_probs.argmax(1) == labels["train"].numpy()).mean())
        va = metrics_section(y_val, va_probs, CLASS_NAMES, 0.0, 0.0)
        scheduler.step(va["f1"])
        for k, v in (("train_loss", tr_loss), ("val_loss", va_loss), ("train_acc", tr_acc),
                     ("val_acc", va["accuracy"]), ("val_f1", va["f1"])):
            hist[k].append(v)
        print(f"[{TAG}] epoch {epoch:03d}/{args.epochs} | loss {tr_loss:.4f}/{va_loss:.4f} | "
              f"acc {tr_acc:.4f}/{va['accuracy']:.4f} | val F1 {va['f1']:.4f} | "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
        if va["f1"] > best_f1 + 1e-4:
            best_f1, best_epoch, stale = va["f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            print(f"[{TAG}]   new best (val macro-F1 {best_f1:.4f})")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[{TAG}] early stop at epoch {epoch} (best val macro-F1 {best_f1:.4f})")
                break

    if best_state is None:
        print("ERROR: no epoch improved; nothing to save.")
        return 1
    head.load_state_dict(best_state)
    _, va_probs = run_epoch(head, features["val"], labels["val"], 128, optimizer, criterion, device, False)
    val = metrics_section(y_val, va_probs, CLASS_NAMES, 0.0, 0.0)
    print_section(TAG, f"VALIDATION (best epoch {best_epoch})", val)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # The canonical names the gallery reads: the plots on display must come from the
    # run that produced the checkpoint being served, not from a route that lost.
    save_curve(hist["train_loss"], "Loss", LOGS_DIR / "train_loss.png", second=hist["val_loss"])
    save_curve(hist["train_acc"], "Accuracy", LOGS_DIR / "accuracy.png", second=hist["val_acc"])

    hp = {"backbones": list(backbones), "dropout": args.dropout, "lr": args.lr,
          "weight_decay": args.weight_decay, "batch_size": args.batch_size,
          "label_smoothing": args.label_smoothing, "epochs": args.epochs}
    ckpt = config.M3_WEIGHTS_PATH if not args.skip_test else config.M3_WEIGHTS_PATH.with_name("fusion_baseline.pth")
    meta = build_meta(
        method_id=METHOD3.method_id, architecture=ARCHITECTURE, role="classification",
        class_names=CLASS_NAMES, image_size=DEFAULT_SPEC.image_size, transform_id=DEFAULT_SPEC.id,
        hyperparameters=hp, backbones=list(backbones),
        # Only the trained part is stored; the frozen ImageNet backbones are rebuilt
        # from torchvision at load time (downloaded once, then cached locally).
        stores="fusion head only",
    )
    save_checkpoint(ckpt, head.state_dict(), meta, epoch=best_epoch, val_f1=best_f1)
    print(f"[{TAG}] checkpoint -> {ckpt}")

    metadata = {
        "method_id": METHOD3.method_id, "method_name": METHOD3.display_name,
        "architecture": f"{ARCHITECTURE}: {' + '.join(backbones)} -> concat -> meta-learner",
        "class_order": CLASS_NAMES, "dataset_path": str(split.ds_meta["root"]),
        "dataset_counts": {"train": class_distribution(split.train),
                           "validation": class_distribution(split.val),
                           "test": class_distribution(split.test),
                           "train_copies_of_test_images_dropped": len(split.removed_test_duplicates)},
        "split_strategy": split.split_kind, "preprocessing": DEFAULT_SPEC.to_dict(),
        "augmentation": "none (frozen-backbone features are cached)",
        "frozen_backbones": list(backbones), "trainable_parameters": trainable,
        "hyperparameters": hp, "best_epoch": best_epoch, "epochs_run": len(hist["train_loss"]),
        "training_time_s": round(time.perf_counter() - started, 1), "random_seed": config.SEED,
        "device": str(device), "model_version": meta.model_version,
        "checkpoint_path": str(ckpt), "training_timestamp": meta.created_at,
        "history": hist, "validation_metrics": val, "test_metrics": None, "warnings": warnings,
    }

    if args.skip_test:
        out = LOGS_DIR / "fusion_validation_metrics.json"
        out.write_text(json.dumps(metadata, indent=2, default=str))
        print(f"[{TAG}] validation-only run; test set not loaded. report -> {out}")
        return 0

    # --- The held-out test set is opened here, once. --------------------- #
    t0 = time.perf_counter()
    _, probs = run_epoch(head, features["test"], labels["test"], 128, optimizer, criterion, device, False)
    total = time.perf_counter() - t0
    y_test = labels["test"].numpy()
    test = metrics_section(y_test, probs, CLASS_NAMES, total / max(1, len(y_test)), total)
    print_section(TAG, "FINAL TEST (Testing/ folder, evaluated once)", test)
    labels_display = [METHOD3.class_labels[c] for c in CLASS_NAMES]
    save_confusion_matrix(test["confusion_matrix"], labels_display,
                          config.LOGS_DIR / "method3_cls_confusion_matrix.png")
    save_roc_curves(y_test, probs, labels_display, config.LOGS_DIR / "method3_cls_roc.png")

    section = {
        "split": split.split_kind, "split_level": split.ds_meta.get("split_level"),
        "dataset": metadata["dataset_path"], **test,
        "per_class": {k: [test["per_class"][c][k] for c in CLASS_NAMES]
                      for k in ("precision", "recall", "f1", "specificity")},
        "per_class_detail": test["per_class"], "test_size": len(split.test),
        "architecture": metadata["architecture"], "backbones": list(backbones),
        "model_version": meta.model_version, "warnings": warnings,
    }
    update_metrics(METHOD3.method_id, "classification", section)
    metadata["test_metrics"] = test
    meta_path = config.M3_WEIGHTS_PATH.parent / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    write_run_card(
        config.LOGS_DIR / METHOD3.run_card_filenames["classification"],
        RunCard(method_id=METHOD3.method_id, stage="classification",
                dataset={**split.ds_meta, "name": "Brain Tumor MRI Dataset (BRI)"},
                split_strategy={"type": split.split_kind, **split.summary},
                random_seed=config.SEED, preprocessing=DEFAULT_SPEC.to_dict(),
                image_size=DEFAULT_SPEC.image_size, architecture=metadata["architecture"],
                model_config=hp, optimizer={"name": "AdamW", "lr": args.lr},
                epochs=metadata["epochs_run"], best_epoch=best_epoch, metrics=section,
                checkpoint_path=str(ckpt), inference_time_s=test["avg_inference_time_s"],
                train_duration_s=metadata["training_time_s"], model_version=meta.model_version,
                device=str(device), warnings=warnings),
        root=config.ROOT_DIR,
    )
    print(f"[{TAG}] metadata -> {meta_path}")
    print(f"[{TAG}] done. Held-out test accuracy: {test['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
