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
import json
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


def prepare_mri_shared_split(args, warnings: list[str]):
    """MRI branch: the same BRI split and test images as the other MRI methods."""
    from backend.methods.common.bri_data import cached_arrays, prepare_bri_split
    from backend.methods.method1.datasets import class_weights
    from backend.methods.method2.datasets import CachedSpectClassificationDataset
    from backend.methods.method2.transforms import preprocess_file

    split = prepare_bri_split(args.data_root, args.val_frac, args.test_frac, False, tag="m2-cls")
    warnings.extend(split.warnings)
    # BRI labels use the MRI methods' order; map them onto this method's class list by name.
    bri_names = ["glioma", "meningioma", "notumor", "pituitary"]
    remap = {i: m2.CLASS_NAMES.index("normal" if n == "notumor" else n) for i, n in enumerate(bri_names)}
    key = m2.TRANSFORM.to_dict()
    parts, labels = {}, {}
    for part in ("train", "val", "test"):
        samples = getattr(split, part)
        labels[part] = [remap[y] for _, y in samples]
        arr = cached_arrays(samples, preprocess_file, key, f"method2_{part}")
        parts[part] = CachedSpectClassificationDataset(arr, labels[part], augment=(part == "train"),
                                                        seed=config.SEED)
    # class_weights counts by index on (path, label) pairs, in this method's class order.
    counts = [labels["train"].count(i) for i in range(m2.NUM_CLASSES)]
    total = sum(counts)
    raw = [total / (m2.NUM_CLASSES * max(1, c)) for c in counts]
    weights = [w / (sum(raw) / len(raw)) for w in raw]
    ds_meta = {**split.ds_meta, "modality": "MRI",
               "class_counts": {m2.CLASS_NAMES[i]: c for i, c in enumerate(counts)}}
    return parts, weights, ds_meta, split.split_kind, split.summary, {
        k: {m2.CLASS_NAMES[i]: labels[k].count(i) for i in range(m2.NUM_CLASSES)} for k in labels
    }, len(split.removed_test_duplicates)


def main() -> int:
    from backend.methods.common.cnn_training import metrics_section, print_section, sync

    ap = argparse.ArgumentParser(description="Train Method 4's (MRI–SPECT fusion) dense network")
    ap.add_argument("--modality", choices=["spect", "mri"], default="spect")
    ap.add_argument("--epochs", type=int, default=config.CLS_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=32)
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
    seg_model = load_segmentation(device, warnings)
    spec = m2.TRANSFORM

    if modality == "MRI" and seg_model is None:
        try:
            datasets, weights, ds_meta, split_kind, summary, dist, dropped = prepare_mri_shared_split(args, warnings)
        except Exception as exc:
            print(f"\nERROR: {exc}")
            return 1
        warnings.insert(0, (
            "Trained on MRI only: this is the MRI branch of the MRI–SPECT fusion model. No "
            "SPECT data and no paired MRI–SPECT scans exist, so the SPECT branch and the "
            "fusion step are not trained, and these metrics describe MRI classification, "
            "not multimodal fusion."
        ))
    else:
        root = args.data_root or (m2.SPECT_PATH if modality == "SPECT" else m2.BRI_PATH)
        try:
            samples, ds_meta = discover_classification_samples(root, modality)
        except DatasetError as exc:
            print(f"\nERROR: {exc}")
            return 1
        warnings.extend(ds_meta.get("warnings", []))
        train_s, val_s, test_s = group_train_val_test_split(
            samples, classification_group_of, args.val_frac, args.test_frac, seed=config.SEED
        )
        summary = split_summary({"train": train_s, "val": val_s, "test": test_s}, classification_group_of)
        if not summary["leak_free"]:
            print("ERROR: split produced overlapping groups; refusing to train.")
            return 1
        split_kind, dropped = "group-level three-way split", 0
        datasets = {name: SpectClassificationDataset(part, spec, seg_model, device, augment=(name == "train"))
                    for name, part in (("train", train_s), ("val", val_s), ("test", test_s))}
        dist = {k: {c: sum(1 for _, y in part if y == i) for i, c in enumerate(m2.CLASS_NAMES)}
                for k, part in (("train", train_s), ("val", val_s), ("test", test_s))}
        counts = [dist["train"][c] for c in m2.CLASS_NAMES]
        raw = [sum(counts) / (m2.NUM_CLASSES * max(1, c)) for c in counts]
        weights = [w / (sum(raw) / len(raw)) for w in raw]

    print(f"[m2-cls] {modality} | split: {split_kind} | {summary['counts']}")
    gen = torch.Generator().manual_seed(config.SEED)
    loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=(name == "train"),
                         num_workers=args.workers, drop_last=(name == "train"),
                         generator=gen if name == "train" else None)
        for name, ds in datasets.items()
    }

    hp = dict(m2.DCN_DEFAULTS)
    model = DenseConvNetClassifier(
        in_channels=m2.DCN_IN_CHANNELS, num_classes=m2.NUM_CLASSES,
        growth_rate=int(hp["growth_rate"]), block_config=tuple(hp["block_config"]),
        num_init_features=int(hp["num_init_features"]), compression=float(hp["compression"]),
        dropout=float(hp["dropout"]), feature_dim=m2.DCN_FEATURE_DIM,
    ).to(device)
    print(f"[m2-cls] DCN on {device}: {sum(p.numel() for p in model.parameters()):,} parameters | "
          f"class weights {dict(zip(m2.CLASS_NAMES, [round(w, 3) for w in weights]))}")
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP)
    # Model selection watches validation macro-F1, like the other MRI methods.
    stopper = EarlyStopping(patience=args.patience, mode="max")

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_epoch, best_state = 0, None
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_m = run_epoch(model, loaders["train"], criterion, optimizer, scaler, True, device)
        va_loss, va_m = run_epoch(model, loaders["val"], criterion, optimizer, scaler, False, device)
        scheduler.step()
        for k, v in (("train_loss", tr_loss), ("val_loss", va_loss), ("train_acc", tr_m["accuracy"]),
                     ("val_acc", va_m["accuracy"]), ("val_f1", va_m["f1"])):
            hist[k].append(v)
        print(f"[m2-cls] epoch {epoch:03d}/{args.epochs} | loss {tr_loss:.4f}/{va_loss:.4f} | "
              f"acc {tr_m['accuracy']:.4f}/{va_m['accuracy']:.4f} | val F1 {va_m['f1']:.4f} | "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
        if stopper.step(va_m["f1"]):
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[m2-cls]   new best (val macro-F1 {va_m['f1']:.4f})")
        if stopper.should_stop:
            print(f"[m2-cls] early stop at epoch {epoch} (best val macro-F1 {stopper.best:.4f})")
            break

    if best_state is None:
        print("ERROR: no epoch improved; nothing to save.")
        return 1
    model.load_state_dict(best_state)

    def full_eval(loader):
        model.eval()
        ys, probs, model_time, n = [], [], 0.0, 0
        t_start = time.perf_counter()
        with torch.no_grad():
            for x, feats, y in loader:
                x, feats = x.to(device), feats.to(device)
                sync(device)
                t0 = time.perf_counter()
                p = F.softmax(model(x, feats).float(), dim=1)
                sync(device)
                model_time += time.perf_counter() - t0
                n += x.size(0)
                probs.append(p.cpu().numpy())
                ys.extend(y.numpy().tolist())
        probs = np.concatenate(probs)
        return np.array(ys), probs, metrics_section(ys, probs, m2.CLASS_NAMES, model_time / max(1, n),
                                                     time.perf_counter() - t_start)

    _, _, val = full_eval(loaders["val"])
    print_section("m2-cls", f"VALIDATION (best epoch {best_epoch})", val)

    meta = build_meta(
        method_id=m2.METHOD_ID, architecture=m2.CLS_ARCHITECTURE, role="classification",
        class_names=m2.CLASS_NAMES, image_size=spec.image_size, transform_id=spec.id,
        hyperparameters=hp, modality=modality, branch=("MRI" if modality == "MRI" else modality),
        fusion_trained=False, feature_stage_active=seg_model is not None,
    )
    save_checkpoint(m2.DCN_WEIGHTS_PATH, model.state_dict(), meta, epoch=best_epoch, val_f1=stopper.best)
    print(f"[m2-cls] checkpoint -> {m2.DCN_WEIGHTS_PATH}")

    # --- The held-out test set is opened here, once. --------------------- #
    y_true, y_prob, test = full_eval(loaders["test"])
    print_section("m2-cls", f"FINAL TEST ({modality}, evaluated once)", test)
    labels = m2.SPEC.labels
    save_curve(hist["train_loss"], "M4 DCN Loss", config.LOGS_DIR / "method2_cls_loss.png", second=hist["val_loss"])
    save_curve(hist["train_acc"], "M4 Accuracy", config.LOGS_DIR / "method2_cls_acc.png", second=hist["val_acc"])
    save_confusion_matrix(test["confusion_matrix"], labels, config.LOGS_DIR / "method2_cls_confusion_matrix.png")
    save_roc_curves(y_true, y_prob, labels, config.LOGS_DIR / "method2_cls_roc.png")

    section = {
        "split": split_kind, "dataset": f"{ds_meta['root']} ({modality})", "modality": modality,
        "branch": "MRI branch of the MRI–SPECT fusion model" if modality == "MRI" else modality,
        **test,
        "per_class": {k: [test["per_class"][c][k] for c in m2.CLASS_NAMES]
                      for k in ("precision", "recall", "f1", "specificity")},
        "per_class_detail": test["per_class"],
        "model_version": meta.model_version, "feature_stage_active": seg_model is not None,
        "warnings": warnings,
    }
    update_metrics(m2.METHOD_ID, "classification", section)

    metadata = {
        "method_id": m2.METHOD_ID, "method_name": m2.SPEC.display_name,
        "architecture": f"{m2.CLS_ARCHITECTURE} (DenseNet-BC style)", "trained_branch": modality,
        "fusion_trained": False, "class_order": list(m2.CLASS_NAMES), "dataset": ds_meta.get("root"),
        "dataset_counts": {**dist, "train_copies_of_test_images_dropped": dropped},
        "split_strategy": split_kind, "preprocessing": spec.to_dict(), "hyperparameters": hp,
        "optimizer": {"name": "Adam", "lr": args.lr, "weight_decay": 1e-4,
                      "scheduler": "CosineAnnealingLR", "batch_size": args.batch_size},
        "class_weights": dict(zip(m2.CLASS_NAMES, weights)), "best_epoch": best_epoch,
        "epochs_run": len(hist["train_loss"]), "training_time_s": round(time.perf_counter() - started, 1),
        "random_seed": config.SEED, "device": str(device), "model_version": meta.model_version,
        "checkpoint_path": str(m2.DCN_WEIGHTS_PATH), "history": hist,
        "validation_metrics": val, "test_metrics": test, "warnings": warnings,
    }
    meta_path = m2.DCN_WEIGHTS_PATH.with_name("method2_model_metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    write_run_card(
        m2.run_card_path("classification"),
        RunCard(
            method_id=m2.METHOD_ID, stage="classification", dataset=ds_meta,
            split_strategy={"type": split_kind, **summary}, random_seed=config.SEED,
            preprocessing=spec.to_dict(), image_size=spec.image_size, architecture=m2.CLS_ARCHITECTURE,
            model_config={**hp, "in_channels": m2.DCN_IN_CHANNELS, "feature_dim": m2.DCN_FEATURE_DIM},
            optimizer=metadata["optimizer"], epochs=metadata["epochs_run"], best_epoch=best_epoch,
            metrics=section, checkpoint_path=str(m2.DCN_WEIGHTS_PATH),
            inference_time_s=test["avg_inference_time_s"], train_duration_s=metadata["training_time_s"],
            model_version=meta.model_version, device=str(device), warnings=warnings,
        ),
        root=config.ROOT_DIR,
    )
    print(f"[m2-cls] metadata -> {meta_path}")
    print(f"[m2-cls] done. Held-out test accuracy ({modality}): {test['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
