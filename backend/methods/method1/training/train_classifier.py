"""Train Method 1's ConvLSTM classifier.

    python -m backend.methods.method1.training.train_classifier --epochs 40
    python -m backend.methods.method1.training.train_classifier --skip-test   # baseline / tuning

What this script guarantees:

* **The dataset's own split is kept.** Validation is carved from ``Training/``
  only; ``Testing/`` is opened once, after training, and never when
  ``--skip-test`` is given. Training copies that are byte-identical to a test
  image are dropped from the training pool (the test set is left untouched), and
  the train/validation split groups byte-identical images together.
* **Training geometry == serving geometry.** Images are preprocessed with the
  same ``preprocess`` function inference calls (cached once, not recomputed per
  epoch) and the transform spec is written into the checkpoint. If ROI cropping
  is requested but no segmentation checkpoint exists, the run downgrades to the
  whole-slice spec and says so.
* **Honest labels.** Discovery refuses binary tumour/no-tumour datasets.
* **Provenance.** A run card and ``model_metadata.json`` record dataset, split,
  seed, preprocessing, architecture, optimiser, epochs, best epoch, metrics,
  checkpoint and warnings.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

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
    CachedClassificationDataset,
    ClassificationDataset,
    DatasetError,
    build_roi_cache,
    class_distribution,
    class_weights,
    content_group_of,
    discover_classification_samples,
    drop_test_duplicates,
    file_digest,
    load_preprocessed,
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


# --------------------------------------------------------------------------- #
# Data preparation shared by training and the SFLA search
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    spec: TransformSpec
    ds_meta: dict
    split_kind: str
    summary: dict
    train: list
    val: list
    test: list
    removed_test_duplicates: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    roi_boxes: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)  # part -> (N, H, W) float32

    def dataset(self, part: str, augment: bool = False):
        samples = getattr(self, part)
        if part in self.arrays:
            return CachedClassificationDataset(
                self.arrays[part], [y for _, y in samples], augment=augment, seed=config.SEED
            )
        return ClassificationDataset(samples, self.spec, augment=augment, roi_boxes=self.roi_boxes)


def prepare_data(args, *, load_test: bool, device=None) -> PreparedData:
    """Discover, split, de-leak and preprocess. Raises DatasetError on bad data.

    ``load_test=False`` still *identifies* the test files (so leaking training
    copies can be dropped by hash) but never decodes or preprocesses them.
    """
    warnings: list[str] = []
    root = args.data_root or m1.BRI_PATH
    samples, ds_meta = discover_classification_samples(root)
    warnings.extend(ds_meta.get("warnings", []))
    spec, _ = resolve_spec(args.transform, warnings)

    digests = {p: file_digest(p) for p, _ in samples}
    group_of = content_group_of(digests)
    removed: list[str] = []

    use_official = ds_meta.get("has_official_split") and not args.ignore_official_split
    if use_official:
        train_pool, test_s = partition_by_official_split(samples)
        train_pool, removed = drop_test_duplicates(train_pool, test_s, digests)
        if removed:
            warnings.append(
                f"Dropped {len(removed)} Training images that are byte-identical to a "
                f"Testing image, so the test score is not inflated by memorised copies. "
                f"The Testing set itself was not modified."
            )
        train_s, val_s = group_split(
            train_pool, group_of, [1.0 - args.val_frac, args.val_frac], seed=config.SEED
        )
        # Byte-identical copies are additionally kept on one side of train/val
        # (see the run card's group_key); that is not patient-level separation.
        split_kind = ("dataset Training/Testing split; validation carved from Training "
                      "only; image-level split; no patient identifiers")
    else:
        train_s, val_s, test_s = group_train_val_test_split(
            samples, group_of, args.val_frac, args.test_frac, seed=config.SEED
        )
        split_kind = "content-hash grouped random three-way split (no Training/Testing folders)"

    summary = split_summary({"train": train_s, "val": val_s, "test": test_s}, group_of)
    print(f"[cls] split: {split_kind}")
    print(f"[cls]   counts {summary['counts']} | leak-free={summary['leak_free']}"
          f" | train copies of test images dropped={len(removed)}")
    print(f"[cls]   per-class train: {class_distribution(train_s)}")
    print(f"[cls]   per-class val  : {class_distribution(val_s)}")
    print(f"[cls]   per-class test : {class_distribution(test_s)}")
    if not summary["leak_free"]:
        raise DatasetError(f"split shares images between parts: {summary['group_overlap']}")

    data = PreparedData(spec, ds_meta, split_kind, summary, train_s, val_s, test_s,
                        removed, warnings)

    if spec.roi_crop:
        seg = UNet(m1.SEG_IN_CHANNELS, m1.SEG_OUT_CHANNELS, m1.BASE_FILTERS).to(device or config.DEVICE)
        from backend.methods.common.checkpoint import load_checkpoint

        meta, warns = load_checkpoint(
            seg, m1.SEG_WEIGHTS_PATH,
            expected_method=m1.METHOD_ID, expected_architecture=m1.SEG_ARCHITECTURE,
            expected_role="segmentation",
        )
        warnings.extend(warns)
        parts = train_s + val_s + (test_s if load_test else [])
        data.roi_boxes = build_roi_cache(
            parts, seg, spec, str(meta.get("model_version", "unknown")), device
        )
    else:
        parts = ["train", "val"] + (["test"] if load_test else [])
        for part in parts:
            data.arrays[part] = load_preprocessed(getattr(data, part), spec)
    return data


def make_loader(dataset, batch_size: int, train: bool, workers: int) -> DataLoader:
    kwargs: dict = {"num_workers": workers, "pin_memory": config.PIN_MEMORY}
    if workers > 0:
        # Respawning workers every epoch dominates runtime on a small dataset.
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    generator = torch.Generator().manual_seed(config.SEED) if train else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=train,
                      generator=generator, **kwargs)


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict, np.ndarray, np.ndarray, float, float]:
    """Return ``(metrics, y_true, y_prob, avg_seconds_per_image, total_seconds)``."""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    model_time, n = 0.0, 0
    started = time.perf_counter()
    for x, y in loader:
        x = x.to(device, non_blocking=config.PIN_MEMORY)
        _sync(device)
        t0 = time.perf_counter()
        probs = F.softmax(model(x).float(), dim=1)
        _sync(device)
        model_time += time.perf_counter() - t0
        n += x.size(0)
        probs = probs.cpu().numpy()
        y_prob.extend(probs.tolist())
        y_pred.extend(probs.argmax(1).tolist())
        y_true.extend(y.numpy().tolist())
    total = time.perf_counter() - started
    metrics = classification_metrics(y_true, y_pred, m1.NUM_CLASSES)
    metrics["auc"] = macro_auc(np.array(y_true), np.array(y_prob), m1.NUM_CLASSES)
    return metrics, np.array(y_true), np.array(y_prob), model_time / max(1, n), total


def run_epoch(model, loader, criterion, optimizer, scaler, train: bool, device):
    model.train(train)
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=config.PIN_MEMORY)
        y = y.to(device, non_blocking=config.PIN_MEMORY)
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


def build_model(hp: dict, device) -> ConvLSTMClassifier:
    return ConvLSTMClassifier(
        in_channels=1,
        num_classes=m1.NUM_CLASSES,
        base=int(hp["base"]),
        lstm_hidden=int(hp["lstm_hidden"]),
        lstm_steps=int(hp["lstm_steps"]),
        dropout=float(hp["dropout"]),
    ).to(device)


def build_criterion(train_samples, device, enabled: bool = True):
    weights = class_weights(train_samples) if enabled else None
    if not weights:
        return torch.nn.CrossEntropyLoss(), None
    return torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    ), weights


def _metrics_section(m: dict, avg_time: float, total_time: float) -> dict:
    per_class = {
        name: {
            "precision": round(m["per_class"]["precision"][i], 4),
            "recall": round(m["per_class"]["recall"][i], 4),
            "sensitivity": round(m["per_class"]["recall"][i], 4),
            "specificity": round(m["per_class"]["specificity"][i], 4),
            "f1": round(m["per_class"]["f1"][i], 4),
            "support": int(sum(m["confusion_matrix"][i])),
        }
        for i, name in enumerate(m1.CLASS_NAMES)
    }
    support = np.array([per_class[c]["support"] for c in m1.CLASS_NAMES], dtype=float)
    w = support / support.sum() if support.sum() else support

    def weighted(key: str) -> float:
        return round(float(np.dot(w, m["per_class"][key])), 4)

    return {
        "accuracy": round(m["accuracy"], 4),
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "sensitivity": round(m["sensitivity"], 4),
        "specificity": round(m["specificity"], 4),
        "f1": round(m["f1"], 4),
        "weighted": {
            "precision": weighted("precision"),
            "recall": weighted("recall"),
            "specificity": weighted("specificity"),
            "f1": weighted("f1"),
        },
        "auc": round(m["auc"], 4) if m.get("auc") is not None else None,
        "avg_inference_time_s": round(avg_time, 6),
        "total_eval_time_s": round(total_time, 3),
        "confusion_matrix": m["confusion_matrix"],
        "class_order": list(m1.CLASS_NAMES),
        "per_class": per_class,
    }


def _print_section(title: str, s: dict) -> None:
    print(f"\n[cls] ===== {title} =====")
    print(f"[cls] accuracy {s['accuracy']:.4f} | macro P {s['precision']:.4f} R/sens "
          f"{s['recall']:.4f} spec {s['specificity']:.4f} F1 {s['f1']:.4f} | AUC {s['auc']}")
    print(f"[cls] weighted P {s['weighted']['precision']:.4f} R {s['weighted']['recall']:.4f} "
          f"F1 {s['weighted']['f1']:.4f}")
    print(f"[cls] {'class':12s} {'prec':>7s} {'recall':>7s} {'spec':>7s} {'f1':>7s} {'n':>5s}")
    for name, c in s["per_class"].items():
        print(f"[cls] {name:12s} {c['precision']:7.4f} {c['recall']:7.4f} "
              f"{c['specificity']:7.4f} {c['f1']:7.4f} {c['support']:5d}")
    print(f"[cls] confusion matrix (rows=true, cols=pred, order={m1.CLASS_NAMES}):")
    for row in s["confusion_matrix"]:
        print(f"[cls]   {row}")
    print(f"[cls] avg inference {s['avg_inference_time_s'] * 1000:.3f} ms/image | "
          f"total {s['total_eval_time_s']:.2f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Method 1's ConvLSTM classifier")
    ap.add_argument("--epochs", type=int, default=config.CLS_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers; 0 is fastest with the in-memory preprocessed cache")
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
    ap.add_argument("--skip-test", action="store_true",
                    help="validation only: never load the test set (baseline / tuning runs)")
    ap.add_argument("--checkpoint", default=None,
                    help="output path (default weights/method1/best_classifier.pth, or "
                         "baseline_classifier.pth with --skip-test)")
    args = ap.parse_args()

    print("[cls] device report:")
    for k, v in config.device_report().items():
        print(f"[cls]   {k:18s}= {v}")

    seed_everything(config.SEED)
    device = config.DEVICE
    started = time.perf_counter()

    try:
        data = prepare_data(args, load_test=not args.skip_test, device=device)
    except DatasetError as exc:
        print(f"\nERROR: {exc}")
        return 1
    warnings = data.warnings
    spec = data.spec
    print(f"[cls] preprocessing spec: {spec.id} (roi_crop={spec.roi_crop}) "
          f"ready in {time.perf_counter() - started:.1f}s")

    hp = m1.classifier_hyperparams()
    if args.batch_size:
        hp["batch_size"] = args.batch_size
    if args.lr is not None:
        hp["lr"] = args.lr
    batch_size, lr = int(hp["batch_size"]), float(hp["lr"])
    sfla_applied = config.SFLA_ENABLED and m1.SFLA_RESULT_PATH.exists()
    print(f"[cls] hyper-parameters ({'SFLA result ' + str(m1.SFLA_RESULT_PATH) if sfla_applied else 'defaults'}): {hp}")

    train_loader = make_loader(data.dataset("train", augment=True), batch_size, True, args.workers)
    val_loader = make_loader(data.dataset("val"), batch_size, False, args.workers)

    model = build_model(hp, device)
    print(f"[cls] model on {next(model.parameters()).device} | "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")
    criterion, weights = build_criterion(data.train, device, not args.no_class_weights)
    if weights:
        print(f"[cls] class weights (inverse frequency): "
              f"{dict(zip(m1.CLASS_NAMES, [round(w, 3) for w in weights]))}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(hp["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(2, args.patience // 2)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP)
    # Early stopping and checkpoint selection watch validation macro-F1, the same
    # quantity SFLA maximises.
    stopper = EarlyStopping(patience=args.patience, mode="max")

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": [], "lr": []}
    best_epoch, best_state = 0, None
    train_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_m = run_epoch(model, train_loader, criterion, optimizer, scaler, True, device)
        va_loss, va_m = run_epoch(model, val_loader, criterion, optimizer, scaler, False, device)
        scheduler.step(va_m["f1"])

        hist["train_loss"].append(tr_loss); hist["val_loss"].append(va_loss)
        hist["train_acc"].append(tr_m["accuracy"]); hist["val_acc"].append(va_m["accuracy"])
        hist["val_f1"].append(va_m["f1"]); hist["lr"].append(optimizer.param_groups[0]["lr"])
        print(f"[cls] epoch {epoch:03d}/{args.epochs} | loss {tr_loss:.4f}/{va_loss:.4f} | "
              f"acc {tr_m['accuracy']:.4f}/{va_m['accuracy']:.4f} | val F1 {va_m['f1']:.4f} | "
              f"lr {hist['lr'][-1]:.2e} | {time.perf_counter() - t0:.1f}s", flush=True)

        if stopper.step(va_m["f1"]):
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[cls]   new best (val macro-F1 {va_m['f1']:.4f})")
        if stopper.should_stop:
            print(f"[cls] early stop at epoch {epoch} (best val macro-F1 {stopper.best:.4f})")
            break

    epochs_run = len(hist["train_loss"])
    train_time = time.perf_counter() - train_started
    if best_state is None:
        print("ERROR: no epoch improved; nothing to save.")
        return 1
    model.load_state_dict(best_state)

    labels = m1.SPEC.labels
    log_prefix = "method1_baseline" if args.skip_test else "method1"
    val_raw, _, _, v_avg, v_total = evaluate(model, val_loader, device)
    val_section = _metrics_section(val_raw, v_avg, v_total)
    _print_section("VALIDATION (best checkpoint)", val_section)
    save_curve(hist["train_loss"], "Classifier Loss", config.LOGS_DIR / f"{log_prefix}_cls_loss.png",
               second=hist["val_loss"])
    save_curve(hist["train_acc"], "Accuracy", config.LOGS_DIR / f"{log_prefix}_cls_acc.png",
               second=hist["val_acc"])
    save_confusion_matrix(val_section["confusion_matrix"], labels,
                          m1.LOGS_DIR / f"{'baseline' if args.skip_test else 'final'}_val_confusion_matrix.png")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else (
        config.METHOD1_WEIGHTS_DIR / ("baseline_classifier.pth" if args.skip_test else "best_classifier.pth")
    )
    meta = build_meta(
        method_id=m1.METHOD_ID,
        architecture=m1.CLS_ARCHITECTURE,
        role="classification",
        class_names=m1.CLASS_NAMES,
        image_size=spec.image_size,
        transform_id=spec.id,
        hyperparameters=hp,
        val_f1=round(stopper.best, 4),
    )
    save_checkpoint(ckpt_path, model.state_dict(), meta, epoch=best_epoch, val_f1=stopper.best)
    print(f"[cls] checkpoint -> {ckpt_path}")

    sfla_info = {"enabled": config.SFLA_ENABLED, "applied": sfla_applied, "applied_parameters": hp}
    if sfla_applied:
        sfla = json.loads(m1.SFLA_RESULT_PATH.read_text())
        sfla_info.update({
            "result_file": str(m1.SFLA_RESULT_PATH),
            "population": sfla.get("population"),
            "memeplexes": sfla.get("memeplexes"),
            "iterations": sfla.get("iterations"),
            "local_iterations": sfla.get("local_iterations"),
            "evaluations": sfla.get("evaluations"),
            "seed": sfla.get("seed"),
            "best_validation_fitness": sfla.get("best_fitness"),
            "objective": sfla.get("objective"),
        })

    metadata = {
        "method_id": m1.METHOD_ID,
        "method_name": m1.SPEC.display_name,
        "architecture": m1.CLS_ARCHITECTURE,
        "pipeline": "U-Net → ROI crop → ConvLSTM → SFLA-tuned → 4-class softmax",
        "roi_crop_used_in_training": spec.roi_crop,
        "class_order": list(m1.CLASS_NAMES),
        "class_labels": [m1.CLASS_LABELS[c] for c in m1.CLASS_NAMES],
        "dataset_path": str(Path(data.ds_meta["root"]).resolve()),
        "dataset_counts": {
            "train": class_distribution(data.train),
            "validation": class_distribution(data.val),
            "test": class_distribution(data.test),
            "total_files": data.ds_meta["num_samples"],
            "train_copies_of_test_images_dropped": len(data.removed_test_duplicates),
        },
        "split_strategy": data.split_kind,
        "split_level": data.ds_meta.get("split_level"),
        "preprocessing": spec.to_dict(),
        "input_size": spec.image_size,
        "augmentation": "train only: hflip p=0.5, rotate ±20° p=0.5, random crop 0.8–1.0 p=0.3; no vflip",
        "batch_size": batch_size,
        "optimizer": "Adam",
        "learning_rate": lr,
        "weight_decay": hp["weight_decay"],
        "scheduler": f"ReduceLROnPlateau(max val macro-F1, factor 0.5, patience {max(2, args.patience // 2)})",
        "loss": "CrossEntropyLoss" + (" (inverse-frequency class weights)" if weights else ""),
        "class_weights": dict(zip(m1.CLASS_NAMES, weights)) if weights else None,
        "model_hyperparameters": hp,
        "max_epochs": args.epochs,
        "epochs": epochs_run,
        "early_stopping_patience": args.patience,
        "best_epoch": best_epoch,
        "selection_metric": "validation macro-F1",
        "training_time_s": round(train_time, 1),
        "random_seed": config.SEED,
        "device": str(device),
        "device_report": config.device_report(),
        "sfla": sfla_info,
        "model_version": meta.model_version,
        "checkpoint_path": str(ckpt_path),
        "training_timestamp": meta.created_at,
        "history": hist,
        "validation_metrics": val_section,
        "test_metrics": None,
        "warnings": warnings,
    }

    if args.skip_test:
        out = m1.LOGS_DIR / "baseline_validation_metrics.json"
        out.write_text(json.dumps(metadata, indent=2, default=str))
        print(f"[cls] validation-only run; test set not loaded. report -> {out}")
        return 0

    # --- The held-out test set is opened here, once. --------------------- #
    test_loader = make_loader(data.dataset("test"), batch_size, False, args.workers)
    test_raw, y_true, y_prob, avg_time, total_time = evaluate(model, test_loader, device)
    test_section = _metrics_section(test_raw, avg_time, total_time)
    _print_section("FINAL TEST (Testing/ folder, evaluated once)", test_section)
    save_confusion_matrix(test_section["confusion_matrix"], labels,
                          config.LOGS_DIR / "method1_cls_confusion_matrix.png")
    save_roc_curves(y_true, y_prob, labels, config.LOGS_DIR / "method1_cls_roc.png")

    section = {
        # Describe the split that was actually used. Calling an image-level
        # split "patient-grouped" would overstate how independent the test set is.
        "split": data.split_kind,
        "split_level": data.ds_meta.get("split_level", "unknown"),
        "dataset": metadata["dataset_path"],
        **test_section,
        "per_class": {k: [test_section["per_class"][c][k] for c in m1.CLASS_NAMES]
                      for k in ("precision", "recall", "f1", "specificity")},
        "per_class_detail": test_section["per_class"],
        "test_size": len(data.test),
        "best_epoch": best_epoch,
        "model_version": meta.model_version,
        "warnings": warnings,
    }
    update_metrics(m1.METHOD_ID, "classification", section)

    metadata["test_metrics"] = test_section
    m1.METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    m1.METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"[cls] metadata   -> {m1.METADATA_PATH}")
    print(f"[cls] metrics    -> {m1.METRICS_PATH}")

    write_run_card(
        m1.run_card_path("classification"),
        RunCard(
            method_id=m1.METHOD_ID,
            stage="classification",
            dataset={**data.ds_meta, "name": "Brain Tumor MRI Dataset (BRI)"},
            split_strategy={
                "type": data.split_kind,
                "level": data.ds_meta.get("split_level", "unknown"),
                "group_key": "image content hash (dataset has no patient ids)",
                "val_frac": args.val_frac,
                "train_copies_of_test_images_dropped": len(data.removed_test_duplicates),
                **data.summary,
            },
            random_seed=config.SEED,
            preprocessing=spec.to_dict(),
            image_size=spec.image_size,
            architecture=m1.CLS_ARCHITECTURE,
            model_config=hp,
            optimizer={"name": "Adam", "lr": lr, "weight_decay": hp["weight_decay"],
                       "scheduler": metadata["scheduler"], "batch_size": batch_size},
            epochs=epochs_run,
            best_epoch=best_epoch,
            metrics=section,
            checkpoint_path=str(ckpt_path),
            inference_time_s=round(avg_time, 6),
            train_duration_s=round(time.perf_counter() - started, 1),
            model_version=meta.model_version,
            device=str(device),
            warnings=warnings,
        ),
        root=config.ROOT_DIR,
    )
    print(f"[cls] done. Held-out test accuracy: {test_section['accuracy']:.4f}")
    for w in warnings:
        print(f"[cls] WARNING: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
