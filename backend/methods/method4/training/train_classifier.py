"""Train the ZFNet classifier with Red Fox–optimised hyper-parameters.

    python -m backend.methods.method4.optimization.run_rfo          # search first
    python -m backend.methods.method4.training.train_classifier     # uses rfo_results.json

Hyper-parameters come from ``backend/logs/method4/rfo_results.json`` when it
exists (override any with CLI flags, or ``--ignore-rfo``). Early stopping and
checkpoint selection watch validation macro-F1; the test set is loaded only for
the final evaluation and never with ``--skip-test``.

Same split as every MRI method (``backend.methods.common.bri_data``).
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from backend import config
from backend.methods.common.bri_data import cached_arrays, prepare_bri_split
from backend.methods.common.checkpoint import build_meta, save_checkpoint
from backend.methods.common.cnn_training import fit, metrics_section, predict_probs, print_section
from backend.methods.common.metrics_store import update_metrics
from backend.methods.common.runcard import RunCard, write_run_card
from backend.methods.method1.datasets import DatasetError, class_distribution, class_weights
from backend.methods.method4.models import ZFNet
from backend.methods.method4.transforms import DEFAULT_SPEC, preprocess_file, to_input
from backend.methods.registry import METHOD4
from backend.training.common import seed_everything
from backend.utils.metrics import save_confusion_matrix, save_curve, save_roc_curves

TAG = "m4"
CLASS_NAMES = list(METHOD4.class_names)
LOGS_DIR = config.LOGS_DIR / "method4"
RFO_RESULT_PATH = LOGS_DIR / "rfo_results.json"

DEFAULT_HP = {"lr": 3e-4, "weight_decay": 5e-4, "dropout": 0.5, "fc_units": 4096, "batch_size": 64}


def load_split_tensors(data_root=None, val_frac: float = 0.15, load_test: bool = False):
    split = prepare_bri_split(data_root, val_frac, 0.15, False, tag=TAG)
    key = DEFAULT_SPEC.to_dict()
    tensors = {}
    for part in ("train", "val") + (("test",) if load_test else ()):
        samples = getattr(split, part)
        tensors[f"x_{part}"] = torch.from_numpy(cached_arrays(samples, preprocess_file, key, f"method4_{part}"))
        tensors[f"y_{part}"] = torch.tensor([y for _, y in samples], dtype=torch.long)
    return split, tensors


def build_model(hp: dict, device) -> ZFNet:
    return ZFNet(num_classes=len(CLASS_NAMES), in_channels=1, fc_units=int(hp["fc_units"]),
                 dropout=float(hp["dropout"])).to(device)


def make_optimizer(model, hp: dict):
    return torch.optim.AdamW(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))


def make_criterion(train_samples, device):
    return torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights(train_samples), dtype=torch.float32, device=device)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the Red Fox optimised ZFNet")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    for k, v in DEFAULT_HP.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=None)
    ap.add_argument("--ignore-rfo", action="store_true", help="use defaults, not rfo_results.json")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--skip-test", action="store_true", help="never load the test set")
    args = ap.parse_args()

    hp, rfo = dict(DEFAULT_HP), None
    if RFO_RESULT_PATH.exists() and not args.ignore_rfo:
        rfo = json.loads(RFO_RESULT_PATH.read_text())
        hp.update({k: v for k, v in rfo.get("best_parameters", {}).items() if k in hp})
        print(f"[{TAG}] using Red Fox Optimization result {RFO_RESULT_PATH}")
    for k in DEFAULT_HP:
        v = getattr(args, k)
        if v is not None:
            hp[k] = v
    print(f"[{TAG}] hyper-parameters: {hp}")

    device = config.DEVICE
    started = time.perf_counter()
    try:
        split, tensors = load_split_tensors(args.data_root, args.val_frac, load_test=not args.skip_test)
    except DatasetError as exc:
        print(f"\nERROR: {exc}")
        return 1
    warnings = list(split.warnings)

    seed_everything(config.SEED)
    model = build_model(hp, device)
    optimizer = make_optimizer(model, hp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                           patience=max(2, args.patience // 3))
    data = {k: v for k, v in tensors.items() if not k.endswith("_test")}
    print(f"[{TAG}] ZFNet on {device}: {sum(p.numel() for p in model.parameters()):,} parameters")
    result = fit(model, data, batch_size=int(hp["batch_size"]), optimizer=optimizer,
                 criterion=make_criterion(split.train, device), device=device, to_input=to_input,
                 epochs=args.epochs, patience=args.patience, class_names=CLASS_NAMES, tag=TAG,
                 scheduler=scheduler)
    if result.best_state is None:
        print("ERROR: no epoch improved; nothing to save.")
        return 1
    model.load_state_dict(result.best_state)

    probs, avg, total = predict_probs(model, tensors["x_val"], 128, device, to_input)
    val = metrics_section(tensors["y_val"].numpy(), probs, CLASS_NAMES, avg, total)
    print_section(TAG, f"VALIDATION (best epoch {result.best_epoch})", val)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    save_curve(result.history["train_loss"], "Training loss", LOGS_DIR / "train_loss.png")
    save_curve(result.history["train_acc"], "Accuracy", LOGS_DIR / "accuracy.png",
               second=result.history["val_acc"])

    ckpt = config.M4_WEIGHTS_PATH if not args.skip_test else config.M4_WEIGHTS_PATH.with_name("baseline_classifier.pth")
    meta = build_meta(method_id=METHOD4.method_id, architecture="ZFNet", role="classification",
                      class_names=CLASS_NAMES, image_size=DEFAULT_SPEC.image_size,
                      transform_id=DEFAULT_SPEC.id, hyperparameters=hp)
    save_checkpoint(ckpt, model.state_dict(), meta, epoch=result.best_epoch, val_f1=result.best_val_f1)
    print(f"[{TAG}] checkpoint -> {ckpt}")

    rfo_info = None
    if rfo:
        rfo_info = {k: rfo.get(k) for k in ("population", "iterations", "evaluations", "seed",
                                              "best_fitness", "objective", "proxy_epochs")}
        rfo_info["result_file"] = str(RFO_RESULT_PATH)
    metadata = {
        "method_id": METHOD4.method_id,
        "method_name": METHOD4.display_name,
        "architecture": "ZFNet (BatchNorm, single-channel input)",
        "class_order": CLASS_NAMES,
        "dataset_path": str(split.ds_meta["root"]),
        "dataset_counts": {"train": class_distribution(split.train),
                           "validation": class_distribution(split.val),
                           "test": class_distribution(split.test),
                           "train_copies_of_test_images_dropped": len(split.removed_test_duplicates)},
        "split_strategy": split.split_kind,
        "preprocessing": DEFAULT_SPEC.to_dict(),
        "augmentation": "train only, on GPU: hflip, rotate ±15°, scale 0.9–1.1, translate ±6%, brightness/contrast; no vflip",
        "loss": "CrossEntropyLoss (inverse-frequency class weights)",
        "optimizer": "AdamW + ReduceLROnPlateau (val macro-F1)",
        "hyperparameters": hp,
        "red_fox_optimization": rfo_info,
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "training_time_s": round(time.perf_counter() - started, 1),
        "random_seed": config.SEED,
        "device": str(device),
        "model_version": meta.model_version,
        "checkpoint_path": str(ckpt),
        "training_timestamp": meta.created_at,
        "history": result.history,
        "validation_metrics": val,
        "test_metrics": None,
        "warnings": warnings,
    }

    if args.skip_test:
        out = LOGS_DIR / "baseline_validation_metrics.json"
        out.write_text(json.dumps(metadata, indent=2, default=str))
        print(f"[{TAG}] validation-only run; test set not loaded. report -> {out}")
        return 0

    # --- The held-out test set is opened here, once. --------------------- #
    probs, avg, total = predict_probs(model, tensors["x_test"], 128, device, to_input)
    y_test = tensors["y_test"].numpy()
    test = metrics_section(y_test, probs, CLASS_NAMES, avg, total)
    print_section(TAG, "FINAL TEST (Testing/ folder, evaluated once)", test)
    labels = [METHOD4.class_labels[c] for c in CLASS_NAMES]
    save_confusion_matrix(test["confusion_matrix"], labels, config.LOGS_DIR / "method4_cls_confusion_matrix.png")
    save_roc_curves(y_test, probs, labels, config.LOGS_DIR / "method4_cls_roc.png")

    section = {
        "split": split.split_kind, "split_level": split.ds_meta.get("split_level"),
        "dataset": metadata["dataset_path"], **test,
        "per_class": {k: [test["per_class"][c][k] for c in CLASS_NAMES]
                      for k in ("precision", "recall", "f1", "specificity")},
        "per_class_detail": test["per_class"], "test_size": len(split.test),
        "model_version": meta.model_version, "warnings": warnings,
    }
    update_metrics(METHOD4.method_id, "classification", section)
    metadata["test_metrics"] = test
    meta_path = config.M4_WEIGHTS_PATH.parent / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    write_run_card(
        config.LOGS_DIR / METHOD4.run_card_filenames["classification"],
        RunCard(
            method_id=METHOD4.method_id, stage="classification",
            dataset={**split.ds_meta, "name": "Brain Tumor MRI Dataset (BRI)"},
            split_strategy={"type": split.split_kind, **split.summary},
            random_seed=config.SEED, preprocessing=DEFAULT_SPEC.to_dict(),
            image_size=DEFAULT_SPEC.image_size, architecture="ZFNet", model_config=hp,
            optimizer={"name": "AdamW", "red_fox_optimization": rfo_info},
            epochs=result.epochs_run, best_epoch=result.best_epoch, metrics=section,
            checkpoint_path=str(ckpt), inference_time_s=round(avg, 6),
            train_duration_s=metadata["training_time_s"], model_version=meta.model_version,
            device=str(device), warnings=warnings,
        ),
        root=config.ROOT_DIR,
    )
    print(f"[{TAG}] metadata -> {meta_path}")
    print(f"[{TAG}] done. Held-out test accuracy: {test['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
