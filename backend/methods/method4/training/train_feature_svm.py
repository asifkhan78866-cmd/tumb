"""Red Fox feature selection on ZFNet features, classified by an SVM.

    python -m backend.methods.method4.training.train_feature_svm
    python -m backend.methods.method4.training.train_feature_svm --skip-test

The trained ZFNet from ``train_classifier`` is used as a fixed feature
extractor: every image becomes the ``fc_units`` numbers its last fully
connected layer reads. Red Fox Optimization then searches for the subset of
those features — together with the SVM's ``C``, ``gamma`` and kernel — that
maximises *validation* macro-F1, and an SVM fitted on the selected columns is
the classifier.

Two heads therefore exist for this method: the network's own softmax layer and
this SVM. **Which one the API serves is decided on validation macro-F1 alone**,
before the test set is touched. The test split is then opened once and both
heads are reported, so the choice can be checked rather than taken on trust.

The ZFNet weights are not modified. The SVM is a separate artefact beside the
checkpoint and is only used when its recorded ZFNet model version matches the
checkpoint the engine loaded — features from a different network mean nothing
to it.
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
from backend.methods.common.checkpoint import load_checkpoint
from backend.methods.common.cnn_training import metrics_section, predict_probs, print_section, sync
from backend.methods.common.metrics_store import update_metrics
from backend.methods.common.runcard import RunCard, write_run_card
from backend.methods.method1.datasets import DatasetError, class_distribution
from backend.methods.method4.models import ZFNet
from backend.methods.method4.optimization.feature_selection import (
    SVM_PARAMETERS, feature_space, fitted_pipeline, make_objective, mask_of,
)
from backend.methods.method4.optimization.rfo import RedFoxOptimizer
from backend.methods.method4.transforms import DEFAULT_SPEC, preprocess_file, to_input
from backend.methods.registry import METHOD4
from backend.training.common import seed_everything
from backend.utils.metrics import save_confusion_matrix, save_curve, save_roc_curves

TAG = "m4-svm"
CLASS_NAMES = list(METHOD4.class_names)
LOGS_DIR = config.LOGS_DIR / "method4"
ARCHITECTURE = "ZFNet features + RFO feature selection + SVM"
SVM_PATH = config.M4_WEIGHTS_PATH.with_name("rfo_svm.joblib")
RESULT_PATH = LOGS_DIR / "rfo_feature_selection.json"


def load_zfnet(device) -> tuple[ZFNet, dict]:
    """The trained ZFNet, exactly as the engine loads it."""
    meta = json.loads((config.M4_WEIGHTS_PATH.parent / "model_metadata.json").read_text()) \
        if (config.M4_WEIGHTS_PATH.parent / "model_metadata.json").exists() else {}
    hp = meta.get("hyperparameters") or {}
    model = ZFNet(num_classes=len(CLASS_NAMES), in_channels=1,
                  fc_units=int(hp.get("fc_units", 4096)),
                  dropout=float(hp.get("dropout", 0.5))).to(device)
    ckpt_meta, warns = load_checkpoint(
        model, config.M4_WEIGHTS_PATH, expected_method=METHOD4.method_id,
        expected_architecture="ZFNet", expected_role="classification",
        expected_class_names=CLASS_NAMES,
    )
    for w in warns:
        print(f"[{TAG}] {w}")
    model.eval()
    return model, ckpt_meta


@torch.no_grad()
def extract_features(model: ZFNet, images: torch.Tensor, device, version: str,
                     paths: list[str], batch_size: int = 64) -> np.ndarray:
    """Penultimate ZFNet features per image, computed once and cached on disk.

    The cache key includes the checkpoint's model version: features from a
    different network must never be read back as if they were these.
    """
    key = hashlib.sha1(json.dumps([version, sorted(paths), DEFAULT_SPEC.to_dict()],
                                  sort_keys=True).encode()).hexdigest()[:16]
    cache = config.MODEL_CACHE_DIR / f"method4_zfnet_feat_{key}.npy"
    if cache.exists():
        print(f"[{TAG}] feature cache hit ({cache.name})")
        return np.load(cache)
    out, t0 = [], time.perf_counter()
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(device).float().unsqueeze(1) / 255.0
        out.append(model.penultimate(to_input(batch)).float().cpu().numpy())
    sync(device)
    features = np.concatenate(out)
    config.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, features)
    print(f"[{TAG}] {features.shape} features in {time.perf_counter() - t0:.0f}s -> {cache.name}",
          flush=True)
    return features


def probs_from_svm(svm, x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Class probabilities, timed. The served class is the argmax of these."""
    t0 = time.perf_counter()
    probs = svm.predict_proba(x[:, mask])
    total = time.perf_counter() - t0
    return probs.astype(np.float64), total / max(1, len(x)), total


def section_for(split, test: dict, *, head: str, architecture: str, model_version: str,
                warnings: list[str], extra: dict) -> dict:
    return {
        "split": split.split_kind, "split_level": split.ds_meta.get("split_level"),
        "dataset": str(split.ds_meta["root"]), **test,
        "per_class": {k: [test["per_class"][c][k] for c in CLASS_NAMES]
                      for k in ("precision", "recall", "f1", "specificity")},
        "per_class_detail": test["per_class"], "test_size": len(split.test),
        "head": head, "architecture": architecture,
        "model_version": model_version, "warnings": warnings, **extra,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Red Fox feature selection + SVM on ZFNet features")
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--replace-fraction", type=float, default=0.25)
    ap.add_argument("--sparsity-penalty", type=float, default=0.02,
                    help="fitness penalty for the fraction of features kept")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--skip-test", action="store_true", help="never load the test set")
    args = ap.parse_args()

    try:
        import joblib  # noqa: F401
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("ERROR: scikit-learn and joblib are required: pip install scikit-learn joblib")
        return 1

    if not config.M4_WEIGHTS_PATH.exists():
        print(f"ERROR: {config.M4_WEIGHTS_PATH} not found. Train the ZFNet first:\n"
              f"  python -m backend.methods.method4.training.train_classifier")
        return 1

    device = config.DEVICE
    print(f"[{TAG}] device report: {config.device_report()}")
    started = time.perf_counter()
    seed_everything(args.seed)

    try:
        split = prepare_bri_split(args.data_root, args.val_frac, 0.15, False, tag=TAG)
    except DatasetError as exc:
        print(f"\nERROR: {exc}")
        return 1
    warnings = list(split.warnings)

    model, ckpt_meta = load_zfnet(device)
    zfnet_version = str(ckpt_meta.get("model_version") or "unknown")
    print(f"[{TAG}] feature extractor: ZFNet {zfnet_version} "
          f"({model.hparams['fc_units']} features per image)")

    parts = ("train", "val") + (() if args.skip_test else ("test",))
    images, labels, features = {}, {}, {}
    for part in parts:
        samples = getattr(split, part)
        images[part] = torch.from_numpy(
            cached_arrays(samples, preprocess_file, DEFAULT_SPEC.to_dict(), f"method4_{part}")
        )
        labels[part] = np.array([y for _, y in samples])
        features[part] = extract_features(model, images[part], device, zfnet_version,
                                          [p for p, _ in samples])

    # Standardise on the training split only, and drop features that are constant
    # there: a feature that never varies carries no information and would make the
    # F-statistics and the SVM's scaling meaningless.
    scaler = StandardScaler().fit(features["train"])
    # StandardScaler reports scale_ = 1 for a constant feature, so variance is what
    # identifies them.
    live = scaler.var_ > 1e-12
    n_live = int(live.sum())
    print(f"[{TAG}] {n_live} of {features['train'].shape[1]} features vary on the training split; "
          f"{features['train'].shape[1] - n_live} are constant and are dropped before the search")
    scaled = {p: scaler.transform(features[p])[:, live] for p in parts}

    # --- Red Fox feature selection: train fits, validation scores. -------- #
    objective = make_objective(scaled["train"], labels["train"], scaled["val"], labels["val"],
                               sparsity_penalty=args.sparsity_penalty, seed=args.seed)
    space = feature_space(n_live, SVM_PARAMETERS)
    rfo = RedFoxOptimizer(space, objective, population=args.population,
                          iterations=args.iterations, replace_fraction=args.replace_fraction,
                          seed=args.seed,
                          objective_name="validation macro-F1 − 0.02·(features kept / features available)",
                          progress=lambda h: print(
                              f"[{TAG}] iteration {h['iteration']:03d}/{args.iterations} | "
                              f"best {h['best_fitness']:.4f} mean {h['mean_fitness']:.4f} | "
                              f"{h['evaluations']} evaluations", flush=True))
    print(f"[{TAG}] searching {len(space)} dimensions ({n_live} features + "
          f"{len(SVM_PARAMETERS)} SVM hyper-parameters) with {args.population} foxes "
          f"× {args.iterations} iterations", flush=True)
    result = rfo.run()
    mask = mask_of(result.best_parameters, n_live)
    svm_hp = {k: result.best_parameters[k] for k in ("C", "gamma", "kernel")}
    kept = int(mask.sum())
    print(f"[{TAG}] selected {kept}/{n_live} features | SVM {svm_hp} | "
          f"fitness {result.best_fitness:.4f} after {result.evaluations} evaluations "
          f"in {result.duration_s:.0f}s")

    # The search's fitness carries a sparsity penalty, so refit and score the chosen
    # model plainly. probability=True (Platt scaling) because the API serves
    # probabilities, and the reported metrics must be the ones it serves.
    svm = fitted_pipeline(result.best_parameters, mask, scaled["train"], labels["train"],
                          seed=args.seed, probability=True)
    val_probs, val_avg, val_total = probs_from_svm(svm, scaled["val"], mask)
    svm_val = metrics_section(labels["val"], val_probs, CLASS_NAMES, val_avg, val_total)
    print_section(TAG, "VALIDATION — RFO-selected features + SVM", svm_val)

    zf_val_probs, zf_avg, zf_total = predict_probs(model, images["val"], 128, device, to_input)
    zf_val = metrics_section(labels["val"], zf_val_probs, CLASS_NAMES, zf_avg, zf_total)
    print_section(TAG, "VALIDATION — ZFNet softmax (for comparison)", zf_val)

    # --- Head choice, made on validation only. ---------------------------- #
    serve_svm = svm_val["f1"] > zf_val["f1"]
    chosen = "rfo_svm" if serve_svm else "zfnet_softmax"
    print(f"[{TAG}] validation macro-F1: SVM {svm_val['f1']:.4f} vs ZFNet softmax "
          f"{zf_val['f1']:.4f} -> serving the {chosen} head")

    warnings.append(
        "Two classifier heads exist for this method: the ZFNet softmax layer and an SVM on "
        f"Red Fox–selected ZFNet features. The {chosen} head is served; the choice was made on "
        f"validation macro-F1 ({svm_val['f1']:.4f} SVM vs {zf_val['f1']:.4f} softmax) before the "
        "test set was opened."
    )
    if serve_svm:
        warnings.append(
            "The SVM's probabilities come from Platt scaling cross-fitted on the training "
            "split; they rank classes well but are not calibrated risk estimates."
        )
        warnings.append(
            "Grad-CAM explains the ZFNet backbone that produced the features, not the SVM's "
            "decision boundary."
        )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    save_curve([h["best_fitness"] for h in result.history],
               "Red Fox feature selection — best fitness per iteration",
               LOGS_DIR / "rfo_feature_selection.png",
               second=[h["mean_fitness"] for h in result.history])

    kept_columns = np.flatnonzero(live)[mask]  # indices into the raw feature vector
    report = {
        **result.to_dict(),
        "stage": "feature_selection",
        "feature_extractor": {"architecture": "ZFNet", "layer": "penultimate fully connected",
                              "model_version": zfnet_version,
                              "features_per_image": int(features["train"].shape[1])},
        "features_available_to_search": n_live,
        "features_selected": kept,
        "selected_feature_indices": kept_columns.tolist(),
        "svm": {**svm_hp, "class_weight": "balanced", "probability": "Platt scaling"},
        "sparsity_penalty": args.sparsity_penalty,
        "validation": {"rfo_svm": svm_val, "zfnet_softmax": zf_val},
        "served_head": chosen,
        "head_choice": "higher validation macro-F1; decided before the test set was loaded",
        "split": split.split_kind,
        "notes": [
            "Fitness is validation macro-F1 of an SVM fitted on the training split only.",
            "The test split was not loaded during the search.",
            "Red Fox's local search moves on a hypersphere, whose per-coordinate step shrinks "
            "with dimension; in a space this wide the global search and the habitat "
            "reproduction do most of the work.",
        ],
    }
    # The evaluation log holds one entry per candidate, each with a full keep/drop
    # dictionary; that is hundreds of thousands of lines for no benefit, so only the
    # scores are kept and the best position is recorded as selected indices above.
    report["evaluation_log"] = [{k: v for k, v in e.items() if k != "parameters"}
                                for e in result.evaluation_log]
    report["best_parameters"] = {**svm_hp, "features_selected": kept}
    report["best_position"] = "see selected_feature_indices"
    RESULT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"[{TAG}] search result -> {RESULT_PATH}")

    bundle = {
        "method_id": METHOD4.method_id,
        "role": "classification",
        "architecture": ARCHITECTURE,
        "class_names": CLASS_NAMES,
        "zfnet_model_version": zfnet_version,
        "zfnet_fc_units": int(features["train"].shape[1]),
        "transform_id": DEFAULT_SPEC.id,
        "scaler": scaler,
        "kept_columns": kept_columns,
        "svm": svm,
        "svm_hyperparameters": svm_hp,
        "features_selected": kept,
        "features_available": n_live,
        "validation_macro_f1": svm_val["f1"],
        "zfnet_validation_macro_f1": zf_val["f1"],
        "serve": bool(serve_svm),
        "model_version": time.strftime("%Y%m%d-%H%M%S"),
        "search_result": str(RESULT_PATH),
        "warnings": warnings,
    }

    if args.skip_test:
        out = LOGS_DIR / "feature_svm_validation_metrics.json"
        out.write_text(json.dumps({k: v for k, v in bundle.items()
                                   if k not in ("scaler", "svm", "kept_columns")},
                                  indent=2, default=str))
        print(f"[{TAG}] validation-only run; test set not loaded. report -> {out}")
        return 0

    # --- The held-out test set is opened here, once, for both heads. ------ #
    svm_probs, svm_t_avg, svm_t_total = probs_from_svm(svm, scaled["test"], mask)
    svm_test = metrics_section(labels["test"], svm_probs, CLASS_NAMES, svm_t_avg, svm_t_total)
    print_section(TAG, "FINAL TEST — RFO-selected features + SVM (evaluated once)", svm_test)
    zf_probs, zf_t_avg, zf_t_total = predict_probs(model, images["test"], 128, device, to_input)
    zf_test = metrics_section(labels["test"], zf_probs, CLASS_NAMES, zf_t_avg, zf_t_total)
    print_section(TAG, "FINAL TEST — ZFNet softmax (evaluated once)", zf_test)

    served_test, other_test = (svm_test, zf_test) if serve_svm else (zf_test, svm_test)
    served_probs = svm_probs if serve_svm else zf_probs
    served_arch = ARCHITECTURE if serve_svm else "ZFNet (BatchNorm, single-channel input)"
    other_arch = "ZFNet (BatchNorm, single-channel input)" if serve_svm else ARCHITECTURE
    model_version = bundle["model_version"] if serve_svm else zfnet_version

    display = [METHOD4.class_labels[c] for c in CLASS_NAMES]
    save_confusion_matrix(served_test["confusion_matrix"], display,
                          config.LOGS_DIR / "method4_cls_confusion_matrix.png")
    save_roc_curves(labels["test"], served_probs, display, config.LOGS_DIR / "method4_cls_roc.png")

    extra = {"features_selected": kept, "features_available": n_live, "svm": svm_hp,
             "feature_extractor_version": zfnet_version} if serve_svm else {}
    section = section_for(split, served_test, head=chosen, architecture=served_arch,
                          model_version=model_version, warnings=warnings, extra=extra)
    update_metrics(METHOD4.method_id, "classification", section)
    update_metrics(METHOD4.method_id, "classification_other_head", section_for(
        split, other_test, head="zfnet_softmax" if serve_svm else "rfo_svm",
        architecture=other_arch, model_version=zfnet_version if serve_svm else bundle["model_version"],
        warnings=["Recorded for comparison; this head is not the one the API serves."],
        extra={},
    ))

    bundle["test_metrics"] = svm_test
    bundle["zfnet_test_metrics"] = zf_test
    import joblib

    joblib.dump(bundle, SVM_PATH, compress=3)
    print(f"[{TAG}] SVM bundle -> {SVM_PATH} ({SVM_PATH.stat().st_size:,} bytes)")

    report["test"] = {"rfo_svm": svm_test, "zfnet_softmax": zf_test}
    RESULT_PATH.write_text(json.dumps(report, indent=2, default=str))

    meta_path = config.M4_WEIGHTS_PATH.parent / "feature_svm_metadata.json"
    meta_path.write_text(json.dumps({
        "method_id": METHOD4.method_id, "method_name": METHOD4.display_name,
        "architecture": ARCHITECTURE, "class_order": CLASS_NAMES,
        "dataset_path": str(split.ds_meta["root"]),
        "dataset_counts": {"train": class_distribution(split.train),
                           "validation": class_distribution(split.val),
                           "test": class_distribution(split.test),
                           "train_copies_of_test_images_dropped": len(split.removed_test_duplicates)},
        "split_strategy": split.split_kind, "preprocessing": DEFAULT_SPEC.to_dict(),
        "feature_extractor": report["feature_extractor"],
        "feature_selection": {"algorithm": "Red Fox Optimization",
                              "population": args.population, "iterations": args.iterations,
                              "evaluations": result.evaluations,
                              "features_available": n_live, "features_selected": kept,
                              "duration_s": round(result.duration_s, 1)},
        "svm": bundle["svm_hyperparameters"], "served_head": chosen,
        "validation_metrics": {"rfo_svm": svm_val, "zfnet_softmax": zf_val},
        "test_metrics": {"rfo_svm": svm_test, "zfnet_softmax": zf_test},
        "random_seed": args.seed, "device": str(device),
        "model_version": bundle["model_version"],
        "artefact_path": str(SVM_PATH),
        "total_time_s": round(time.perf_counter() - started, 1),
        "warnings": warnings,
    }, indent=2, default=str))

    write_run_card(
        config.LOGS_DIR / METHOD4.run_card_filenames["feature_selection"],
        RunCard(method_id=METHOD4.method_id, stage="feature_selection",
                dataset={**split.ds_meta, "name": "Brain Tumor MRI Dataset (BRI)"},
                split_strategy={"type": split.split_kind, **split.summary},
                random_seed=args.seed, preprocessing=DEFAULT_SPEC.to_dict(),
                image_size=DEFAULT_SPEC.image_size, architecture=ARCHITECTURE,
                model_config={"features_available": n_live, "features_selected": kept, **svm_hp},
                optimizer={"name": "Red Fox Optimization (feature selection)",
                           "population": args.population, "iterations": args.iterations,
                           "evaluations": result.evaluations,
                           "objective": rfo.objective_name},
                epochs=args.iterations, best_epoch=args.iterations, metrics=section,
                checkpoint_path=str(SVM_PATH), inference_time_s=served_test["avg_inference_time_s"],
                train_duration_s=round(time.perf_counter() - started, 1),
                model_version=model_version, device=str(device), warnings=warnings),
        root=config.ROOT_DIR,
    )
    print(f"[{TAG}] metadata -> {meta_path}")
    print(f"[{TAG}] done. Served head: {chosen}. Held-out test accuracy: "
          f"{served_test['accuracy']:.4f} (SVM {svm_test['accuracy']:.4f}, "
          f"ZFNet softmax {zf_test['accuracy']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
