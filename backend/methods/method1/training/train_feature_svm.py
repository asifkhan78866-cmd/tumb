"""SFLA feature selection on ConvLSTM features, classified by an SVM.

    python -m backend.methods.method1.training.train_feature_svm
    python -m backend.methods.method1.training.train_feature_svm --skip-test

This is the selection stage of Method 3's architecture. The trained ConvLSTM
classifier is used as a fixed feature extractor — the 128 numbers its final
dense layer reads — and the Shuffled Frog Leaping Algorithm searches for the
subset of those features, and the SVM hyper-parameters, that maximise
*validation* macro-F1.

The geometry is taken from the classifier's own checkpoint, so the features are
the ones that network actually produces at serving time: whole slices for a
legacy checkpoint, U-Net ROI crops for a ``m1-v2-roi`` one.

Two heads therefore exist: the ConvLSTM's softmax layer and this SVM. **The one
the API serves is chosen on validation macro-F1 before the test set is opened.**
The test split is then evaluated once for both, and the losing head's numbers
are kept beside the served ones rather than discarded.

Nothing about the ConvLSTM checkpoint is modified: the SVM is a separate
artefact, used only while its recorded classifier version matches the
checkpoint the engine loaded.
"""
from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend import config
from backend.methods.common.checkpoint import load_checkpoint, peek_meta
from backend.methods.common.cnn_training import metrics_section, print_section, sync
from backend.methods.common.feature_selection import (
    SVM_PARAMETERS, feature_space, fitted_pipeline, make_objective, mask_of,
)
from backend.methods.common.metrics_store import update_metrics
from backend.methods.common.runcard import RunCard, write_run_card
from backend.methods.method1 import config as m1
from backend.methods.method1.datasets import DatasetError, class_distribution
from backend.methods.method1.models import ConvLSTMClassifier
from backend.methods.method1.optimization.sfla import SFLA
from backend.methods.method1.training.train_classifier import prepare_data
from backend.methods.registry import METHOD1
from backend.training.common import seed_everything
from backend.utils.metrics import save_confusion_matrix, save_curve, save_roc_curves

TAG = "m1-svm"
CLASS_NAMES = list(METHOD1.class_names)
LOGS_DIR = config.LOGS_DIR / "method1"
ARCHITECTURE = "ConvLSTM features + SFLA feature selection + SVM"
SVM_PATH = m1.CLS_WEIGHTS_PATH.with_name("feature_svm.joblib")
RESULT_PATH = LOGS_DIR / "sfla_feature_selection.json"


def load_classifier(device) -> tuple[ConvLSTMClassifier, dict, str]:
    """The trained ConvLSTM and the transform id its checkpoint was trained with."""
    peek = peek_meta(m1.CLS_WEIGHTS_PATH) or {}
    hp = dict((peek.get("extra") or {}).get("hyperparameters") or {})
    model = ConvLSTMClassifier(
        in_channels=1, num_classes=m1.NUM_CLASSES,
        base=int(hp.get("base", 32)), lstm_hidden=int(hp.get("lstm_hidden", 64)),
        lstm_steps=int(hp.get("lstm_steps", 3)), dropout=float(hp.get("dropout", 0.4)),
    ).to(device)
    meta, warns = load_checkpoint(
        model, m1.CLS_WEIGHTS_PATH, expected_method=m1.METHOD_ID,
        expected_architecture=m1.CLS_ARCHITECTURE, expected_role="classification",
        expected_class_names=CLASS_NAMES,
    )
    for w in warns:
        print(f"[{TAG}] {w}")
    model.eval()
    return model, meta, str(meta.get("transform_id") or "")


@torch.no_grad()
def extract(model: ConvLSTMClassifier, dataset, device, batch_size: int = 64):
    """``(features, logits, labels)`` for a dataset, in dataset order.

    The logits are the final dense layer applied to the same features, so the
    network's own softmax head is evaluated on exactly the vector the SVM sees —
    no second forward pass, and no chance of the two disagreeing about input.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, logits, labels = [], [], []
    t0 = time.perf_counter()
    for x, y in loader:
        f = model.penultimate(x.to(device)).float()
        feats.append(f.cpu().numpy())
        logits.append(model.head[-1](f).cpu().numpy())
        labels.append(y.numpy())
    sync(device)
    print(f"[{TAG}] {len(dataset)} images -> features in {time.perf_counter() - t0:.0f}s", flush=True)
    return np.concatenate(feats), np.concatenate(logits), np.concatenate(labels)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def probs_from_svm(svm, x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    t0 = time.perf_counter()
    probs = svm.predict_proba(x[:, mask])
    total = time.perf_counter() - t0
    return probs.astype(np.float64), total / max(1, len(x)), total


def section_for(data, test: dict, *, head: str, architecture: str, model_version: str,
                warnings: list[str], extra: dict) -> dict:
    return {
        "split": data.split_kind, "split_level": data.ds_meta.get("split_level"),
        "dataset": str(data.ds_meta["root"]), **test,
        "per_class": {k: [test["per_class"][c][k] for c in CLASS_NAMES]
                      for k in ("precision", "recall", "f1", "specificity")},
        "per_class_detail": test["per_class"], "test_size": len(data.test),
        "head": head, "architecture": architecture, "transform": data.spec.to_dict(),
        "model_version": model_version, "warnings": warnings, **extra,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="SFLA feature selection + SVM on ConvLSTM features")
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--memeplexes", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--local-iterations", type=int, default=6)
    ap.add_argument("--max-step", type=float, default=0.5)
    ap.add_argument("--sparsity-penalty", type=float, default=0.02,
                    help="fitness penalty for the fraction of features kept")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ignore-official-split", action="store_true")
    ap.add_argument("--skip-test", action="store_true", help="never load the test set")
    args = ap.parse_args()

    try:
        import joblib  # noqa: F401
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("ERROR: scikit-learn and joblib are required: pip install scikit-learn joblib")
        return 1

    if not m1.CLS_WEIGHTS_PATH.exists():
        print(f"ERROR: {m1.CLS_WEIGHTS_PATH} not found. Train the classifier first:\n"
              f"  python -m backend.methods.method1.training.train_classifier")
        return 1

    device = config.DEVICE
    started = time.perf_counter()
    seed_everything(args.seed)

    model, ckpt_meta, transform_id = load_classifier(device)
    cls_version = str(ckpt_meta.get("model_version") or "unknown")
    print(f"[{TAG}] feature extractor: ConvLSTM {cls_version} (transform {transform_id or 'legacy'})")

    # The geometry comes from the checkpoint, not from a flag: features must be the
    # ones this network is actually fed at serving time.
    prep_args = SimpleNamespace(
        transform="v2" if transform_id == "m1-v2-roi" else "v1",
        data_root=args.data_root, val_frac=args.val_frac, test_frac=args.test_frac,
        ignore_official_split=args.ignore_official_split,
    )
    try:
        data = prepare_data(prep_args, load_test=not args.skip_test, device=device)
    except DatasetError as exc:
        print(f"\nERROR: {exc}")
        return 1
    if data.spec.id != (transform_id or data.spec.id):
        print(f"ERROR: the checkpoint was trained with {transform_id} but this machine can only "
              f"reproduce {data.spec.id} (is the U-Net checkpoint present?). Refusing to select "
              f"features on a geometry the network was not trained on.")
        return 1
    warnings = list(data.warnings)

    parts = ("train", "val") + (() if args.skip_test else ("test",))
    features, logits, labels = {}, {}, {}
    for part in parts:
        features[part], logits[part], labels[part] = extract(model, data.dataset(part), device)

    scaler = StandardScaler().fit(features["train"])
    live = scaler.var_ > 1e-12
    n_live = int(live.sum())
    print(f"[{TAG}] {n_live} of {features['train'].shape[1]} features vary on the training split; "
          f"{features['train'].shape[1] - n_live} are constant and are dropped before the search")
    scaled = {p: scaler.transform(features[p])[:, live] for p in parts}

    # --- SFLA feature selection: train fits, validation scores. ----------- #
    objective = make_objective(scaled["train"], labels["train"], scaled["val"], labels["val"],
                               sparsity_penalty=args.sparsity_penalty, seed=args.seed)
    space = feature_space(n_live, SVM_PARAMETERS)
    sfla = SFLA(space, objective, population=args.population, memeplexes=args.memeplexes,
                iterations=args.iterations, local_iterations=args.local_iterations,
                max_step=args.max_step, seed=args.seed,
                objective_name=("validation macro-F1 − "
                                f"{args.sparsity_penalty}·(features kept / features available)"),
                progress=lambda h: print(
                    f"[{TAG}] iteration {h['iteration']:03d}/{args.iterations} | "
                    f"best {h['best_fitness']:.4f} mean {h['iteration_mean']:.4f} | "
                    f"{h['evaluations']} evaluations", flush=True))
    print(f"[{TAG}] searching {len(space)} dimensions ({n_live} features + "
          f"{len(SVM_PARAMETERS)} SVM hyper-parameters) with {sfla.population} frogs in "
          f"{args.memeplexes} memeplexes × {args.iterations} iterations", flush=True)
    result = sfla.run()
    mask = mask_of(result.best_parameters, n_live)
    svm_hp = {k: result.best_parameters[k] for k in ("C", "gamma", "kernel")}
    kept = int(mask.sum())
    print(f"[{TAG}] selected {kept}/{n_live} features | SVM {svm_hp} | "
          f"fitness {result.best_fitness:.4f} after {result.evaluations} evaluations "
          f"in {result.duration_s:.0f}s")

    svm = fitted_pipeline(result.best_parameters, mask, scaled["train"], labels["train"],
                          seed=args.seed, probability=True)
    val_probs, val_avg, val_total = probs_from_svm(svm, scaled["val"], mask)
    svm_val = metrics_section(labels["val"], val_probs, CLASS_NAMES, val_avg, val_total)
    print_section(TAG, "VALIDATION — SFLA-selected features + SVM", svm_val)
    cls_val = metrics_section(labels["val"], softmax(logits["val"]), CLASS_NAMES, 0.0, 0.0)
    print_section(TAG, "VALIDATION — ConvLSTM softmax (for comparison)", cls_val)

    # --- Head choice, made on validation only. ---------------------------- #
    serve_svm = svm_val["f1"] > cls_val["f1"]
    chosen = "sfla_svm" if serve_svm else "convlstm_softmax"
    print(f"[{TAG}] validation macro-F1: SVM {svm_val['f1']:.4f} vs ConvLSTM softmax "
          f"{cls_val['f1']:.4f} -> serving the {chosen} head")

    warnings.append(
        "Two classifier heads exist for this method: the ConvLSTM's softmax layer and an SVM on "
        f"SFLA-selected ConvLSTM features. The {chosen} head is served; the choice was made on "
        f"validation macro-F1 ({svm_val['f1']:.4f} SVM vs {cls_val['f1']:.4f} softmax) before the "
        "test set was opened."
    )
    if serve_svm:
        warnings.append(
            "The SVM's probabilities come from Platt scaling cross-fitted on the training split; "
            "they rank classes well but are not calibrated risk estimates."
        )
        warnings.append(
            "Grad-CAM explains the ConvLSTM that produced the features, not the SVM's decision "
            "boundary."
        )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    save_curve([h["best_fitness"] for h in result.history],
               "SFLA feature selection — fitness per iteration",
               LOGS_DIR / "sfla_feature_selection.png",
               second=[h["iteration_mean"] for h in result.history],
               labels=("best", "population mean"))

    kept_columns = np.flatnonzero(live)[mask]
    report = {
        **result.to_dict(),
        "stage": "feature_selection",
        "feature_extractor": {"architecture": m1.CLS_ARCHITECTURE,
                              "layer": "penultimate dense layer",
                              "model_version": cls_version, "transform_id": data.spec.id,
                              "features_per_image": int(features["train"].shape[1])},
        "features_available_to_search": n_live,
        "features_selected": kept,
        "selected_feature_indices": kept_columns.tolist(),
        "svm": {**svm_hp, "class_weight": "balanced", "probability": "Platt scaling"},
        "sparsity_penalty": args.sparsity_penalty,
        "validation": {"sfla_svm": svm_val, "convlstm_softmax": cls_val},
        "served_head": chosen,
        "head_choice": "higher validation macro-F1; decided before the test set was loaded",
        "split": data.split_kind,
        "notes": [
            "Fitness is validation macro-F1 of an SVM fitted on the training split only.",
            "The test split was not loaded during the search.",
            "Features are the ConvLSTM's own penultimate activations on the geometry recorded in "
            "its checkpoint, so they are the vector it produces when serving.",
        ],
    }
    # One evaluation log entry per candidate would carry a full keep/drop dictionary
    # each; only the scores are kept, and the winning subset is the index list above.
    report["evaluation_log"] = [{k: v for k, v in e.items() if k != "parameters"}
                               for e in result.evaluation_log]
    report["best_parameters"] = {**svm_hp, "features_selected": kept}
    report["best_position"] = "see selected_feature_indices"
    RESULT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"[{TAG}] search result -> {RESULT_PATH}")

    bundle = {
        "method_id": METHOD1.method_id, "role": "classification", "architecture": ARCHITECTURE,
        "class_names": CLASS_NAMES, "convlstm_model_version": cls_version,
        "transform_id": data.spec.id, "features_per_image": int(features["train"].shape[1]),
        "scaler": scaler, "kept_columns": kept_columns, "svm": svm,
        "svm_hyperparameters": svm_hp, "features_selected": kept, "features_available": n_live,
        "validation_macro_f1": svm_val["f1"], "softmax_validation_macro_f1": cls_val["f1"],
        "serve": bool(serve_svm), "model_version": time.strftime("%Y%m%d-%H%M%S"),
        "search_result": str(RESULT_PATH), "warnings": warnings,
    }

    if args.skip_test:
        out = LOGS_DIR / "feature_svm_validation_metrics.json"
        out.write_text(json.dumps({k: v for k, v in bundle.items()
                                   if k not in ("scaler", "svm", "kept_columns")},
                                  indent=2, default=str))
        print(f"[{TAG}] validation-only run; test set not loaded. report -> {out}")
        return 0

    # --- The held-out test set is opened here, once, for both heads. ------ #
    svm_probs, svm_avg, svm_total = probs_from_svm(svm, scaled["test"], mask)
    svm_test = metrics_section(labels["test"], svm_probs, CLASS_NAMES, svm_avg, svm_total)
    print_section(TAG, "FINAL TEST — SFLA-selected features + SVM (evaluated once)", svm_test)
    cls_probs = softmax(logits["test"])
    cls_test = metrics_section(labels["test"], cls_probs, CLASS_NAMES, 0.0, 0.0)
    print_section(TAG, "FINAL TEST — ConvLSTM softmax (evaluated once)", cls_test)

    served_test, other_test = (svm_test, cls_test) if serve_svm else (cls_test, svm_test)
    served_probs = svm_probs if serve_svm else cls_probs
    served_arch = ARCHITECTURE if serve_svm else "U-Net → ROI crop → ConvLSTM → softmax"
    model_version = bundle["model_version"] if serve_svm else cls_version

    display = [METHOD1.class_labels[c] for c in CLASS_NAMES]
    save_confusion_matrix(served_test["confusion_matrix"], display,
                          config.LOGS_DIR / "method1_cls_confusion_matrix.png")
    save_roc_curves(labels["test"], served_probs, display,
                    config.LOGS_DIR / "method1_cls_roc.png")

    extra = {"features_selected": kept, "features_available": n_live, "svm": svm_hp,
             "feature_extractor_version": cls_version} if serve_svm else {}
    section = section_for(data, served_test, head=chosen, architecture=served_arch,
                          model_version=model_version, warnings=warnings, extra=extra)
    update_metrics(METHOD1.method_id, "classification", section)
    update_metrics(METHOD1.method_id, "classification_other_head", section_for(
        data, other_test, head="convlstm_softmax" if serve_svm else "sfla_svm",
        architecture="U-Net → ROI crop → ConvLSTM → softmax" if serve_svm else ARCHITECTURE,
        model_version=cls_version if serve_svm else bundle["model_version"],
        warnings=["Recorded for comparison; this head is not the one the API serves."],
        extra={},
    ))

    bundle["test_metrics"] = svm_test
    bundle["softmax_test_metrics"] = cls_test
    import joblib

    joblib.dump(bundle, SVM_PATH, compress=3)
    print(f"[{TAG}] SVM bundle -> {SVM_PATH} ({SVM_PATH.stat().st_size:,} bytes)")

    report["test"] = {"sfla_svm": svm_test, "convlstm_softmax": cls_test}
    RESULT_PATH.write_text(json.dumps(report, indent=2, default=str))

    meta_path = m1.CLS_WEIGHTS_PATH.parent / "feature_svm_metadata.json"
    meta_path.write_text(json.dumps({
        "method_id": METHOD1.method_id, "method_name": METHOD1.display_name,
        "architecture": ARCHITECTURE, "class_order": CLASS_NAMES,
        "dataset_path": str(data.ds_meta["root"]),
        "dataset_counts": {"train": class_distribution(data.train),
                           "validation": class_distribution(data.val),
                           "test": class_distribution(data.test),
                           "train_copies_of_test_images_dropped": len(data.removed_test_duplicates)},
        "split_strategy": data.split_kind, "preprocessing": data.spec.to_dict(),
        "feature_extractor": report["feature_extractor"],
        "feature_selection": {"algorithm": "Shuffled Frog Leaping Algorithm",
                              "population": sfla.population, "memeplexes": args.memeplexes,
                              "iterations": args.iterations,
                              "local_iterations": args.local_iterations,
                              "evaluations": result.evaluations,
                              "features_available": n_live, "features_selected": kept,
                              "duration_s": round(result.duration_s, 1)},
        "svm": svm_hp, "served_head": chosen,
        "validation_metrics": {"sfla_svm": svm_val, "convlstm_softmax": cls_val},
        "test_metrics": {"sfla_svm": svm_test, "convlstm_softmax": cls_test},
        "random_seed": args.seed, "device": str(device),
        "model_version": bundle["model_version"], "artefact_path": str(SVM_PATH),
        "total_time_s": round(time.perf_counter() - started, 1), "warnings": warnings,
    }, indent=2, default=str))

    write_run_card(
        config.LOGS_DIR / METHOD1.run_card_filenames["feature_selection"],
        RunCard(method_id=METHOD1.method_id, stage="feature_selection",
                dataset={**data.ds_meta, "name": "Brain Tumor MRI Dataset (BRI)"},
                split_strategy={"type": data.split_kind, **data.summary},
                random_seed=args.seed, preprocessing=data.spec.to_dict(),
                image_size=data.spec.image_size, architecture=ARCHITECTURE,
                model_config={"features_available": n_live, "features_selected": kept, **svm_hp},
                optimizer={"name": "SFLA (feature selection)", "population": sfla.population,
                           "memeplexes": args.memeplexes, "iterations": args.iterations,
                           "local_iterations": args.local_iterations,
                           "evaluations": result.evaluations, "objective": sfla.objective_name},
                epochs=args.iterations, best_epoch=args.iterations, metrics=section,
                checkpoint_path=str(SVM_PATH),
                inference_time_s=served_test["avg_inference_time_s"],
                train_duration_s=round(time.perf_counter() - started, 1),
                model_version=model_version, device=str(device), warnings=warnings),
        root=config.ROOT_DIR,
    )
    print(f"[{TAG}] metadata -> {meta_path}")
    print(f"[{TAG}] done. Served head: {chosen}. Held-out test accuracy: "
          f"{served_test['accuracy']:.4f} (SVM {svm_test['accuracy']:.4f}, "
          f"ConvLSTM softmax {cls_test['accuracy']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
