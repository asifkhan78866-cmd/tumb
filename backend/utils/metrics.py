"""Classification metrics and plotting helpers.

Computes accuracy, precision, recall, F1, specificity, sensitivity, the confusion
matrix and ROC curves, and saves the plots to disk.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def classification_metrics(y_true, y_pred, num_classes: int) -> dict:
    """Return a dict of macro-averaged metrics plus the confusion matrix.

    Sensitivity == recall (macro). Specificity is macro-averaged one-vs-rest.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    precisions, recalls, f1s, specificities = [], [], [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        spec = _safe_div(tn, tn + fp)
        f1 = _safe_div(2 * prec * rec, prec + rec)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        specificities.append(spec)

    accuracy = _safe_div(np.trace(cm), cm.sum())
    return {
        "accuracy": accuracy,
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "sensitivity": float(np.mean(recalls)),
        "specificity": float(np.mean(specificities)),
        "f1": float(np.mean(f1s)),
        "per_class": {
            "precision": precisions,
            "recall": recalls,
            "f1": f1s,
            "specificity": specificities,
        },
        "confusion_matrix": cm.tolist(),
    }


def macro_auc(y_true, y_score, num_classes: int):
    """Macro-averaged one-vs-rest ROC AUC, or ``None`` if it is undefined.

    Returns ``None`` rather than a placeholder when no class has both positive
    and negative examples — an AUC that could not be computed must not reach a
    report as a number.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_score.ndim != 2 or y_score.shape[1] != num_classes or len(y_true) == 0:
        return None
    aucs = []
    for c in range(num_classes):
        binary = (y_true == c).astype(int)
        if binary.sum() in (0, len(binary)):
            continue  # undefined for this class
        _, _, auc = _roc(binary, y_score[:, c])
        aucs.append(auc)
    return float(np.mean(aucs)) if aucs else None


def _pyplot():
    """Return pyplot, or None when matplotlib is unavailable.

    Plots are diagnostics. A training run that has produced a checkpoint must
    never be lost because a plotting library is missing, so every plotting
    helper degrades to a warning instead of raising.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[metrics] skipping plot ({type(exc).__name__}: {exc}); install matplotlib to enable plots.")
        return None


def save_confusion_matrix(cm, class_names, out_path: Path):
    plt = _pyplot()
    if plt is None:
        return
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_roc_curves(y_true, y_score, class_names, out_path: Path):
    """One-vs-rest ROC curves. ``y_score`` is (N, num_classes) softmax probs."""
    plt = _pyplot()
    if plt is None:
        return
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(n):
        binary = (y_true == c).astype(int)
        fpr, tpr, auc = _roc(binary, y_score[:, c])
        ax.plot(fpr, tpr, label=f"{class_names[c]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (one-vs-rest)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _roc(binary, scores):
    """Compute ROC curve + AUC without sklearn (trapezoidal AUC)."""
    order = np.argsort(-scores)
    binary = binary[order]
    P = binary.sum()
    N = len(binary) - P
    if P == 0 or N == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.5
    tps = np.cumsum(binary)
    fps = np.cumsum(1 - binary)
    tpr = np.concatenate([[0], tps / P])
    fpr = np.concatenate([[0], fps / N])
    # np.trapz was removed in NumPy 2.0 in favour of np.trapezoid.
    _trap = getattr(np, "trapezoid", None) or np.trapz
    auc = _trap(tpr, fpr)
    return fpr, tpr, auc


def save_curve(values, ylabel: str, out_path: Path, second=None, labels=("train", "val")):
    plt = _pyplot()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(values) + 1), values, label=labels[0])
    if second is not None:
        ax.plot(range(1, len(second) + 1), second, label=labels[1])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " per epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
