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


def save_confusion_matrix(cm, class_names, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
