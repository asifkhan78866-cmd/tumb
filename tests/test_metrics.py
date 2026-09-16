"""9. Metric calculations."""
from __future__ import annotations

import numpy as np
import pytest

from backend.utils.metrics import _roc, classification_metrics, macro_auc


def test_perfect_prediction_scores_one():
    y = [0, 1, 2, 3, 0, 1]
    m = classification_metrics(y, y, 4)
    for key in ("accuracy", "precision", "recall", "f1", "specificity"):
        assert m[key] == pytest.approx(1.0)


def test_confusion_matrix_is_true_by_predicted():
    m = classification_metrics([0, 0, 1], [0, 1, 1], 2)
    assert m["confusion_matrix"] == [[1, 1], [0, 1]]
    assert m["accuracy"] == pytest.approx(2 / 3)


def test_hand_computed_binary_case():
    # TP=2 (class1), FP=1, FN=1, TN=2
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0]
    m = classification_metrics(y_true, y_pred, 2)
    per = m["per_class"]
    assert per["precision"][1] == pytest.approx(2 / 3)
    assert per["recall"][1] == pytest.approx(2 / 3)
    assert per["specificity"][1] == pytest.approx(2 / 3)
    assert per["f1"][1] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(4 / 6)


def test_sensitivity_equals_macro_recall():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, 200).tolist()
    y_pred = rng.integers(0, 4, 200).tolist()
    m = classification_metrics(y_true, y_pred, 4)
    assert m["sensitivity"] == pytest.approx(m["recall"])


def test_absent_class_contributes_zero_not_nan():
    m = classification_metrics([0, 0, 0], [0, 0, 0], 4)
    assert np.isfinite(m["precision"]) and np.isfinite(m["f1"])
    assert m["per_class"]["recall"][1] == 0.0


def test_metrics_never_exceed_one():
    rng = np.random.default_rng(5)
    for _ in range(20):
        y_true = rng.integers(0, 4, 60).tolist()
        y_pred = rng.integers(0, 4, 60).tolist()
        m = classification_metrics(y_true, y_pred, 4)
        for key in ("accuracy", "precision", "recall", "f1", "specificity"):
            assert 0.0 <= m[key] <= 1.0


def test_roc_auc_of_a_perfect_separator_is_one():
    binary = np.array([0, 0, 1, 1])
    _, _, auc = _roc(binary, np.array([0.1, 0.2, 0.8, 0.9]))
    assert auc == pytest.approx(1.0)


def test_roc_auc_of_a_reversed_separator_is_zero():
    binary = np.array([0, 0, 1, 1])
    _, _, auc = _roc(binary, np.array([0.9, 0.8, 0.2, 0.1]))
    assert auc == pytest.approx(0.0, abs=1e-9)


def test_macro_auc_is_none_when_undefined():
    """An AUC that cannot be computed must not reach a report as a number."""
    y_true = np.array([0, 0, 0])
    y_score = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
    assert macro_auc(y_true, y_score, 4) is None


def test_macro_auc_perfect_case():
    y_true = np.array([0, 1, 2, 3])
    y_score = np.eye(4) * 0.9 + 0.025
    assert macro_auc(y_true, y_score, 4) == pytest.approx(1.0)


def test_dice_and_iou_agree_on_a_known_overlap():
    torch = pytest.importorskip("torch")
    from backend.utils.losses import dice_coefficient, iou_score

    # prediction and target overlap on half of a 4-pixel target
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, :2, :] = 1.0          # 8 positive pixels
    logits = torch.full((1, 1, 4, 4), -10.0)
    logits[0, 0, :1, :] = 10.0         # 4 predicted, all correct
    # Dice = 2*4/(4+8) = 0.667, IoU = 4/8 = 0.5 (smoothing shifts both slightly)
    assert dice_coefficient(logits, target) == pytest.approx(0.667, abs=0.05)
    assert iou_score(logits, target) == pytest.approx(0.5, abs=0.06)


def test_metrics_store_merges_instead_of_clobbering(tmp_path, monkeypatch):
    """Evaluating one stage must not wipe another stage's numbers."""
    from backend import config
    from backend.methods.common import metrics_store

    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    metrics_store.update_metrics("method1", "classification", {"accuracy": 0.84})
    metrics_store.update_metrics("method1", "segmentation", {"dice": 0.71})

    merged = metrics_store.metrics_for("method1")
    assert merged["accuracy"] == 0.84, "writing segmentation erased classification"
    assert merged["dice"] == 0.71
    assert merged["evaluated"] is True


def test_unevaluated_metrics_are_none_not_zero(tmp_path, monkeypatch):
    from backend import config
    from backend.methods.common import metrics_store

    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    m = metrics_store.metrics_for("method2")
    assert m["evaluated"] is False
    for key in ("dice", "iou", "accuracy", "precision", "recall", "f1", "auc"):
        assert m[key] is None, f"{key} must be null when not evaluated, not 0.0"
    assert m["warnings"]
