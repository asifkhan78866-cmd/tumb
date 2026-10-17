"""SFLA feature selection for Method 1: the feature vector is the one the softmax
layer reads, and the SVM head is only served when the artefact matches."""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import requires_torch

sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")

from backend.methods.common.feature_selection import (  # noqa: E402
    feature_space, fitted_pipeline, make_objective, mask_of,
)


def _toy(n_per_class: int = 40, n_features: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1, 2, 3], n_per_class)
    rng.shuffle(y)
    x = rng.normal(0, 1.0, (len(y), n_features))
    x[:, 0] += y * 3.0
    x[:, 1] -= y * 2.0
    return x, y


# --- the feature vector ----------------------------------------------------- #
@requires_torch
def test_the_selected_features_are_what_the_softmax_layer_reads():
    """The SVM must classify the same vector the dense head classifies; if these
    diverged, the selection would be made on features the network never serves."""
    import torch

    from backend.methods.method1.models import ConvLSTMClassifier

    torch.manual_seed(0)
    model = ConvLSTMClassifier(in_channels=1, num_classes=4, base=8, lstm_hidden=16,
                               lstm_steps=2, dropout=0.0).eval()
    x = torch.randn(3, 1, 64, 64)
    with torch.no_grad():
        features = model.penultimate(x)
        logits_from_features = model.head[-1](features)
        logits = model(x)
    assert features.shape == (3, 128)
    assert torch.allclose(logits, logits_from_features, atol=1e-6)


# --- the search ------------------------------------------------------------- #
def test_sfla_searches_the_feature_space_and_improves_its_objective():
    from backend.methods.method1.optimization.sfla import SFLA

    x, y = _toy(seed=5)
    obj = make_objective(x[:120], y[:120], x[120:], y[120:])
    space = feature_space(x.shape[1])
    seen: list[dict] = []
    result = SFLA(space, obj, population=12, memeplexes=3, iterations=4, local_iterations=3,
                  seed=7, progress=seen.append).run()
    assert len(seen) == 4, "the progress callback reports once per iteration"
    assert result.best_fitness >= result.history[0]["iteration_mean"]
    assert 1 <= mask_of(result.best_parameters, x.shape[1]).sum() <= x.shape[1]


def test_the_search_never_receives_the_test_matrices():
    x, y = _toy()
    obj = make_objective(x[:120], y[:120], x[120:], y[120:])
    assert not {"x_test", "y_test", "test"} & set(vars(obj))


# --- serving ---------------------------------------------------------------- #
def _bundle(tmp_path, **overrides):
    import joblib
    from sklearn.preprocessing import StandardScaler

    x, y = _toy(n_features=6)
    columns = np.array([0, 1])
    svm = fitted_pipeline({"C": 1.0, "gamma": 0.1, "kernel": "rbf"},
                          np.isin(np.arange(6), columns), x, y)
    bundle = {
        "method_id": "method1", "role": "classification",
        "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
        "convlstm_model_version": "test-version", "scaler": StandardScaler().fit(x),
        "kept_columns": columns, "svm": svm, "features_available": 6, "features_selected": 2,
        "svm_hyperparameters": {"C": 1.0, "gamma": 0.1, "kernel": "rbf"},
        "serve": True, "model_version": "svm-version",
    }
    bundle.update(overrides)
    path = tmp_path / "feature_svm.joblib"
    joblib.dump(bundle, path)
    return path


def _engine(monkeypatch, path, cls_version="test-version", loaded=True):
    pytest.importorskip("torch")
    from backend.methods.method1 import inference as m1i

    monkeypatch.setattr(m1i, "SVM_PATH", path)
    engine = m1i.Method1Engine()
    engine.cls_weights_loaded = loaded
    engine.cls_meta = {"model_version": cls_version}
    engine._load_feature_svm()
    return engine


@requires_torch
def test_the_svm_head_is_served_when_the_artefact_matches(tmp_path, monkeypatch):
    engine = _engine(monkeypatch, _bundle(tmp_path))
    assert engine.served_head == "sfla_svm"
    assert engine.model_version == "svm-version"
    assert engine.load_warnings == []
    details = engine._feature_selection_details()
    assert details["feature_selection"]["features_selected"] == 2
    assert "not the SVM's decision boundary" in details["gradcam_note"]


@requires_torch
def test_an_svm_from_another_classifier_version_is_refused_and_said_so(tmp_path, monkeypatch):
    engine = _engine(monkeypatch, _bundle(tmp_path), cls_version="some-other-convlstm")
    assert engine.served_head == "convlstm_softmax"
    assert any("was built from convlstm" in w for w in engine.load_warnings)
    assert engine._feature_selection_details() == {}


@requires_torch
def test_an_svm_belonging_to_another_method_is_refused(tmp_path, monkeypatch):
    engine = _engine(monkeypatch, _bundle(tmp_path, method_id="method4"))
    assert engine.served_head == "convlstm_softmax"
    assert any("belongs to method4" in w for w in engine.load_warnings)


@requires_torch
def test_an_svm_that_lost_on_validation_is_not_served(tmp_path, monkeypatch):
    engine = _engine(monkeypatch, _bundle(tmp_path, serve=False))
    assert engine.served_head == "convlstm_softmax" and engine.load_warnings == []


@requires_torch
def test_no_svm_is_loaded_while_the_classifier_itself_is_untrained(tmp_path, monkeypatch):
    engine = _engine(monkeypatch, _bundle(tmp_path), loaded=False)
    assert engine.served_head == "convlstm_softmax" and engine.load_warnings == []
