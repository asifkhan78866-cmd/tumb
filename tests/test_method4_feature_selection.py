"""Red Fox feature selection + SVM: the search sees validation only, and the
served head is the one the artefact says won — never an unverified artefact."""
from __future__ import annotations

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")

from backend.methods.common.feature_selection import (  # noqa: E402
    SVM_PARAMETERS, build_svm, feature_space, fitted_pipeline, make_objective, mask_of,
)


def _toy(n_per_class: int = 40, n_features: int = 12, seed: int = 0):
    """Two informative features, the rest noise, so a *selection* is detectable."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1, 2], n_per_class)
    rng.shuffle(y)  # so any contiguous slice carries all three classes
    x = rng.normal(0, 1.0, (len(y), n_features))
    x[:, 0] += y * 4.0
    x[:, 1] -= y * 3.0
    return x, y


# --- the search space ------------------------------------------------------- #
def test_space_has_one_coordinate_per_feature_plus_the_svm_hyperparameters():
    space = feature_space(50)
    assert len(space) == 50 + len(SVM_PARAMETERS)
    assert [p.name for p in space.parameters[-3:]] == ["C", "gamma", "kernel"]


def test_a_coordinate_in_the_upper_half_of_the_cube_keeps_its_feature():
    space = feature_space(4)
    params = space.decode([0.9, 0.1, 0.51, 0.49, 0.5, 0.5, 0.5])
    assert mask_of(params, 4).tolist() == [True, False, True, False]


def test_every_feature_can_be_addressed_however_many_there_are():
    # Names are zero-padded, so 1000 features must not collide or misalign.
    space = feature_space(1000)
    position = [1.0] * 1000 + [0.5] * len(SVM_PARAMETERS)
    assert mask_of(space.decode(position), 1000).all()
    position[137] = 0.0
    mask = mask_of(space.decode(position), 1000)
    assert not mask[137] and mask.sum() == 999


def test_an_svm_is_built_from_the_decoded_position():
    svm = build_svm({"C": 3.0, "gamma": 0.01, "kernel": "rbf"})
    assert (svm.C, svm.gamma, svm.kernel, svm.class_weight) == (3.0, 0.01, "rbf", "balanced")


# --- the objective ---------------------------------------------------------- #
def test_fitness_is_validation_score_minus_a_penalty_for_features_kept():
    x, y = _toy()
    obj = make_objective(x[:90], y[:90], x[90:], y[90:], sparsity_penalty=0.1, min_features=2)
    space = feature_space(x.shape[1])
    hp = [0.5] * len(SVM_PARAMETERS)

    informative = obj(space.decode([1.0, 1.0] + [0.0] * 10 + hp))
    everything = obj(space.decode([1.0] * 12 + hp))
    # Both classify this toy problem; the penalty must prefer the smaller subset.
    assert informative > everything


def test_a_subset_smaller_than_min_features_scores_zero_instead_of_raising():
    x, y = _toy()
    obj = make_objective(x[:90], y[:90], x[90:], y[90:], min_features=4)
    space = feature_space(x.shape[1])
    assert obj(space.decode([1.0] + [0.0] * 11 + [0.5] * len(SVM_PARAMETERS))) == 0.0


def test_the_objective_is_only_ever_given_two_matrices():
    """Structural: there is no third matrix it could score on, so the test set
    cannot reach the search even by mistake."""
    x, y = _toy()
    obj = make_objective(x[:90], y[:90], x[90:], y[90:])
    held_out = {"x_test", "y_test", "test"}
    assert not held_out & set(vars(obj))


def test_red_fox_improves_the_objective_it_is_given():
    from backend.methods.method4.optimization.rfo import RedFoxOptimizer

    x, y = _toy(seed=3)
    obj = make_objective(x[:90], y[:90], x[90:], y[90:])
    space = feature_space(x.shape[1])
    result = RedFoxOptimizer(space, obj, population=8, iterations=4, seed=11).run()
    assert result.evaluations > 0
    assert result.best_fitness >= result.history[0]["mean_fitness"]
    assert 1 <= mask_of(result.best_parameters, x.shape[1]).sum() <= x.shape[1]


def test_the_final_classifier_reports_probabilities_over_the_selected_columns():
    x, y = _toy()
    mask = np.zeros(x.shape[1], dtype=bool)
    mask[[0, 1]] = True
    svm = fitted_pipeline({"C": 1.0, "gamma": 0.1, "kernel": "rbf"}, mask, x[:90], y[:90])
    probs = svm.predict_proba(x[90:][:, mask])
    assert probs.shape == (len(y) - 90, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


# --- serving ---------------------------------------------------------------- #
def _bundle(tmp_path, **overrides):
    """A minimal artefact of the shape the trainer writes."""
    import joblib
    from sklearn.preprocessing import StandardScaler

    x, y = _toy(n_features=6)
    scaler = StandardScaler().fit(x)
    mask = np.array([0, 1])
    svm = fitted_pipeline({"C": 1.0, "gamma": 0.1, "kernel": "rbf"},
                          np.isin(np.arange(6), mask), x, y)
    bundle = {
        "method_id": "method4", "role": "classification",
        "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
        "zfnet_model_version": "test-version", "scaler": scaler, "kept_columns": mask,
        "svm": svm, "svm_hyperparameters": {"C": 1.0, "gamma": 0.1, "kernel": "rbf"},
        "features_selected": 2, "features_available": 6, "serve": True,
        "model_version": "svm-version",
    }
    bundle.update(overrides)
    path = tmp_path / "rfo_svm.joblib"
    joblib.dump(bundle, path)
    return path


def _engine(monkeypatch, path, zfnet_version="test-version"):
    pytest.importorskip("torch")
    from backend.methods.method4 import inference as m4

    monkeypatch.setattr(m4, "SVM_PATH", path)
    engine = m4.Method4Engine()
    engine.meta = {"model_version": zfnet_version}
    engine.load_extras()
    return engine


def test_the_svm_head_is_used_when_the_artefact_matches_the_checkpoint(tmp_path, monkeypatch):
    engine = _engine(monkeypatch, _bundle(tmp_path))
    assert engine.served_head == "rfo_svm"
    assert engine.model_version == "svm-version"
    assert engine.load_warnings == []


def test_an_svm_built_from_another_checkpoint_is_refused_and_said_so(tmp_path, monkeypatch):
    engine = _engine(monkeypatch, _bundle(tmp_path), zfnet_version="a-different-zfnet")
    assert engine.served_head == "zfnet_softmax"
    assert any("was built from zfnet" in w for w in engine.load_warnings)


def test_an_svm_that_lost_on_validation_is_not_served(tmp_path, monkeypatch):
    engine = _engine(monkeypatch, _bundle(tmp_path, serve=False))
    assert engine.served_head == "zfnet_softmax" and engine.load_warnings == []


def test_an_svm_for_other_classes_is_refused(tmp_path, monkeypatch):
    engine = _engine(monkeypatch, _bundle(tmp_path, class_names=["tumour", "healthy"]))
    assert engine.served_head == "zfnet_softmax"
    assert any("different classes" in w for w in engine.load_warnings)


def test_a_corrupt_artefact_warns_instead_of_breaking_the_method(tmp_path, monkeypatch):
    path = tmp_path / "rfo_svm.joblib"
    path.write_bytes(b"not a joblib file")
    engine = _engine(monkeypatch, path)
    assert engine.served_head == "zfnet_softmax"
    assert any("could not be loaded" in w for w in engine.load_warnings)


def test_no_artefact_at_all_is_simply_the_softmax_head(tmp_path, monkeypatch):
    engine = _engine(monkeypatch, tmp_path / "absent.joblib")
    assert engine.served_head == "zfnet_softmax" and engine.load_warnings == []
