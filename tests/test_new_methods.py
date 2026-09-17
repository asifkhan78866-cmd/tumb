"""Transfer learning (displayed #1) and Red Fox optimised ZFNet (displayed #2)."""
from __future__ import annotations

import math

import pytest

from tests.conftest import requires_api, requires_torch


# --- Red Fox Optimization ------------------------------------------------- #
def _space():
    from backend.methods.method4.optimization.rfo import Parameter, SearchSpace

    return SearchSpace([
        Parameter("lr", "float", 1e-4, 1e-1, log=True),
        Parameter("dropout", "float", 0.0, 0.9),
        Parameter("width", "choice", choices=(32, 64, 128)),
    ])


def _objective(p):
    return -((math.log10(p["lr"]) + 2) ** 2 + 4 * (p["dropout"] - 0.3) ** 2 + 1e-4 * (p["width"] - 64) ** 2)


def _run(seed):
    from backend.methods.method4.optimization.rfo import RedFoxOptimizer

    return RedFoxOptimizer(_space(), _objective, population=8, iterations=5, seed=seed).run()


def test_rfo_same_seed_is_fully_reproducible():
    a, b = _run(11), _run(11)
    assert a.best_parameters == b.best_parameters and a.best_fitness == b.best_fitness
    assert [(e["parameters"], e["fitness"]) for e in a.evaluation_log] == \
           [(e["parameters"], e["fitness"]) for e in b.evaluation_log]


def test_rfo_improves_and_uses_all_three_phases():
    r = _run(3)
    initial_best = max(e["fitness"] for e in r.evaluation_log if e["phase"] == "init")
    assert r.best_fitness >= initial_best
    assert r.history[-1]["best_fitness"] >= r.history[0]["best_fitness"]
    phases = {e["phase"] for e in r.evaluation_log}
    assert {"init", "global", "local"} <= phases and phases & {"reproduce", "leave_herd"}
    assert 1e-4 <= r.best_parameters["lr"] <= 1e-1 and r.best_parameters["width"] in (32, 64, 128)


def test_rfo_rejects_invalid_configuration():
    from backend.methods.method4.optimization.rfo import RedFoxOptimizer

    with pytest.raises(ValueError):
        RedFoxOptimizer(_space(), _objective, population=2)
    with pytest.raises(ValueError):
        RedFoxOptimizer(_space(), _objective, replace_fraction=1.0)


# --- models ---------------------------------------------------------------- #
@requires_torch
def test_zfnet_layout_and_output_shape():
    import torch

    from backend.methods.method4.models import ZFNet

    m = ZFNet(num_classes=4, fc_units=1024).eval()
    assert m(torch.rand(2, 1, 224, 224)).shape == (2, 4)
    assert m.features[0].kernel_size == (7, 7) and m.features[0].stride == (2, 2)
    assert m.features[4].kernel_size == (5, 5) and m.features[4].stride == (2, 2)
    assert isinstance(m.gradcam_target, torch.nn.Conv2d)


@requires_torch
@pytest.mark.parametrize("name", ["resnet50", "efficientnet_b0", "densenet121"])
def test_transfer_backbones_get_a_four_class_head_and_gradcam(name):
    import torch

    from backend.methods.method3 import models as m3
    from backend.utils.gradcam import GradCAM

    model = m3.build(name, 4, pretrained=False).eval()
    x = torch.rand(1, 3, 224, 224, requires_grad=True)
    assert model(x).shape == (1, 4)
    cam = GradCAM(model, m3.gradcam_layer(model, name))
    assert cam(x, 0).shape == (224, 224)


# --- preprocessing shared by training and serving -------------------------- #
@requires_torch
def test_training_cache_and_serving_produce_the_same_input(tmp_path):
    import cv2
    import numpy as np

    from backend.methods.common.classifier_engine import decode_to_gray224
    from backend.methods.method3.transforms import DEFAULT_SPEC, preprocess_file

    img = np.random.default_rng(0).integers(0, 255, (300, 260, 3), dtype=np.uint8)
    path = tmp_path / "x.png"
    cv2.imwrite(str(path), img)
    cached = preprocess_file((str(path), DEFAULT_SPEC.to_dict()))
    served = decode_to_gray224(path.read_bytes(), DEFAULT_SPEC.image_size)
    assert cached.dtype == np.uint8 and np.array_equal(cached, served)


# --- engines never guess without weights ----------------------------------- #
@requires_api
@pytest.mark.parametrize("module", ["backend.methods.method3.inference", "backend.methods.method4.inference"])
def test_engine_without_checkpoint_reports_no_prediction(module, png_bytes, tmp_path, monkeypatch):
    import importlib

    engine_cls = importlib.import_module(module).engine.__class__
    monkeypatch.setattr(engine_cls, "weights_path", tmp_path / "missing.pth")
    r = engine_cls().predict(png_bytes)
    assert r.prediction is None and r.confidence is None and r.probabilities == {}
    assert r.model_version == "untrained" and r.segmentation_available is False
    assert any("not trained" in w for w in r.warnings)


@requires_api
def test_untrained_new_method_uses_labelled_ai_assessment(monkeypatch, png_bytes, tmp_path):
    from backend import config
    from backend.methods.method2 import ai_assessment
    from backend.methods.method4.inference import Method4Engine

    monkeypatch.setattr(Method4Engine, "weights_path", tmp_path / "missing.pth")
    monkeypatch.setattr(config, "method2_ai_available", lambda: True)
    assessment = ai_assessment.AIAssessment(
        image_is_brain_scan=True, observed_modality="MRI", image_quality="good",
        predicted_class="normal",
        likelihoods={"normal": 90, "glioma": 4, "meningioma": 3, "pituitary": 3},
        key_findings=["no mass"], rationale="Symmetric parenchyma.",
    )
    monkeypatch.setattr(ai_assessment, "assess_image",
                        lambda b: ai_assessment.AIAssessmentResult(assessment, "free-model", 0.1))
    r = Method4Engine().predict(png_bytes)
    assert r.prediction == "No Tumor" and r.prediction_key == "notumor"  # "normal" mapped to this class set
    assert r.confidence == 90.0 and set(r.probabilities) == {"Glioma", "Meningioma", "No Tumor", "Pituitary"}
    assert r.model_version == "ai:free-model" and r.details["result_source"] == "ai_assessment"
    assert r.warnings[0].startswith("AI ASSESSMENT") and "ZFNet" in r.warnings[0]
    assert "not finished training" in r.details["ai_assessment"]["reason"]


@requires_api
def test_engine_picks_up_a_checkpoint_written_after_startup(tmp_path, monkeypatch, png_bytes):
    from backend.methods.common.checkpoint import build_meta, save_checkpoint
    from backend.methods.method4.inference import Method4Engine
    from backend.methods.method4.models import ZFNet

    ckpt = tmp_path / "best_classifier.pth"
    monkeypatch.setattr(Method4Engine, "weights_path", ckpt)
    engine = Method4Engine()
    assert engine.predict(png_bytes).prediction is None  # untrained at "startup"

    hp = {"fc_units": 1024, "dropout": 0.5}
    model = ZFNet(num_classes=4, fc_units=1024)
    meta = build_meta(method_id="method4", architecture="ZFNet", role="classification",
                      class_names=["glioma", "meningioma", "notumor", "pituitary"],
                      image_size=224, transform_id="m4-gray224", hyperparameters=hp)
    save_checkpoint(ckpt, model.state_dict(), meta)

    r = engine.predict(png_bytes)  # same engine object, no restart
    assert r.prediction is not None and r.details["result_source"] == "trained_model"
    assert r.model_version == meta.model_version
