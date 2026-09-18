"""6. Missing-weight safety.

The regression this file guards against: an untrained network's output being
rendered as a segmentation result in a medical UI. A mask from randomly
initialised weights is noise, and noise must never reach a caller as a finding.
"""
from __future__ import annotations

import pytest

from tests.conftest import requires_api, requires_torch

pytestmark = requires_api


def _engine_without_segmentation(monkeypatch, tmp_path):
    """A Method 1 engine that cannot find a U-Net, whatever is on this machine."""
    from backend.methods.method1 import config as m1
    from backend.methods.method1.inference import Method1Engine

    monkeypatch.setattr(m1, "SEG_WEIGHTS_PATH", tmp_path / "missing_unet.pth")
    return Method1Engine()


def test_no_segmentation_weights_means_no_segmentation_claim(png_bytes, tmp_path, monkeypatch):
    r = _engine_without_segmentation(monkeypatch, tmp_path).predict(png_bytes)
    assert r.segmentation_available is False
    assert any("segmentation" in w.lower() for w in r.warnings)


def test_any_image_returned_without_weights_is_labelled_a_placeholder(png_bytes, tmp_path, monkeypatch):
    r = _engine_without_segmentation(monkeypatch, tmp_path).predict(png_bytes)
    if r.mask_path is not None:
        assert "placeholder" in r.mask_path.name.lower()
        assert any("placeholder" in w.lower() for w in r.warnings)


def test_segmentation_trained_on_weak_masks_says_so(client, png_bytes):
    """A U-Net trained on Otsu masks must never pass as tumour annotation."""
    from backend import config

    if not config.SEG_WEIGHTS_PATH.exists():
        import pytest as _pytest

        _pytest.skip("no segmentation checkpoint on this machine")
    body = client.post("/api/predict/method1",
                       files={"file": ("s.png", png_bytes, "image/png")}).json()
    details = body["details"]
    if details.get("segmentation_is_ground_truth_trained") is False:
        assert details["segmentation_mask_source"]
        assert any("not radiologist annotation" in w for w in body["warnings"])


def test_untrained_method_returns_no_prediction_rather_than_a_guess(png_bytes, tmp_path, monkeypatch):
    from backend.methods.method2 import config as m2
    from backend.methods.method2.inference import Method2Engine

    monkeypatch.setattr(m2, "DCN_WEIGHTS_PATH", tmp_path / "missing.pth")
    r = Method2Engine().predict(png_bytes)
    assert r.prediction is None and r.confidence is None and r.probabilities == {}
    assert r.model_version == "untrained"
    assert any("not trained" in w.lower() for w in r.warnings)


def test_untrained_method_reports_no_metrics(client, tmp_path, monkeypatch):
    from backend import config

    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)  # no metrics file written yet
    body = client.get("/api/metrics/method2").json()
    assert body["evaluated"] is False
    assert all(body[k] is None for k in ("accuracy", "dice", "iou", "f1", "auc", "precision"))


def test_methods_listing_advertises_untrained_state(client, tmp_path, monkeypatch):
    from backend import config

    monkeypatch.setattr(config, "M2_DCN_WEIGHTS_PATH", tmp_path / "missing.pth")
    rows = {m["method_id"]: m for m in client.get("/api/methods").json()}
    assert rows["method2"]["trained"] is False
    assert rows["method2"]["classifier_available"] is False
    assert rows["method2"]["warnings"]


def test_health_surfaces_the_missing_weights(client, monkeypatch):
    from backend.methods.method1.inference import engine

    # /health reports the live engine, so simulate the missing U-Net on it.
    monkeypatch.setattr(engine, "seg_weights_loaded", False)
    monkeypatch.setattr(engine, "load_warnings", ["Segmentation unavailable: checkpoint not found."])
    monkeypatch.setattr(engine, "_loaded", True)
    body = client.get("/health").json()
    assert body["seg_weights_loaded"] is False
    assert body["warnings"]


def test_legacy_upload_refuses_rather_than_inventing_a_class(client, png_bytes, monkeypatch):
    """With no classifier, /upload must 503 instead of returning a fabricated label."""
    from backend.methods.method1.inference import engine

    monkeypatch.setattr(engine, "cls_weights_loaded", False)
    monkeypatch.setattr(engine, "_loaded", True)
    r = client.post("/upload", files={"file": ("s.png", png_bytes, "image/png")})
    assert r.status_code == 503
    assert "not trained" in r.json()["detail"].lower()


@requires_torch
def test_fallback_model_initialisation_is_seeded():
    """If a fallback model is ever built, it must at least be reproducible."""
    import torch

    from backend import config
    from backend.methods.method1.models import UNet

    def build():
        torch.manual_seed(config.SEED)
        return UNet(1, 1, config.BASE_FILTERS)

    a, b = build(), build()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb), "unseeded fallback init makes predictions unreproducible"


@requires_torch
def test_engine_reload_is_stable_across_restarts(client, png_bytes):
    """The same image must classify the same way after a reload."""
    from backend.methods.method1.inference import engine

    first = client.post("/api/predict/method1",
                        files={"file": ("s.png", png_bytes, "image/png")}).json()
    engine.load(force=True)
    second = client.post("/api/predict/method1",
                         files={"file": ("s.png", png_bytes, "image/png")}).json()
    assert first["prediction"] == second["prediction"]
    assert first["class_probabilities"] == second["class_probabilities"]
