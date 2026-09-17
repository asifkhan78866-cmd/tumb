"""2, 3, 10. Per-method response schemas and API method switching."""
from __future__ import annotations

import pytest

from tests.conftest import requires_api

pytestmark = requires_api

REQUIRED_PREDICTION_FIELDS = {
    "method_id", "method_name", "prediction", "class_probabilities",
    "segmentation_available", "segmentation_mask_url", "heatmap_url",
    "processing_time_s", "model_version", "dataset_context", "warnings",
}


def test_list_methods_returns_all_four_in_display_order(client):
    r = client.get("/api/methods")
    assert r.status_code == 200
    rows = r.json()
    # ids are stable (baked into checkpoints); display numbers follow the project's list.
    assert [m["method_id"] for m in rows] == ["method3", "method4", "method1", "method2"]
    assert [m["display_number"] for m in rows] == [1, 2, 3, 4]
    seg = {m["method_id"]: m["has_segmentation_stage"] for m in rows}
    assert seg == {"method3": False, "method4": False, "method1": True, "method2": True}


def test_method_detail_includes_pipeline_and_datasets(client):
    r = client.get("/api/methods/method1")
    assert r.status_code == 200
    body = r.json()
    assert body["pipeline_stages"] and body["datasets"] and body["weights"]
    assert body["optimization"] and "SFLA" in body["optimization"]


def test_method2_detail_reports_fusion_modalities_and_no_optimization(client):
    body = client.get("/api/methods/method2").json()
    assert body["modality"] == "MRI + SPECT"
    assert "Fusion" in body["display_name"]
    assert body["optimization"] is None
    assert "DenseNet" in body["classifier_model"] or "Dense" in body["classifier_model"]


def test_unknown_method_is_404_everywhere(client):
    for path in ("/api/methods/nope", "/api/metrics/nope"):
        assert client.get(path).status_code == 404
    assert client.post("/api/predict/nope", files={"file": ("a.png", b"x", "image/png")}).status_code == 404


# --- 2 & 3: response schemas ------------------------------------------- #
@pytest.mark.parametrize("method_id", ["method1", "method2", "method3", "method4"])
def test_prediction_response_uses_the_shared_schema(client, png_bytes, method_id):
    r = client.post(f"/api/predict/{method_id}", files={"file": ("s.png", png_bytes, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert REQUIRED_PREDICTION_FIELDS <= set(body)
    assert body["method_id"] == method_id
    assert isinstance(body["class_probabilities"], dict)
    assert isinstance(body["warnings"], list)
    assert isinstance(body["segmentation_available"], bool)
    assert body["processing_time_s"] >= 0


def test_method1_response_carries_method1_identity(client, png_bytes):
    body = client.post("/api/predict/method1",
                       files={"file": ("s.png", png_bytes, "image/png")}).json()
    assert body["modality"] == "MRI"
    assert "ConvLSTM" in body["details"]["classifier_model"]
    assert "transform" in body["details"]


def test_method2_response_carries_its_own_modality_and_feature_stage(client, png_bytes):
    body = client.post("/api/predict/method2",
                       files={"file": ("s.png", png_bytes, "image/png")}).json()
    assert body["modality"] == "MRI + SPECT"
    assert body["details"]["modality"] == "SPECT"  # the SPECT branch's own feature stage
    assert body["details"]["spec_modality_label"] == "PECT"
    feature = body["details"]["feature_stage"]
    assert feature["dimension"] == len(feature["values"]) == len(feature["names"])
    assert "Dense" in body["details"]["classifier_model"]


# --- 10: switching methods keeps them separate -------------------------- #
def test_switching_methods_does_not_mix_results(client, png_bytes):
    files = {"file": ("s.png", png_bytes, "image/png")}
    a = client.post("/api/predict/method1", files=files).json()
    b = client.post("/api/predict/method2", files=files).json()

    assert a["method_id"] != b["method_id"]
    assert a["method_name"] != b["method_name"]
    assert a["modality"] != b["modality"]
    assert a["prediction_id"] != b["prediction_id"]
    assert a["model_version"] != b["model_version"]


def test_metrics_are_never_shared_between_methods(client, tmp_path, monkeypatch):
    import json

    from backend import config

    # Each method reads only its own file: write one method's metrics and check
    # no other method picks them up.
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    (tmp_path / "method1_metrics.json").write_text(json.dumps(
        {"method_id": "method1", "classification": {"accuracy": 0.9, "f1": 0.9, "split": "x"}}))
    m1 = client.get("/api/metrics/method1").json()
    assert m1["method_id"] == "method1" and m1["accuracy"] == 0.9
    for other in ("method2", "method3", "method4"):
        m = client.get(f"/api/metrics/{other}").json()
        assert m["method_id"] == other and m["evaluated"] is False
        for key in ("accuracy", "dice", "f1", "auc"):
            assert m[key] is None, f"{other} must not report {key} from another method's file"


def test_metrics_list_endpoint_returns_one_row_per_method(client):
    rows = client.get("/api/metrics").json()
    assert [r["method_id"] for r in rows] == ["method3", "method4", "method1", "method2"]


# --- uploads ------------------------------------------------------------ #
def test_unsupported_content_type_rejected(client):
    r = client.post("/api/predict/method1", files={"file": ("a.txt", b"hi", "text/plain")})
    assert r.status_code == 415


def test_empty_upload_rejected(client):
    r = client.post("/api/predict/method1", files={"file": ("a.png", b"", "image/png")})
    assert r.status_code == 400


def test_oversized_upload_rejected(client):
    from backend import config

    blob = b"\x89PNG\r\n\x1a\n" + b"0" * (config.MAX_UPLOAD_BYTES + 1024)
    r = client.post("/api/predict/method1", files={"file": ("big.png", blob, "image/png")})
    assert r.status_code == 413


def test_undecodable_image_rejected(client):
    r = client.post("/api/predict/method1",
                    files={"file": ("a.png", b"not-an-image", "image/png")})
    assert r.status_code == 422


# --- training endpoint safety ------------------------------------------- #
def test_training_endpoint_does_not_launch_by_default(client):
    body = client.post("/api/train/method1?stage=classification").json()
    assert body["started"] is False
    assert "backend.methods.method1.training.train_classifier" in body["command"]


def test_training_endpoint_rejects_an_unknown_stage(client):
    assert client.post("/api/train/method1?stage=bogus").status_code == 400


def test_method2_has_no_sfla_stage(client):
    assert client.post("/api/train/method2?stage=sfla").status_code == 400


def test_optimisation_stages_belong_to_their_own_methods(client):
    assert client.post("/api/train/method4?stage=sfla").status_code == 400
    assert client.post("/api/train/method1?stage=rfo").status_code == 400
    body = client.post("/api/train/method4?stage=rfo").json()
    assert body["started"] is False and "method4.optimization.run_rfo" in body["command"]
    assert client.post("/api/train/method3?stage=segmentation").status_code == 400


def test_history_can_be_filtered_by_method(client, png_bytes):
    client.post("/api/predict/method2", files={"file": ("s.png", png_bytes, "image/png")})
    rows = client.get("/history?limit=50&method_id=method2").json()
    assert rows and all(r["method_id"] == "method2" for r in rows)
