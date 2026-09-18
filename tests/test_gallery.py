"""Per-method galleries: only real files are served, and only from their own roots."""
from __future__ import annotations

import pytest

from tests.conftest import requires_api

pytestmark = requires_api

METHOD_IDS = ["method1", "method2", "method3", "method4"]


@pytest.mark.parametrize("method_id", METHOD_IDS)
def test_plot_listing_only_advertises_files_that_exist(client, method_id):
    from backend import config
    from backend.api.gallery_routes import _PLOTS

    rows = client.get(f"/api/gallery/{method_id}/plots").json()
    assert {r["key"] for r in rows} <= set(_PLOTS[method_id])
    for row in rows:
        assert (config.LOGS_DIR / _PLOTS[method_id][row["key"]][0]).exists()
        assert row["label"] and row["url"].endswith(row["key"])
        assert client.get(row["url"]).status_code == 200


def test_missing_plot_is_404_not_a_broken_image(client, tmp_path, monkeypatch):
    from backend import config

    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    assert client.get("/api/gallery/method1/plots").json() == []
    assert client.get("/api/gallery/method1/plots/confusion_matrix").status_code == 404


@pytest.mark.parametrize("method_id", METHOD_IDS)
def test_dataset_samples_cover_every_class_of_that_method(client, method_id):
    from backend import config
    from backend.methods.registry import get_method

    body = client.get(f"/api/gallery/{method_id}/dataset?per_class=2").json()
    if not config.BRI_DATASET_PATH.exists():
        pytest.skip("dataset not present on this machine")
    spec = get_method(method_id)
    # Method 4 names the tumour-free class "normal"; the dataset folder is "notumor".
    assert {i["class_label"] for i in body["images"]} == set(spec.labels)
    assert len(body["images"]) == 2 * spec.num_classes
    r = client.get(body["images"][0]["url"])
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/")


def test_sample_listing_says_so_when_the_dataset_is_absent(client, tmp_path, monkeypatch):
    from backend import config

    monkeypatch.setattr(config, "BRI_DATASET_PATH", tmp_path / "gone")
    body = client.get("/api/gallery/method3/dataset").json()
    assert body["present"] is False and body["images"] == []
    assert "download_dataset" in body["note"]


@pytest.mark.parametrize("path", [
    "/api/gallery/method3/plots/../../../etc/passwd",
    "/api/gallery/method3/dataset/..%2F..%2Fetc/0",
    "/api/gallery/method3/dataset/glioma/99",
    "/api/gallery/nope/plots",
    "/api/gallery/nope/dataset",
])
def test_no_path_escapes_and_unknown_ids_are_404(client, path):
    assert client.get(path).status_code in (404, 422)
