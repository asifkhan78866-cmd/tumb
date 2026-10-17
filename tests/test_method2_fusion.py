"""MRI–SPECT fusion: it keeps what each source carries, and it never becomes
the thing that produced the class."""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import requires_api, requires_cv2

pytestmark = requires_cv2


def _mri(size: int = 128) -> np.ndarray:
    """Structure: edges and texture, the thing MRI contributes."""
    import cv2

    rng = np.random.default_rng(3)
    img = np.zeros((size, size), np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 3, 180, -1)
    cv2.rectangle(img, (20, 20), (46, 46), 240, 2)
    cv2.line(img, (0, size - 10), (size, size - 30), 255, 1)
    return np.clip(img.astype(int) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)


def _spect(size: int = 128) -> np.ndarray:
    """Function: smooth, low-detail, strongly coloured."""
    import cv2

    y, x = np.mgrid[0:size, 0:size]
    blob = np.exp(-(((x - size * 0.4) ** 2 + (y - size * 0.6) ** 2) / (2 * (size / 5) ** 2)))
    grey = cv2.GaussianBlur((blob * 255).astype(np.uint8), (0, 0), 3)
    return cv2.applyColorMap(grey, cv2.COLORMAP_JET)


# --- the transform ---------------------------------------------------------- #
def test_fusion_keeps_the_spect_colour_and_takes_the_mri_detail():
    from backend.methods.method2 import fusion as F

    mri, spect = _mri(), _spect()
    pair = F.fuse(mri, spect)
    assert pair.fused_rgb.shape == (128, 128, 3)

    # The SPECT measurement is its colour, so the hue must survive the fusion.
    # (Chrominance is carried through untouched; what the 8-bit RGB rendering can
    # shift is magnitude, where a bright fused luminance clips against the gamut.)
    import cv2

    before = cv2.cvtColor(F.to_uint8(pair.spect_rgb), cv2.COLOR_BGR2HSV)
    after = cv2.cvtColor(F.to_uint8(pair.fused_rgb), cv2.COLOR_BGR2HSV)
    coloured = before[..., 1] > 30
    hue_shift = np.abs(before[..., 0].astype(int) - after[..., 0].astype(int))[coloured]
    assert np.median(hue_shift) <= 2 and np.percentile(hue_shift, 95) <= 10

    # The detail must come from the MRI, which the blurred SPECT does not have.
    m = F.metrics(pair.mri, pair.spect_luminance, pair.fused_luminance)
    spect_only = F.metrics(pair.mri, pair.spect_luminance, pair.spect_luminance)
    assert m["spatial_frequency"] > spect_only["spatial_frequency"]
    assert m["ssim_mri"] > m["ssim_spect"]


def test_fusion_preserves_more_edge_information_than_averaging():
    """Q_abf is the claim this stage makes; averaging is the thing it must beat."""
    from backend.methods.method2 import fusion as F

    pair = F.fuse(_mri(), _spect())
    fused_q = F.metrics(pair.mri, pair.spect_luminance, pair.fused_luminance)["qabf"]
    mean_q = F.metrics(pair.mri, pair.spect_luminance,
                       F.average_fuse(pair.mri, pair.spect_luminance))["qabf"]
    assert fused_q > mean_q


def test_a_mismatched_size_is_resampled_and_said_so():
    import cv2

    from backend.methods.method2 import fusion as F

    pair = F.fuse(_mri(128), cv2.resize(_spect(128), (64, 64)))
    assert pair.fused_rgb.shape[:2] == (128, 128)
    assert any("resampled to the MRI" in n for n in pair.notes)
    assert any("does not register them" in n for n in pair.notes)


def test_identical_sources_fuse_to_themselves():
    """A sanity bound: with nothing to choose between, fusion must not invent detail."""
    import cv2

    from backend.methods.method2 import fusion as F

    mri = _mri()
    grey_spect = cv2.cvtColor(mri, cv2.COLOR_GRAY2BGR)
    pair = F.fuse(mri, grey_spect)
    assert np.abs(pair.fused_luminance - pair.mri).mean() < 0.02


def test_metrics_are_all_finite_and_qabf_is_a_fraction():
    from backend.methods.method2 import fusion as F

    pair = F.fuse(_mri(), _spect())
    m = F.metrics(pair.mri, pair.spect_luminance, pair.fused_luminance)
    assert set(m) == {"entropy", "std", "spatial_frequency", "mutual_information_mri",
                      "mutual_information_spect", "ssim_mri", "ssim_spect", "qabf"}
    assert all(np.isfinite(v) for v in m.values())
    assert 0.0 <= m["qabf"] <= 1.0
    assert 0.0 <= m["entropy"] <= 8.0


def test_undecodable_bytes_return_nothing_rather_than_raising():
    from backend.methods.method2 import fusion as F

    assert F.fuse_bytes(b"not an image", b"also not an image") is None


# --- serving ---------------------------------------------------------------- #
@requires_api
def test_a_second_image_produces_a_fused_output_that_is_not_the_classifier_input(png_bytes):
    import cv2

    from backend.methods.method2.inference import Method2Engine

    spect_bytes = cv2.imencode(".png", _spect())[1].tobytes()
    result = Method2Engine().predict(png_bytes, spect_bytes)
    assert result.fused_path is not None and result.fused_path.exists()
    assert result.details["fusion_performed"] is True
    assert result.details["fusion"]["used_for_classification"] is False
    assert any("predicted from the MRI alone" in w for w in result.warnings)
    assert any("does not align them" in w for w in result.warnings)


@requires_api
def test_without_a_second_image_nothing_claims_fusion_happened(png_bytes):
    from backend.methods.method2.inference import Method2Engine

    result = Method2Engine().predict(png_bytes)
    assert result.fused_path is None
    assert result.details["fusion_performed"] is False
    assert result.details["fusion"] is None


@requires_api
def test_an_undecodable_second_image_warns_and_keeps_the_prediction(png_bytes):
    from backend.methods.method2.inference import Method2Engine

    result = Method2Engine().predict(png_bytes, b"not an image at all")
    assert result.fused_path is None
    assert any("could not be decoded" in w for w in result.warnings)


@requires_api
def test_only_the_fusion_method_offers_a_second_image(client):
    for method_id in ("method1", "method2", "method3", "method4"):
        body = client.get(f"/api/methods/{method_id}").json()
        assert body["accepts_secondary_image"] is (method_id == "method2")
        assert body["secondary_image_label"] == ("SPECT" if method_id == "method2" else "")


@requires_api
def test_a_second_image_sent_to_a_one_image_method_is_refused_in_words(client, png_bytes):
    files = {"file": ("a.png", png_bytes, "image/png"),
             "secondary": ("b.png", png_bytes, "image/png")}
    body = client.post("/api/predict/method1", files=files).json()
    assert body["fused_image_url"] is None
    assert any("takes one image, so it was ignored" in w for w in body["warnings"])


@requires_api
def test_the_second_image_gets_the_same_limits_as_the_first(client, png_bytes):
    files = {"file": ("a.png", png_bytes, "image/png"),
             "secondary": ("b.txt", b"x" * 10, "text/plain")}
    assert client.post("/api/predict/method2", files=files).status_code == 415


@requires_api
@pytest.mark.parametrize("path", ["/api/gallery/method2/fusion/999",
                                  "/api/gallery/nope/fusion",
                                  "/api/gallery/method2/fusion/-1"])
def test_fusion_samples_are_served_by_index_only(client, path):
    assert client.get(path).status_code in (404, 422)


@requires_api
def test_methods_that_do_not_fuse_list_no_fusion_panels(client):
    for method_id in ("method1", "method3", "method4"):
        assert client.get(f"/api/gallery/{method_id}/fusion").json() == []
