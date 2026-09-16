"""5. Preprocessing consistency — training and inference must share one geometry."""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import requires_cv2

pytestmark = requires_cv2


@pytest.fixture
def raw():
    rng = np.random.default_rng(3)
    return rng.integers(0, 255, (200, 170), dtype=np.uint8)


def test_preprocess_is_deterministic(raw):
    from backend.methods.method1.transforms import M1_V2, preprocess

    a, b = preprocess(raw, M1_V2), preprocess(raw, M1_V2)
    assert np.array_equal(a, b)


def test_preprocess_output_contract(raw):
    from backend.methods.method1.transforms import M1_V2, preprocess

    out = preprocess(raw, M1_V2)
    assert out.dtype == np.float32
    assert out.shape == (M1_V2.image_size, M1_V2.image_size)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_training_and_inference_call_the_same_implementation(raw, tmp_path):
    """The dataset loader must reproduce what the engine computes, byte for byte."""
    import cv2

    from backend.methods.method1.datasets import _load_prepared
    from backend.methods.method1.transforms import M1_V2, preprocess

    path = tmp_path / "slice.png"
    cv2.imwrite(str(path), raw)

    from_disk = _load_prepared(str(path), M1_V2)          # training path
    from_memory = preprocess(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), M1_V2)  # inference path
    assert np.allclose(from_disk, from_memory, atol=1e-6)


def test_the_two_specs_are_genuinely_different(raw):
    from backend.methods.method1.transforms import M1_V1_LEGACY, M1_V2, preprocess

    assert not np.allclose(preprocess(raw, M1_V1_LEGACY), preprocess(raw, M1_V2)), (
        "if the orders produced identical output the spec would be meaningless"
    )


def test_legacy_spec_never_crops_even_when_a_mask_is_supplied(raw):
    from backend.methods.method1.transforms import M1_V1_LEGACY, classifier_input, preprocess

    image = preprocess(raw, M1_V1_LEGACY)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[20:60, 20:60] = 1

    out, cropped = classifier_input(image, mask, M1_V1_LEGACY)
    assert cropped is False
    assert np.array_equal(out, image), "a whole-slice model must never be served a crop"


def test_v2_spec_crops_when_a_mask_is_available(raw):
    from backend.methods.method1.transforms import M1_V2, classifier_input, preprocess

    image = preprocess(raw, M1_V2)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[30:70, 30:70] = 1

    out, cropped = classifier_input(image, mask, M1_V2)
    assert cropped is True
    assert out.shape == (M1_V2.image_size, M1_V2.image_size)


def test_empty_or_tiny_mask_falls_back_to_the_whole_image(raw):
    from backend.methods.method1.transforms import M1_V2, classifier_input, preprocess

    image = preprocess(raw, M1_V2)
    for mask in (np.zeros_like(image, np.uint8), None):
        out, cropped = classifier_input(image, mask, M1_V2)
        assert cropped is False and out.shape == image.shape


def test_spec_resolution_maps_untagged_checkpoints_to_the_legacy_geometry():
    from backend.methods.method1.transforms import M1_V1_LEGACY, M1_V2, get_spec

    assert get_spec(None).id == M1_V1_LEGACY.id
    assert get_spec("").id == M1_V1_LEGACY.id
    assert get_spec("m1-v2-roi").id == M1_V2.id


def test_scale_to_unit_is_pure_min_max(raw):
    """The old z-score-then-min-max was a provable no-op; verify the simplification."""
    from backend.methods.method1.transforms import scale_to_unit

    out = scale_to_unit(raw)
    expected = (raw.astype(np.float32) / 255.0)
    expected = (expected - expected.min()) / (expected.max() - expected.min())
    assert np.allclose(out, expected, atol=1e-6)


def test_flat_image_does_not_divide_by_zero():
    from backend.methods.method1.transforms import scale_to_unit

    out = scale_to_unit(np.full((16, 16), 128, np.uint8))
    assert np.isfinite(out).all()


def test_method2_preprocessing_is_independent_of_method1(raw):
    from backend.methods.method1.transforms import M1_V2
    from backend.methods.method1.transforms import preprocess as m1_preprocess
    from backend.methods.method2.transforms import M2_V1
    from backend.methods.method2.transforms import preprocess as m2_preprocess

    a, b = m1_preprocess(raw, M1_V2), m2_preprocess(raw, M2_V1)
    assert a.shape == b.shape
    assert not np.allclose(a, b), "the two methods must not share a preprocessing pipeline"
