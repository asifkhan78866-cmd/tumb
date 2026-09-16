"""4. Checkpoint isolation — one method's weights must never load into another's."""
from __future__ import annotations

import pytest

from backend.methods.common.checkpoint import (
    CheckpointError,
    build_meta,
    validate_meta,
)
from tests.conftest import requires_torch

M1 = dict(expected_method="method1", expected_architecture="ConvLSTMClassifier",
          expected_role="classification")


def test_matching_metadata_validates_cleanly():
    meta = build_meta("method1", "ConvLSTMClassifier", "classification",
                      class_names=("glioma", "meningioma", "notumor", "pituitary")).to_dict()
    assert validate_meta(meta, **M1) == []


def test_foreign_method_checkpoint_is_rejected():
    meta = build_meta("method2", "DenseConvNetClassifier", "classification").to_dict()
    with pytest.raises(CheckpointError, match="belongs to 'method2'"):
        validate_meta(meta, **M1)


def test_wrong_architecture_is_rejected():
    meta = build_meta("method1", "UNet", "classification").to_dict()
    with pytest.raises(CheckpointError, match="architecture"):
        validate_meta(meta, **M1)


def test_wrong_role_is_rejected():
    meta = build_meta("method1", "ConvLSTMClassifier", "segmentation").to_dict()
    with pytest.raises(CheckpointError, match="role"):
        validate_meta(meta, **M1)


def test_reordered_class_list_is_rejected():
    """Silently relabelling every prediction is worse than refusing to load."""
    meta = build_meta("method1", "ConvLSTMClassifier", "classification",
                      class_names=("meningioma", "glioma", "notumor", "pituitary")).to_dict()
    with pytest.raises(CheckpointError, match="class order"):
        validate_meta(meta, **M1,
                      expected_class_names=("glioma", "meningioma", "notumor", "pituitary"))


def test_legacy_untagged_checkpoint_loads_with_a_warning():
    warnings = validate_meta(None, **M1, path="best_classifier.pth")
    assert len(warnings) == 1 and "legacy" in warnings[0].lower()


def test_malformed_meta_is_rejected():
    with pytest.raises(CheckpointError):
        validate_meta("not-a-dict", **M1)


@requires_torch
def test_method2_weights_refuse_to_load_into_method1(tmp_path):
    from backend.methods.common.checkpoint import load_checkpoint, save_checkpoint
    from backend.methods.method1.models import ConvLSTMClassifier
    from backend.methods.method2.models import DenseConvNetClassifier

    dcn = DenseConvNetClassifier(in_channels=4, num_classes=4, feature_dim=8)
    path = tmp_path / "method2_dcn.pth"
    save_checkpoint(path, dcn.state_dict(),
                    build_meta("method2", "DenseConvNetClassifier", "classification"))

    with pytest.raises(CheckpointError, match="belongs to 'method2'"):
        load_checkpoint(ConvLSTMClassifier(), path, **M1)


@requires_torch
def test_round_trip_save_and_load_preserves_metadata(tmp_path):
    from backend.methods.common.checkpoint import load_checkpoint, peek_meta, save_checkpoint
    from backend.methods.method1.models import ConvLSTMClassifier

    model = ConvLSTMClassifier()
    path = tmp_path / "m1.pth"
    meta = build_meta("method1", "ConvLSTMClassifier", "classification",
                      class_names=("glioma", "meningioma", "notumor", "pituitary"),
                      image_size=128, transform_id="m1-v2-roi")
    save_checkpoint(path, model.state_dict(), meta)

    assert peek_meta(path)["transform_id"] == "m1-v2-roi"
    loaded, warnings = load_checkpoint(ConvLSTMClassifier(), path, **M1)
    assert loaded["method_id"] == "method1" and warnings == []


@requires_torch
def test_missing_checkpoint_raises_a_readable_error(tmp_path):
    from backend.methods.common.checkpoint import load_checkpoint
    from backend.methods.method1.models import ConvLSTMClassifier

    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(ConvLSTMClassifier(), tmp_path / "nope.pth", **M1)
