"""1. Method registry."""
from __future__ import annotations

import pytest

from backend.methods.registry import METHOD_IDS, METHODS, get_method, list_methods


def test_four_methods_registered_in_project_order():
    assert set(METHOD_IDS) == {"method1", "method2", "method3", "method4"}
    ordered = list_methods()
    assert [m.display_number for m in ordered] == [1, 2, 3, 4]
    assert [m.method_id for m in ordered] == ["method3", "method4", "method1", "method2"]
    titles = [m.display_name for m in ordered]
    assert "Transfer Learning" in titles[0]
    assert "Red Fox" in titles[1] and "ZFNet" in titles[1]
    assert "U-Net" in titles[2] and "ConvLSTM" in titles[2] and "SFLA" in titles[2]
    assert "MRI–SPECT" in titles[3] and "Fusion" in titles[3]


def test_unknown_method_raises_with_a_useful_message():
    with pytest.raises(KeyError) as exc:
        get_method("method9")
    assert all(m in str(exc.value) for m in ("method1", "method2", "method3", "method4"))


@pytest.mark.parametrize("method_id", METHOD_IDS)
def test_every_method_is_fully_described(method_id):
    spec = get_method(method_id)
    assert spec.display_name and spec.short_name and spec.summary
    assert spec.datasets, "a method must declare its datasets"
    assert spec.pipeline, "a method must declare its pipeline"
    assert spec.segmentation_model and spec.classifier_model
    assert spec.num_classes == len(spec.class_names) == len(spec.labels)
    assert spec.training_entrypoints
    assert spec.metrics_filename
    assert spec.weights, "a method must declare its weight roles"
    roles = {w.role for w in spec.weights}
    assert "classification" in roles and roles <= {"segmentation", "classification"}


def test_methods_do_not_share_weights_or_metrics_files():
    specs = list_methods()
    assert len({s.metrics_filename for s in specs}) == len(specs)
    paths = [w.path_attr for s in specs for w in s.weights]
    archs = [w.architecture for s in specs for w in s.weights]
    assert len(set(paths)) == len(paths) and len(set(archs)) == len(archs)


def test_method1_class_order_is_frozen():
    # The shipped checkpoint was trained with this exact index order; changing it
    # silently relabels every prediction.
    assert get_method("method1").class_names == ("glioma", "meningioma", "notumor", "pituitary")


def test_method2_declares_spect_not_pect():
    spec = get_method("method2")
    assert spec.modality == "MRI + SPECT"
    assert any("PECT" in n for n in spec.notes), "the SPECT/PECT distinction must be recorded"


def test_method2_marks_mri_dataset_as_incompatible():
    bri = next(d for d in get_method("method2").datasets if d.key == "bri")
    assert bri.compatible is False
