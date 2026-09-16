"""1. Method registry."""
from __future__ import annotations

import pytest

from backend.methods.registry import METHOD_IDS, METHODS, get_method, list_methods


def test_exactly_two_methods_registered():
    assert set(METHOD_IDS) == {"method1", "method2"}
    assert len(list_methods()) == 2


def test_unknown_method_raises_with_a_useful_message():
    with pytest.raises(KeyError) as exc:
        get_method("method3")
    assert "method1" in str(exc.value) and "method2" in str(exc.value)


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
    assert {w.role for w in spec.weights} == {"segmentation", "classification"}


def test_methods_do_not_share_weights_or_metrics_files():
    m1, m2 = get_method("method1"), get_method("method2")
    assert m1.metrics_filename != m2.metrics_filename
    assert {w.path_attr for w in m1.weights}.isdisjoint({w.path_attr for w in m2.weights})
    assert {w.architecture for w in m1.weights}.isdisjoint({w.architecture for w in m2.weights})


def test_method1_class_order_is_frozen():
    # The shipped checkpoint was trained with this exact index order; changing it
    # silently relabels every prediction.
    assert get_method("method1").class_names == ("glioma", "meningioma", "notumor", "pituitary")


def test_method2_declares_spect_not_pect():
    spec = get_method("method2")
    assert spec.modality == "SPECT"
    assert any("PECT" in n for n in spec.notes), "the SPECT/PECT distinction must be recorded"


def test_method2_marks_mri_dataset_as_incompatible():
    bri = next(d for d in get_method("method2").datasets if d.key == "bri")
    assert bri.compatible is False
