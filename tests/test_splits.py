"""8. Patient-level split utility."""
from __future__ import annotations

import pytest

from backend.methods.common.splits import (
    brats_volume_id,
    bri_patient_id,
    group_split,
    group_train_val_test_split,
    split_summary,
)


def test_brats_volume_id_from_h5_slice_names():
    assert brats_volume_id("/d/volume_37_slice_102.h5") == "volume_37"
    assert brats_volume_id("/d/volume_37_slice_9.h5") == "volume_37"
    # every slice of a volume maps to one key
    keys = {brats_volume_id(f"/d/volume_5_slice_{i}.h5") for i in range(50)}
    assert keys == {"volume_5"}


def test_brats_volume_id_from_nifti_subject_dirs():
    assert "brats20_training_037" in brats_volume_id(
        "/d/BraTS20_Training_037/BraTS20_Training_037_flair.nii"
    )


def test_bri_patient_id_strips_trailing_counters():
    assert bri_patient_id("Tr-gl_0123_2.jpg") == "tr-gl_0123"
    assert bri_patient_id("Tr-gl_0123_3.jpg") == "tr-gl_0123"


def test_no_group_is_ever_split_across_parts():
    samples = [f"volume_{v}_slice_{s}.h5" for v in range(40) for s in range(10)]
    train, val, test = group_train_val_test_split(samples, brats_volume_id, 0.2, 0.2, seed=1)
    groups = [{brats_volume_id(s) for s in part} for part in (train, val, test)]
    assert groups[0] & groups[1] == set()
    assert groups[0] & groups[2] == set()
    assert groups[1] & groups[2] == set()
    assert len(train) + len(val) + len(test) == len(samples)


def test_split_is_deterministic_for_a_seed_and_order_independent():
    samples = [f"volume_{v}_slice_{s}.h5" for v in range(30) for s in range(8)]
    a = group_train_val_test_split(samples, brats_volume_id, 0.2, 0.2, seed=99)
    b = group_train_val_test_split(list(reversed(samples)), brats_volume_id, 0.2, 0.2, seed=99)
    assert [sorted(p) for p in a] == [sorted(p) for p in b]


def test_different_seeds_give_different_splits():
    samples = [f"volume_{v}_slice_{s}.h5" for v in range(30) for s in range(8)]
    a = group_train_val_test_split(samples, brats_volume_id, 0.2, 0.2, seed=1)[0]
    b = group_train_val_test_split(samples, brats_volume_id, 0.2, 0.2, seed=2)[0]
    assert sorted(a) != sorted(b)


def test_realised_sizes_are_close_to_the_requested_fractions():
    samples = [f"volume_{v}_slice_{s}.h5" for v in range(100) for s in range(10)]
    train, val, test = group_train_val_test_split(samples, brats_volume_id, 0.15, 0.15, seed=3)
    n = len(samples)
    assert 0.60 <= len(train) / n <= 0.80
    assert 0.08 <= len(val) / n <= 0.24
    assert 0.08 <= len(test) / n <= 0.24


def test_split_summary_detects_leakage():
    clean = split_summary(
        {"train": ["volume_1_slice_1.h5"], "test": ["volume_2_slice_1.h5"]}, brats_volume_id
    )
    assert clean["leak_free"] is True

    leaky = split_summary(
        {"train": ["volume_1_slice_1.h5"], "test": ["volume_1_slice_2.h5"]}, brats_volume_id
    )
    assert leaky["leak_free"] is False
    assert "volume_1" in str(leaky["group_overlap"])


def test_rejects_impossible_fractions():
    with pytest.raises(ValueError):
        group_train_val_test_split(["a"], lambda s: s, 0.6, 0.6)
    with pytest.raises(ValueError):
        group_split(["a"], lambda s: s, [])


# --- official Training/Testing split -------------------------------------- #
def test_official_split_detection():
    from backend.methods.method1.datasets import official_split_of

    assert official_split_of("archive/Training/glioma/a.jpg") == "train"
    assert official_split_of("archive/Testing/notumor/b.jpg") == "test"
    assert official_split_of("data/bri/archive/Training/pituitary/c.jpg") == "train"
    assert official_split_of("somewhere/else/d.jpg") is None


def test_partition_keeps_testing_folder_out_of_the_train_pool():
    from backend.methods.method1.datasets import partition_by_official_split

    samples = [("archive/Training/glioma/a.jpg", 0), ("archive/Testing/glioma/b.jpg", 0),
               ("archive/Training/notumor/c.jpg", 2), ("archive/Testing/notumor/d.jpg", 2)]
    train_pool, test = partition_by_official_split(samples)
    assert [p for p, _ in train_pool] == ["archive/Training/glioma/a.jpg",
                                          "archive/Training/notumor/c.jpg"]
    assert [p for p, _ in test] == ["archive/Testing/glioma/b.jpg",
                                    "archive/Testing/notumor/d.jpg"]
    assert not (set(train_pool) & set(test))


def test_class_weights_counter_imbalance():
    from backend.methods.method1.datasets import class_weights

    # notumor is 3x the others -> it must receive the smallest weight.
    samples = [("a.jpg", 0)] * 10 + [("b.jpg", 1)] * 10 + \
              [("c.jpg", 2)] * 30 + [("d.jpg", 3)] * 10
    w = class_weights(samples)
    assert len(w) == 4
    assert w[2] == min(w), "the majority class must be down-weighted"
    assert abs(sum(w) / len(w) - 1.0) < 1e-6, "weights are normalised to mean 1"


def test_class_weights_are_flat_when_balanced():
    from backend.methods.method1.datasets import class_weights

    samples = [("x.jpg", i) for i in range(4) for _ in range(25)]
    assert all(abs(w - 1.0) < 1e-6 for w in class_weights(samples))
