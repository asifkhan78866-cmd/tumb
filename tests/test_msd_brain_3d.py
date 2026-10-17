"""The Decathlon (BraTS) reader and the volumetric U-Net.

Two failures matter most here and both are silent: label values that do not
mean what the project thinks they mean, and a split that puts one patient on
both sides. Everything else is shape bookkeeping.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import requires_torch

nib = pytest.importorskip("nibabel", reason="nibabel not installed")


def _write_volume(directory: Path, name: str, shape=(24, 24, 16), labels=(1, 2, 3)) -> None:
    """A tiny four-sequence volume with a labelled blob, in MSD layout."""
    directory.joinpath("imagesTr").mkdir(parents=True, exist_ok=True)
    directory.joinpath("labelsTr").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    image = np.zeros((*shape, 4), np.float32)
    image[4:-4, 4:-4, 2:-2] = rng.uniform(40, 200, image[4:-4, 4:-4, 2:-2].shape)
    label = np.zeros(shape, np.uint8)
    for i, value in enumerate(labels):
        label[8 + i : 10 + i, 8:12, 6:9] = value
    nib.save(nib.Nifti1Image(image, np.eye(4)), directory / "imagesTr" / f"{name}.nii.gz")
    nib.save(nib.Nifti1Image(label, np.eye(4)), directory / "labelsTr" / f"{name}.nii.gz")


@pytest.fixture
def msd_root(tmp_path) -> Path:
    root = tmp_path / "Task01_BrainTumour"
    for i in range(1, 5):
        _write_volume(root, f"BRATS_{i:03d}")
    return root


# --- the reader ------------------------------------------------------------- #
def test_labels_are_remapped_to_this_projects_class_order():
    """MSD: 1 oedema, 2 non-enhancing, 3 enhancing. Here: 1 necrotic, 2 oedema,
    3 enhancing. Trusting the integers would relabel two regions silently."""
    from backend.methods.common.msd_brain import MSD_TO_PROJECT, remap_labels

    out = remap_labels(np.array([0, 1, 2, 3], dtype=np.uint8))
    assert out.tolist() == [0, 2, 1, 3]
    assert MSD_TO_PROJECT[1] == 2 and MSD_TO_PROJECT[2] == 1


def test_discovery_finds_pairs_and_skips_resource_forks(msd_root):
    from backend.methods.common.msd_brain import discover_msd

    (msd_root / "imagesTr" / "._BRATS_001.nii.gz").write_bytes(b"not a volume")
    pairs = discover_msd(msd_root)
    assert [p.volume_id for p in pairs] == ["BRATS_001", "BRATS_002", "BRATS_003", "BRATS_004"]
    assert all(p.image.exists() and p.label.exists() for p in pairs)


def test_discovery_descends_to_the_folder_that_holds_the_data(tmp_path):
    """An archive unpacked one level deeper must still be found."""
    from backend.methods.common.msd_brain import discover_msd

    inner = tmp_path / "download" / "Task01_BrainTumour"
    _write_volume(inner, "BRATS_001")
    (inner / "dataset.json").write_text("{}")
    assert len(discover_msd(tmp_path / "download")) == 1


def test_one_modality_is_selected_by_name_not_by_index(msd_root):
    from backend.methods.common.msd_brain import MODALITIES, discover_msd, load_volume

    pair = discover_msd(msd_root)[0]
    flair, _ = load_volume(pair, "FLAIR")
    t2, _ = load_volume(pair, "T2w")
    assert flair.ndim == 3 and t2.ndim == 3
    assert not np.array_equal(flair, t2)
    everything, _ = load_volume(pair, None)
    assert everything.shape[-1] == len(MODALITIES)
    with pytest.raises(ValueError, match="Unknown modality"):
        load_volume(pair, "PET")


def test_an_empty_volume_normalises_to_zeros_rather_than_dividing_by_nothing():
    from backend.methods.common.msd_brain import normalise

    assert normalise(np.zeros((4, 4, 4), np.float32)).max() == 0.0
    out = normalise(np.linspace(0, 100, 64, dtype=np.float32).reshape(4, 4, 4))
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_the_brain_box_crops_the_air_around_the_head():
    from backend.methods.common.msd_brain import brain_bbox

    volume = np.zeros((20, 20, 20), np.float32)
    volume[8:12, 6:14, 5:9] = 1.0
    box = brain_bbox(volume, pad=1)
    assert volume[box].shape == (6, 10, 6)
    assert brain_bbox(np.zeros((5, 5, 5), np.float32))[0] == slice(0, 5)


def test_slice_extraction_keeps_annotated_slices_and_says_what_it_sampled(msd_root, tmp_path):
    from backend.methods.common.msd_brain import discover_msd, materialise_slices

    out = tmp_path / "slices"
    index = materialise_slices(discover_msd(msd_root), out, stride=1, size=32,
                               min_tumour_pixels=1, background_every=0, verbose=False)
    assert index.files and all((out / f).exists() for f in index.files)
    assert index.volume_of(index.files[0]) == "BRATS_001"
    with np.load(out / index.files[0]) as data:
        assert data["image"].shape == (32, 32) and data["label"].max() > 0
    assert any("sampled, not taken whole" in w for w in index.warnings)
    assert any("resampled" in w for w in index.warnings)

    # A second call reuses the cache rather than re-decoding the volumes.
    again = materialise_slices([], out, verbose=False)
    assert again.files == index.files


def test_volume_caching_round_trips_image_and_labels(msd_root, tmp_path):
    from backend.methods.common.msd_brain import discover_msd, materialise_volumes

    out = tmp_path / "volumes"
    names = materialise_volumes(discover_msd(msd_root), out, size=16, verbose=False)
    assert len(names) == 4
    with np.load(out / names[0]) as data:
        assert data["image"].shape == (16, 16, 16) and data["label"].shape == (16, 16, 16)
        # Nearest-neighbour resampling: no class index may be invented.
        assert set(np.unique(data["label"])) <= {0, 1, 2, 3}


# --- the model and its training loop ---------------------------------------- #
@requires_torch
def test_the_3d_unet_returns_a_label_map_the_size_of_its_input():
    import torch

    from backend.models.unet3d import UNet3D

    model = UNet3D(in_channels=1, num_classes=2, base=4).eval()
    with torch.no_grad():
        assert tuple(model(torch.randn(1, 1, 32, 32, 32)).shape) == (1, 2, 32, 32, 32)
        # Odd sizes survive: the decoder pads rather than demanding a multiple of 8.
        assert tuple(model(torch.randn(1, 1, 24, 28, 20)).shape) == (1, 2, 24, 28, 20)


@requires_torch
def test_patches_are_drawn_on_the_tumour_when_asked_and_are_always_the_right_size():
    from backend.methods.method1.training.train_segmentation_3d import sample_patch

    rng = np.random.default_rng(0)
    image = rng.random((40, 40, 40)).astype(np.float32)
    label = np.zeros((40, 40, 40), np.uint8)
    label[30:34, 30:34, 30:34] = 1

    on_tumour = [sample_patch(image, label, 16, rng, 1.0)[1].sum() for _ in range(12)]
    assert all(t > 0 for t in on_tumour), "tumour_prob=1 must centre every patch on tumour"
    patch, target = sample_patch(image, label, 16, rng, 0.0)
    assert patch.shape == (16, 16, 16) and target.shape == (16, 16, 16)

    # A volume smaller than the patch is padded, not refused.
    small, small_label = sample_patch(image[:8, :8, :8], label[:8, :8, :8], 16, rng, 0.0)
    assert small.shape == (16, 16, 16) and small_label.shape == (16, 16, 16)


@requires_torch
def test_dice_is_one_only_when_both_are_empty_or_both_agree():
    from backend.methods.method1.training.train_segmentation_3d import dice_of

    empty = np.zeros((4, 4, 4), np.uint8)
    blob = empty.copy()
    blob[1:3, 1:3, 1:3] = 1
    assert dice_of(empty, empty) == 1.0          # correctly found nothing
    assert dice_of(blob, blob) == 1.0
    assert dice_of(empty, blob) == 0.0           # missed it entirely
    assert 0.0 < dice_of(blob, np.roll(blob, 1, axis=0)) < 1.0


@requires_torch
def test_sliding_window_inference_covers_a_volume_larger_than_its_patch():
    import torch

    from backend.methods.method1.training.train_segmentation_3d import segment_volume
    from backend.models.unet3d import UNet3D

    model = UNet3D(1, 2, base=4)
    with torch.no_grad():  # make the model answer "tumour" everywhere
        model.head.bias.copy_(torch.tensor([-10.0, 10.0]))
        model.head.weight.zero_()
    pred = segment_volume(model, np.zeros((40, 36, 32), np.float32), patch=16,
                          device=torch.device("cpu"), overlap=4)
    assert pred.shape == (40, 36, 32)
    assert pred.all(), "every voxel must be covered by at least one window"


@requires_torch
def test_the_volume_store_binarises_sub_regions_into_whole_tumour(tmp_path):
    from backend.methods.method1.training.train_segmentation_3d import VolumeStore

    np.savez_compressed(tmp_path / "BRATS_001.npz",
                        image=np.ones((4, 4, 4), np.float16),
                        label=np.array([[[0, 1, 2, 3]] * 4] * 4, np.uint8))
    image, label = VolumeStore(tmp_path, ["BRATS_001.npz"]).load("BRATS_001.npz")
    assert image.dtype == np.float32
    assert set(np.unique(label)) == {0, 1}, "this model is binary whole-tumour"


@requires_torch
def test_volumes_are_split_by_patient_so_no_slice_of_a_test_patient_is_trained_on():
    from backend.methods.common.splits import group_train_val_test_split, split_summary

    names = [f"BRATS_{i:03d}.npz" for i in range(1, 41)]
    key = lambda n: Path(n).stem.lower()  # noqa: E731
    train, val, test = group_train_val_test_split(names, key, 0.15, 0.15, seed=7)
    summary = split_summary({"train": train, "val": val, "test": test}, key)
    assert summary["leak_free"] is True
    assert not ({key(n) for n in train} & {key(n) for n in test})
