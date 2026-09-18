"""Dataset download: layout normalisation and verification (no network)."""
from __future__ import annotations

from pathlib import Path

CLASSES = ("glioma", "meningioma", "notumor", "pituitary")


def _make_split(root: Path, per_class: int = 15) -> None:
    for split in ("Training", "Testing"):
        for c in CLASSES:
            d = root / split / c
            d.mkdir(parents=True)
            for i in range(per_class):
                (d / f"{c}_{i}.jpg").write_bytes(b"x")


def test_verify_accepts_the_expected_layout(tmp_path):
    from backend.utils.dataset_download import _verify_classification

    _make_split(tmp_path)
    ok, detail = _verify_classification(tmp_path)
    assert ok, detail


def test_verify_rejects_a_missing_testing_split(tmp_path):
    from backend.utils.dataset_download import _verify_classification

    _make_split(tmp_path, per_class=30)  # Training alone stays above the truncation threshold
    import shutil

    shutil.rmtree(tmp_path / "Testing")
    ok, detail = _verify_classification(tmp_path)
    assert not ok and "Testing" in detail


def test_wrapper_folder_is_flattened(tmp_path):
    from backend.utils.dataset_download import _flatten_single_wrapper, _verify_classification

    _make_split(tmp_path / "archive")
    _flatten_single_wrapper(tmp_path)
    assert (tmp_path / "Training").is_dir() and not (tmp_path / "archive").exists()
    assert _verify_classification(tmp_path)[0]


def test_default_dataset_path_matches_the_documented_layout(monkeypatch):
    from backend import config

    assert config.BRI_DATASET_PATH.parts[-3:] == ("data", "bri", "archive")


# --- finding the dataset a clone already has -------------------------------- #
def test_dataset_is_found_when_the_folder_is_just_archive(tmp_path):
    """The layout a teammate ends up with after unzipping the Kaggle file."""
    from backend.config import resolve_bri_path

    _make_split(tmp_path / "archive")
    assert resolve_bri_path(tmp_path) == (tmp_path / "archive").resolve()


def test_documented_layout_wins_over_other_copies(tmp_path):
    from backend.config import resolve_bri_path

    _make_split(tmp_path / "archive")
    _make_split(tmp_path / "data" / "bri" / "archive")
    assert resolve_bri_path(tmp_path) == (tmp_path / "data" / "bri" / "archive").resolve()


def test_an_explicit_setting_is_always_honoured(tmp_path):
    from backend.config import resolve_bri_path

    _make_split(tmp_path / "archive")            # would be found by the search
    chosen = tmp_path / "elsewhere" / "mri"      # but this was asked for, and does not exist
    assert resolve_bri_path(tmp_path, str(chosen)) == chosen
    assert resolve_bri_path(tmp_path, "some/relative/dir") == (tmp_path / "some/relative/dir").resolve()


def test_a_folder_without_both_splits_is_not_mistaken_for_the_dataset(tmp_path):
    from backend.config import looks_like_bri, resolve_bri_path
    import shutil

    _make_split(tmp_path / "archive")
    shutil.rmtree(tmp_path / "archive" / "Testing")
    assert looks_like_bri(tmp_path / "archive") is False
    # Falls back to the documented path so the message names one place to look.
    assert resolve_bri_path(tmp_path) == (tmp_path / "data" / "bri" / "archive").resolve()
