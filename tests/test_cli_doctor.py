"""The doctor: an honest report of what this copy can run.

Its whole job is to be right about absence, so these tests mostly remove things
and check that it says so rather than reporting health it cannot see.
"""
from __future__ import annotations

import sys

import pytest

from tests.conftest import requires_api

pytestmark = requires_api


def _run(monkeypatch, capsys) -> tuple[int, str]:
    from backend.cli import doctor

    monkeypatch.setattr(sys, "argv", ["doctor"])
    code = doctor.main()
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def test_it_reports_every_component_and_succeeds_on_a_working_copy(monkeypatch, capsys):
    code, out = _run(monkeypatch, capsys)
    for label in ("Python", "Python packages", "Compute device", "Recorded metrics",
                  "ImageNet backbones", "Sample images", "Full dataset"):
        assert label in out
    assert "Method 1" in out and "Method 4" in out
    assert code == 0


def test_a_missing_classifier_is_an_error_with_the_command_that_fixes_it(monkeypatch, capsys,
                                                                        tmp_path):
    """A method whose classifier is gone cannot predict — that is not a warning."""
    from backend import config

    monkeypatch.setattr(config, "M4_WEIGHTS_PATH", tmp_path / "gone.pth")
    code, out = _run(monkeypatch, capsys)
    assert code == 1
    assert "no classification checkpoint" in out
    assert "train_classifier" in out, "it must name the command that rebuilds it"
    assert "Something a prediction needs is missing" in out


def test_a_missing_segmentation_is_only_a_warning(monkeypatch, capsys, tmp_path):
    """Method 3 still classifies without its U-Net; it just cannot outline."""
    from backend import config

    monkeypatch.setattr(config, "SEG_WEIGHTS_PATH", tmp_path / "gone.pth")
    code, out = _run(monkeypatch, capsys)
    assert "no segmentation checkpoint" in out
    assert code == 0, "a missing mask must not be reported as a broken install"


def test_absent_backbones_are_explained_rather_than_failed(monkeypatch, capsys, tmp_path):
    """Method 1's ImageNet weights are a download, not part of the repository."""
    import torch

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    code, out = _run(monkeypatch, capsys)
    assert "0/4 cached" in out and "800 MB" in out
    assert "every other method works offline" in out
    assert code == 0


def test_a_clone_without_the_dataset_is_told_it_can_still_predict(monkeypatch, capsys, tmp_path):
    from backend import config

    monkeypatch.setattr(config, "BRI_DATASET_PATH", tmp_path / "no-dataset")
    code, out = _run(monkeypatch, capsys)
    assert "not downloaded" in out and "training does not" in out
    assert "download_dataset.sh" in out
    assert code == 0


def test_the_documented_sample_images_are_checked_by_name(monkeypatch, capsys):
    """Counting whatever sits in data/ would call a machine healthy because it
    has the full dataset, which is not what a clone has."""
    from backend.cli import doctor

    assert any("Te-me_0014" in s for s in doctor.SAMPLES)
    _, out = _run(monkeypatch, capsys)
    assert f"{len(doctor.SAMPLES)}/{len(doctor.SAMPLES)} documented examples present" in out


def test_missing_packages_stop_the_report_instead_of_guessing(monkeypatch, capsys):
    import builtins

    real = builtins.__import__

    def fail_on_sklearn(name, *args, **kwargs):
        if name == "sklearn":
            raise ImportError("no sklearn")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_on_sklearn)
    monkeypatch.delitem(sys.modules, "sklearn", raising=False)
    code, out = _run(monkeypatch, capsys)
    assert code == 1
    assert "sklearn" in out and "requirements.txt" in out
    assert "Fix the above first" in out
