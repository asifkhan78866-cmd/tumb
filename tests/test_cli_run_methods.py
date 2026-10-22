"""The terminal runner: every method gets a block, every image gets a file, and
nothing is printed that the engine did not return."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import requires_api

pytestmark = requires_api


@pytest.fixture
def image(tmp_path, png_bytes) -> Path:
    path = tmp_path / "scan.png"
    path.write_bytes(png_bytes)
    return path


def _invoke(monkeypatch, capsys, *argv) -> tuple[int, str]:
    import sys

    from backend.cli import run_methods

    monkeypatch.setattr(sys, "argv", ["run_methods", *argv])
    code = run_methods.main()
    captured = capsys.readouterr()
    return code, captured.out + captured.err


# --- structure -------------------------------------------------------------- #
def test_one_block_per_method_and_a_summary(monkeypatch, capsys, image, tmp_path):
    code, out = _invoke(monkeypatch, capsys, str(image), "--outdir", str(tmp_path / "runs"))
    assert code == 0
    for number in (1, 2, 3, 4):
        assert f"METHOD {number}" in out
    assert "SUMMARY" in out
    # Every method names what it is, what its pictures show, and where they went.
    assert out.count("What it is") == 4
    assert out.count("What you see") == 4
    assert out.count("Saved to") == 4


def test_each_method_writes_its_images_into_its_own_folder(monkeypatch, capsys, image, tmp_path):
    out_root = tmp_path / "runs"
    _invoke(monkeypatch, capsys, str(image), "--outdir", str(out_root))
    run_dir = next(out_root.iterdir())
    folders = sorted(p.name for p in run_dir.iterdir() if p.is_dir())
    assert len(folders) == 4
    assert folders[0].startswith("method1_") and "method" in folders[-1]
    for folder in folders:
        files = {p.name for p in (run_dir / folder).iterdir()}
        assert "input.png" in files, "the preprocessed input is always written"
        assert any(f.startswith("source") for f in files), "the original is kept beside it"


def test_the_run_json_matches_what_was_printed(monkeypatch, capsys, image, tmp_path):
    out_root = tmp_path / "runs"
    _, out = _invoke(monkeypatch, capsys, str(image), "--outdir", str(out_root))
    payload = json.loads((next(out_root.iterdir()) / "run.json").read_text())
    assert payload["image"] == str(image)
    assert [m["display_number"] for m in payload["methods"]] == [1, 2, 3, 4]
    for method in payload["methods"]:
        assert method["short_name"] in out
        for path in method["images"].values():
            assert Path(path).exists()


def test_one_method_can_be_selected_by_number_or_id(monkeypatch, capsys, image, tmp_path):
    _, out = _invoke(monkeypatch, capsys, str(image), "--methods", "2",
                     "--outdir", str(tmp_path / "a"))
    assert "METHOD 2" in out and "METHOD 1" not in out

    _, out = _invoke(monkeypatch, capsys, str(image), "--methods", "method1",
                     "--outdir", str(tmp_path / "b"))
    assert "METHOD 3" in out, "method1 is displayed as Method 3"
    assert "METHOD 2" not in out


def test_unknown_inputs_fail_loudly(monkeypatch, capsys, image, tmp_path):
    code, out = _invoke(monkeypatch, capsys, str(image), "--methods", "nope")
    assert code == 1 and "unknown method" in out.lower()

    code, out = _invoke(monkeypatch, capsys, str(tmp_path / "missing.png"))
    assert code == 1 and "no such file" in out.lower()


def test_listing_shows_every_method_with_its_recorded_accuracy(monkeypatch, capsys):
    from backend.methods.common.metrics_store import metrics_for

    code, out = _invoke(monkeypatch, capsys, "--list")
    assert code == 0
    for method_id in ("method1", "method2", "method3", "method4"):
        assert method_id in out
        accuracy = metrics_for(method_id).get("accuracy")
        expected = f"{accuracy * 100:.2f}%" if accuracy is not None else "not evaluated"
        assert expected in out


# --- content ---------------------------------------------------------------- #
def test_a_methods_caveats_are_printed_not_dropped(monkeypatch, capsys, image, tmp_path):
    from backend.methods.registry import load_engine

    engine = load_engine("method4")
    result = engine.predict(image.read_bytes())
    _, out = _invoke(monkeypatch, capsys, str(image), "--methods", "2",
                     "--outdir", str(tmp_path / "runs"))
    if result.warnings:
        assert "Caveats" in out
        # The first few words of each warning survive the wrapping.
        assert all(w.split(".")[0][:40] in out.replace("\n", " ").replace("  ", " ")
                   or w[:30] in out for w in result.warnings[:1])


def test_a_borrowed_mask_is_labelled_as_borrowed(monkeypatch, capsys, image, tmp_path):
    _, out = _invoke(monkeypatch, capsys, str(image), "--methods", "1",
                     "--images", "off", "--outdir", str(tmp_path / "runs"))
    flat = " ".join(out.split())
    if "mask " in flat:
        assert "borrowed from the U-Net" in flat or "Not part of this method" in flat


def test_the_fusion_method_takes_a_second_image(monkeypatch, capsys, image, tmp_path):
    import cv2
    import numpy as np

    spect = tmp_path / "spect.png"
    cv2.imwrite(str(spect), cv2.applyColorMap(
        (np.random.default_rng(0).random((128, 128)) * 255).astype(np.uint8), cv2.COLORMAP_JET))
    out_root = tmp_path / "runs"
    _, out = _invoke(monkeypatch, capsys, str(image), "--methods", "4",
                     "--spect", str(spect), "--outdir", str(out_root))
    assert "fused" in out
    folder = next(p for p in next(out_root.iterdir()).iterdir() if p.is_dir())
    assert (folder / "fused.png").exists()


def test_no_prediction_is_said_plainly_rather_than_guessed(monkeypatch, capsys, image, tmp_path):
    """With the checkpoint hidden, the block must say so instead of inventing a class."""
    from backend.methods.method4 import inference as m4

    engine = m4.Method4Engine()
    monkeypatch.setattr(m4, "engine", engine)
    monkeypatch.setattr(type(engine), "weights_path",
                        tmp_path / "__absent__.pth")
    _, out = _invoke(monkeypatch, capsys, str(image), "--methods", "2",
                     "--outdir", str(tmp_path / "runs"))
    assert "none — this method produced no class" in out or "no prediction" in out
