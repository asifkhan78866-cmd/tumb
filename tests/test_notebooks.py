"""The generated notebooks: one per method, each a valid notebook that quotes
the registry rather than a copy of it."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat", reason="nbformat not installed")

from tests.conftest import requires_api  # noqa: E402

pytestmark = requires_api

EXPECTED = {
    "00_overview.ipynb": None,
    "01_method1_pretrained_transfer.ipynb": "method3",
    "02_method2_zfnet_redfox.ipynb": "method4",
    "03_method3_unet_convlstm_sfla.ipynb": "method1",
    "04_method4_mri_spect_fusion.ipynb": "method2",
}


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    import sys

    from backend.cli import build_notebooks

    out = tmp_path_factory.mktemp("notebooks")
    argv = ["build_notebooks", "--outdir", str(out)]
    old, sys.argv = sys.argv, argv
    try:
        assert build_notebooks.main() == 0
    finally:
        sys.argv = old
    return out


def test_one_notebook_per_method_plus_an_overview(built):
    assert sorted(p.name for p in built.glob("*.ipynb")) == sorted(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_notebook_is_valid_and_runnable_by_a_python_kernel(built, name):
    nb = nbformat.read(built / name, as_version=4)
    nbformat.validate(nb)
    assert nb.metadata["kernelspec"]["name"] == "python3"
    assert sum(1 for c in nb.cells if c.cell_type == "code") >= 8
    assert sum(1 for c in nb.cells if c.cell_type == "markdown") >= 5


@pytest.mark.parametrize("name,method_id",
                         [(n, m) for n, m in EXPECTED.items() if m])
def test_a_method_notebook_targets_that_method_and_names_it(built, name, method_id):
    from backend.methods.registry import get_method

    spec = get_method(method_id)
    source = "\n".join(c.source for c in nbformat.read(built / name, as_version=4).cells)
    assert f'METHOD = "{method_id}"' in source
    assert spec.short_name in source
    assert f"Method {spec.display_number}" in source


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_sections_follow_the_same_order_in_every_method_notebook(built, name):
    if name == "00_overview.ipynb":
        pytest.skip("the overview has its own outline")
    headings = [c.source.splitlines()[0] for c in nbformat.read(built / name, as_version=4).cells
                if c.cell_type == "markdown" and c.source.startswith("## ")]
    numbers = [h.split("·")[0].strip("# ").strip() for h in headings]
    assert numbers == sorted(numbers, key=int), "sections must be in order"
    joined = " ".join(headings).lower()
    for expected in ("setup", "data", "preprocessing", "model", "prediction", "caveats"):
        assert expected in joined, f"every method notebook needs a {expected} section"


def test_no_notebook_trains_anything(built):
    """A notebook that trained on open would make its own recorded metrics stale."""
    for path in built.glob("*.ipynb"):
        for cell in nbformat.read(path, as_version=4).cells:
            if cell.cell_type != "code":
                continue
            assert ".backward()" not in cell.source
            assert "optimizer.step" not in cell.source
            assert "train_classifier" not in cell.source.replace(
                "python -m backend.methods", "#"), "training is printed, not run"


def test_metrics_come_from_the_recorded_evaluation_not_from_literals(built):
    """No accuracy may be typed into a notebook: it must be read back."""
    import re

    for path in built.glob("*.ipynb"):
        for cell in nbformat.read(path, as_version=4).cells:
            if cell.cell_type == "code":
                assert not re.search(r"accuracy\s*=\s*0\.\d+", cell.source)
    overview = "\n".join(c.source for c in
                         nbformat.read(built / "00_overview.ipynb", as_version=4).cells)
    assert "metrics_for(" in overview


def test_the_checked_in_notebooks_carry_their_outputs():
    """The point of shipping them executed is that images show without running."""
    from backend import config

    shipped = config.ROOT_DIR / "notebooks"
    for name in EXPECTED:
        path = shipped / name
        if not path.exists():
            pytest.skip(f"{name} has not been generated on this machine")
        nb = json.loads(path.read_text())
        images = sum(1 for c in nb["cells"] if c["cell_type"] == "code"
                     for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
        assert images >= 2, f"{name} should show pictures without being run"


def test_each_method_notebook_looks_at_different_images(built):
    """Four notebooks that all show the same glioma teach less than four that
    show four different scans — and that is what they did before this test."""
    offsets, classes = set(), set()
    for name, method_id in EXPECTED.items():
        if not method_id:
            continue
        source = "\n".join(c.source for c in nbformat.read(built / name, as_version=4).cells)
        offset = next(line for line in source.splitlines() if line.startswith("OFFSET = "))
        focus = next(line for line in source.splitlines() if line.startswith("FOCUS = "))
        offsets.add(offset)
        classes.add(focus)
    assert len(offsets) == 4, "every method must pick its examples from a different position"
    assert len(classes) == 4, "every method must demonstrate a different class"


def test_the_shipped_notebooks_share_no_rendered_image():
    import hashlib

    from backend import config

    shipped = config.ROOT_DIR / "notebooks"
    rendered = {}
    for name, method_id in EXPECTED.items():
        if not method_id:
            continue  # the overview runs all four on one image, by design
        if not (shipped / name).exists():
            pytest.skip("notebooks have not been generated on this machine")
        nb = json.loads((shipped / name).read_text())
        rendered[name] = {
            hashlib.sha1(o["data"]["image/png"].encode()).hexdigest()
            for c in nb["cells"] if c["cell_type"] == "code"
            for o in c.get("outputs", []) if "image/png" in o.get("data", {})
        }
    everywhere = set.intersection(*rendered.values())
    assert not everywhere, f"{len(everywhere)} image(s) appear in all four notebooks"
