"""Drawing images in a terminal: the right size, the right colours, and never
a wall of base64 in a terminal that cannot show an image."""
from __future__ import annotations

import io
import re

import numpy as np
import pytest

from tests.conftest import requires_cv2

pytestmark = requires_cv2

ANSI = re.compile(r"\033\[[0-9;]*m")


def _square(colour=(0, 0, 255), size=64) -> np.ndarray:
    image = np.zeros((size, size, 3), np.uint8)
    image[:, :] = colour
    return image


# --- geometry --------------------------------------------------------------- #
def test_each_cell_carries_two_pixel_rows_so_the_aspect_ratio_survives(monkeypatch):
    from backend.cli import render

    monkeypatch.setenv("COLORTERM", "truecolor")
    lines = render.blocks(_square(size=64), columns=40).split("\n")
    # 40 columns of a square image is 40 pixel rows, two per text row.
    assert len(lines) == 20
    assert all(len(ANSI.sub("", line)) == 40 for line in lines)


def test_a_wide_image_gets_fewer_rows_than_a_tall_one(monkeypatch):
    from backend.cli import render

    monkeypatch.setenv("COLORTERM", "truecolor")
    wide = render.blocks(np.zeros((40, 160, 3), np.uint8), columns=40).split("\n")
    tall = render.blocks(np.zeros((160, 40, 3), np.uint8), columns=40).split("\n")
    assert len(wide) < len(tall)


def test_greyscale_images_are_accepted(monkeypatch):
    from backend.cli import render

    monkeypatch.setenv("COLORTERM", "truecolor")
    assert render.blocks(np.full((32, 32), 128, np.uint8), columns=16)


# --- colour ----------------------------------------------------------------- #
def test_true_colour_terminals_get_the_exact_pixel(monkeypatch):
    from backend.cli import render

    monkeypatch.setenv("COLORTERM", "truecolor")
    out = render.blocks(_square((0, 0, 255)), columns=8)  # BGR red
    assert "38;2;255;0;0" in out and "48;2;255;0;0" in out


def test_a_terminal_without_true_colour_falls_back_to_the_256_cube(monkeypatch):
    from backend.cli import render

    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    out = render.blocks(_square((0, 0, 255)), columns=8)
    assert "38;5;" in out and "38;2;" not in out


def test_grey_pixels_use_the_grey_ramp_not_the_colour_cube(monkeypatch):
    from backend.cli import render

    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    out = render.blocks(_square((128, 128, 128)), columns=4)
    indices = {int(m) for m in re.findall(r"38;5;(\d+)", out)}
    assert all(index >= 232 for index in indices), "greys belong on the 232-255 ramp"


# --- montage and captions --------------------------------------------------- #
def test_panels_are_placed_in_order_at_a_common_height():
    from backend.cli import render

    combined, placement = render.montage(
        [_square(size=64), np.zeros((32, 96, 3), np.uint8)], ["a", "b"], panel=100, gap=4
    )
    assert combined.shape[0] == 100
    labels = [label for label, _ in placement]
    offsets = [offset for _, offset in placement]
    assert labels == ["a", "b"] and offsets[0] == 0 and offsets[1] > 0


def test_a_caption_sits_under_its_own_panel():
    from backend.cli import render

    row = render.caption_row([("left", 0), ("right", 500)], total_width=1000, columns=40)
    assert row.startswith("left")
    assert row.index("right") == 20


# --- choosing how to draw --------------------------------------------------- #
def test_inline_is_only_claimed_where_the_escape_is_understood(monkeypatch):
    from backend.cli import render

    monkeypatch.delenv("LC_TERMINAL", raising=False)
    for program, expected in (("iTerm.app", True), ("WezTerm", True),
                              ("vscode", False), ("Apple_Terminal", False)):
        monkeypatch.setenv("TERM_PROGRAM", program)
        assert render.supports_inline() is expected


def test_show_draws_a_caption_row_and_one_row_per_two_pixel_rows(tmp_path, monkeypatch):
    import cv2

    from backend.cli import render

    monkeypatch.setenv("COLORTERM", "truecolor")
    paths = []
    for i, colour in enumerate(((255, 0, 0), (0, 255, 0))):
        path = tmp_path / f"{i}.png"
        cv2.imwrite(str(path), _square(colour))
        paths.append(path)
    stream = io.StringIO()
    assert render.show(paths, ["input", "mask"], columns=40, stream=stream) is True
    lines = stream.getvalue().rstrip("\n").split("\n")
    assert "input" in lines[0] and "mask" in lines[0]
    assert len(lines) > 2 and "\033[38;2;" in lines[1]


def test_show_says_no_when_nothing_could_be_read(tmp_path):
    from backend.cli import render

    missing = tmp_path / "gone.png"
    assert render.show([missing], ["input"], stream=io.StringIO()) is False


def test_inline_mode_emits_the_escape_once_per_image(tmp_path, monkeypatch):
    import cv2

    from backend.cli import render

    path = tmp_path / "a.png"
    cv2.imwrite(str(path), _square())
    stream = io.StringIO()
    render.show([path, path], ["input", "mask"], mode="inline", stream=stream)
    assert stream.getvalue().count("\033]1337;File=inline=1") == 2
