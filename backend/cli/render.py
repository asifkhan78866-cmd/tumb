"""Draw images in a terminal.

Two ways, because terminals differ:

``inline``
    The iTerm2 inline-image escape. iTerm2, WezTerm and recent VS Code
    terminals (with ``terminal.integrated.enableImages``) draw the real image.
    A terminal that does not understand it prints the base64 payload as noise,
    so this is only used when the terminal is known to support it.

``blocks``
    Half-block rendering: each character cell is ``▀`` with the upper pixel as
    the foreground colour and the lower pixel as the background colour, so one
    cell carries two pixels and the aspect ratio comes out square. This needs
    nothing from the terminal but colour, and is the default for that reason.

Colour depth is detected once: 24-bit where the terminal advertises it, the
256-colour cube otherwise.
"""
from __future__ import annotations

import base64
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

__all__ = ["supports_inline", "inline", "blocks", "montage", "show", "terminal_width"]

UPPER_HALF = "▀"
RESET = "\033[0m"


def terminal_width(default: int = 100) -> int:
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except OSError:
        return default


def supports_inline() -> bool:
    """True when the terminal understands the iTerm2 inline-image escape."""
    if os.environ.get("TERM_PROGRAM") in {"iTerm.app", "WezTerm"}:
        return True
    return os.environ.get("LC_TERMINAL", "") == "iTerm2"


def _truecolour() -> bool:
    if os.environ.get("COLORTERM", "").lower() in {"truecolor", "24bit"}:
        return True
    # VS Code's terminal and iTerm2 are 24-bit even when COLORTERM is unset.
    return os.environ.get("TERM_PROGRAM") in {"vscode", "iTerm.app", "WezTerm", "Apple_Terminal"}


def _cube(value: int) -> int:
    """One channel onto the 6 levels of the 256-colour cube."""
    return min(5, max(0, (value - 35) // 40 + (1 if value >= 75 else 0)))


def _ansi(rgb: tuple[int, int, int], background: bool) -> str:
    layer = 48 if background else 38
    r, g, b = (int(v) for v in rgb)
    if _truecolour():
        return f"\033[{layer};2;{r};{g};{b}m"
    if abs(r - g) < 12 and abs(g - b) < 12:  # grey ramp reads better than the cube
        index = 232 + min(23, max(0, (r - 8) // 10))
    else:
        index = 16 + 36 * _cube(r) + 6 * _cube(g) + _cube(b)
    return f"\033[{layer};5;{index}m"


def blocks(image: np.ndarray, columns: Optional[int] = None) -> str:
    """An image as coloured half-block characters, ready to print.

    ``image`` is BGR or greyscale, any size. Two pixel rows per text row.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    columns = max(8, columns or min(terminal_width() - 2, 96))
    height, width = image.shape[:2]
    rows = max(2, int(round(columns * height / width / 2)) * 2)
    resized = cv2.resize(image, (columns, rows), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    lines = []
    for y in range(0, rows - 1, 2):
        parts = []
        for x in range(columns):
            top, bottom = rgb[y, x], rgb[y + 1, x]
            parts.append(f"{_ansi(tuple(top), False)}{_ansi(tuple(bottom), True)}{UPPER_HALF}")
        lines.append("".join(parts) + RESET)
    return "\n".join(lines)


def inline(path: Path, height_rows: int = 16) -> str:
    payload = base64.b64encode(Path(path).read_bytes()).decode()
    return (f"\033]1337;File=inline=1;height={height_rows};preserveAspectRatio=1:"
            f"{payload}\a")


def montage(images: Sequence[np.ndarray], labels: Sequence[str], panel: int = 220,
            gap: int = 6) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Panels side by side at a common height, plus where each label starts.

    Returns ``(image, [(label, x_offset_in_panels)])`` so the caller can write a
    caption row that lines up with the panels.
    """
    prepared = []
    for image in images:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        height, width = image.shape[:2]
        scaled = cv2.resize(image, (max(1, int(width * panel / height)), panel),
                            interpolation=cv2.INTER_AREA)
        prepared.append(scaled)

    separator = np.zeros((panel, gap, 3), np.uint8)
    canvas, offsets, x = [], [], 0
    for i, image in enumerate(prepared):
        if i:
            canvas.append(separator)
            x += gap
        offsets.append(x)
        canvas.append(image)
        x += image.shape[1]
    combined = np.hstack(canvas)
    return combined, list(zip(labels, offsets))


def caption_row(placement: Iterable[tuple[str, int]], total_width: int, columns: int) -> str:
    """A text row whose labels sit under their panels."""
    line = [" "] * columns
    for label, offset in placement:
        start = min(columns - 1, max(0, round(offset * columns / total_width)))
        for i, char in enumerate(label[: columns - start]):
            line[start + i] = char
    return "".join(line).rstrip()


def show(paths: Sequence[Path], labels: Sequence[str], mode: str = "blocks",
         columns: Optional[int] = None, stream=None) -> bool:
    """Draw the images, captioned. Returns False when there was nothing to draw."""
    stream = stream or sys.stdout
    images, kept_labels, kept_paths = [], [], []
    for path, label in zip(paths, labels):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        images.append(image)
        kept_labels.append(label)
        kept_paths.append(path)
    if not images:
        return False

    if mode == "inline":
        for path, label in zip(kept_paths, kept_labels):
            stream.write(f"  {label}\n")
            stream.write(inline(path))
            stream.write("\n")
        stream.flush()
        return True

    columns = max(24, columns or min(terminal_width() - 4, 100))
    combined, placement = montage(images, kept_labels)
    stream.write("  " + caption_row(placement, combined.shape[1], columns) + "\n")
    for line in blocks(combined, columns).split("\n"):
        stream.write("  " + line + "\n")
    stream.flush()
    return True
