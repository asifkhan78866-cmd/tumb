"""Training run cards — the provenance record for every trained checkpoint.

A metric without its provenance is not a result. Each training entry point
writes one JSON run card next to the checkpoint recording exactly what produced
it: dataset, split strategy, seed, preprocessing, architecture, optimiser,
epochs, best epoch, metrics, checkpoint path, timing, version and any warnings
raised along the way (unverifiable patient grouping, weak masks, and so on).

The ``warnings`` list travels all the way through to the API response and the
PDF report, so a caveat recorded at training time cannot be lost by the time a
reader sees the number.
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

__all__ = ["RunCard", "write_run_card", "read_run_card", "git_revision"]


def git_revision(root: Optional[Path] = None) -> str:
    """Short git SHA of the working tree, or ``"unknown"`` outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root) if root else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@dataclass
class RunCard:
    method_id: str
    stage: str  # "segmentation" | "classification" | "sfla"
    dataset: dict = field(default_factory=dict)
    split_strategy: dict = field(default_factory=dict)
    random_seed: int = 0
    preprocessing: dict = field(default_factory=dict)
    image_size: int = 0
    architecture: str = ""
    model_config: dict = field(default_factory=dict)
    optimizer: dict = field(default_factory=dict)
    epochs: int = 0
    best_epoch: int = 0
    metrics: dict = field(default_factory=dict)
    checkpoint_path: str = ""
    inference_time_s: Optional[float] = None
    train_duration_s: Optional[float] = None
    model_version: str = ""
    created_at: str = ""
    git_revision: str = ""
    device: str = ""
    python: str = ""
    warnings: list[str] = field(default_factory=list)

    def finalise(self, root: Optional[Path] = None) -> "RunCard":
        self.created_at = self.created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.git_revision = self.git_revision or git_revision(root)
        self.python = self.python or platform.python_version()
        self.model_version = self.model_version or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def write_run_card(path: Path, card: RunCard, root: Optional[Path] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card.finalise(root).to_dict(), indent=2, default=str))
    print(f"[runcard] wrote {path}")
    return path


def read_run_card(path: Path) -> Optional[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
