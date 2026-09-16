"""Method 1 pipeline description, shared by the API, the UI and the PDF report.

Kept separate from the engine so the diagram can be rendered without importing
torch or loading any weights.
"""
from __future__ import annotations

from backend.methods.registry import METHOD1

SPEC = METHOD1


def stages() -> list[dict]:
    return [
        {"id": s.id, "label": s.label, "kind": s.kind, "description": s.description}
        for s in METHOD1.pipeline
    ]


def flow() -> list[str]:
    """Condensed flow used by the architecture card in the frontend."""
    return ["Input", "Preprocess", "U-Net", "ROI Crop", "ConvLSTM", "SFLA", "Classes"]
