"""Method 2 pipeline description, shared by the API, the UI and the PDF report."""
from __future__ import annotations

from backend.methods.registry import METHOD2

SPEC = METHOD2


def stages() -> list[dict]:
    return [
        {"id": s.id, "label": s.label, "kind": s.kind, "description": s.description}
        for s in METHOD2.pipeline
    ]


def flow() -> list[str]:
    """Condensed flow used by the architecture card in the frontend."""
    from backend import config

    return [
        "Input",
        "Grayscale/Filtering",
        "Multi-class Segmentation",
        f"{config.METHOD2_MODALITY} Feature Stage",
        "DCN",
        "Classes",
    ]
