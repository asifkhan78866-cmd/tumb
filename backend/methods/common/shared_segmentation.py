"""Show one method's trained U-Net alongside methods that have no segmenter.

Methods 1 and 2 classify whole images and Method 4's fusion segmenter is
untrained, so none of them can outline a tumour. Rather than leave the panel
empty, they may display the mask produced by **Method 3's** U-Net.

Two properties keep this honest:

* The mask is **not part of the prediction**. It is computed after the class has
  been decided, never feeds the classifier, and the response says so.
* The mask is **attributed**, including what it was trained on, so a heuristic
  mask can never be read as a radiologist's outline.

Nothing here runs if Method 3 has no segmentation checkpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from backend import config

__all__ = ["shared_mask"]


def shared_mask(image_bytes: bytes, prediction_id: str, method_id: str
                ) -> tuple[Optional[Path], dict, list[str]]:
    """Return ``(overlay_path, details, warnings)`` for a borrowed U-Net mask."""
    try:
        from backend.methods.method1.inference import Method1Engine
        from backend.methods.method1.inference import engine as m1_engine
        from backend.methods.method1.transforms import decode_image_bytes, preprocess, to_tensor
    except Exception:  # pragma: no cover - import guard only
        return None, {}, []

    m1_engine.load()
    if not m1_engine.seg_weights_loaded or m1_engine.seg_model is None:
        return None, {}, []

    try:
        image = preprocess(decode_image_bytes(image_bytes), m1_engine.transform)
        with torch.no_grad():
            probability = torch.sigmoid(m1_engine.seg_model(to_tensor(image).to(m1_engine.device)))
        mask = (probability.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    except Exception as exc:  # pragma: no cover - display feature, never fatal
        print(f"[{method_id}] shared segmentation failed: {exc}")
        return None, {}, []

    path = config.PREDICTIONS_DIR / f"{prediction_id}_{method_id}_shared_mask.png"
    cv2.imwrite(str(path), Method1Engine._mask_overlay(image, mask))

    extra = (m1_engine.seg_meta.get("extra") or {})
    source = str(extra.get("mask_source") or "unknown")
    details = {
        "segmentation_from_other_method": "method1",  # id of the U-Net + ConvLSTM method
        "segmentation_mask_source": source,
        "segmentation_is_ground_truth_trained": bool(extra.get("ground_truth_masks", False)),
        "segmentation_label": "Tumour region (U-Net + ConvLSTM method)",
        "tumor_pixel_fraction": round(float((mask > 0).mean()), 4),
    }
    warnings = [
        f"The outline shown is produced by the U-Net + ConvLSTM method's segmentation model "
        f"(trained on {source}). It is not part of this method's pipeline and did not "
        f"influence the class predicted above."
    ]
    if not details["segmentation_is_ground_truth_trained"]:
        warnings.append(
            "That segmentation model was not trained on radiologist annotation, so the "
            "outline is approximate and is not a validated tumour boundary."
        )
    return path, details, warnings
