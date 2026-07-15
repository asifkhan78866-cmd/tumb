"""Command-line inference on a single MRI image.

Run:
    python -m backend.predict path/to/mri.png
Outputs the predicted class, confidence, inference time, and writes the mask,
Grad-CAM overlay and a PDF report next to the input file.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from backend import config
from backend.services.inference import engine
from backend.utils.report import generate_report


def main():
    ap = argparse.ArgumentParser(description="Single-image brain-tumor inference")
    ap.add_argument("image", help="Path to an MRI image (png/jpg/tif)")
    ap.add_argument("--outdir", default=None, help="Output directory (default: alongside input)")
    ap.add_argument("--report", action="store_true", help="Also generate a PDF report")
    args = ap.parse_args()

    src = Path(args.image)
    if not src.exists():
        print(f"ERROR: file not found: {src}", file=sys.stderr)
        sys.exit(1)

    result = engine.predict(src.read_bytes())
    outdir = Path(args.outdir) if args.outdir else src.parent
    outdir.mkdir(parents=True, exist_ok=True)

    mask_out = outdir / f"{src.stem}_mask.png"
    overlay_out = outdir / f"{src.stem}_gradcam.png"
    shutil.copy(result.mask_path, mask_out)
    shutil.copy(result.overlay_path, overlay_out)

    summary = {
        "class": result.prediction,
        "confidence": f"{result.confidence:.2f}%",
        "inference_time": f"{result.inference_time_s:.2f} sec",
        "probabilities": result.probabilities,
        "mask": str(mask_out),
        "gradcam": str(overlay_out),
        "weights": {
            "segmentation_trained": engine.seg_weights_loaded,
            "classifier_trained": engine.cls_weights_loaded,
        },
    }
    print(json.dumps(summary, indent=2))

    if args.report:
        report_out = outdir / f"{src.stem}_report.pdf"
        generate_report(
            out_path=report_out,
            original_path=result.original_path,
            mask_path=result.mask_path,
            overlay_path=result.overlay_path,
            prediction=result.prediction,
            confidence=result.confidence,
            inference_time=f"{result.inference_time_s:.2f} sec",
        )
        print(f"Report written to {report_out}")


if __name__ == "__main__":
    main()
