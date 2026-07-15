"""PDF report generation for a single prediction using ReportLab."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


def generate_report(
    out_path: Path,
    original_path: Path,
    mask_path: Path,
    overlay_path: Path | None,
    prediction: str,
    confidence: float,
    inference_time: str,
    model_used: str = "U-Net + ConvLSTM",
) -> Path:
    """Render a one-page clinical-style PDF report and return its path."""
    c = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4

    # Header
    c.setFillColor(colors.HexColor("#0f766e"))
    c.rect(0, height - 3 * cm, width, 3 * cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(2 * cm, height - 1.9 * cm, "Brain Tumor Analysis Report")
    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, height - 2.5 * cm, f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")

    # Prediction summary block
    y = height - 4.5 * cm
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2 * cm, y, "Prediction Summary")
    c.setFont("Helvetica", 11)
    rows = [
        ("Tumor Class", prediction),
        ("Confidence", f"{confidence:.2f} %"),
        ("Inference Time", inference_time),
        ("Model Used", model_used),
    ]
    for i, (k, v) in enumerate(rows):
        yy = y - (i + 1) * 0.8 * cm
        c.setFont("Helvetica-Bold", 11)
        c.drawString(2 * cm, yy, f"{k}:")
        c.setFont("Helvetica", 11)
        c.drawString(6 * cm, yy, str(v))

    # Images
    img_y = y - 5.5 * cm
    img_w = 5.0 * cm
    labels = [("Original MRI", original_path), ("Segmentation", mask_path)]
    if overlay_path is not None:
        labels.append(("Grad-CAM", overlay_path))
    for i, (label, path) in enumerate(labels):
        x = 2 * cm + i * (img_w + 0.6 * cm)
        try:
            c.drawImage(str(path), x, img_y, width=img_w, height=img_w, preserveAspectRatio=True)
        except Exception:
            c.rect(x, img_y, img_w, img_w)
        c.setFont("Helvetica", 9)
        c.drawString(x, img_y - 0.5 * cm, label)

    # Disclaimer
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.grey)
    c.drawString(
        2 * cm, 1.5 * cm,
        "For research and educational use only. Not a substitute for professional medical diagnosis.",
    )
    c.showPage()
    c.save()
    return out_path
