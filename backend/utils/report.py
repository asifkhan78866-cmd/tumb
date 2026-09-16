"""PDF report generation for a single prediction using ReportLab.

The report is method-aware: it names the method that produced the result, prints
that method's full pipeline and dataset context, and carries every warning
attached to the prediction — including "segmentation unavailable" and "this model
is not trained". A reader holding only the PDF can tell what was computed, what
was not, and on what data.

Metrics printed here are the ones stored for *that method*. Metrics that were
never computed print as "not evaluated", never as 0.00.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

DISCLAIMER_LINES = [
    "RESEARCH / DECISION-SUPPORT OUTPUT — NOT A CLINICAL DIAGNOSIS.",
    "This system is not a certified medical device and has not been clinically validated.",
    "It must not be used to diagnose, treat, or make any care decision for a patient.",
    "All findings require interpretation by a qualified radiologist.",
]

_METRIC_ROWS = [
    ("Dice", "dice"), ("IoU", "iou"), ("Accuracy", "accuracy"),
    ("Precision", "precision"), ("Recall / Sensitivity", "recall"),
    ("Specificity", "specificity"), ("F1", "f1"), ("AUC", "auc"),
]


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "not evaluated"
    try:
        return f"{float(value) * 100:.2f} %"
    except (TypeError, ValueError):
        return str(value)


def _wrap(c: canvas.Canvas, text: str, x: float, y: float, width: float,
          font: str = "Helvetica", size: int = 9, leading: float = 0.42) -> float:
    """Draw ``text`` wrapped to ``width``; returns the new y cursor."""
    c.setFont(font, size)
    words, line = text.split(), ""
    for word in words:
        trial = f"{line} {word}".strip()
        if c.stringWidth(trial, font, size) <= width:
            line = trial
        else:
            c.drawString(x, y, line)
            y -= leading * cm
            line = word
    if line:
        c.drawString(x, y, line)
        y -= leading * cm
    return y


def generate_report(
    out_path: Path,
    original_path: Optional[Path],
    mask_path: Optional[Path],
    overlay_path: Optional[Path],
    prediction: Optional[str],
    confidence: Optional[float],
    inference_time: str,
    model_used: str = "",
    method_spec: Any = None,
    segmentation_available: bool = True,
    probabilities: Optional[dict] = None,
    metrics: Optional[dict] = None,
    model_version: str = "",
    warnings: Optional[list[str]] = None,
) -> Path:
    """Render a one-page research report and return its path."""
    probabilities = probabilities or {}
    warnings = list(warnings or [])
    metrics = metrics or {}

    method_name = getattr(method_spec, "display_name", None) or model_used or "U-Net + ConvLSTM"
    short_name = getattr(method_spec, "short_name", None) or method_name
    pipeline = " → ".join(getattr(method_spec, "pipeline_labels", lambda: [])()) or "—"
    datasets = "; ".join(
        f"{d.name} ({d.role})" for d in getattr(method_spec, "datasets", ()) if d.compatible
    ) or "not recorded"
    modality = getattr(method_spec, "modality", "")

    c = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4
    left, text_w = 2 * cm, width - 4 * cm

    # ---- Header -------------------------------------------------------- #
    c.setFillColor(colors.HexColor("#0f766e"))
    c.rect(0, height - 3 * cm, width, 3 * cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(left, height - 1.6 * cm, "Brain Tumor Analysis Report")
    c.setFont("Helvetica", 9)
    c.drawString(left, height - 2.2 * cm, f"{short_name}   ·   modality: {modality or 'n/a'}")
    c.drawString(left, height - 2.7 * cm,
                 f"Generated {datetime.now():%Y-%m-%d %H:%M:%S}   ·   model version: {model_version or 'untrained'}")

    y = height - 3.8 * cm
    c.setFillColor(colors.black)

    # ---- Method & pipeline --------------------------------------------- #
    c.setFont("Helvetica-Bold", 11)
    c.drawString(left, y, "Method")
    y -= 0.45 * cm
    y = _wrap(c, method_name, left, y, text_w, size=9)
    y = _wrap(c, f"Pipeline: {pipeline}", left, y, text_w, size=8)
    y = _wrap(c, f"Dataset context: {datasets}", left, y, text_w, size=8)
    y -= 0.25 * cm

    # ---- Prediction ----------------------------------------------------- #
    c.setFont("Helvetica-Bold", 11)
    c.drawString(left, y, "Prediction")
    y -= 0.5 * cm
    rows = [
        ("Tumor Class", prediction if prediction else "NOT AVAILABLE — model not trained"),
        ("Confidence", f"{float(confidence):.2f} %" if confidence is not None else "n/a"),
        ("Inference Time", inference_time),
        ("Segmentation", "available" if segmentation_available else "UNAVAILABLE — no trained weights"),
    ]
    for key, value in rows:
        c.setFont("Helvetica-Bold", 9)
        c.drawString(left, y, f"{key}:")
        c.setFont("Helvetica", 9)
        if key == "Segmentation" and not segmentation_available:
            c.setFillColor(colors.HexColor("#b45309"))
        if key == "Tumor Class" and not prediction:
            c.setFillColor(colors.HexColor("#b45309"))
        c.drawString(left + 4 * cm, y, str(value))
        c.setFillColor(colors.black)
        y -= 0.42 * cm

    if probabilities:
        y -= 0.1 * cm
        c.setFont("Helvetica-Bold", 9)
        c.drawString(left, y, "Class probabilities:")
        c.setFont("Helvetica", 9)
        y -= 0.4 * cm
        for name, prob in probabilities.items():
            try:
                c.drawString(left + 0.4 * cm, y, f"{name}: {float(prob):.2f} %")
            except (TypeError, ValueError):
                c.drawString(left + 0.4 * cm, y, f"{name}: {prob}")
            y -= 0.36 * cm

    # ---- Images --------------------------------------------------------- #
    y -= 0.2 * cm
    img_w = 4.4 * cm
    panels = [("Original", original_path)]
    panels.append(("Segmentation" if segmentation_available else "Segmentation (unavailable)", mask_path))
    if overlay_path is not None:
        panels.append(("Grad-CAM", overlay_path))

    img_y = y - img_w
    for i, (label, path) in enumerate(panels):
        x = left + i * (img_w + 0.5 * cm)
        drawn = False
        if path is not None and Path(path).exists():
            try:
                c.drawImage(str(path), x, img_y, width=img_w, height=img_w, preserveAspectRatio=True)
                drawn = True
            except Exception:
                drawn = False
        if not drawn:
            c.setStrokeColor(colors.HexColor("#cbd5e1"))
            c.rect(x, img_y, img_w, img_w)
            c.setFont("Helvetica-Oblique", 7)
            c.setFillColor(colors.HexColor("#94a3b8"))
            c.drawCentredString(x + img_w / 2, img_y + img_w / 2, "not available")
            c.setFillColor(colors.black)
            c.setStrokeColor(colors.black)
        c.setFont("Helvetica", 7.5)
        c.drawString(x, img_y - 0.4 * cm, label)
    y = img_y - 1.0 * cm

    # ---- Method metrics -------------------------------------------------- #
    c.setFont("Helvetica-Bold", 11)
    c.drawString(left, y, f"{short_name} — held-out metrics")
    y -= 0.45 * cm
    if not metrics.get("evaluated"):
        c.setFillColor(colors.HexColor("#b45309"))
        y = _wrap(c, "This method has not been evaluated. No metrics are reported.",
                  left, y, text_w, "Helvetica-Bold", 9)
        c.setFillColor(colors.black)
    else:
        c.setFont("Helvetica", 8.5)
        col = 0
        start_y = y
        for label, key in _METRIC_ROWS:
            x = left + (col % 2) * (text_w / 2)
            row_y = start_y - (col // 2) * 0.38 * cm
            c.drawString(x, row_y, f"{label}: {_fmt_metric(metrics.get(key))}")
            col += 1
        y = start_y - ((col + 1) // 2) * 0.38 * cm - 0.1 * cm
        y = _wrap(c, f"Split: {metrics.get('split', 'n/a')}", left, y, text_w, size=8)

    # ---- Warnings -------------------------------------------------------- #
    all_warnings = warnings + list(metrics.get("warnings", []) or [])
    if all_warnings:
        y -= 0.2 * cm
        c.setFillColor(colors.HexColor("#b45309"))
        c.setFont("Helvetica-Bold", 9)
        c.drawString(left, y, "Warnings")
        y -= 0.4 * cm
        for w in all_warnings[:6]:
            y = _wrap(c, f"• {w}", left, y, text_w, "Helvetica", 7.5, leading=0.33)
        c.setFillColor(colors.black)

    # ---- Disclaimer ------------------------------------------------------ #
    c.setStrokeColor(colors.HexColor("#b45309"))
    c.setFillColor(colors.HexColor("#fef3c7"))
    c.rect(left - 0.3 * cm, 1.0 * cm, text_w + 0.6 * cm, 2.1 * cm, fill=1, stroke=1)
    c.setFillColor(colors.HexColor("#7c2d12"))
    dy = 2.75 * cm
    for i, line in enumerate(DISCLAIMER_LINES):
        c.setFont("Helvetica-Bold" if i == 0 else "Helvetica", 7.5 if i else 8)
        c.drawString(left, dy, line)
        dy -= 0.42 * cm

    c.showPage()
    c.save()
    return out_path
