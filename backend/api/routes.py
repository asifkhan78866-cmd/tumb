"""API routes: /upload, /health, /model-info, /train-status, /metrics, /report, /history."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend import config
from backend.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TrainStatusResponse,
)
from backend.services import store
from backend.services.inference import engine
from backend.utils.report import generate_report

router = APIRouter()

ALLOWED_CONTENT = {"image/png", "image/jpeg", "image/jpg", "image/tiff", "image/bmp", "application/octet-stream"}


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters()) if model is not None else 0


@router.post("/upload", response_model=PredictionResponse, tags=["inference"])
async def upload(file: UploadFile = File(...)):
    """Accept an MRI image and return segmentation + classification results."""
    if file.content_type not in ALLOWED_CONTENT:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    try:
        result = engine.predict(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    record = {
        "prediction_id": result.prediction_id,
        "class": result.prediction,
        "confidence": result.confidence,
        "inference_time": f"{result.inference_time_s:.2f} sec",
        "original_image": f"/predictions/{result.original_path.name}",
        "segmentation_mask": f"/predictions/{result.mask_path.name}",
        "gradcam_overlay": f"/predictions/{result.overlay_path.name}",
        "probabilities": result.probabilities,
    }
    store.add_prediction(record)
    return PredictionResponse(**{**record, "class": result.prediction})


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    engine.load()
    return HealthResponse(
        status="ok",
        device=str(config.DEVICE),
        seg_weights_loaded=engine.seg_weights_loaded,
        cls_weights_loaded=engine.cls_weights_loaded,
    )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["system"])
async def model_info():
    engine.load()
    return ModelInfoResponse(
        segmentation_model="2D U-Net (encoder-decoder, skip connections, BN, dropout)",
        classification_model="ConvLSTM (Conv + BatchNorm + ConvLSTM + Dense + Softmax)",
        classes=[config.CLASS_LABELS[c] for c in config.CLASS_NAMES],
        image_size=config.IMAGE_SIZE,
        device=str(config.DEVICE),
        parameters={
            "segmentation": count_params(engine.seg_model),
            "classification": count_params(engine.cls_model),
        },
    )


@router.get("/train-status", response_model=TrainStatusResponse, tags=["training"])
async def train_status():
    return TrainStatusResponse(**store.get_train_status())


@router.get("/metrics", tags=["metrics"])
async def metrics():
    return store.get_metrics()


@router.get("/history", tags=["metrics"])
async def history(limit: int = 50):
    return store.get_history(limit)


@router.get("/report/{prediction_id}", tags=["inference"])
async def report(prediction_id: str):
    """Generate and download a PDF report for a stored prediction."""
    records = store.get_history(200)
    rec = next((r for r in records if r.get("prediction_id") == prediction_id), None)
    if rec is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    def _p(url: str) -> Path:
        return config.PREDICTIONS_DIR / Path(url).name

    out = config.PREDICTIONS_DIR / f"{prediction_id}_report.pdf"
    generate_report(
        out_path=out,
        original_path=_p(rec["original_image"]),
        mask_path=_p(rec["segmentation_mask"]),
        overlay_path=_p(rec.get("gradcam_overlay", "")) if rec.get("gradcam_overlay") else None,
        prediction=rec["class"],
        confidence=float(rec["confidence"]),
        inference_time=rec["inference_time"],
    )
    return FileResponse(str(out), media_type="application/pdf", filename=out.name)
