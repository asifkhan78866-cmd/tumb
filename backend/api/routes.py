"""Legacy (method-unaware) API routes, kept for backwards compatibility.

These endpoints predate the two-method split. They are preserved so existing
clients keep working, and they now delegate to Method 1 rather than owning their
own inference path. New clients should use the ``/api/*`` routes in
:mod:`backend.api.methods_routes`, which carry the method id explicitly.

Behaviour changes forced by correctness:

* ``/upload`` is ``def`` rather than ``async def`` so CPU-bound inference runs in
  Starlette's threadpool instead of blocking the event loop.
* Uploads larger than ``MAX_UPLOAD_MB`` are rejected before decoding.
* The response carries ``segmentation_available``. When it is false the mask URL
  is absent — this endpoint no longer returns a random-weight mask as a result.
* When Method 1's classifier is not trained, ``/upload`` returns 503 instead of
  inventing a class.
"""
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
from backend.methods.common.metrics_store import metrics_for
from backend.methods.method1.inference import engine as method1_engine
from backend.methods.registry import METHOD1, dataset_context
from backend.services import store
from backend.utils.report import generate_report

router = APIRouter()

ALLOWED_CONTENT = {
    "image/png", "image/jpeg", "image/jpg", "image/tiff", "image/bmp",
    "image/webp", "application/octet-stream",
}


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters()) if model is not None else 0


@router.post("/upload", response_model=PredictionResponse, tags=["inference"])
def upload(file: UploadFile = File(...)):
    """Method 1 inference (legacy route). Prefer ``POST /api/predict/method1``."""
    if file.content_type not in ALLOWED_CONTENT:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")

    data = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {config.MAX_UPLOAD_MB:g} MB limit (MAX_UPLOAD_MB).",
        )

    try:
        result = method1_engine.predict(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if result.prediction is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Method 1's classifier is not trained, so no class can be returned. "
                "Train it with `python -m backend.methods.method1.training.train_classifier`. "
                "Use GET /api/methods/method1 to inspect the current state."
            ),
        )

    record = {
        "prediction_id": result.prediction_id,
        "method_id": METHOD1.method_id,
        "method_name": METHOD1.short_name,
        "class": result.prediction,
        "prediction_key": result.prediction_key,
        "confidence": result.confidence,
        "inference_time": f"{result.inference_time_s:.2f} sec",
        "original_image": f"/predictions/{result.original_path.name}" if result.original_path else None,
        "segmentation_mask": f"/predictions/{result.mask_path.name}" if result.mask_path else None,
        "gradcam_overlay": f"/predictions/{result.overlay_path.name}" if result.overlay_path else None,
        "segmentation_available": result.segmentation_available,
        "probabilities": result.probabilities,
        "model_version": result.model_version,
        "warnings": result.warnings,
    }
    store.add_prediction(record)
    return PredictionResponse(**record)


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    status = method1_engine.status()
    return HealthResponse(
        status="ok",
        device=str(config.DEVICE),
        seg_weights_loaded=status["segmentation_available"],
        cls_weights_loaded=status["classifier_available"],
        warnings=status["warnings"],
    )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["system"])
def model_info():
    method1_engine.load()
    return ModelInfoResponse(
        segmentation_model=METHOD1.segmentation_model,
        classification_model=METHOD1.classifier_model,
        classes=METHOD1.labels,
        image_size=config.IMAGE_SIZE,
        device=str(config.DEVICE),
        parameters={
            "segmentation": count_params(method1_engine.seg_model),
            "classification": count_params(method1_engine.cls_model),
        },
    )


@router.get("/train-status", response_model=TrainStatusResponse, tags=["training"])
def train_status():
    return TrainStatusResponse(**store.get_train_status())


@router.get("/metrics", tags=["metrics"])
def metrics():
    """Method 1's metrics (legacy route). Prefer ``GET /api/metrics/{method_id}``."""
    return metrics_for(METHOD1.method_id)


@router.get("/history", tags=["metrics"])
def history(limit: int = 50, method_id: str | None = None):
    items = store.get_history(max(1, min(limit, 200)))
    if method_id:
        items = [i for i in items if i.get("method_id", "method1") == method_id]
    return items


@router.get("/report/{prediction_id}", tags=["inference"])
def report(prediction_id: str):
    """PDF report for a stored prediction (legacy route)."""
    rec = next(
        (r for r in store.get_history(200) if r.get("prediction_id") == prediction_id), None
    )
    if rec is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    method_id = rec.get("method_id", METHOD1.method_id)
    from backend.methods.registry import get_method

    try:
        spec = get_method(method_id)
    except KeyError:
        spec = METHOD1

    def _local(url):
        return config.PREDICTIONS_DIR / Path(url).name if url else None

    out = config.PREDICTIONS_DIR / f"{prediction_id}_report.pdf"
    generate_report(
        out_path=out,
        original_path=_local(rec.get("original_image")),
        mask_path=_local(rec.get("segmentation_mask")),
        overlay_path=_local(rec.get("gradcam_overlay")),
        prediction=rec.get("class"),
        confidence=rec.get("confidence"),
        inference_time=rec.get("inference_time", "n/a"),
        method_spec=spec,
        segmentation_available=bool(rec.get("segmentation_available")),
        probabilities=rec.get("probabilities") or {},
        metrics=metrics_for(method_id),
        model_version=rec.get("model_version", "untrained"),
        warnings=rec.get("warnings") or [],
    )
    return FileResponse(str(out), media_type="application/pdf", filename=out.name)


@router.get("/dataset-context", tags=["system"])
def dataset_ctx():
    return {"method1": dataset_context(METHOD1)}
