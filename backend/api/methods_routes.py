"""Method-aware API surface.

    GET  /api/methods                 list every registered method
    GET  /api/methods/{method_id}     full detail for one method
    POST /api/predict/{method_id}     run that method's pipeline on an upload
    POST /api/train/{method_id}       safe training endpoint (see below)
    GET  /api/metrics/{method_id}     that method's metrics, never mixed

Two deliberate choices:

*Routes are ``def``, not ``async def``.* Inference is CPU-bound and takes
hundreds of milliseconds to seconds. A synchronous path lets Starlette run it in
its worker threadpool; declaring it ``async`` would block the event loop and
serialise every other request behind it.

*Training is not launched from HTTP by default.* ``POST /api/train`` returns the
exact command to run and refuses to spawn anything unless
``TRAINING_API_ENABLED=true``, because an unauthenticated endpoint that starts a
multi-hour GPU job is a denial-of-service primitive.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Any

from fastapi import APIRouter, File, HTTPException, Path as PathParam, UploadFile
from fastapi.responses import FileResponse

from backend import config
from backend.methods.common.metrics_store import metrics_for
from backend.methods.common.runcard import read_run_card
from backend.methods.common.schemas import (
    MethodDetail,
    MethodMetricsResponse,
    MethodPredictionResponse,
    MethodSummary,
    TrainRequestResponse,
)
from backend.methods.registry import (
    METHOD_IDS,
    dataset_context,
    get_method,
    list_methods,
    load_engine,
    runtime_status,
)
from backend.services import store

router = APIRouter(prefix="/api", tags=["methods"])

ALLOWED_CONTENT = {
    "image/png", "image/jpeg", "image/jpg", "image/tiff", "image/bmp",
    "image/webp", "application/octet-stream",
}


def _spec_or_404(method_id: str):
    try:
        return get_method(method_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None


def _asset_url(path) -> str | None:
    return f"/predictions/{path.name}" if path else None


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
@router.get("/methods", response_model=list[MethodSummary])
def list_all_methods() -> list[MethodSummary]:
    out: list[MethodSummary] = []
    for spec in list_methods():
        status = runtime_status(spec)
        out.append(
            MethodSummary(
                method_id=spec.method_id,
                display_name=spec.display_name,
                short_name=spec.short_name,
                summary=spec.summary,
                modality=spec.modality,
                num_classes=spec.num_classes,
                class_labels=spec.labels,
                pipeline=spec.pipeline_labels(),
                segmentation_model=spec.segmentation_model,
                classifier_model=spec.classifier_model,
                optimization=spec.optimization,
                trained=status["trained"],
                segmentation_available=status["segmentation_available"],
                classifier_available=status["classifier_available"],
                warnings=status["warnings"] + list(spec.notes),
            )
        )
    return out


@router.get("/methods/{method_id}", response_model=MethodDetail)
def method_detail(method_id: str = PathParam(..., description=f"one of {METHOD_IDS}")) -> MethodDetail:
    spec = _spec_or_404(method_id)
    status = runtime_status(spec)

    run_cards: dict[str, Any] = {}
    for stage, filename in spec.run_card_filenames.items():
        card = read_run_card(config.LOGS_DIR / filename)
        if card:
            run_cards[stage] = card

    optimization_result = None
    if spec.optimization_result_filename:
        optimization_result = read_run_card(config.LOGS_DIR / spec.optimization_result_filename)

    preprocessing: dict[str, Any] = {}
    try:
        import importlib

        transforms = importlib.import_module(spec.transforms_module)
        preprocessing = transforms.DEFAULT_SPEC.to_dict()
    except Exception as exc:  # pragma: no cover - descriptive only
        preprocessing = {"error": f"could not describe preprocessing: {exc}"}

    return MethodDetail(
        method_id=spec.method_id,
        display_name=spec.display_name,
        short_name=spec.short_name,
        summary=spec.summary,
        modality=spec.modality,
        num_classes=spec.num_classes,
        class_labels=spec.labels,
        pipeline=spec.pipeline_labels(),
        segmentation_model=spec.segmentation_model,
        classifier_model=spec.classifier_model,
        optimization=spec.optimization,
        trained=status["trained"],
        segmentation_available=status["segmentation_available"],
        classifier_available=status["classifier_available"],
        warnings=status["warnings"] + list(spec.notes),
        pipeline_stages=[
            {"id": s.id, "label": s.label, "kind": s.kind, "description": s.description}
            for s in spec.pipeline
        ],
        datasets=status["datasets"],
        weights=status["weights"],
        preprocessing=preprocessing,
        training_entrypoints=list(spec.training_entrypoints),
        run_cards=run_cards,
        optimization_result=optimization_result,
    )


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
@router.post("/predict/{method_id}", response_model=MethodPredictionResponse)
def predict(method_id: str, file: UploadFile = File(...)) -> MethodPredictionResponse:
    """Run one method's pipeline. Synchronous on purpose — see the module docstring."""
    spec = _spec_or_404(method_id)

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
        engine = load_engine(method_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load {method_id}: {exc}") from exc

    try:
        result = engine.predict(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    response = MethodPredictionResponse(
        method_id=spec.method_id,
        method_name=spec.display_name,
        prediction=result.prediction,
        prediction_key=result.prediction_key,
        confidence=result.confidence,
        class_probabilities=result.probabilities,
        segmentation_available=result.segmentation_available,
        segmentation_mask_url=_asset_url(result.mask_path),
        heatmap_url=_asset_url(result.overlay_path),
        original_image_url=_asset_url(result.original_path),
        processing_time_s=result.inference_time_s,
        model_version=result.model_version,
        dataset_context=dataset_context(spec),
        modality=spec.modality,
        warnings=result.warnings,
        prediction_id=result.prediction_id,
        details=result.details,
    )
    store.add_prediction(
        {
            "prediction_id": response.prediction_id,
            "method_id": spec.method_id,
            "method_name": spec.short_name,
            "class": response.prediction,
            "prediction_key": response.prediction_key,
            "confidence": response.confidence,
            "inference_time": f"{response.processing_time_s:.2f} sec",
            "original_image": response.original_image_url,
            "segmentation_mask": response.segmentation_mask_url,
            "gradcam_overlay": response.heatmap_url,
            "segmentation_available": response.segmentation_available,
            "probabilities": response.class_probabilities,
            "model_version": response.model_version,
            "warnings": response.warnings,
        }
    )
    return response


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@router.get("/metrics/{method_id}", response_model=MethodMetricsResponse)
def method_metrics(method_id: str) -> MethodMetricsResponse:
    _spec_or_404(method_id)
    return MethodMetricsResponse(**metrics_for(method_id))


@router.get("/metrics", response_model=list[MethodMetricsResponse])
def all_method_metrics() -> list[MethodMetricsResponse]:
    """Every method's metrics, for the comparison page. Never merged together."""
    return [MethodMetricsResponse(**metrics_for(mid)) for mid in METHOD_IDS]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@router.post("/train/{method_id}", response_model=TrainRequestResponse)
def train(method_id: str, stage: str = "classification") -> TrainRequestResponse:
    """Start (or, by default, just describe) a training run.

    Training is long-running and resource-hungry. Unless ``TRAINING_API_ENABLED``
    is explicitly set, this returns the command to run instead of executing it.
    """
    spec = _spec_or_404(method_id)
    module_stage = {"classification": "train_classifier", "segmentation": "train_segmentation",
                    "sfla": "optimization.run_sfla"}.get(stage)
    if module_stage is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown stage '{stage}'. Use: classification, segmentation, sfla.",
        )
    if stage == "sfla" and spec.optimization is None:
        raise HTTPException(status_code=400, detail=f"{spec.short_name} has no optimisation stage.")

    module = (
        f"backend.methods.{method_id}.{module_stage}"
        if stage == "sfla"
        else f"backend.methods.{method_id}.training.{module_stage}"
    )
    command = f"{sys.executable} -m {module}"

    if not config.TRAINING_API_ENABLED:
        return TrainRequestResponse(
            method_id=method_id, stage=stage, started=False,
            message=(
                "Training over HTTP is disabled. Run the command below on the host, "
                "or set TRAINING_API_ENABLED=true to allow this endpoint to launch it."
            ),
            command=command,
            run_card=read_run_card(config.LOGS_DIR / spec.run_card_filenames.get(stage, "")),
        )

    try:
        subprocess.Popen(
            [sys.executable, "-m", module],
            cwd=str(config.ROOT_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not start training: {exc}") from exc

    return TrainRequestResponse(
        method_id=method_id, stage=stage, started=True,
        message="Training started in a detached process. Follow progress in the backend logs.",
        command=command,
    )


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
@router.get("/report/{method_id}/{prediction_id}")
def method_report(method_id: str, prediction_id: str):
    """Generate a PDF report for a stored prediction made with ``method_id``."""
    from pathlib import Path

    from backend.utils.report import generate_report

    spec = _spec_or_404(method_id)
    record = next(
        (
            r
            for r in store.get_history(200)
            if r.get("prediction_id") == prediction_id
            and r.get("method_id", "method1") == method_id
        ),
        None,
    )
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction '{prediction_id}' recorded for {method_id}.",
        )

    def _local(url: str | None):
        return config.PREDICTIONS_DIR / Path(url).name if url else None

    out = config.PREDICTIONS_DIR / f"{prediction_id}_{method_id}_report.pdf"
    generate_report(
        out_path=out,
        original_path=_local(record.get("original_image")),
        mask_path=_local(record.get("segmentation_mask")),
        overlay_path=_local(record.get("gradcam_overlay")),
        prediction=record.get("class"),
        confidence=record.get("confidence"),
        inference_time=record.get("inference_time", "n/a"),
        method_spec=spec,
        segmentation_available=bool(record.get("segmentation_available")),
        probabilities=record.get("probabilities") or {},
        metrics=metrics_for(method_id),
        model_version=record.get("model_version", "untrained"),
        warnings=record.get("warnings") or [],
    )
    return FileResponse(str(out), media_type="application/pdf", filename=out.name)
