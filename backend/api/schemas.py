"""Pydantic response models for the API (drives the Swagger docs)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictionResponse(BaseModel):
    """Legacy ``/upload`` shape. Optional URL fields are ``null`` when the
    corresponding stage did not run — they are never filled with a placeholder."""

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    prediction_id: str = Field(..., examples=["a1b2c3d4e5f6"])
    class_: str = Field(..., alias="class", examples=["Glioma"])
    confidence: float = Field(..., examples=[98.2])
    inference_time: str = Field(..., examples=["0.32 sec"])
    segmentation_mask: Optional[str] = Field(None, examples=["/predictions/a1b2c3d4e5f6_mask.png"])
    original_image: Optional[str] = Field(None, examples=["/predictions/a1b2c3d4e5f6_original.png"])
    gradcam_overlay: Optional[str] = Field(None, examples=["/predictions/a1b2c3d4e5f6_overlay.png"])
    segmentation_available: bool = False
    probabilities: dict = Field(default_factory=dict)
    method_id: str = "method1"
    method_name: str = ""
    prediction_key: Optional[str] = None
    model_version: str = "untrained"
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    device: str
    seg_weights_loaded: bool
    cls_weights_loaded: bool
    warnings: list[str] = Field(default_factory=list)


class ModelInfoResponse(BaseModel):
    segmentation_model: str
    classification_model: str
    classes: list[str]
    image_size: int
    device: str
    parameters: dict


class TrainStatusResponse(BaseModel):
    state: str
    message: str = ""
    seg: dict = Field(default_factory=dict)
    cls: dict = Field(default_factory=dict)
    updated_at: str | None = None


# NOTE: the per-method metrics contract lives in
# backend.methods.common.schemas.MethodMetricsResponse, where every field is
# Optional so "not evaluated" can be expressed as null rather than 0.0.
