"""Pydantic response models for the API (drives the Swagger docs)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    prediction_id: str = Field(..., example="a1b2c3d4e5f6")
    class_: str = Field(..., alias="class", example="Glioma")
    confidence: float = Field(..., example=98.2)
    inference_time: str = Field(..., example="0.32 sec")
    segmentation_mask: str = Field(..., example="/predictions/a1b2c3d4e5f6_mask.png")
    original_image: str = Field(..., example="/predictions/a1b2c3d4e5f6_original.png")
    gradcam_overlay: str = Field(..., example="/predictions/a1b2c3d4e5f6_overlay.png")
    probabilities: dict = Field(default_factory=dict)

    class Config:
        populate_by_name = True


class HealthResponse(BaseModel):
    status: str = "ok"
    device: str
    seg_weights_loaded: bool
    cls_weights_loaded: bool


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


class MetricsResponse(BaseModel):
    accuracy: float
    dice: float
    sensitivity: float
    specificity: float
    precision: float
    recall: float
    f1: float
    avg_inference_time_s: float
