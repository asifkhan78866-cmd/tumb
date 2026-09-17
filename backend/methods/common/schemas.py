"""The prediction contract shared by every method.

Both methods answer with the same envelope so the frontend can switch between
them without special-casing, and so a reader can always tell what was actually
computed from what was merely configured:

* ``segmentation_available`` is ``False`` whenever no validated segmentation
  checkpoint was loaded. A mask produced by an untrained network is never
  reported as a result — see ``PLACEHOLDER`` in the method engines.
* ``prediction`` is ``None`` when the classifier could not run at all. It is
  never filled with a guess.
* ``warnings`` carries every caveat attached to the answer: missing weights,
  unverifiable patient grouping, placeholder visualisations, modality
  mismatches. It is rendered in the UI and printed in the PDF report.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MethodPredictionResponse",
    "MethodSummary",
    "MethodDetail",
    "MethodMetricsResponse",
    "TrainRequestResponse",
]


class MethodPredictionResponse(BaseModel):
    """Uniform prediction envelope returned by ``POST /api/predict/{method_id}``."""

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    method_id: str = Field(..., examples=["method1"])
    method_name: str = Field(..., examples=["Method 1 — U-Net + ConvLSTM + SFLA"])
    prediction: Optional[str] = Field(
        None,
        description="Predicted class label, or null when no classifier ran.",
        examples=["Glioma"],
    )
    prediction_key: Optional[str] = Field(
        None, description="Stable class key for the prediction (UI comparisons)."
    )
    confidence: Optional[float] = Field(
        None, description="Probability of the predicted class, in percent."
    )
    class_probabilities: dict[str, float] = Field(default_factory=dict)
    segmentation_available: bool = False
    segmentation_mask_url: Optional[str] = None
    heatmap_url: Optional[str] = None
    original_image_url: Optional[str] = None
    processing_time_s: float = 0.0
    model_version: str = "untrained"
    dataset_context: str = ""
    modality: str = ""
    warnings: list[str] = Field(default_factory=list)
    prediction_id: str = ""
    # Extra per-method detail (e.g. Method 2 segmentation class areas, the SFLA
    # parameter set in force). Deliberately untyped so neither method has to
    # widen the shared contract to say something specific.
    details: dict[str, Any] = Field(default_factory=dict)


class DatasetSummary(BaseModel):
    name: str
    role: str
    source: str = ""
    present: bool = False
    path: str = ""
    notes: str = ""


class MethodSummary(BaseModel):
    """Compact record returned by ``GET /api/methods``."""

    model_config = ConfigDict(protected_namespaces=())

    method_id: str
    display_name: str
    short_name: str
    summary: str
    modality: str
    num_classes: int
    class_labels: list[str]
    pipeline: list[str]
    segmentation_model: str
    classifier_model: str
    optimization: Optional[str] = None
    trained: bool = False
    segmentation_available: bool = False
    classifier_available: bool = False
    # Name, version and creation time of the classifier checkpoint on disk, so a
    # client can show which trained model it is talking to. None when absent.
    classifier_checkpoint: Optional[str] = None
    classifier_model_version: Optional[str] = None
    classifier_trained_at: Optional[str] = None
    # True when uploads get an AI model's assessment instead of a trained classifier.
    ai_assessment_available: bool = False
    # Position in the numbered method list (1-4); ids are stable, numbers are display.
    display_number: int = 0
    has_segmentation_stage: bool = True
    warnings: list[str] = Field(default_factory=list)


class MethodDetail(MethodSummary):
    """Full record returned by ``GET /api/methods/{method_id}``."""

    pipeline_stages: list[dict[str, Any]] = Field(default_factory=list)
    datasets: list[DatasetSummary] = Field(default_factory=list)
    weights: dict[str, Any] = Field(default_factory=dict)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    training_entrypoints: list[str] = Field(default_factory=list)
    run_cards: dict[str, Any] = Field(default_factory=dict)
    optimization_result: Optional[dict[str, Any]] = None


class MethodMetricsResponse(BaseModel):
    """Metrics for one method. Absent metrics are ``None``, never zero."""

    model_config = ConfigDict(protected_namespaces=())

    method_id: str
    method_name: str
    evaluated: bool = False
    split: str = "not evaluated"
    dice: Optional[float] = None
    iou: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    f1: Optional[float] = None
    auc: Optional[float] = None
    avg_inference_time_s: Optional[float] = None
    confusion_matrix: Optional[list[list[int]]] = None
    class_labels: list[str] = Field(default_factory=list)
    per_class: Optional[dict[str, Any]] = None
    model_version: Optional[str] = None
    dataset_context: str = ""
    warnings: list[str] = Field(default_factory=list)


class TrainRequestResponse(BaseModel):
    method_id: str
    stage: str
    started: bool
    message: str
    command: str
    run_card: Optional[dict[str, Any]] = None
