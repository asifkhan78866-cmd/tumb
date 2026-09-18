"""Per-method image galleries: training plots and dataset samples.

    GET /api/gallery/{method_id}/plots            what plots exist, with URLs
    GET /api/gallery/{method_id}/plots/{key}      one plot image
    GET /api/gallery/{method_id}/dataset          sample images, with URLs
    GET /api/gallery/{method_id}/dataset/{cls}/{i}  one sample image

Two rules hold throughout:

* **No path comes from the client.** A request names a *key* (``confusion_matrix``)
  or a class and an index; the server resolves that to a file it chose. Nothing
  concatenates user input into a path, and every resolved file is checked to sit
  under the directory it is supposed to be in before it is served.
* **Only what exists is advertised.** A plot that was never produced is absent
  from the listing rather than returned as a broken image, so the UI shows what a
  method actually has rather than implying missing artefacts exist.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Path as PathParam, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend import config
from backend.methods.registry import get_method

router = APIRouter(prefix="/api/gallery", tags=["gallery"])

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

# key -> (relative path under backend/logs, label, what the picture shows)
_PLOTS: dict[str, dict[str, tuple[str, str, str]]] = {
    "method1": {
        "confusion_matrix": ("method1_cls_confusion_matrix.png", "Confusion matrix (held-out test)",
                             "Rows are the true class, columns the prediction."),
        "roc": ("method1_cls_roc.png", "ROC curves (held-out test)", "One-vs-rest, with per-class AUC."),
        "accuracy": ("method1_cls_acc.png", "Accuracy per epoch", "Training vs validation."),
        "loss": ("method1_cls_loss.png", "Loss per epoch", "Training vs validation."),
    },
    "method2": {
        "confusion_matrix": ("method2_cls_confusion_matrix.png", "Confusion matrix (held-out test)",
                             "Rows are the true class, columns the prediction."),
        "roc": ("method2_cls_roc.png", "ROC curves (held-out test)", "One-vs-rest, with per-class AUC."),
        "accuracy": ("method2_cls_acc.png", "Accuracy per epoch", "Training vs validation."),
        "loss": ("method2_cls_loss.png", "Loss per epoch", "Training vs validation."),
    },
    "method3": {
        "confusion_matrix": ("method3_cls_confusion_matrix.png", "Confusion matrix (held-out test)",
                             "Rows are the true class, columns the prediction."),
        "roc": ("method3_cls_roc.png", "ROC curves (held-out test)", "One-vs-rest, with per-class AUC."),
        "accuracy": ("method3/accuracy.png", "Accuracy per epoch", "Selected backbone, training vs validation."),
        "loss": ("method3/train_loss.png", "Training loss per epoch", "Selected backbone."),
    },
    "method4": {
        "confusion_matrix": ("method4_cls_confusion_matrix.png", "Confusion matrix (held-out test)",
                             "Rows are the true class, columns the prediction."),
        "roc": ("method4_cls_roc.png", "ROC curves (held-out test)", "One-vs-rest, with per-class AUC."),
        "accuracy": ("method4/accuracy.png", "Accuracy per epoch", "Training vs validation."),
        "loss": ("method4/train_loss.png", "Training loss per epoch", "ZFNet trained from scratch."),
    },
}

# Folder names differ from a method's class keys where the wording differs
# (this dataset calls the tumour-free class "notumor"; Method 4 calls it "normal").
_CLASS_FOLDER = {"normal": "notumor"}


class PlotImage(BaseModel):
    key: str
    label: str
    description: str = ""
    url: str


class SampleImage(BaseModel):
    class_key: str
    class_label: str
    url: str


class DatasetSamples(BaseModel):
    dataset: str
    path: str
    present: bool
    split: str = "Testing"
    note: str = ""
    images: list[SampleImage] = []


def _spec_or_404(method_id: str):
    try:
        return get_method(method_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None


def _under(path: Path, root: Path) -> bool:
    """True when ``path`` really sits inside ``root`` (symlinks resolved)."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _class_dir(method_id: str, class_key: str) -> Optional[Path]:
    spec = _spec_or_404(method_id)
    if class_key not in spec.class_names:
        return None
    folder = _CLASS_FOLDER.get(class_key, class_key)
    root = config.BRI_DATASET_PATH / "Testing"
    candidate = root / folder
    if not (candidate.is_dir() and _under(candidate, root)):
        return None
    return candidate


def _sample_files(directory: Path, count: int) -> list[Path]:
    """``count`` files spread evenly through the folder, in a stable order."""
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files or count <= 0:
        return []
    if len(files) <= count:
        return files
    step = len(files) / count
    return [files[int(i * step)] for i in range(count)]


@router.get("/{method_id}/plots", response_model=list[PlotImage])
def list_plots(method_id: str = PathParam(...)) -> list[PlotImage]:
    """Training and evaluation plots this method has actually produced."""
    _spec_or_404(method_id)
    out: list[PlotImage] = []
    for key, (rel, label, description) in _PLOTS.get(method_id, {}).items():
        if (config.LOGS_DIR / rel).exists():
            out.append(PlotImage(key=key, label=label, description=description,
                                 url=f"/api/gallery/{method_id}/plots/{key}"))
    return out


@router.get("/{method_id}/plots/{key}")
def get_plot(method_id: str = PathParam(...), key: str = PathParam(...)):
    _spec_or_404(method_id)
    entry = _PLOTS.get(method_id, {}).get(key)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No plot '{key}' for {method_id}.")
    path = config.LOGS_DIR / entry[0]
    if not path.exists() or not _under(path, config.LOGS_DIR):
        raise HTTPException(status_code=404, detail=f"Plot '{key}' has not been produced yet.")
    return FileResponse(str(path), media_type="image/png")


@router.get("/{method_id}/dataset", response_model=DatasetSamples)
def list_dataset_samples(
    method_id: str = PathParam(...),
    per_class: int = Query(3, ge=1, le=8),
) -> DatasetSamples:
    """Sample images from the dataset's Testing split, one row per class."""
    spec = _spec_or_404(method_id)
    root = config.BRI_DATASET_PATH
    note = ("Images from the dataset's own Testing split — the held-out set these metrics "
            "were measured on. They are examples only, not predictions.")
    if spec.method_id == "method2":
        note += (" Method 4's SPECT branch has no dataset; these are the MRI images its MRI "
                 "branch was trained and tested on.")
    samples: list[SampleImage] = []
    for class_key in spec.class_names:
        directory = _class_dir(method_id, class_key)
        if directory is None:
            continue
        for i in range(len(_sample_files(directory, per_class))):
            samples.append(SampleImage(
                class_key=class_key,
                class_label=spec.class_labels[class_key],
                url=f"/api/gallery/{method_id}/dataset/{class_key}/{i}?per_class={per_class}",
            ))
    return DatasetSamples(
        dataset="Brain Tumor MRI Dataset (BRI)",
        path=str(root),
        present=bool(samples),
        note=note if samples else "Dataset not present on this machine — run scripts/download_dataset.sh.",
        images=samples,
    )


@router.get("/{method_id}/dataset/{class_key}/{index}")
def get_dataset_sample(
    method_id: str = PathParam(...),
    class_key: str = PathParam(...),
    index: int = PathParam(..., ge=0, le=7),
    per_class: int = Query(3, ge=1, le=8),
):
    directory = _class_dir(method_id, class_key)
    if directory is None:
        raise HTTPException(status_code=404, detail=f"No class '{class_key}' for {method_id}.")
    files = _sample_files(directory, per_class)
    if index >= len(files):
        raise HTTPException(status_code=404, detail="No such sample image.")
    path = files[index]
    if not _under(path, config.BRI_DATASET_PATH):
        raise HTTPException(status_code=404, detail="No such sample image.")
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(str(path), media_type=media)
