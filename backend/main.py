"""FastAPI application entrypoint.

Run with:
    uvicorn backend.main:app --reload
or from inside the backend/ directory:
    uvicorn main:app --reload
Swagger UI is served at /docs, ReDoc at /redoc.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Make ``import backend.*`` work regardless of the working directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.api.routes import router  # noqa: E402

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=(
        "Production-ready Brain Tumor Segmentation (U-Net) & Classification "
        "(ConvLSTM) API. Upload an MRI slice to get a tumor mask, class, "
        "confidence, Grad-CAM overlay and inference time."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated images so the frontend can display them by URL.
app.mount("/predictions", StaticFiles(directory=str(config.PREDICTIONS_DIR)), name="predictions")
app.mount("/uploads", StaticFiles(directory=str(config.UPLOADS_DIR)), name="uploads")

app.include_router(router)


@app.get("/", tags=["system"])
async def root():
    return {
        "name": config.API_TITLE,
        "version": config.API_VERSION,
        "docs": "/docs",
        "config": config.summary(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
