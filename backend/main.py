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
from backend.api.methods_routes import router as methods_router  # noqa: E402
from backend.api.routes import router  # noqa: E402

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=(
        "Two independent brain-tumor analysis methods behind one API.\n\n"
        "* **Method 1** — U-Net segmentation → ROI crop → ConvLSTM classification, "
        "with SFLA hyper-parameter optimisation and Grad-CAM.\n"
        "* **Method 2** — multi-class segmentation → SPECT feature stage → Dense "
        "Convolutional Network classification.\n\n"
        "Use `/api/methods` to discover both, and `/api/predict/{method_id}` to run "
        "one. Metrics are never mixed between methods, and a stage without trained "
        "weights reports itself unavailable rather than returning an untrained "
        "model's output.\n\n"
        "Research / decision-support only — not a clinical diagnostic device."
    ),
)

# CORS. In development the frontend often cannot use port 3000 (another project
# may already own it), so any localhost/127.0.0.1 port is accepted there —
# otherwise every API call fails as an opaque browser-level "failed to fetch"
# that looks like the backend being down. Production keeps the explicit
# ALLOWED_ORIGINS list; the regex is never applied outside development.
_cors: dict = {
    "allow_origins": config.ALLOWED_ORIGINS,
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}
if config.APP_ENV == "development":
    _cors["allow_origin_regex"] = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"

app.add_middleware(CORSMiddleware, **_cors)

# Serve generated images so the frontend can display them by URL.
app.mount("/predictions", StaticFiles(directory=str(config.PREDICTIONS_DIR)), name="predictions")
app.mount("/uploads", StaticFiles(directory=str(config.UPLOADS_DIR)), name="uploads")

app.include_router(router)
app.include_router(methods_router)


@app.get("/", tags=["system"])
def root():
    from backend.methods.registry import METHOD_IDS

    return {
        "name": config.API_TITLE,
        "version": config.API_VERSION,
        "docs": "/docs",
        "methods": list(METHOD_IDS),
        "methods_endpoint": "/api/methods",
        "config": config.summary(),
        "disclaimer": (
            "Research / decision-support output only. Not a clinical diagnosis and "
            "not a certified medical device."
        ),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=config.BACKEND_HOST, port=config.BACKEND_PORT, reload=True)
