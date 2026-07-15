# Backend: FastAPI + PyTorch (CPU). For GPU, use an nvidia/cuda base image and
# install the matching torch build.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System libs required by OpenCV / matplotlib / reportlab.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --upgrade pip && pip install -r backend/requirements.txt

COPY backend ./backend

EXPOSE 8000
# Run from the repo root so `import backend.*` resolves.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
