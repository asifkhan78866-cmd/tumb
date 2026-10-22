# Backend: FastAPI + PyTorch (CPU), with the trained checkpoints baked in.
#
# The image is meant to run on a machine that has neither Python nor the
# dataset, so it carries everything a prediction needs: the four methods'
# weights, the metrics they were measured with, and a handful of sample images.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production

# System libraries OpenCV, matplotlib and reportlab link against.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch first, from the CPU index. The default linux/amd64 wheel bundles CUDA
# libraries and is several gigabytes; nothing in a container here uses a GPU,
# and the difference decides whether this image can be sent to someone.
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install -r backend/requirements.txt

# The code, the trained weights, the recorded metrics and the sample images.
# What is excluded is in .dockerignore — datasets, caches, environments.
COPY backend ./backend
COPY data ./data

EXPOSE 8000

# A quick sanity line at start-up: which methods this image can actually serve.
CMD ["sh", "-c", "python -c \"from backend.methods.registry import list_methods, runtime_status; [print(f'[image] {s.display_number}. {s.short_name}: ' + ('ready' if runtime_status(s)['classifier_available'] else 'NO CHECKPOINT')) for s in sorted(list_methods(), key=lambda s: s.display_number)]\" && exec uvicorn backend.main:app --host 0.0.0.0 --port 8000"]
