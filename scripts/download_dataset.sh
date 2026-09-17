#!/usr/bin/env bash
# Download the Brain Tumor MRI Dataset (Kaggle) to data/bri/archive/{Training,Testing}.
#
# The images are not stored in this repository. You need a Kaggle API token:
#   https://www.kaggle.com/settings -> "Create New API Token"
# then either set KAGGLE_USERNAME and KAGGLE_KEY in .env, or place kaggle.json
# at ~/.kaggle/kaggle.json (chmod 600).
#
# Usage (from the repository root):
#   scripts/download_dataset.sh            # download and verify
#   scripts/download_dataset.sh --dry-run  # show what would happen
#   scripts/download_dataset.sh --force    # replace an existing copy
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
"$PY" -m backend.utils.dataset_download --kind classification "$@"
"$PY" -m backend.methods.method1.inventory
