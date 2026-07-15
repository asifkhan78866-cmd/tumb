"""Shim so `python train_classifier.py` works from inside backend/.

Prefer `python -m backend.training.train_classifier` from the repo root.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.training.train_classifier import main  # noqa: E402

if __name__ == "__main__":
    main()
