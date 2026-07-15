"""Shim so `python evaluate.py` works from inside backend/.

Prefer `python -m backend.training.evaluate` from the repo root.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.training.evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
