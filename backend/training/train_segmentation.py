"""DEPRECATED — delegates to Method 1's segmentation training.

The previous implementation split BraTS slices at random, putting neighbouring
slices of one volume on both sides of the split. Method 1's trainer splits by
volume instead, so this file forwards to it.

    python -m backend.methods.method1.training.train_segmentation
"""
from __future__ import annotations

import sys

from backend.methods.method1.training.train_segmentation import main as _main


def main() -> int:
    print(
        "[deprecated] backend.training.train_segmentation now delegates to\n"
        "             backend.methods.method1.training.train_segmentation\n"
        "             (volume-level splits, held-out test set).\n",
        file=sys.stderr,
    )
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
