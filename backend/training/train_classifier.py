"""DEPRECATED — delegates to Method 1's classifier training.

This module used to own its own training loop with a random slice-level split
and a geometry that did not match inference. Both are fixed in
``backend.methods.method1.training.train_classifier``, so this file forwards to
it rather than keeping a second, worse implementation alive.

    python -m backend.methods.method1.training.train_classifier
"""
from __future__ import annotations

import sys

from backend.methods.method1.training.train_classifier import main as _main


def main() -> int:
    print(
        "[deprecated] backend.training.train_classifier now delegates to\n"
        "             backend.methods.method1.training.train_classifier\n"
        "             (patient-grouped splits, held-out test set, shared geometry).\n",
        file=sys.stderr,
    )
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
