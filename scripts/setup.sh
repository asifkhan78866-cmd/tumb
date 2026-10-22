#!/usr/bin/env bash
# One-time setup on a fresh clone: a virtual environment with everything the
# trained models need to run.
#
#   scripts/setup.sh
#   PYTHON=python3.12 scripts/setup.sh     # pick the interpreter yourself
#
# macOS ships no `python` on PATH, only `python3`, and this project is never
# installed into the system interpreter — everything lives in .venv, which is
# not in git because it holds gigabytes of compiled wheels.
#
# PyTorch does not publish a wheel for every (Python version, CPU) pair, and
# pip's message when none matches — "No matching distribution found" — does not
# say which half is wrong. So the combination is checked here, before anything
# is downloaded, and the fix is named.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: '$PY' is not on PATH." >&2
  echo "       macOS has no 'python', only 'python3'. Install one with: brew install python@3.12" >&2
  exit 1
fi

ARCH="$(uname -m)"
OS="$(uname -s)"
PYVER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PYMAJ="${PYVER%%.*}"; PYMIN="${PYVER##*.}"
echo "[setup] $("$PY" -c 'import sys; print(sys.executable)')"
echo "[setup] Python $PYVER · $OS $ARCH"

fail_with() {
  echo >&2
  echo "ERROR: $1" >&2
  echo >&2
  echo "  Fix:  $2" >&2
  echo "        PYTHON=$3 scripts/setup.sh" >&2
  exit 1
}

if [ "$PYMAJ" -lt 3 ] || { [ "$PYMAJ" -eq 3 ] && [ "$PYMIN" -lt 10 ]; }; then
  fail_with "Python $PYVER is too old; this project needs 3.10 or newer." \
            "brew install python@3.12" "python3.12"
fi

# PyTorch stopped shipping macOS Intel (x86_64) wheels after 2.2.2, and 2.2.2
# supports Python 3.8-3.11. Those two facts together are what produces
# "(from versions: none)" on an Intel Mac running a recent Python.
TORCH_PINS=""
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "x86_64" ]; then
  if [ "$PYMIN" -ge 12 ]; then
    fail_with "PyTorch publishes no macOS Intel (x86_64) wheel for Python $PYVER — its last Intel release, 2.2.2, supports Python 3.8-3.11." \
              "brew install python@3.11" "python3.11"
  fi
  TORCH_PINS="torch==2.2.2 torchvision==0.17.2"
  echo "[setup] macOS Intel: pinning $TORCH_PINS (the last release with Intel wheels)"
elif [ "$PYMIN" -ge 14 ]; then
  echo "[setup] WARNING: Python $PYVER is very new; if torch has no wheel yet, re-run with"
  echo "[setup]          PYTHON=python3.12 scripts/setup.sh"
fi

if [ ! -x .venv/bin/python ]; then
  echo "[setup] creating .venv"
  "$PY" -m venv .venv
else
  existing="$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  if [ "$existing" != "$PYVER" ]; then
    echo "[setup] .venv holds Python $existing but you asked for $PYVER — rebuilding it"
    rm -rf .venv
    "$PY" -m venv .venv
  else
    echo "[setup] reusing the existing .venv (Python $existing)"
  fi
fi
VENV=.venv/bin/python

# An old pip cannot read the wheel tags newer releases use, and a half-written
# cache entry ("Cache entry deserialization failed") makes it skip wheels it
# already has. Both are fixed before anything else is installed.
echo "[setup] updating pip"
"$VENV" -m pip install --quiet --upgrade pip setuptools wheel

install() {
  if [ -n "$TORCH_PINS" ]; then
    # shellcheck disable=SC2086
    "$VENV" -m pip install $1 $TORCH_PINS
  fi
  "$VENV" -m pip install $1 -r backend/requirements.txt
}

echo "[setup] installing dependencies (this pulls torch, so give it a few minutes)"
if ! install --quiet; then
  echo "[setup] install failed — clearing the pip cache and trying once more"
  "$VENV" -m pip cache purge >/dev/null 2>&1 || true
  if ! install "--no-cache-dir"; then
    echo >&2
    echo "ERROR: dependencies could not be installed on Python $PYVER / $OS $ARCH." >&2
    echo "       Versions of torch pip can see for this interpreter:" >&2
    "$VENV" -m pip index versions torch 2>&1 | sed 's/^/       /' >&2 || true
    echo "       If that list is empty, this Python has no torch build: use 3.12 (or 3.11 on" >&2
    echo "       an Intel Mac) with  PYTHON=python3.12 scripts/setup.sh" >&2
    exit 1
  fi
fi

echo
"$VENV" - <<'EOF'
import torch

from backend import config
from backend.methods.registry import list_methods, runtime_status

print(f"[setup] torch    : {torch.__version__}")
print(f"[setup] device   : {config.DEVICE}"
      f"{'' if config.DEVICE.type != 'cpu' else '   (no GPU here — predictions are slower, not different)'}")
print(f"[setup] dataset  : {config.BRI_DATASET_PATH}"
      f"{'' if config.BRI_DATASET_PATH.exists() else '   (not downloaded — see data/README.md)'}")
print("[setup] methods  :")
for spec in sorted(list_methods(), key=lambda s: s.display_number):
    status = runtime_status(spec)
    state = "ready" if status["classifier_available"] else "NO CHECKPOINT"
    print(f"           {spec.display_number}. {spec.short_name:<36} {state}")
EOF

cat <<EOF

[setup] done. Run things with .venv/bin/python — no activation needed:

  .venv/bin/python -m backend.cli.run_methods data/bri/archive/Testing/meningioma/Te-me_0014.jpg
  .venv/bin/python -m uvicorn backend.main:app --reload --port 8000

Or activate the environment once per terminal, and plain \`python\` will work:

  source .venv/bin/activate
EOF
