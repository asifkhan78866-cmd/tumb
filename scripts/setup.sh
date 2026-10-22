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

ARCH="$(uname -m)"
OS="$(uname -s)"

# PyTorch stopped shipping macOS Intel (x86_64) wheels after 2.2.2, and 2.2.2
# supports Python 3.8-3.11. Those two facts together are what produces
# "No matching distribution found for torch (from versions: none)" on an Intel
# Mac running a recent Python — so which interpreters can work is decided here,
# before anything is downloaded.
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "x86_64" ]; then
  WANTED="3.11 3.10"
  WANTED_LABEL="3.11 or 3.10 (PyTorch's last Intel build, 2.2.2, supports Python 3.8-3.11)"
else
  WANTED="3.13 3.12 3.11 3.10"
  WANTED_LABEL="3.13, 3.12, 3.11 or 3.10"
fi

version_of() { "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null; }

usable() {
  local found; found="$(version_of "$1")" || return 1
  [ -n "$found" ] || return 1
  for want in $WANTED; do [ "$found" = "$want" ] && return 0; done
  return 1
}

# Everywhere a Python may live on a Mac that has no package manager.
candidates() {
  [ -n "${PYTHON:-}" ] && printf '%s\n' "$PYTHON"
  for want in $WANTED; do
    printf '%s\n' "python$want"
    printf '%s\n' "/Library/Frameworks/Python.framework/Versions/$want/bin/python3"
    printf '%s\n' "/opt/homebrew/bin/python$want" "/usr/local/bin/python$want"
  done
  printf '%s\n' python3 /usr/bin/python3
  # A venv built earlier is itself a usable interpreter, and may be the only one
  # on a machine whose system Python cannot install torch.
  printf '%s\n' .venv/bin/python
}

PY=""
while read -r candidate; do
  [ -n "$candidate" ] || continue
  command -v "$candidate" >/dev/null 2>&1 || [ -x "$candidate" ] || continue
  if usable "$candidate"; then PY="$candidate"; break; fi
done <<EOF
$(candidates)
EOF

if [ -z "$PY" ]; then
  echo "ERROR: no usable Python found. This project needs $WANTED_LABEL." >&2
  echo >&2
  echo "       Interpreters on this machine:" >&2
  while read -r candidate; do
    [ -n "$candidate" ] || continue
    if command -v "$candidate" >/dev/null 2>&1 || [ -x "$candidate" ]; then
      found="$(version_of "$candidate")"
      [ -n "$found" ] && echo "         $candidate -> $found" >&2
    fi
  done <<EOF2
$(candidates)
EOF2
  echo >&2
  echo "       Install one, no package manager needed — either:" >&2
  echo >&2
  echo "       a) python.org installer (double-click, works on Intel and Apple Silicon)" >&2
  echo "          https://www.python.org/downloads/macos/" >&2
  echo "          Take the latest ${WANTED%% *}.x 'macOS 64-bit universal2 installer', then re-run:" >&2
  echo "            scripts/setup.sh" >&2
  echo >&2
  echo "       b) uv, a single binary that installs Pythons for you:" >&2
  echo "            curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  echo "            ~/.local/bin/uv python install ${WANTED%% *}" >&2
  echo "            ~/.local/bin/uv venv --python ${WANTED%% *} .venv" >&2
  echo "            scripts/setup.sh" >&2
  exit 1
fi

PYVER="$(version_of "$PY")"
echo "[setup] $("$PY" -c 'import sys; print(sys.executable)')"
echo "[setup] Python $PYVER · $OS $ARCH"

TORCH_PINS=""
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "x86_64" ]; then
  TORCH_PINS="torch==2.2.2 torchvision==0.17.2"
  echo "[setup] macOS Intel: pinning $TORCH_PINS (the last release with Intel wheels)"
fi

if [ ! -x .venv/bin/python ]; then
  echo "[setup] creating .venv"
  "$PY" -m venv .venv
else
  existing="$(version_of .venv/bin/python)"
  if ! usable .venv/bin/python; then
    echo "[setup] .venv holds Python $existing, which cannot install torch here — rebuilding it"
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
