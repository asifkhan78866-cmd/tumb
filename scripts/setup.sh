#!/usr/bin/env bash
# One-time setup on a fresh clone: a virtual environment with everything the
# trained models need to run.
#
#   scripts/setup.sh
#
# macOS ships no `python` on PATH, only `python3`, and this project is never
# installed into the system interpreter — everything lives in .venv, which is
# not in git because it holds gigabytes of compiled wheels.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: $PY is not on PATH. Install Python 3.11+ (brew install python) and re-run." >&2
  exit 1
fi

version="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "[setup] $PY is Python $version"
"$PY" - <<'EOF'
import sys
if sys.version_info < (3, 10):
    sys.exit("ERROR: Python 3.10 or newer is required.")
EOF

if [ ! -x .venv/bin/python ]; then
  echo "[setup] creating .venv"
  "$PY" -m venv .venv
else
  echo "[setup] reusing the existing .venv"
fi

echo "[setup] installing dependencies (this pulls torch, so give it a few minutes)"
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r backend/requirements.txt

echo
.venv/bin/python - <<'EOF'
from backend import config
from backend.methods.registry import list_methods, runtime_status

print(f"[setup] device   : {config.DEVICE}")
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

  cd "$ROOT"
  .venv/bin/python -m backend.cli.run_methods data/bri/archive/Testing/meningioma/Te-me_0014.jpg
  .venv/bin/python -m uvicorn backend.main:app --reload --port 8000

Or activate the environment once per terminal, and plain \`python\` will work:

  source .venv/bin/activate
EOF
