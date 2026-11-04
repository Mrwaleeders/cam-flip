#!/bin/bash
# Trains the personal eye model on capture sessions and installs it for the plugin (see train.py).
#   tools/model/train.sh ~/Downloads/camflip-diag/captures/<session> [more sessions...]
# First run sets up Python packages in obs-plugin/build/venv (a few minutes). Everything stays on this Mac.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-$(command -v python3.12 || command -v python3.11 || command -v python3)}"
if [ ! -x build/venv/bin/python ]; then
  mkdir -p build
  "$PY" -m venv build/venv
  build/venv/bin/pip install -q --upgrade pip
  build/venv/bin/pip install -q -r tools/model/requirements.txt
fi
# Anything Apple's GPU can't run falls back to the CPU instead of stopping training.
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec build/venv/bin/python tools/model/train.py "$@"
