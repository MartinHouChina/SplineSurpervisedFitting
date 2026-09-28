#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$SCRIPT_DIR/.."
exec "${SPLINE_PYTHON:-python}" scripts/run_v16_saved_models_evaluation.py "$@"
