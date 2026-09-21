#!/usr/bin/env bash
# Run after activating the server's existing PyTorch environment.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd -- "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
exec "${SPLINE_PYTHON:-python}" scripts/run_v16_paper_training.py "$@"
