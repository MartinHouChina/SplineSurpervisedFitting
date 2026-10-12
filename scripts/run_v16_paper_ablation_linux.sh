#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd -- "$ROOT"

RUN_NAME="v16_paper_ablation_$(date -u +%Y%m%d_%H%M%S)"
PYTHON_BIN="${SPLINE_PYTHON:-python}"
ARGS=()
while (($#)); do
  case "$1" in
    --run-name)
      [[ $# -ge 2 && -n "$2" ]] || { echo "ERROR: --run-name requires a value" >&2; exit 2; }
      RUN_NAME="$2"; shift 2 ;;
    --python)
      [[ $# -ge 2 && -n "$2" ]] || { echo "ERROR: --python requires a value" >&2; exit 2; }
      PYTHON_BIN="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/run_v16_paper_ablation_linux.sh [--run-name NAME] [Python options]

Runs paired feature and module ablations from the selected K32 checkpoint.
Every other option is forwarded to scripts/run_v16_paper_ablation.py.
The destination is outputs/ablations/<run-name> and is never overwritten.
EOF
      exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "ERROR: invalid run name" >&2; exit 2;
}
OUTPUT="outputs/ablations/$RUN_NAME"
[[ ! -e "$OUTPUT" ]] || {
  echo "ERROR: refusing to overwrite $OUTPUT" >&2; exit 2;
}
mkdir -p -- "$OUTPUT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
"$PYTHON_BIN" -u scripts/run_v16_paper_ablation.py \
  --output-dir "$OUTPUT" "${ARGS[@]}" 2>&1 | tee "$OUTPUT.run.log"
