#!/usr/bin/env bash
# Short warm-start ablations; activate the existing PyTorch environment first.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd -- "$ROOT"

usage() {
    cat <<'HELP'
Usage: bash scripts/run_v16_short_ablation_linux.sh [options]

  --run-name NAME      New outputs/ablations/NAME run (default: UTC timestamp).
  --output-dir PATH    Explicit run directory instead of --run-name.
  --resume             Resume the same run with its original configuration.
  --dry-run            Print the command without creating files or training.
  --help               Show this launcher help.

All other arguments are forwarded unchanged to run_v16_short_ablation.py,
including --checkpoint, --data-root, --device, --epochs, --modes, and
--skip-real-data. Set SPLINE_PYTHON to select the Python executable.

Defaults: existing Stable K32 checkpoint; fixed, legacy, geometry; 8 epochs
each; no Proposal retraining; train/val/test 256/64/64; CUDA; batch size 32.
Logs: <output-dir>.run.log (outside the run directory; appended on resume).
Existing results are not overwritten. See docs/v16_short_ablation.md.
HELP
}

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
require_value() {
    [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires a value"
}

RUN_NAME="v16_short_ablation_$(date -u +%Y%m%d_%H%M%S)_$$"
OUTPUT_DIR=""
EXPLICIT_RUN_NAME=0
RESUME=0
DRY_RUN=0
PYTHON_ARGS=()
while (($#)); do
    case "$1" in
        --run-name)
            require_value "$@"
            RUN_NAME="$2"; EXPLICIT_RUN_NAME=1; shift 2 ;;
        --run-name=*)
            RUN_NAME="${1#*=}"; EXPLICIT_RUN_NAME=1; shift ;;
        --output-dir)
            require_value "$@"
            OUTPUT_DIR="$2"; shift 2 ;;
        --output-dir=*)
            OUTPUT_DIR="${1#*=}"
            [[ -n "$OUTPUT_DIR" ]] || fail '--output-dir requires a value'
            shift ;;
        --resume)
            RESUME=1; PYTHON_ARGS+=("$1"); shift ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            PYTHON_ARGS+=("$1"); shift ;;
    esac
done

[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail '--run-name must start with a letter or digit and contain only letters, digits, _, -, or .'
if [[ -n "$OUTPUT_DIR" && "$EXPLICIT_RUN_NAME" == 1 ]]; then
    fail 'use either --run-name or --output-dir, not both'
fi
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ablations/$RUN_NAME}"
# A trailing slash must not accidentally place the log inside the run directory.
while [[ "$OUTPUT_DIR" == */ && "$OUTPUT_DIR" != / ]]; do OUTPUT_DIR="${OUTPUT_DIR%/}"; done
[[ "$OUTPUT_DIR" != / && "$OUTPUT_DIR" != . && "$OUTPUT_DIR" != .. ]] || fail '--output-dir must identify a dedicated run directory'
LOG_PATH="$OUTPUT_DIR.run.log"
COMMAND=("${SPLINE_PYTHON:-python}" -u scripts/run_v16_short_ablation.py --output-dir "$OUTPUT_DIR" "${PYTHON_ARGS[@]}")

printf '[short_ablation] '
printf '%q ' "${COMMAND[@]}"
printf '\n[log] %s\n' "$LOG_PATH"
if ((DRY_RUN)); then exit 0; fi
if ((!RESUME)); then
    [[ ! -e "$OUTPUT_DIR" ]] || fail "run directory already exists: $OUTPUT_DIR; use a new name or --resume"
    [[ ! -e "$LOG_PATH" ]] || fail "run log already exists: $LOG_PATH; use a new name or --resume"
fi
mkdir -p -- "$(dirname -- "$LOG_PATH")"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND="${MPLBACKEND:-Agg}"
if ((RESUME)); then
    "${COMMAND[@]}" 2>&1 | tee -a -- "$LOG_PATH"
else
    "${COMMAND[@]}" 2>&1 | tee -- "$LOG_PATH"
fi
