#!/usr/bin/env bash
# Native Linux runner for the formal v16 K=4..56, MSE=1e-4 experiment.

set -Eeuo pipefail

PYTHON_BIN="python"
SIMPLIFICATION_CONTRACT="synthetic_ground_truth_ordered_keep_and_relocation_v4"
RUN_NAME="candidate_selection_v16_mse1e-4_k56_supervised_linux"
DEVICE="cuda"
EPOCHS=104
PROPOSAL_EPOCHS=40
TRAIN_SIZE=3000
VAL_SIZE=600
REAL_VAL_SIZE=100
PROPOSAL_HIGH_K_FRACTION=0.50
PROPOSAL_HIGH_K_MIN_KNOTS=40
BATCH_SIZE=64
NUM_WORKERS=4
SYNTHETIC_SAMPLES_PER_K=5
REAL_SAMPLES_PER_DATASET=20
VISUAL_SAMPLES_PER_DATASET=2
KANG_ADMM_ITERATIONS=1000
LUO_DE_POPULATION=20
LUO_DE_ITERATIONS=100
NETWORK_REPEATS=100
END_TO_END_REPEATS=3
OUTPUT_ROOT=""
INIT_CHECKPOINT=""
PREPARE_REAL_DATA=0
DIAGNOSTIC=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_v16_mse1e-4_3090.sh [options]

Main options:
  --python PATH                   Python executable (default: python)
  --run-name NAME                Fresh output name
  --device auto|cpu|cuda         Training/evaluation device (default: cuda)
  --epochs N                     Total epochs (default: 104)
  --proposal-epochs N            Proposal-stage epochs (default: 40)
  --train-size N                 Training draws per epoch (default: 3000)
  --val-size N                   Synthetic validation curves (default: 600)
  --real-val-size N              Validation curves per real source (default: 100)
                                  Training is always 100% labelled Synthetic;
                                  real manifests are validation/test only.
  --proposal-high-k-fraction X   Proposal synthetic high-K share (default: 0.50)
  --proposal-high-k-min-knots N  High-K stratum begins here (default: 40)
  --batch-size N                 Batch size (default: 64)
  --num-workers N                DataLoader workers (default: 4)
  --output-root PATH             Output root (default: repository outputs/)
  --init-checkpoint PATH         Optional proposal-only warm start
  --no-init-checkpoint           Train all modules from scratch
  --prepare-real-data            Download/prepare missing UJI, Natural Earth and USGS data
  --diagnostic                   Continue after a structural-integrity audit failure
  --dry-run                      Print every command without executing Python
  -h, --help                     Show this help

Evaluation-size options:
  --synthetic-samples-per-k N    Synthetic cases for each K=4..56
  --real-samples-per-dataset N   Real benchmark cases per dataset
  --visual-samples-per-dataset N Real plotted cases per dataset
  --kang-admm-iterations N
  --luo-de-population N
  --luo-de-iterations N
  --network-repeats N
  --end-to-end-repeats N
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

need_value() {
  [[ $# -ge 2 ]] || die "option $1 requires a value"
}

while (($#)); do
  case "$1" in
    --python) need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --run-name) need_value "$@"; RUN_NAME="$2"; shift 2 ;;
    --device) need_value "$@"; DEVICE="$2"; shift 2 ;;
    --epochs) need_value "$@"; EPOCHS="$2"; shift 2 ;;
    --proposal-epochs) need_value "$@"; PROPOSAL_EPOCHS="$2"; shift 2 ;;
    --train-size) need_value "$@"; TRAIN_SIZE="$2"; shift 2 ;;
    --val-size) need_value "$@"; VAL_SIZE="$2"; shift 2 ;;
    --real-val-size) need_value "$@"; REAL_VAL_SIZE="$2"; shift 2 ;;
    --proposal-high-k-fraction) need_value "$@"; PROPOSAL_HIGH_K_FRACTION="$2"; shift 2 ;;
    --proposal-high-k-min-knots) need_value "$@"; PROPOSAL_HIGH_K_MIN_KNOTS="$2"; shift 2 ;;
    --batch-size) need_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) need_value "$@"; NUM_WORKERS="$2"; shift 2 ;;
    --synthetic-samples-per-k) need_value "$@"; SYNTHETIC_SAMPLES_PER_K="$2"; shift 2 ;;
    --real-samples-per-dataset) need_value "$@"; REAL_SAMPLES_PER_DATASET="$2"; shift 2 ;;
    --visual-samples-per-dataset) need_value "$@"; VISUAL_SAMPLES_PER_DATASET="$2"; shift 2 ;;
    --kang-admm-iterations) need_value "$@"; KANG_ADMM_ITERATIONS="$2"; shift 2 ;;
    --luo-de-population) need_value "$@"; LUO_DE_POPULATION="$2"; shift 2 ;;
    --luo-de-iterations) need_value "$@"; LUO_DE_ITERATIONS="$2"; shift 2 ;;
    --network-repeats) need_value "$@"; NETWORK_REPEATS="$2"; shift 2 ;;
    --end-to-end-repeats) need_value "$@"; END_TO_END_REPEATS="$2"; shift 2 ;;
    --output-root) need_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --init-checkpoint) need_value "$@"; INIT_CHECKPOINT="$2"; shift 2 ;;
    --no-init-checkpoint) INIT_CHECKPOINT=""; shift ;;
    --prepare-real-data) PREPARE_REAL_DATA=1; shift ;;
    --diagnostic) DIAGNOSTIC=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

positive_integer() {
  local name="$1" value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer"
}

nonnegative_integer() {
  local name="$1" value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer"
}

for item in \
  "EPOCHS:$EPOCHS" \
  "PROPOSAL_EPOCHS:$PROPOSAL_EPOCHS" \
  "TRAIN_SIZE:$TRAIN_SIZE" \
  "VAL_SIZE:$VAL_SIZE" \
  "REAL_VAL_SIZE:$REAL_VAL_SIZE" \
  "PROPOSAL_HIGH_K_MIN_KNOTS:$PROPOSAL_HIGH_K_MIN_KNOTS" \
  "BATCH_SIZE:$BATCH_SIZE" \
  "SYNTHETIC_SAMPLES_PER_K:$SYNTHETIC_SAMPLES_PER_K" \
  "REAL_SAMPLES_PER_DATASET:$REAL_SAMPLES_PER_DATASET" \
  "VISUAL_SAMPLES_PER_DATASET:$VISUAL_SAMPLES_PER_DATASET" \
  "KANG_ADMM_ITERATIONS:$KANG_ADMM_ITERATIONS" \
  "LUO_DE_POPULATION:$LUO_DE_POPULATION" \
  "LUO_DE_ITERATIONS:$LUO_DE_ITERATIONS" \
  "NETWORK_REPEATS:$NETWORK_REPEATS" \
  "END_TO_END_REPEATS:$END_TO_END_REPEATS"; do
  positive_integer "${item%%:*}" "${item#*:}"
done
nonnegative_integer "NUM_WORKERS" "$NUM_WORKERS"
((PROPOSAL_EPOCHS < EPOCHS)) || die "proposal epochs must be smaller than total epochs"
((LUO_DE_POPULATION >= 5)) || die "Luo DE population must be at least 5"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "run name contains unsupported characters"
[[ "$DEVICE" == "auto" || "$DEVICE" == "cpu" || "$DEVICE" == "cuda" ]] || \
  die "device must be auto, cpu or cuda"
[[ "$PROPOSAL_HIGH_K_FRACTION" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] || \
  die "proposal high-K fraction must lie in [0,1]"
((PROPOSAL_HIGH_K_MIN_KNOTS >= 4 && PROPOSAL_HIGH_K_MIN_KNOTS <= 56)) || \
  die "proposal high-K minimum must lie inside source K=4..56"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd -- "$REPOSITORY_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export MPLBACKEND=Agg

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"
command -v tee >/dev/null 2>&1 || die "tee is required"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$REPOSITORY_ROOT/outputs"
elif [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="$REPOSITORY_ROOT/$OUTPUT_ROOT"
fi
OUTPUT_ROOT="$(readlink -m -- "$OUTPUT_ROOT")"

CHECKPOINT_DIRECTORY="$OUTPUT_ROOT/checkpoints"
LOG_DIRECTORY="$OUTPUT_ROOT/logs/$RUN_NAME"
COMPARISON_ROOT="$OUTPUT_ROOT/comparisons/$RUN_NAME"
FIGURE_ROOT="$OUTPUT_ROOT/figures/$RUN_NAME"
CHECKPOINT_PATH="$CHECKPOINT_DIRECTORY/$RUN_NAME.pt"
LAST_PATH="$CHECKPOINT_DIRECTORY/$RUN_NAME.last.pt"
PROPOSAL_PATH="$CHECKPOINT_DIRECTORY/$RUN_NAME.proposal.pt"
HISTORY_PATH="$CHECKPOINT_DIRECTORY/$RUN_NAME.history.json"

for owned in \
  "$CHECKPOINT_PATH" "$LAST_PATH" "$PROPOSAL_PATH" "$HISTORY_PATH" \
  "$LOG_DIRECTORY" "$COMPARISON_ROOT" "$FIGURE_ROOT"; do
  [[ ! -e "$owned" ]] || die "refusing to overwrite existing run artifact: $owned"
done

UJI_MANIFEST="$REPOSITORY_ROOT/data/splits/uji_pen_v2.jsonl"
NATURAL_EARTH_MANIFEST="$REPOSITORY_ROOT/data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl"
USGS_MANIFEST="$REPOSITORY_ROOT/data/processed/usgs_contours/large_scale/manifest.jsonl"

prepare_missing_real_data() {
  if [[ ! -f "$UJI_MANIFEST" ]]; then
    printf '\n[prepare_uji] downloading and preparing UJI Pen Characters v2\n'
    if ((DRY_RUN)); then
      printf ' %q' "$PYTHON_BIN" scripts/prepare_uji_pen.py --download
      printf '\n'
    else
      "$PYTHON_BIN" scripts/prepare_uji_pen.py --download
    fi
  fi
  if [[ ! -f "$NATURAL_EARTH_MANIFEST" ]]; then
    printf '\n[prepare_natural_earth] downloading and preparing 10m coastline\n'
    if ((DRY_RUN)); then
      printf ' %q' "$PYTHON_BIN" scripts/prepare_natural_earth.py \
        --resolution 10m --layer coastline --reference-points 768
      printf '\n'
    else
      "$PYTHON_BIN" scripts/prepare_natural_earth.py \
        --resolution 10m --layer coastline --reference-points 768
    fi
  fi
  if [[ ! -f "$USGS_MANIFEST" ]]; then
    printf '\n[prepare_usgs] downloading and preparing example contour regions\n'
    if ((DRY_RUN)); then
      printf ' %q' "$PYTHON_BIN" scripts/prepare_usgs_contours.py \
        --bbox-file configs/usgs_contour_regions.example.json \
        --max-features-per-region 2000 \
        --output-dir data/processed/usgs_contours/large_scale
      printf '\n'
    else
      "$PYTHON_BIN" scripts/prepare_usgs_contours.py \
        --bbox-file configs/usgs_contour_regions.example.json \
        --max-features-per-region 2000 \
        --output-dir data/processed/usgs_contours/large_scale
    fi
  fi
}

if ((PREPARE_REAL_DATA)); then
  prepare_missing_real_data
fi
if ((DRY_RUN == 0)); then
  for manifest in "$UJI_MANIFEST" "$NATURAL_EARTH_MANIFEST" "$USGS_MANIFEST"; do
    [[ -f "$manifest" ]] || die \
      "required real-data manifest does not exist: $manifest; rerun with --prepare-real-data"
  done
  if [[ "$DEVICE" == "cuda" ]]; then
    "$PYTHON_BIN" -c \
      'import torch; assert torch.cuda.is_available(), "CUDA requested but unavailable"; print(torch.cuda.get_device_name(0))'
  fi
fi

mkdir -p -- "$CHECKPOINT_DIRECTORY" "$LOG_DIRECTORY"
CURRENT_PHASE="initialization"
trap 'code=$?; printf "\nPipeline failed during %s (exit %d). Logs: %s\n" "$CURRENT_PHASE" "$code" "$LOG_DIRECTORY" >&2; exit "$code"' ERR

run_logged() {
  local phase="$1"
  shift
  local log_path="$LOG_DIRECTORY/$phase.log"
  CURRENT_PHASE="$phase"
  printf '\n[%s]' "$phase"
  printf ' %q' "$@"
  printf '\n'
  if ((DRY_RUN)); then
    return 0
  fi
  set +e
  "$@" 2>&1 | tee "$log_path"
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

printf 'Fresh v16 Linux profile: MSE=1e-4, Kc=56, source K=4..56, train/val=%s/%s, batch=%s.\n' \
  "$TRAIN_SIZE" "$VAL_SIZE" "$BATCH_SIZE"
printf 'Simplification contract: %s\n' "$SIMPLIFICATION_CONTRACT"
printf 'Checkpoint selection: mean_per_curve_subset_cost_v1; aggregate pass is reporting only.\n'
printf 'Training supervision: certified Synthetic labels only; online Teacher disabled; real data is validation/test only.\n'
printf 'Proposal synthetic high-K share=%s at K>=%s; Joint restores K=4..56.\n' \
  "$PROPOSAL_HIGH_K_FRACTION" "$PROPOSAL_HIGH_K_MIN_KNOTS"

TRAIN_ARGS=(
  "$PYTHON_BIN" scripts/train_v16.py
  --epochs "$EPOCHS"
  --proposal-epochs "$PROPOSAL_EPOCHS"
  --train-size "$TRAIN_SIZE"
  --val-size "$VAL_SIZE"
  --synthetic-boundary-val-size 32
  --real-val-size "$REAL_VAL_SIZE"
  --proposal-high-k-fraction "$PROPOSAL_HIGH_K_FRACTION"
  --proposal-high-k-min-knots "$PROPOSAL_HIGH_K_MIN_KNOTS"
  --batch-size "$BATCH_SIZE"
  --num-points 192
  --min-control-points 8
  --max-control-points 60
  --candidate-knots 56
  --knot-min-span 0.01
  --mse-tolerance 1e-4
  --knot-match-tolerance 0.01
  --tolerance-factor-min 1
  --tolerance-factor-max 1
  --certified-minimal-source
  --minimality-margin 0.2
  --minimality-max-attempts 16
  --minimality-audit-points 512
  --oscillation-amplitude 0.3
  --proposal-knot-assignment-weight 1.0
  --one-shot-selection-policy mass_topk
  --initial-keep-fraction 0.5357142857142857
  --joint-supervision synthetic_ground_truth
  --synthetic-count-role exact
  --no-synthetic-geometry-oracle-teacher
  --one-shot-coverage-bins 0
  --min-selected-knots 4
  --one-shot-safety-sigma 0
  --one-shot-safety-knots 0
  --final-safety-sigma 0
  --final-safety-knots 0
  --safety-anneal-epochs 12
  --complexity-ramp-epochs 12
  --complexity-max-scale 1.0
  --complexity-weight 0
  --real-fraction 0
  --real-manifest "$UJI_MANIFEST"
  --real-manifest "$NATURAL_EARTH_MANIFEST"
  --real-manifest "$USGS_MANIFEST"
  --resample-train-each-epoch
  --num-workers "$NUM_WORKERS"
  --torch-num-threads 4
  --device "$DEVICE"
  --output "$CHECKPOINT_PATH"
)
if [[ -n "$INIT_CHECKPOINT" ]]; then
  if [[ "$INIT_CHECKPOINT" != /* ]]; then
    INIT_CHECKPOINT="$REPOSITORY_ROOT/$INIT_CHECKPOINT"
  fi
  INIT_CHECKPOINT="$(readlink -m -- "$INIT_CHECKPOINT")"
  if [[ -f "$INIT_CHECKPOINT" ]]; then
    printf 'Using proposal-only warm start: %s\n' "$INIT_CHECKPOINT"
    TRAIN_ARGS+=(--init-checkpoint "$INIT_CHECKPOINT")
  else
    printf 'WARNING: optional proposal initializer not found; training from scratch: %s\n' \
      "$INIT_CHECKPOINT" >&2
  fi
fi
run_logged train_fresh "${TRAIN_ARGS[@]}"

if ((DRY_RUN == 0)); then
  [[ -f "$CHECKPOINT_PATH" ]] || die "training completed without best checkpoint: $CHECKPOINT_PATH"
fi

INSPECT_ARGS=(
  "$PYTHON_BIN" scripts/inspect_v16_checkpoint.py
  --checkpoint "$CHECKPOINT_PATH"
  --mse-tolerance 1e-4
)
if run_logged inspect_checkpoint "${INSPECT_ARGS[@]}"; then
  CHECKPOINT_INTEGRITY_OK=1
else
  CHECKPOINT_INTEGRITY_OK=0
fi
if ((CHECKPOINT_INTEGRITY_OK == 0 && DIAGNOSTIC == 0)); then
  die "checkpoint failed structural-integrity audit; benchmark and figures were not produced"
fi

if ((DRY_RUN)); then
  CHECKPOINT_HASH="0000000000000000000000000000000000000000000000000000000000000000"
else
  CHECKPOINT_HASH="$(sha256sum -- "$CHECKPOINT_PATH" | awk '{print $1}')"
fi
if ((DIAGNOSTIC || CHECKPOINT_INTEGRITY_OK == 0)); then
  TAG="diagnostic_${CHECKPOINT_HASH:0:12}"
  DIAGNOSTIC_ARGS=(--allow-unqualified-diagnostic)
else
  TAG="formal_${CHECKPOINT_HASH:0:12}"
  DIAGNOSTIC_ARGS=()
fi

COMPARISON_DIRECTORY="$COMPARISON_ROOT/$TAG"
METRIC_FIGURE_DIRECTORY="$FIGURE_ROOT/$TAG/four_metrics"
REAL_FIGURE_DIRECTORY="$FIGURE_ROOT/$TAG/six_method_real_cases"
for directory in "$COMPARISON_DIRECTORY" "$METRIC_FIGURE_DIRECTORY" "$REAL_FIGURE_DIRECTORY"; do
  [[ ! -e "$directory" ]] || die "refusing to overwrite evaluation output: $directory"
  mkdir -p -- "$directory"
done

MANIFEST_ARGS=(
  --manifest "UJI=$UJI_MANIFEST"
  --manifest "NaturalEarth=$NATURAL_EARTH_MANIFEST"
  --manifest "USGS=$USGS_MANIFEST"
)
BASELINE_ARGS=(
  --max-internal-knots 56
  --gradient-steps 12
  --paper-initial-knots 56
  --paper-admm-iterations "$KANG_ADMM_ITERATIONS"
  --paper-lambda-bisections 10
  --paper-relocation-iterations 12
  --liang-dense-knots 56
  --liang-feature-samples 1025
  --dung-scan-intervals 10
  --dung-optimization-iterations 10
  --luo-eta 0.5
  --luo-de-population "$LUO_DE_POPULATION"
  --luo-de-iterations "$LUO_DE_ITERATIONS"
)

run_logged benchmark_six_methods \
  "$PYTHON_BIN" scripts/benchmark_v16_datasets.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --output-dir "$COMPARISON_DIRECTORY" \
  --method-set published \
  --samples-per-knot-count "$SYNTHETIC_SAMPLES_PER_K" \
  --min-knot-count 4 --max-knot-count 56 \
  --real-samples-per-dataset "$REAL_SAMPLES_PER_DATASET" \
  --mse-tolerance 1e-4 \
  "${MANIFEST_ARGS[@]}" "${BASELINE_ARGS[@]}" \
  --network-warmups 10 --network-repeats "$NETWORK_REPEATS" \
  --end-to-end-repeats "$END_TO_END_REPEATS" \
  --torch-num-threads 4 --device "$DEVICE" \
  "${DIAGNOSTIC_ARGS[@]}"

run_logged plot_four_metrics \
  "$PYTHON_BIN" scripts/plot_v16_method_comparison.py \
  --input "$COMPARISON_DIRECTORY/comparison.json" \
  --output-dir "$METRIC_FIGURE_DIRECTORY" \
  --method-set published --dpi 300 --reference \
  "${DIAGNOSTIC_ARGS[@]}"

run_logged visualize_six_method_real_cases \
  "$PYTHON_BIN" scripts/visualize_v16_real_deployments.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --output-dir "$REAL_FIGURE_DIRECTORY" \
  --real-samples-per-dataset "$VISUAL_SAMPLES_PER_DATASET" \
  --selection-seed 20260910 --mse-tolerance 1e-4 \
  "${MANIFEST_ARGS[@]}" "${BASELINE_ARGS[@]}" \
  --network-warmups 10 --network-repeats "$NETWORK_REPEATS" \
  --end-to-end-repeats "$END_TO_END_REPEATS" \
  --torch-num-threads 4 --device "$DEVICE" --dpi 300 \
  "${DIAGNOSTIC_ARGS[@]}"

CURRENT_PHASE="completed"
trap - ERR
printf '\nv16 Linux pipeline completed.\n'
printf '  checkpoint: %s\n' "$CHECKPOINT_PATH"
printf '  diagnostic: %s\n' "$([[ ${#DIAGNOSTIC_ARGS[@]} -gt 0 ]] && printf true || printf false)"
printf '  comparison: %s\n' "$COMPARISON_DIRECTORY"
printf '  figures:    %s\n' "$FIGURE_ROOT/$TAG"
printf '  logs:       %s\n' "$LOG_DIRECTORY"
