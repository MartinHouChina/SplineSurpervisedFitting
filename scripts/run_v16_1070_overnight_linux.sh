#!/usr/bin/env bash
# Historical 6413ee0 architecture: online Teacher, Kc64, source K4..24.
# New Linux warm-start run, not relocation/resume of the old Windows .last.pt.
set -Eeuo pipefail

PYTHON_BIN=python
RUN_NAME=""
DEVICE=cuda
EPOCHS=""
PROPOSAL_EPOCHS=""
TRAIN_SIZE=1500
VAL_SIZE=500
BATCH_SIZE=32
MSE_TOLERANCE=5e-5
NUM_WORKERS=0
INIT_CHECKPOINT=outputs/checkpoints/candidate_selection_v16.proposal.pt
INIT_SPECIFIED=0
WARM_START_CHECKPOINT=""
ENHANCED_SELECTION=0
RELIABLE_SELECTION=0
COMPACT_SELECTION=0
REAL_VAL_SIZE=32
NATIVE_BASELINES=0
CHECKPOINT=""
DATA_ROOT=""
OUTPUT_ROOT=""
PREPARE=0
RESUME=0
DRY_RUN=0
BENCHMARK_PROFILE=full
SYNTHETIC_SAMPLES=""
REAL_SAMPLES=""
VISUAL_SAMPLES=""
NETWORK_REPEATS=""
END_TO_END_REPEATS=""

usage() {
  cat <<'EOF'
Usage: bash scripts/run_v16_1070_overnight_linux.sh [options]
Historical warm start -> Synthetic + UJI/NaturalEarth/USGS six-method benchmark
-> four-metric PNGs -> Ours cases -> six-method real-case PNGs.
No current K24/K48 or offline-teacher model code is used.

  --python PATH                    Python 3.11+ executable (default python)
  --run-name NAME                  Fresh name (default overnight_1070_arch_3090_r1)
  --device auto|cpu|cuda           Default cuda
  --epochs N --proposal-epochs N   Total/Proposal epochs (64/4; Joint=60)
  --enhanced-selection            Stronger online Teacher + boundary ranking;
                                  defaults to 96/12 epochs, new run name
  --reliable-selection            Parameter trust + geometry Teacher + local loss;
                                  defaults to 32/4 epochs, real validation only
  --compact-selection             Per-curve feasible simplification + bounded greedy
                                  Teacher; defaults 24/4, explicit full warm start
  --real-val-size N               Reliable/compact: val curves per real source (32)
  --native-baselines              Disable explicit MSE repair for Dung/Kang/Luo;
                                  corrected Kang clustering remains enabled
  --train-size N --val-size N      Synthetic training/validation (1500/500)
  --batch-size N --num-workers N   Default 32/0
  --mse-tolerance X                Shared MSE threshold, not RMS (5e-5)
  --init-checkpoint PATH           Historical Proposal warm-start checkpoint
  --warm-start-checkpoint PATH     Copy ALL compatible overnight model weights
                                  into a NEW experiment (not optimizer resume)
  --no-init-checkpoint             Explicit from-scratch ablation (not original run)
  --checkpoint PATH               Skip training and evaluate existing Kc64 weights
  --resume-run                     Resume same run; skip training if epochs completed
  --data-root PATH                 Complete data tree (default repository data/)
  --prepare-real-data              Prepare missing real manifests without overwriting
  --output-root PATH               Default repository outputs/
  --benchmark-profile quick|full  Default full; quick is visibly diagnostic
  --synthetic-samples-per-k N       Full 2 / quick 1 (source K=4..24)
  --real-samples-per-dataset N      Full 8 / quick 2 (test split only)
  --visual-samples-per-dataset N    Full 2 / quick 1
  --network-repeats N              Full 100 / quick 5
  --end-to-end-repeats N            Full 3 / quick 1, same across all methods
  --dry-run                        Print commands only; no Python execution/writes
  -h, --help

Six methods = Ours + Park, Liang, Dung, Kang, Luo (paper adaptations).
Timing plot uses complete fitting time; Ours network-only time is separate.
Fresh names never overwrite existing experiments. Resume retains old logs and
uses fresh attempt directories for figures. Windows .last.pt cannot be resumed
under a different Linux output path; use --checkpoint for evaluation instead.
EOF
}
die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
need_value() { [[ $# -ge 2 ]] || die "option $1 requires a value"; }
while (($#)); do
  case "$1" in
    --python) need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --run-name) need_value "$@"; RUN_NAME="$2"; shift 2 ;;
    --device) need_value "$@"; DEVICE="$2"; shift 2 ;;
    --epochs) need_value "$@"; EPOCHS="$2"; shift 2 ;;
    --proposal-epochs) need_value "$@"; PROPOSAL_EPOCHS="$2"; shift 2 ;;
    --enhanced-selection) ENHANCED_SELECTION=1; shift ;;
    --reliable-selection) RELIABLE_SELECTION=1; shift ;;
    --compact-selection) COMPACT_SELECTION=1; shift ;;
    --real-val-size) need_value "$@"; REAL_VAL_SIZE="$2"; shift 2 ;;
    --native-baselines) NATIVE_BASELINES=1; shift ;;
    --train-size) need_value "$@"; TRAIN_SIZE="$2"; shift 2 ;;
    --val-size) need_value "$@"; VAL_SIZE="$2"; shift 2 ;;
    --batch-size) need_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) need_value "$@"; NUM_WORKERS="$2"; shift 2 ;;
    --mse-tolerance) need_value "$@"; MSE_TOLERANCE="$2"; shift 2 ;;
    --init-checkpoint) need_value "$@"; INIT_CHECKPOINT="$2"; INIT_SPECIFIED=1; shift 2 ;;
    --no-init-checkpoint) INIT_CHECKPOINT=""; INIT_SPECIFIED=1; shift ;;
    --warm-start-checkpoint) need_value "$@"; WARM_START_CHECKPOINT="$2"; shift 2 ;;
    --checkpoint) need_value "$@"; CHECKPOINT="$2"; shift 2 ;;
    --data-root) need_value "$@"; DATA_ROOT="$2"; shift 2 ;;
    --output-root) need_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --prepare-real-data) PREPARE=1; shift ;;
    --resume-run) RESUME=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --benchmark-profile) need_value "$@"; BENCHMARK_PROFILE="$2"; shift 2 ;;
    --synthetic-samples-per-k) need_value "$@"; SYNTHETIC_SAMPLES="$2"; shift 2 ;;
    --real-samples-per-dataset) need_value "$@"; REAL_SAMPLES="$2"; shift 2 ;;
    --visual-samples-per-dataset) need_value "$@"; VISUAL_SAMPLES="$2"; shift 2 ;;
    --network-repeats) need_value "$@"; NETWORK_REPEATS="$2"; shift 2 ;;
    --end-to-end-repeats) need_value "$@"; END_TO_END_REPEATS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done
((ENHANCED_SELECTION + RELIABLE_SELECTION + COMPACT_SELECTION <= 1)) || die "choose only one of --enhanced-selection, --reliable-selection and --compact-selection"
SAFETY_ARGS=(--one-shot-safety-sigma 0.2 --safety-anneal-epochs 8)
RESAMPLE_ARGS=(--no-resample-train-each-epoch)
if ((COMPACT_SELECTION)); then
  RUN_NAME=${RUN_NAME:-overnight_compact_3090_r1}
  EPOCHS=${EPOCHS:-24}
  PROPOSAL_EPOCHS=${PROPOSAL_EPOCHS:-4}
  LEARNING_ARGS=(--policy-samples 2 --counterfactual-edits 4 --teacher-prefix-search-steps 6
    --lr 2e-5 --joint-lr 3e-5 --teacher-refinement-steps 1 --teacher-refinement-candidates 3
    --boundary-ranking-weight 0.5 --boundary-ranking-candidates 4
    --joint-proposal-lr-scale 0.1 --joint-decoder-lr-scale 0.25 --joint-final-lr-ratio 0.25
    --parameter-trust-enabled --parameter-trust-initial 0.25
    --parameter-counterfactual-weight 0.25 --teacher-geometry-candidates 4
    --local-fit-weight 0.1 --proposal-ordered-weight 0.5 --allow-infeasible-proposals
    --simplification-controller per_curve --complexity-max-scale 1.0
    --feasible-objective --feasible-fit-margin 0.8 --feasible-fit-weight 0.02
    --teacher-greedy-steps 16 --teacher-greedy-max-curves 2
    --teacher-geometry-distillation-weight 0.2 --count-reserve-alignment
    --synthetic-simple-fraction 0.35 --synthetic-shape-fraction 0.25)
  SAFETY_ARGS=(--one-shot-safety-sigma 0 --one-shot-safety-knots 0
    --final-safety-sigma 0 --final-safety-knots 0 --safety-anneal-epochs 8)
  RESAMPLE_ARGS=(--resample-train-each-epoch)
  if [[ -z "$CHECKPOINT" && "$RESUME" == 0 ]]; then
    [[ -n "$WARM_START_CHECKPOINT" ]] || die "--compact-selection requires an explicit --warm-start-checkpoint for a new run (use the selected best .pt, not .last.pt)"
  fi
elif ((RELIABLE_SELECTION)); then
  RUN_NAME=${RUN_NAME:-overnight_reliable_3090_r1}
  EPOCHS=${EPOCHS:-32}
  PROPOSAL_EPOCHS=${PROPOSAL_EPOCHS:-4}
  LEARNING_ARGS=(--policy-samples 2 --counterfactual-edits 4 --teacher-prefix-search-steps 6
    --lr 5e-5 --joint-lr 5e-5 --teacher-refinement-steps 1 --teacher-refinement-candidates 3
    --boundary-ranking-weight 0.5 --boundary-ranking-candidates 4
    --joint-proposal-lr-scale 0.25 --joint-decoder-lr-scale 0.5 --joint-final-lr-ratio 0.25
    --parameter-trust-enabled --parameter-trust-initial 0.25
    --parameter-counterfactual-weight 0.25 --teacher-geometry-candidates 4
    --local-fit-weight 0.1 --proposal-ordered-weight 0.5 --allow-infeasible-proposals)
elif ((ENHANCED_SELECTION)); then
  RUN_NAME=${RUN_NAME:-overnight_plus_3090_r1}
  EPOCHS=${EPOCHS:-96}
  PROPOSAL_EPOCHS=${PROPOSAL_EPOCHS:-12}
  LEARNING_ARGS=(--policy-samples 4 --counterfactual-edits 6 --teacher-prefix-search-steps 8
    --lr 5e-5 --joint-lr 5e-5
    --teacher-refinement-steps 1 --teacher-refinement-candidates 3
    --boundary-ranking-weight 0.5 --boundary-ranking-candidates 4
    --joint-proposal-lr-scale 0.25 --joint-decoder-lr-scale 0.5 --joint-final-lr-ratio 0.25)
else
  RUN_NAME=${RUN_NAME:-overnight_1070_arch_3090_r1}
  EPOCHS=${EPOCHS:-64}
  PROPOSAL_EPOCHS=${PROPOSAL_EPOCHS:-4}
  LEARNING_ARGS=(--policy-samples 2 --counterfactual-edits 2 --teacher-prefix-search-steps 6)
fi
if [[ -n "$WARM_START_CHECKPOINT" ]]; then
  ((INIT_SPECIFIED == 0)) || die "--warm-start-checkpoint cannot be combined with --init-checkpoint or --no-init-checkpoint"
  [[ -z "$CHECKPOINT" && "$RESUME" == 0 ]] || die "warm-start, evaluation and resume are mutually exclusive"
  INIT_CHECKPOINT=""
fi
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "invalid run name"
[[ "$DEVICE" =~ ^(auto|cpu|cuda)$ ]] || die "device must be auto, cpu or cuda"
[[ "$BENCHMARK_PROFILE" =~ ^(quick|full)$ ]] || die "benchmark profile must be quick or full"
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || die "num-workers must be non-negative"
[[ "$MSE_TOLERANCE" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || die "invalid MSE tolerance"
awk 'BEGIN { n=ARGV[1]+0; exit !(n>0 && (n-n)==0) }' "$MSE_TOLERANCE" || die "MSE tolerance must be finite and positive"
FORCE_DIAGNOSTIC=()
if [[ "$BENCHMARK_PROFILE" == quick ]]; then
  SYNTHETIC_SAMPLES=${SYNTHETIC_SAMPLES:-1}
  REAL_SAMPLES=${REAL_SAMPLES:-2}
  VISUAL_SAMPLES=${VISUAL_SAMPLES:-1}
  NETWORK_REPEATS=${NETWORK_REPEATS:-5}
  END_TO_END_REPEATS=${END_TO_END_REPEATS:-1}
  KANG_ITERATIONS=100
  LUO_ITERATIONS=10
  FORCE_DIAGNOSTIC=(--force-diagnostic)
else
  SYNTHETIC_SAMPLES=${SYNTHETIC_SAMPLES:-2}
  REAL_SAMPLES=${REAL_SAMPLES:-8}
  VISUAL_SAMPLES=${VISUAL_SAMPLES:-2}
  NETWORK_REPEATS=${NETWORK_REPEATS:-100}
  END_TO_END_REPEATS=${END_TO_END_REPEATS:-3}
  KANG_ITERATIONS=400
  LUO_ITERATIONS=50
fi
for number in "$EPOCHS" "$PROPOSAL_EPOCHS" "$TRAIN_SIZE" "$VAL_SIZE" "$BATCH_SIZE" \
  "$REAL_VAL_SIZE" "$SYNTHETIC_SAMPLES" "$REAL_SAMPLES" "$VISUAL_SAMPLES" "$NETWORK_REPEATS" "$END_TO_END_REPEATS"; do
  [[ "$number" =~ ^[1-9][0-9]*$ ]] || die "sizes, epochs and repeats must be positive integers"
done
((PROPOSAL_EPOCHS < EPOCHS)) || die "Proposal epochs must be less than total epochs"
[[ -z "$CHECKPOINT" || "$RESUME" == 0 ]] || die "--checkpoint and --resume-run are mutually exclusive"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd -- "$ROOT"
absolute_path() {
  local value="$1"
  [[ "$value" == /* ]] || value="$ROOT/$value"
  readlink -m -- "$value"
}
DATA_ROOT="$(absolute_path "${DATA_ROOT:-data}")"
OUTPUT_ROOT="$(absolute_path "${OUTPUT_ROOT:-outputs}")"
if [[ -n "$INIT_CHECKPOINT" ]]; then INIT_CHECKPOINT="$(absolute_path "$INIT_CHECKPOINT")"; fi
if [[ -n "$WARM_START_CHECKPOINT" ]]; then
  WARM_START_CHECKPOINT="$(absolute_path "$WARM_START_CHECKPOINT")"
  [[ -f "$WARM_START_CHECKPOINT" ]] || die "full-model warm-start checkpoint is missing: $WARM_START_CHECKPOINT"
fi
TRAIN_OUTPUT="$OUTPUT_ROOT/checkpoints/$RUN_NAME.pt"
LAST_PATH="$OUTPUT_ROOT/checkpoints/$RUN_NAME.last.pt"
LOG_DIR="$OUTPUT_ROOT/logs/$RUN_NAME"
COMPARISON_DIR="$OUTPUT_ROOT/comparisons/$RUN_NAME"
FIGURE_DIR="$OUTPUT_ROOT/figures/$RUN_NAME"
if [[ -n "$CHECKPOINT" ]]; then
  CHECKPOINT="$(absolute_path "$CHECKPOINT")"
  [[ -f "$CHECKPOINT" ]] || die "evaluation checkpoint is missing: $CHECKPOINT"
else
  CHECKPOINT="$TRAIN_OUTPUT"
  if ((RESUME)); then
    [[ -f "$LAST_PATH" ]] || die "--resume-run requires $LAST_PATH"
  elif [[ -n "$INIT_CHECKPOINT" ]]; then
    [[ -f "$INIT_CHECKPOINT" ]] || die "initializer is missing: $INIT_CHECKPOINT; use --no-init-checkpoint only for a fresh ablation"
  fi
fi
if ((RESUME == 0)); then
  for target in "$TRAIN_OUTPUT" "$LAST_PATH" "$OUTPUT_ROOT/checkpoints/$RUN_NAME.proposal.pt" \
    "$OUTPUT_ROOT/checkpoints/$RUN_NAME.history.json" "$LOG_DIR" "$COMPARISON_DIR" "$FIGURE_DIR"; do
    [[ ! -e "$target" ]] || die "refusing to overwrite existing run artifact: $target"
  done
fi
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
PHASE=initializing
trap 'status=$?; printf "Pipeline failed during %s (exit %s). Logs: %s\n" "$PHASE" "$status" "$LOG_DIR" >&2; exit "$status"' ERR
run_logged() {
  PHASE="$1"; shift
  local command_text
  printf -v command_text ' %q' "$@"
  if ((DRY_RUN)); then
    printf '\n[%s]%s\n' "$PHASE" "$command_text"
    return 0
  fi
  printf '\n[%s]%s\n' "$PHASE" "$command_text" | tee -a "$LOG_DIR/$PHASE.log"
  if "$@" 2>&1 | tee -a "$LOG_DIR/$PHASE.log"; then return 0; else return $?; fi
}
if ((DRY_RUN == 0)); then
  command -v "$PYTHON_BIN" >/dev/null || die "Python executable not found: $PYTHON_BIN"
  mkdir -p -- "$LOG_DIR" "$OUTPUT_ROOT/checkpoints"
fi
printf 'Historical overnight: online Teacher; Kc=64; source K=4..24; MSE=%s.\n' "$MSE_TOLERANCE"
if ((ENHANCED_SELECTION)); then
  printf 'Enhanced selection: feasible teacher remove/add/swap refinement + boundary ranking; unchanged one-shot deployment.\n'
fi
if ((RELIABLE_SELECTION)); then
  printf 'Reliable selection: trust gates, ranking-independent Teacher, local fit and ordered coverage; one-shot deployment.\n'
fi
if ((COMPACT_SELECTION)); then
  printf 'Compact selection: per-curve feasible simplification, bounded greedy Teacher and count reserve alignment; unchanged one-shot deployment.\n'
  printf 'Teacher geometry is checked separately from decoded deployment; no global-minimum or pass-rate guarantee.\n'
fi
if ((RELIABLE_SELECTION || COMPACT_SELECTION)); then
  printf 'Real data: validation-only (%s/source); synthetic-only training; held-out test for comparison.\n' "$REAL_VAL_SIZE"
fi
printf 'Proposal %s + Joint %s = %s epochs; train/val=%s/%s, batch=%s.\n' \
  "$PROPOSAL_EPOCHS" "$((EPOCHS-PROPOSAL_EPOCHS))" "$EPOCHS" "$TRAIN_SIZE" "$VAL_SIZE" "$BATCH_SIZE"
printf 'Six methods, four datasets. Profile=%s (full is an experimental budget, not a claim of qualification).\n' "$BENCHMARK_PROFILE"
run_logged check_environment "$PYTHON_BIN" scripts/overnight_linux_preflight.py --runtime-only --device "$DEVICE"

UJI="$DATA_ROOT/splits/uji_pen_v2.jsonl"
NATURAL="$DATA_ROOT/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl"
USGS="$DATA_ROOT/processed/usgs_contours/large_scale/manifest.jsonl"
if ((PREPARE)); then
  if [[ ! -f "$UJI" ]]; then
    run_logged prepare_uji "$PYTHON_BIN" scripts/prepare_uji_pen.py --download \
      --raw-dir "$DATA_ROOT/raw/uji_pen_v2" --output-dir "$DATA_ROOT/processed/uji_pen_v2" \
      --manifest "$UJI" --split-summary "$DATA_ROOT/splits/uji_pen_v2_split.json"
  fi
  if [[ ! -f "$NATURAL" ]]; then
    run_logged prepare_natural_earth "$PYTHON_BIN" scripts/prepare_natural_earth.py \
      --raw-dir "$DATA_ROOT/raw/natural_earth" --output-dir "$(dirname -- "$NATURAL")" \
      --resolution 10m --layer coastline --reference-points 768
  fi
  if [[ ! -f "$USGS" ]]; then
    run_logged prepare_usgs "$PYTHON_BIN" scripts/prepare_usgs_contours.py \
      --bbox-file configs/usgs_contour_regions.example.json --max-features-per-region 2000 \
      --raw-dir "$DATA_ROOT/raw/usgs_contours" --output-dir "$(dirname -- "$USGS")"
  fi
fi
if ((DRY_RUN == 0)); then
  for manifest in "$UJI" "$NATURAL" "$USGS"; do
    [[ -f "$manifest" ]] || die "real-data manifest missing: $manifest; use --data-root or --prepare-real-data"
  done
fi
STAMP="$(date -u +%Y%m%d_%H%M%S)_$$"
PREFLIGHT_ARGS=(--data-root "$DATA_ROOT" --device "$DEVICE" --mse-tolerance "$MSE_TOLERANCE")
if ((RELIABLE_SELECTION || COMPACT_SELECTION)); then PREFLIGHT_ARGS+=(--validate-real-splits); fi
if [[ "$CHECKPOINT" != "$TRAIN_OUTPUT" ]]; then
  PREFLIGHT_ARGS+=(--checkpoint "$CHECKPOINT")
elif [[ -n "$INIT_CHECKPOINT" && "$RESUME" == 0 ]]; then
  PREFLIGHT_ARGS+=(--initializer "$INIT_CHECKPOINT")
elif [[ -n "$WARM_START_CHECKPOINT" ]]; then
  PREFLIGHT_ARGS+=(--warm-start-checkpoint "$WARM_START_CHECKPOINT")
fi
PREFLIGHT_OUTPUT="$LOG_DIR/preflight.json"
if ((RESUME)); then PREFLIGHT_OUTPUT="$LOG_DIR/preflight_$STAMP.json"; fi
run_logged check_data_and_provenance "$PYTHON_BIN" scripts/overnight_linux_preflight.py \
  "${PREFLIGHT_ARGS[@]}" --output "$PREFLIGHT_OUTPUT"

if [[ "$CHECKPOINT" == "$TRAIN_OUTPUT" ]]; then
  TRAIN_ARGS=(scripts/train_v16.py --epochs "$EPOCHS" --proposal-epochs "$PROPOSAL_EPOCHS"
    --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" --batch-size "$BATCH_SIZE"
    --num-points 192 --min-control-points 8 --max-control-points 28 --candidate-knots 64
    --mse-tolerance "$MSE_TOLERANCE" --real-fraction 0 --proposal-pass-target 0.90
    --deployment-pass-target 0.90 "${LEARNING_ARGS[@]}"
    "${SAFETY_ARGS[@]}"
    --complexity-ramp-epochs 8 "${RESAMPLE_ARGS[@]}" --num-workers "$NUM_WORKERS"
    --torch-num-threads 4 --device "$DEVICE" --output "$TRAIN_OUTPUT")
  if ((RELIABLE_SELECTION || COMPACT_SELECTION)); then
    TRAIN_ARGS+=(--real-val-size "$REAL_VAL_SIZE"
      --real-manifest "$UJI" --real-manifest "$NATURAL" --real-manifest "$USGS")
  fi
  if ((RESUME)); then
    TRAIN_ARGS+=(--resume "$LAST_PATH")
    run_logged check_resume_status "$PYTHON_BIN" scripts/overnight_linux_preflight.py \
      --resume-status --checkpoint "$LAST_PATH" -- "${TRAIN_ARGS[@]:1}"
    RESUME_STATUS=needed
    if ((DRY_RUN == 0)); then
      RESUME_STATUS="$(tail -n 1 "$LOG_DIR/check_resume_status.log")"
      RESUME_STATUS=${RESUME_STATUS%$'\r'}
    fi
    case "$RESUME_STATUS" in
      completed)
        printf 'Same-run checkpoint already completed the requested epochs; skip training and continue benchmark/plots.\n' \
          | tee -a "$LOG_DIR/check_resume_status.log"
        ;;
      needed) run_logged train_resume "$PYTHON_BIN" "${TRAIN_ARGS[@]}" ;;
      *) die "unexpected resume status: $RESUME_STATUS" ;;
    esac
  else
    if [[ -n "$INIT_CHECKPOINT" ]]; then TRAIN_ARGS+=(--init-checkpoint "$INIT_CHECKPOINT"); fi
    if [[ -n "$WARM_START_CHECKPOINT" ]]; then TRAIN_ARGS+=(--warm-start-checkpoint "$WARM_START_CHECKPOINT"); fi
    run_logged train_fresh "$PYTHON_BIN" "${TRAIN_ARGS[@]}"
  fi
fi
if ((DRY_RUN == 0)); then
  [[ -f "$CHECKPOINT" ]] || die "training produced no deployable checkpoint; inspect training logs"
fi
if run_logged inspect_checkpoint "$PYTHON_BIN" scripts/inspect_v16_checkpoint.py \
  --checkpoint "$CHECKPOINT" --mse-tolerance "$MSE_TOLERANCE"; then
  printf 'Checkpoint inspection completed.\n'
else
  printf 'Checkpoint is not qualified for formal reporting; continue only with diagnostic labels.\n'
fi
MANIFEST_ARGS=(--manifest "UJI=$UJI" --manifest "NaturalEarth=$NATURAL" --manifest "USGS=$USGS")
BASELINE_ARGS=(--max-internal-knots 64 --paper-initial-knots 64 --liang-dense-knots 64
  --gradient-steps 12 --paper-admm-iterations "$KANG_ITERATIONS" --paper-lambda-bisections 8
  --paper-relocation-iterations 8 --liang-feature-samples 1025 --dung-scan-intervals 10
  --dung-optimization-iterations 10 --luo-de-population 10 --luo-de-iterations "$LUO_ITERATIONS")
if ((NATIVE_BASELINES)); then
  BASELINE_ARGS+=(--no-published-feasibility-safeguard)
else
  BASELINE_ARGS+=(--published-feasibility-safeguard)
fi
TIMING_ARGS=(--network-warmups 3 --network-repeats "$NETWORK_REPEATS"
  --end-to-end-repeats "$END_TO_END_REPEATS" --torch-num-threads 4 --device "$DEVICE")
BENCHMARK_RESUME=()
if ((RESUME)) && [[ -f "$COMPARISON_DIR/experiment.json" ]]; then BENCHMARK_RESUME=(--resume); fi
run_logged benchmark_six_methods "$PYTHON_BIN" scripts/benchmark_v16_datasets.py \
  --checkpoint "$CHECKPOINT" --output-dir "$COMPARISON_DIR" --method-set published \
  --samples-per-knot-count "$SYNTHETIC_SAMPLES" --min-knot-count 4 --max-knot-count 24 \
  --real-samples-per-dataset "$REAL_SAMPLES" --mse-tolerance "$MSE_TOLERANCE" \
  "${MANIFEST_ARGS[@]}" "${BASELINE_ARGS[@]}" "${TIMING_ARGS[@]}" \
  --allow-unqualified-diagnostic "${FORCE_DIAGNOSTIC[@]}" "${BENCHMARK_RESUME[@]}"
if ((RESUME)); then FIGURE_DIR="$FIGURE_DIR/attempt_$STAMP"; fi
run_logged plot_four_metrics "$PYTHON_BIN" scripts/plot_v16_method_comparison.py \
  --input "$COMPARISON_DIR/comparison.json" --output-dir "$FIGURE_DIR/four_metrics" \
  --method-set published --dpi 300 --reference --allow-unqualified-diagnostic
CASE_ARGS=(--checkpoint "$CHECKPOINT" --real-samples-per-dataset "$VISUAL_SAMPLES"
  --selection-seed 20260909 --mse-tolerance "$MSE_TOLERANCE"
  "${MANIFEST_ARGS[@]}" "${BASELINE_ARGS[@]}" "${TIMING_ARGS[@]}"
  --dpi 240 --allow-unqualified-diagnostic "${FORCE_DIAGNOSTIC[@]}")
run_logged plot_ours_cases "$PYTHON_BIN" scripts/visualize_v16_ours_cases.py \
  "${CASE_ARGS[@]}" --output-dir "$FIGURE_DIR/ours_cases"
run_logged plot_six_method_real_cases "$PYTHON_BIN" scripts/visualize_v16_real_deployments.py \
  "${CASE_ARGS[@]}" --method-set published --output-dir "$FIGURE_DIR/six_method_real_cases"
PHASE=completed
printf '\nCompleted: checkpoint=%s\ncomparison=%s\nfigures=%s\nlogs=%s\n' \
  "$CHECKPOINT" "$COMPARISON_DIR" "$FIGURE_DIR" "$LOG_DIR"
