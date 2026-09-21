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
CANDIDATE_KNOTS=64
CANDIDATE_SPECIFIED=0
SOURCE_MAX_KNOTS=24
TRAINING_SOURCE=synthetic
TRAINING_REAL_FRACTION=0.5
VALIDATION_SOURCE=all
EXTRA_LEARNING_ARGS=()
RESIZE_CANDIDATE_WARM_START=0
NUM_WORKERS=0
INIT_CHECKPOINT=outputs/checkpoints/candidate_selection_v16.proposal.pt
INIT_SPECIFIED=0
WARM_START_CHECKPOINT=""
ENHANCED_SELECTION=0
RELIABLE_SELECTION=0
COMPACT_SELECTION=0
STABLE_SELECTION=0
ANCHORED_SELECTION=0
JOINT_GEOMETRY_CALIBRATION_EPOCHS=""
COMPLEXITY_RAMP_EPOCHS=8
REAL_VAL_SIZE=32
NATIVE_BASELINES=0
BASELINE_PROTOCOL=adaptation
PAPER_OUTPUT=0
ALL_REAL_TEST_SAMPLES=0
CHECKPOINT=""
DATA_ROOT=""
EXTRA_MANIFESTS=()
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
Historical warm start -> Synthetic + UJI/NaturalEarth/USGS/IndustrialOffset benchmark
-> four/five-metric PNGs -> Ours cases -> six-method real-case PNGs.
Historical online-teacher architecture; capacity is explicit and shared by all methods.

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
  --stable-selection              Compact + decoded trajectory checks (4) and
                                  trajectory targets (2); explicit full warm start
  --anchored-selection            Short K32 warm-start calibration: 16/4 epochs,
                                  6 frozen-Proposal joint epochs + 6 low-LR joint;
                                  compact-priority Teacher and anchored geometry
  --joint-geometry-calibration-epochs N
                                  Freeze dense geometry and pause complexity for
                                  first N Joint epochs (anchored 6; otherwise 0)
  --real-val-size N               Reliable/compact/stable: real val curves/source (32)
  --native-baselines              Disable explicit MSE repair for Dung/Kang/Luo;
                                  NOT a certificate of original-paper fidelity
  --baseline-protocol adaptation|native
                                  Native rejects unverified reproductions (N/A), never silently repairs
  --paper-output                  Save full per-case results; draw from saved fits, without reruns
  --all-real-test-samples          Evaluate every held-out test record in every real manifest
  --train-size N --val-size N      Synthetic training/validation (1500/500)
  --batch-size N --num-workers N   Default 32/0
  --mse-tolerance X                Shared MSE threshold, not RMS (5e-5)
  --candidate-knots N              Shared INTERNAL knot cap for network + baselines
                                  (legacy default 64; use 32 for the new experiment)
  --source-max-knots N             Synthetic training source maximum (24); <= candidate cap
                                  Comparison still tests the SAME synthetic K4..24 curves
  --training-source NAME           synthetic (default), UJI, NaturalEarth, USGS,
                                  IndustrialOffset; specialist uses ONLY that train split
  --training-real-fraction X       Specialist train-split fraction (0.5); remainder synthetic
  --validation-source NAME         all (three observed sources), or one source above
  --synthetic-shape-domain NAME    mixed|handwriting|terrain|industrial; synthetic-only specialization
  --candidate-refinement-layers N  Extra gated candidate interaction blocks (0)
  --selection-refinement-layers N  Extra gated pre-mask interaction blocks (0)
  --decoder-refinement-layers N    Extra survivor/parameter interaction blocks (0)
  --coupled-proposal-steps N       Actual parameter/knot updates before selection (0)
  --coupled-subset-steps N         Actual parameter/knot updates after selection (0)
  --parameter-chord-blend X        Chord anchor fraction for the new run (0)
  --max-point-error-weight X       Optional peak/tail squared-error training weight (0)
  --max-point-error-tolerance X    Explicit peak SQUARED-distance target; MSE pass unchanged
  --max-point-error-tail-fraction X  Worst-point training fraction (0.05)
  --init-checkpoint PATH           Historical Proposal warm-start checkpoint
  --warm-start-checkpoint PATH     Copy ALL compatible overnight model weights
                                  into a NEW experiment (not optimizer resume)
  --resize-candidate-warm-start    Explicit smaller-capacity migration; resample only
                                  candidate interval queries, then retrain (not resume)
  --no-init-checkpoint             Explicit from-scratch ablation (not original run)
  --checkpoint PATH               Skip training; checkpoint capacity must match N
  --resume-run                     Resume same run; skip training if epochs completed
  --data-root PATH                 Complete data tree (default repository data/)
  --manifest NAME=PATH             Add an independent evaluation-only source;
                                  repeat; never changes training/validation data
  --prepare-real-data              Prepare missing real manifests without overwriting
                                  IndustrialOffset is generated locally, not measured
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
    --stable-selection) STABLE_SELECTION=1; shift ;;
    --anchored-selection) ANCHORED_SELECTION=1; shift ;;
    --joint-geometry-calibration-epochs) need_value "$@"; JOINT_GEOMETRY_CALIBRATION_EPOCHS="$2"; shift 2 ;;
    --source-max-knots) need_value "$@"; SOURCE_MAX_KNOTS="$2"; shift 2 ;;
    --training-source) need_value "$@"; TRAINING_SOURCE="$2"; shift 2 ;;
    --training-real-fraction) need_value "$@"; TRAINING_REAL_FRACTION="$2"; shift 2 ;;
    --validation-source) need_value "$@"; VALIDATION_SOURCE="$2"; shift 2 ;;
    --synthetic-shape-domain|--synthetic-shape-fraction|--synthetic-simple-fraction|--candidate-refinement-layers|--selection-refinement-layers|--decoder-refinement-layers|--max-point-error-weight|--max-point-error-tolerance|--max-point-error-tail-fraction|--coupled-proposal-steps|--coupled-subset-steps|--parameter-chord-blend|--joint-lr|--joint-proposal-lr-scale|--joint-decoder-lr-scale|--teacher-geometry-distillation-weight|--seed)
      need_value "$@"; EXTRA_LEARNING_ARGS+=("$1" "$2"); shift 2 ;;
    --real-val-size) need_value "$@"; REAL_VAL_SIZE="$2"; shift 2 ;;
    --native-baselines) NATIVE_BASELINES=1; shift ;;
    --baseline-protocol) need_value "$@"; BASELINE_PROTOCOL="$2"; shift 2 ;;
    --paper-output) PAPER_OUTPUT=1; shift ;;
    --all-real-test-samples) ALL_REAL_TEST_SAMPLES=1; shift ;;
    --train-size) need_value "$@"; TRAIN_SIZE="$2"; shift 2 ;;
    --val-size) need_value "$@"; VAL_SIZE="$2"; shift 2 ;;
    --batch-size) need_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) need_value "$@"; NUM_WORKERS="$2"; shift 2 ;;
    --mse-tolerance) need_value "$@"; MSE_TOLERANCE="$2"; shift 2 ;;
    --candidate-knots) need_value "$@"; CANDIDATE_KNOTS="$2"; CANDIDATE_SPECIFIED=1; shift 2 ;;
    --resize-candidate-warm-start) RESIZE_CANDIDATE_WARM_START=1; shift ;;
    --init-checkpoint) need_value "$@"; INIT_CHECKPOINT="$2"; INIT_SPECIFIED=1; shift 2 ;;
    --no-init-checkpoint) INIT_CHECKPOINT=""; INIT_SPECIFIED=1; shift ;;
    --warm-start-checkpoint) need_value "$@"; WARM_START_CHECKPOINT="$2"; shift 2 ;;
    --checkpoint) need_value "$@"; CHECKPOINT="$2"; shift 2 ;;
    --data-root) need_value "$@"; DATA_ROOT="$2"; shift 2 ;;
    --manifest) need_value "$@"; EXTRA_MANIFESTS+=("$2"); shift 2 ;;
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
((ENHANCED_SELECTION + RELIABLE_SELECTION + COMPACT_SELECTION + STABLE_SELECTION + ANCHORED_SELECTION <= 1)) || die "choose only one of --enhanced-selection, --reliable-selection, --compact-selection, --stable-selection and --anchored-selection"
if ((ANCHORED_SELECTION && CANDIDATE_SPECIFIED == 0)); then CANDIDATE_KNOTS=32; fi
[[ "$CANDIDATE_KNOTS" =~ ^[1-9][0-9]*$ ]] || die "candidate-knots must be a positive integer"
[[ "$SOURCE_MAX_KNOTS" =~ ^[1-9][0-9]*$ ]] || die "source-max-knots must be a positive integer"
((SOURCE_MAX_KNOTS >= 4 && SOURCE_MAX_KNOTS <= 188)) || die "source-max-knots must be 4..188"
((CANDIDATE_KNOTS >= SOURCE_MAX_KNOTS && CANDIDATE_KNOTS <= 188)) || die "candidate-knots must be $SOURCE_MAX_KNOTS..188 for this synthetic source range and 192 input points"
case "$TRAINING_SOURCE" in synthetic|UJI|NaturalEarth|USGS|IndustrialOffset) ;; *) die "unknown training source: $TRAINING_SOURCE" ;; esac
case "$VALIDATION_SOURCE" in all|UJI|NaturalEarth|USGS|IndustrialOffset) ;; *) die "unknown validation source: $VALIDATION_SOURCE" ;; esac
if [[ "$TRAINING_SOURCE" != synthetic && "$VALIDATION_SOURCE" != all && "$VALIDATION_SOURCE" != "$TRAINING_SOURCE" ]]; then
  die "specialist real training and validation must name the same source"
fi
if [[ "$TRAINING_SOURCE" != synthetic ]]; then VALIDATION_SOURCE="$TRAINING_SOURCE"; fi
awk 'BEGIN { n=ARGV[1]+0; exit !(n>0 && n<=1) }' "$TRAINING_REAL_FRACTION" || die "training-real-fraction must be in (0,1]"
if ((RESIZE_CANDIDATE_WARM_START)); then
  [[ -n "$WARM_START_CHECKPOINT" && -z "$CHECKPOINT" && "$RESUME" == 0 ]] || die "--resize-candidate-warm-start requires a new --warm-start-checkpoint run, not evaluation or resume"
fi
SAFETY_ARGS=(--one-shot-safety-sigma 0.2 --safety-anneal-epochs 8)
RESAMPLE_ARGS=(--no-resample-train-each-epoch)
if ((COMPACT_SELECTION || STABLE_SELECTION || ANCHORED_SELECTION)); then
  if ((ANCHORED_SELECTION)); then
    RUN_NAME=${RUN_NAME:-overnight_anchored_k32_3090_r1}
    EPOCHS=${EPOCHS:-16}
    JOINT_GEOMETRY_CALIBRATION_EPOCHS=${JOINT_GEOMETRY_CALIBRATION_EPOCHS:-6}
    COMPLEXITY_RAMP_EPOCHS=4
  elif ((STABLE_SELECTION)); then
    RUN_NAME=${RUN_NAME:-overnight_stable_3090_r1}
  else
    RUN_NAME=${RUN_NAME:-overnight_compact_3090_r1}
  fi
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
  if ((STABLE_SELECTION || ANCHORED_SELECTION)); then
    LEARNING_ARGS+=(--teacher-greedy-trajectory-checks 4 --teacher-geometry-trajectory-targets 2)
  fi
  if ((ANCHORED_SELECTION)); then
    # Replace inherited settings rather than emitting duplicate options.
    for ((index=0; index<${#LEARNING_ARGS[@]}; index++)); do
      case "${LEARNING_ARGS[index]}" in
        --teacher-greedy-max-curves) LEARNING_ARGS[index+1]=4 ;;
        --teacher-geometry-distillation-weight) LEARNING_ARGS[index+1]=0.4 ;;
        --joint-decoder-lr-scale) LEARNING_ARGS[index+1]=1.0 ;;
      esac
    done
    LEARNING_ARGS+=(--subset-geometry-mode anchored --subset-geometry-residual-scale 0.25
      --teacher-greedy-priority compact --teacher-compact-mask-weight 0.5)
  fi
  SAFETY_ARGS=(--one-shot-safety-sigma 0 --one-shot-safety-knots 0
    --final-safety-sigma 0 --final-safety-knots 0 --safety-anneal-epochs 8)
  if ((ANCHORED_SELECTION)); then SAFETY_ARGS[${#SAFETY_ARGS[@]}-1]=4; fi
  RESAMPLE_ARGS=(--resample-train-each-epoch)
  if [[ -z "$CHECKPOINT" && "$RESUME" == 0 ]]; then
    [[ -n "$WARM_START_CHECKPOINT" ]] || die "compact/stable/anchored selection requires an explicit --warm-start-checkpoint for a new run (use the selected best .pt, not .last.pt)"
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
JOINT_GEOMETRY_CALIBRATION_EPOCHS=${JOINT_GEOMETRY_CALIBRATION_EPOCHS:-0}
# Replace inherited defaults so manifests/dry-runs contain each option once.
for ((extra=0; extra<${#EXTRA_LEARNING_ARGS[@]}; extra+=2)); do
  replaced=0
  for ((index=0; index<${#LEARNING_ARGS[@]}; index++)); do
    if [[ "${LEARNING_ARGS[index]}" == "${EXTRA_LEARNING_ARGS[extra]}" ]]; then
      LEARNING_ARGS[index+1]="${EXTRA_LEARNING_ARGS[extra+1]}"; replaced=1; break
    fi
  done
  if ((replaced == 0)); then LEARNING_ARGS+=("${EXTRA_LEARNING_ARGS[extra]}" "${EXTRA_LEARNING_ARGS[extra+1]}"); fi
done
[[ "$JOINT_GEOMETRY_CALIBRATION_EPOCHS" =~ ^[0-9]+$ ]] || die "joint-geometry-calibration-epochs must be non-negative"
if ((JOINT_GEOMETRY_CALIBRATION_EPOCHS > 0)); then
  LEARNING_ARGS+=(--joint-geometry-calibration-epochs "$JOINT_GEOMETRY_CALIBRATION_EPOCHS")
fi
if [[ -n "$WARM_START_CHECKPOINT" ]]; then
  ((INIT_SPECIFIED == 0)) || die "--warm-start-checkpoint cannot be combined with --init-checkpoint or --no-init-checkpoint"
  [[ -z "$CHECKPOINT" && "$RESUME" == 0 ]] || die "warm-start, evaluation and resume are mutually exclusive"
  INIT_CHECKPOINT=""
fi
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "invalid run name"
[[ "$DEVICE" =~ ^(auto|cpu|cuda)$ ]] || die "device must be auto, cpu or cuda"
[[ "$BENCHMARK_PROFILE" =~ ^(quick|full)$ ]] || die "benchmark profile must be quick or full"
[[ "$BASELINE_PROTOCOL" =~ ^(adaptation|native)$ ]] || die "baseline-protocol must be adaptation or native"
if [[ "$BASELINE_PROTOCOL" == native ]]; then NATIVE_BASELINES=1; PAPER_OUTPUT=1; fi
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
((JOINT_GEOMETRY_CALIBRATION_EPOCHS <= EPOCHS-PROPOSAL_EPOCHS)) || die "joint-geometry-calibration-epochs cannot exceed total Joint epochs"
[[ -z "$CHECKPOINT" || "$RESUME" == 0 ]] || die "--checkpoint and --resume-run are mutually exclusive"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd -- "$ROOT"
absolute_path() {
  local value="$1"
  if [[ "$value" =~ ^[A-Za-z]:[\\/] ]]; then
    command -v cygpath >/dev/null 2>&1 || die "Windows path requires Git Bash/cygpath; use Linux paths on the server"
    value="$(cygpath -u -- "$value")"
  fi
  [[ "$value" == /* ]] || value="$ROOT/$value"
  readlink -m -- "$value"
}
DATA_ROOT="$(absolute_path "${DATA_ROOT:-data}")"
EXTRA_MANIFEST_ARGS=()
EXTRA_MANIFEST_NAMES=(UJI NaturalEarth USGS IndustrialOffset Synthetic)
for entry in "${EXTRA_MANIFESTS[@]}"; do
  [[ "$entry" == *=* && -n "${entry%%=*}" && -n "${entry#*=}" ]] || die "--manifest must be NAME=PATH"
  name=${entry%%=*}
  for existing in "${EXTRA_MANIFEST_NAMES[@]}"; do
    [[ "$name" != "$existing" ]] || die "duplicate or reserved dataset name: $name"
  done
  EXTRA_MANIFEST_NAMES+=("$name")
  EXTRA_MANIFEST_ARGS+=(--manifest "$name=$(absolute_path "${entry#*=}")")
done
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
printf 'Historical overnight: online Teacher; Kc=%s internal (full cubic vector <= %s); training source K=4..%s; MSE=%s.\n' "$CANDIDATE_KNOTS" "$((CANDIDATE_KNOTS+8))" "$SOURCE_MAX_KNOTS" "$MSE_TOLERANCE"
printf 'All six methods use the same internal-knot cap. Pointwise maximum squared error is reported separately; pass remains an MSE test.\n'
if [[ "$CHECKPOINT" != "$TRAIN_OUTPUT" ]]; then
  printf 'Evaluation-only: using existing checkpoint %s; no training or new model selection.\n' "$CHECKPOINT"
else
if ((ENHANCED_SELECTION)); then
  printf 'Enhanced selection: feasible teacher remove/add/swap refinement + boundary ranking; unchanged one-shot deployment.\n'
fi
if ((RELIABLE_SELECTION)); then
  printf 'Reliable selection: trust gates, ranking-independent Teacher, local fit and ordered coverage; one-shot deployment.\n'
fi
if ((COMPACT_SELECTION || STABLE_SELECTION || ANCHORED_SELECTION)); then
  printf 'Compact selection: per-curve feasible simplification, bounded greedy Teacher and count reserve alignment; unchanged one-shot deployment.\n'
  printf 'Teacher geometry is checked separately from decoded deployment; no global-minimum or pass-rate guarantee.\n'
fi
if ((STABLE_SELECTION || ANCHORED_SELECTION)); then
  printf 'Stable selection: up to 4 decoded trajectory checks and 2 geometry trajectory targets; no deployment search added.\n'
fi
if ((ANCHORED_SELECTION)); then
  printf 'Anchored selection: proposal-relative geometry, bounded residual=0.25; compact-priority Teacher covers <=4 curves/batch. No deployment search.\n'
fi
if ((JOINT_GEOMETRY_CALIBRATION_EPOCHS > 0)); then
  printf 'First %s Joint epochs freeze dense geometry and pause complexity; remaining %s Joint epochs unfreeze at low LR.\n' "$JOINT_GEOMETRY_CALIBRATION_EPOCHS" "$((EPOCHS-PROPOSAL_EPOCHS-JOINT_GEOMETRY_CALIBRATION_EPOCHS))"
fi
if [[ "$TRAINING_SOURCE" != synthetic ]]; then
  printf 'Specialist: %s train split fraction=%s, remaining draws synthetic; selected source val only, ALL test sources held out.\n' "$TRAINING_SOURCE" "$TRAINING_REAL_FRACTION"
elif ((RELIABLE_SELECTION || COMPACT_SELECTION || STABLE_SELECTION || ANCHORED_SELECTION)); then
  printf 'Real data: validation-only (%s/source, selection=%s); synthetic-only training; held-out test for comparison.\n' "$REAL_VAL_SIZE" "$VALIDATION_SOURCE"
fi
printf 'Proposal %s + Joint %s = %s epochs; train/val=%s/%s, batch=%s.\n' \
  "$PROPOSAL_EPOCHS" "$((EPOCHS-PROPOSAL_EPOCHS))" "$EPOCHS" "$TRAIN_SIZE" "$VAL_SIZE" "$BATCH_SIZE"
fi
printf 'Six methods: Synthetic + three observed/derived sources + procedural IndustrialOffset, plus explicit extra sources. Profile=%s.\n' "$BENCHMARK_PROFILE"
printf 'IndustrialOffset is CAD-driven semi-synthetic, not measured industrial data; test split is never used for training or model selection.\n'
run_logged check_environment "$PYTHON_BIN" scripts/overnight_linux_preflight.py --runtime-only --device "$DEVICE"

UJI="$DATA_ROOT/splits/uji_pen_v2.jsonl"
NATURAL="$DATA_ROOT/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl"
USGS="$DATA_ROOT/processed/usgs_contours/large_scale/manifest.jsonl"
INDUSTRIAL="$DATA_ROOT/processed/industrial_offsets/v1/manifest.jsonl"
VALIDATION_MANIFEST_ARGS=()
case "$VALIDATION_SOURCE" in
  all) VALIDATION_MANIFEST_ARGS=(--real-manifest "$UJI" --real-manifest "$NATURAL" --real-manifest "$USGS") ;;
  UJI) VALIDATION_MANIFEST_ARGS=(--real-manifest "$UJI") ;;
  NaturalEarth) VALIDATION_MANIFEST_ARGS=(--real-manifest "$NATURAL") ;;
  USGS) VALIDATION_MANIFEST_ARGS=(--real-manifest "$USGS") ;;
  IndustrialOffset) VALIDATION_MANIFEST_ARGS=(--real-manifest "$INDUSTRIAL") ;;
esac
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
  if [[ ! -f "$INDUSTRIAL" ]]; then
    run_logged prepare_industrial_offsets "$PYTHON_BIN" scripts/prepare_industrial_offsets.py \
      --output-dir "$(dirname -- "$INDUSTRIAL")"
  fi
fi
if ((DRY_RUN == 0)); then
  for manifest in "$UJI" "$NATURAL" "$USGS" "$INDUSTRIAL"; do
    [[ -f "$manifest" ]] || die "required external-data manifest missing: $manifest; use --data-root or --prepare-real-data (industrial offsets are generated locally)"
  done
fi
STAMP="$(date -u +%Y%m%d_%H%M%S)_$$"
PREFLIGHT_ARGS=(--data-root "$DATA_ROOT" --device "$DEVICE" --mse-tolerance "$MSE_TOLERANCE" --candidate-knots "$CANDIDATE_KNOTS" --source-max-knots "$SOURCE_MAX_KNOTS" --training-source "$TRAINING_SOURCE" --validation-source "$VALIDATION_SOURCE" "${EXTRA_MANIFEST_ARGS[@]}")
if ((RESIZE_CANDIDATE_WARM_START)); then PREFLIGHT_ARGS+=(--resize-candidate-warm-start); fi
if ((RELIABLE_SELECTION || COMPACT_SELECTION || STABLE_SELECTION || ANCHORED_SELECTION)); then PREFLIGHT_ARGS+=(--validate-real-splits); fi
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
  EFFECTIVE_REAL_FRACTION=0
  if [[ "$TRAINING_SOURCE" != synthetic ]]; then EFFECTIVE_REAL_FRACTION="$TRAINING_REAL_FRACTION"; fi
  TRAIN_ARGS=(scripts/train_v16.py --epochs "$EPOCHS" --proposal-epochs "$PROPOSAL_EPOCHS"
    --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" --batch-size "$BATCH_SIZE"
    --num-points 192 --min-control-points 8 --max-control-points "$((SOURCE_MAX_KNOTS+4))" --candidate-knots "$CANDIDATE_KNOTS"
    --mse-tolerance "$MSE_TOLERANCE" --real-fraction "$EFFECTIVE_REAL_FRACTION" --proposal-pass-target 0.90
    --deployment-pass-target 0.90 "${LEARNING_ARGS[@]}"
    "${SAFETY_ARGS[@]}"
    --complexity-ramp-epochs "$COMPLEXITY_RAMP_EPOCHS" "${RESAMPLE_ARGS[@]}" --num-workers "$NUM_WORKERS"
    --torch-num-threads 4 --device "$DEVICE" --output "$TRAIN_OUTPUT")
  if [[ "$TRAINING_SOURCE" != synthetic ]]; then
    case "$TRAINING_SOURCE" in
      UJI) TRAIN_MANIFEST="$UJI" ;; NaturalEarth) TRAIN_MANIFEST="$NATURAL" ;;
      USGS) TRAIN_MANIFEST="$USGS" ;; IndustrialOffset) TRAIN_MANIFEST="$INDUSTRIAL" ;;
    esac
    TRAIN_ARGS+=(--real-val-size "$REAL_VAL_SIZE" --real-manifest "$TRAIN_MANIFEST")
  elif [[ "$VALIDATION_SOURCE" != all ]] || ((RELIABLE_SELECTION || COMPACT_SELECTION || STABLE_SELECTION || ANCHORED_SELECTION)); then
    TRAIN_ARGS+=(--real-val-size "$REAL_VAL_SIZE"
      "${VALIDATION_MANIFEST_ARGS[@]}")
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
    if ((RESIZE_CANDIDATE_WARM_START)); then TRAIN_ARGS+=(--resize-candidate-warm-start); fi
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
MANIFEST_ARGS=(--manifest "UJI=$UJI" --manifest "NaturalEarth=$NATURAL" --manifest "USGS=$USGS"
  --manifest "IndustrialOffset=$INDUSTRIAL" "${EXTRA_MANIFEST_ARGS[@]}")
BASELINE_ARGS=(--max-internal-knots "$CANDIDATE_KNOTS" --paper-initial-knots "$CANDIDATE_KNOTS" --liang-dense-knots "$CANDIDATE_KNOTS"
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
PAPER_BENCHMARK_ARGS=()
if ((PAPER_OUTPUT)); then PAPER_BENCHMARK_ARGS+=(--baseline-protocol "$BASELINE_PROTOCOL"); fi
if ((ALL_REAL_TEST_SAMPLES)); then PAPER_BENCHMARK_ARGS+=(--all-real-test-samples); fi
if ((RESUME)) && [[ -f "$COMPARISON_DIR/experiment.json" ]]; then BENCHMARK_RESUME=(--resume); fi
run_logged benchmark_six_methods "$PYTHON_BIN" scripts/benchmark_v16_datasets.py \
  --checkpoint "$CHECKPOINT" --output-dir "$COMPARISON_DIR" --method-set published \
  --samples-per-knot-count "$SYNTHETIC_SAMPLES" --min-knot-count 4 --max-knot-count 24 \
  --synthetic-source-max-knots 24 \
  --real-samples-per-dataset "$REAL_SAMPLES" --mse-tolerance "$MSE_TOLERANCE" \
  "${MANIFEST_ARGS[@]}" "${BASELINE_ARGS[@]}" "${TIMING_ARGS[@]}" \
  --allow-unqualified-diagnostic "${FORCE_DIAGNOSTIC[@]}" "${BENCHMARK_RESUME[@]}" "${PAPER_BENCHMARK_ARGS[@]}"
if ((RESUME)); then FIGURE_DIR="$FIGURE_DIR/attempt_$STAMP"; fi
run_logged plot_four_metrics "$PYTHON_BIN" scripts/plot_v16_method_comparison.py \
  --input "$COMPARISON_DIR/comparison.json" --output-dir "$FIGURE_DIR/four_metrics" \
  --method-set published --dpi 300 --reference --allow-unqualified-diagnostic
CASE_ARGS=(--checkpoint "$CHECKPOINT" --real-samples-per-dataset "$VISUAL_SAMPLES"
  --selection-seed 20260909 --mse-tolerance "$MSE_TOLERANCE"
  "${MANIFEST_ARGS[@]}" "${BASELINE_ARGS[@]}" "${TIMING_ARGS[@]}"
  --dpi 240 --allow-unqualified-diagnostic "${FORCE_DIAGNOSTIC[@]}")
if ((PAPER_OUTPUT)); then
  run_logged plot_saved_six_method_cases "$PYTHON_BIN" scripts/visualize_v16_six_methods.py \
    --benchmark-dir "$COMPARISON_DIR" --output-dir "$FIGURE_DIR/saved_cases" \
    --max-cases-per-dataset "$VISUAL_SAMPLES"
else
  run_logged plot_ours_cases "$PYTHON_BIN" scripts/visualize_v16_ours_cases.py \
    "${CASE_ARGS[@]}" --output-dir "$FIGURE_DIR/ours_cases"
  run_logged plot_six_method_real_cases "$PYTHON_BIN" scripts/visualize_v16_real_deployments.py \
    "${CASE_ARGS[@]}" --method-set published --output-dir "$FIGURE_DIR/six_method_real_cases"
fi
PHASE=completed
printf '\nCompleted: checkpoint=%s\ncomparison=%s\nfigures=%s\nlogs=%s\n' \
  "$CHECKPOINT" "$COMPARISON_DIR" "$FIGURE_DIR" "$LOG_DIR"
