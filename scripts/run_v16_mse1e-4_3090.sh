#!/usr/bin/env bash
# Native Linux runner for v16 fixed-Proposal / offline feasible-subset Teacher.
# The quick benchmark profile is diagnostic; full is the paper-sized protocol.

set -Eeuo pipefail

PYTHON_BIN="python"
SIMPLIFICATION_CONTRACT="fixed_proposal_offline_feasible_subset_v1"
RUN_NAME="candidate_selection_v16_mse1e-4_sourcek56_kc72_feasible_teacher_linux_r1"
DEVICE="cuda"
EPOCHS=128
PROPOSAL_EPOCHS=64
SELECTOR_WARMUP_EPOCHS=8
SELECTOR_LR=2e-4
PROPOSAL_LR=2e-4
PROPOSAL_JOINT_LR=0
PARAMETER_JOINT_LR=0
DECODER_JOINT_LR=5e-5
TRAIN_SIZE=3000
VAL_SIZE=600
REAL_VAL_SIZE=100
PROPOSAL_HIGH_K_FRACTION=0.50
PROPOSAL_HIGH_K_MIN_KNOTS=40
JOINT_HIGH_K_FRACTION_OVERRIDE=""
SYNTHETIC_HIGH_K_VAL_SIZE_OVERRIDE=""
BATCH_SIZE=64
FEASIBLE_TEACHER_BATCH_SIZE=8
MAX_CONTROL_POINTS=60
CANDIDATE_KNOTS=72
BASELINE_CAP=56
# Legacy profiles retain --min-knot-count 4 --max-knot-count 56.
BENCHMARK_MAX_KNOTS=56
INITIAL_COUNT_ARGS=(--initial-keep-fraction 0.4166666666666667)
SYNTHETIC_BOUNDARY_VAL_SIZE=32
SAFETY_SIGMA=0
SAFETY_ANNEAL_EPOCHS=12
COMPLEXITY_RAMP_EPOCHS=12
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
INIT_FULL_CHECKPOINT=""
INIT_FULL_EXPLICIT=0
PAIRED_REFERENCE_CHECKPOINT=""
RESUME_RUN=0
PILOT_KC56=0
KEEP_SWAP_HIGHK72=0
KEEP_STATE_RECOVERY=0
SMALL_MEDIUM_K24=0
RUN_PROFILE=default
PREPARE_REAL_DATA=0
DIAGNOSTIC=0
DRY_RUN=0
BENCHMARK_PROFILE="full"
SYNTHETIC_SAMPLES_EXPLICIT=0
REAL_SAMPLES_EXPLICIT=0
KANG_ITERATIONS_EXPLICIT=0
LUO_ITERATIONS_EXPLICIT=0
NETWORK_REPEATS_EXPLICIT=0
END_TO_END_REPEATS_EXPLICIT=0

# Resolve the opt-in pilot preset before parsing ordinary overrides, so option
# order does not change its meaning. The formal/default profile remains intact.
for option in "$@"; do
  if [[ "$option" == "--pilot-kc56" ]]; then
    PILOT_KC56=1
    RUN_PROFILE=PILOT
    RUN_NAME="candidate_selection_v16_mse1e-4_pilot_sourcek44_kc56_linux_r1"
    EPOCHS=72
    PROPOSAL_EPOCHS=48
    SELECTOR_WARMUP_EPOCHS=6
    TRAIN_SIZE=1500
    VAL_SIZE=300
    REAL_VAL_SIZE=30
    PROPOSAL_HIGH_K_MIN_KNOTS=32
    BATCH_SIZE=64
    FEASIBLE_TEACHER_BATCH_SIZE=8
    MAX_CONTROL_POINTS=48
    CANDIDATE_KNOTS=56
    SYNTHETIC_BOUNDARY_VAL_SIZE=16
    BENCHMARK_PROFILE="quick"
    DIAGNOSTIC=1
  elif [[ "$option" == "--keep-swap-highk72" ]]; then
    KEEP_SWAP_HIGHK72=1
    RUN_PROFILE=KEEP_SWAP_HIGHK72
    RUN_NAME="candidate_selection_v16_mse1e-4_keep_swap_highk72_linux_r1"
    EPOCHS=96
    PROPOSAL_EPOCHS=64
    SELECTOR_WARMUP_EPOCHS=8
    TRAIN_SIZE=1500
    VAL_SIZE=400
    REAL_VAL_SIZE=40
    PROPOSAL_HIGH_K_FRACTION=0.65
    PROPOSAL_HIGH_K_MIN_KNOTS=40
    BATCH_SIZE=64
    FEASIBLE_TEACHER_BATCH_SIZE=8
    MAX_CONTROL_POINTS=60
    CANDIDATE_KNOTS=72
    BASELINE_CAP=72
    SYNTHETIC_BOUNDARY_VAL_SIZE=32
    BENCHMARK_PROFILE="quick"
    DIAGNOSTIC=1
  elif [[ "$option" == "--keep-state-recovery" ]]; then
    KEEP_STATE_RECOVERY=1
    RUN_PROFILE=KEEP_STATE_RECOVERY
    RUN_NAME="candidate_selection_v16_mse1e-4_keep_state_recovery_linux_r1"
    EPOCHS=32
    PROPOSAL_EPOCHS=8
    SELECTOR_WARMUP_EPOCHS=4
    PROPOSAL_LR=5e-5
    SELECTOR_LR=5e-5
    DECODER_JOINT_LR=1e-5
    TRAIN_SIZE=600
    VAL_SIZE=160
    REAL_VAL_SIZE=20
    BATCH_SIZE=32
    MAX_CONTROL_POINTS=60
    CANDIDATE_KNOTS=56
    BASELINE_CAP=56
    SYNTHETIC_BOUNDARY_VAL_SIZE=32
    INIT_FULL_CHECKPOINT="outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_ordered_highk_softcost_linux_r2_joint_fast.pt"
    SAFETY_SIGMA=0.03
    SAFETY_ANNEAL_EPOCHS=1
    COMPLEXITY_RAMP_EPOCHS=0
    BENCHMARK_PROFILE="quick"
    DIAGNOSTIC=1
  elif [[ "$option" == "--small-medium-k24" ]]; then
    SMALL_MEDIUM_K24=1
    RUN_PROFILE=SMALL_MEDIUM_K24
    RUN_NAME="candidate_selection_v16_mse1e-4_small_medium_sourcek20_kc24_linux_r1"
    EPOCHS=48
    PROPOSAL_EPOCHS=24
    SELECTOR_WARMUP_EPOCHS=4
    PROPOSAL_LR=2e-4
    SELECTOR_LR=5e-5
    DECODER_JOINT_LR=1e-5
    TRAIN_SIZE=600
    VAL_SIZE=160
    REAL_VAL_SIZE=20
    BATCH_SIZE=32
    PROPOSAL_HIGH_K_FRACTION=0
    PROPOSAL_HIGH_K_MIN_KNOTS=20
    MAX_CONTROL_POINTS=24
    CANDIDATE_KNOTS=24
    BASELINE_CAP=24
    BENCHMARK_MAX_KNOTS=20
    INITIAL_COUNT_ARGS=(--initial-keep-fraction 0.5)
    SYNTHETIC_BOUNDARY_VAL_SIZE=16
    INIT_CHECKPOINT=""
    INIT_FULL_CHECKPOINT=""
    SAFETY_SIGMA=0.03
    SAFETY_ANNEAL_EPOCHS=1
    COMPLEXITY_RAMP_EPOCHS=0
    BENCHMARK_PROFILE="quick"
    DIAGNOSTIC=1
  fi
done

usage() {
  cat <<'EOF'
Usage: bash scripts/run_v16_mse1e-4_3090.sh [options]

Main options:
  --python PATH                   Python executable (default: python)
  --run-name NAME                Fresh output name
  --device auto|cpu|cuda         Training/evaluation device (default: cuda)
  --epochs N                     Total epochs (default: 128)
  --proposal-epochs N            Proposal-stage epochs (default: 64)
  --selector-warmup-epochs N     Joint selector/decoder-only warmup (default: 8)
  --selector-lr X                Joint Selector learning rate (default: 2e-4)
  --lr X                         Proposal learning rate (default: 2e-4)
  --proposal-joint-lr X          Joint encoder/candidate LR (default: 0; Proposal fixed)
  --parameter-joint-lr X         Joint ParameterHead LR (default: 0; Proposal fixed)
  --decoder-joint-lr X           Joint selected-decoder LR (default: 5e-5)
  --train-size N                 Training draws per epoch (default: 3000)
  --val-size N                   Synthetic validation curves (default: 600)
  --real-val-size N              Validation curves per real source (default: 100)
                                  Training is always 100% labelled Synthetic;
                                  real manifests are validation/test only.
  --proposal-high-k-fraction X   Proposal synthetic high-K share (default: 0.50)
  --proposal-high-k-min-knots N  High-K stratum begins here (default: 40)
  --joint-high-k-fraction X      Override Joint high-K share (K24 requires 0)
  --synthetic-high-k-val-size N  Override dedicated high-K validation size (K24 requires 0)
  --batch-size N                 Batch size (default: 64)
  --num-workers N                DataLoader workers (default: 4)
  --output-root PATH             Output root (default: repository outputs/)
  --init-checkpoint PATH         Optional proposal-only warm start
  --init-full-checkpoint PATH    Full-model warm start, including selector/decoder
  --paired-reference-checkpoint PATH Native paired reference (recovery resume override)
  --no-init-checkpoint           Train all modules from scratch
  --resume-run                   Continue this run from its existing .last.pt
                                  (keeps the original --output and Proposal).
                                  Completed training is validated and skipped;
                                  evaluation/figure stages resume without overwriting.
  --feasible-teacher-batch-size N Numerical Teacher build batch (default: 8)
  --pilot-kc56                  Opt-in quick architecture diagnostic: source K=4..44,
                                  Kc=56, Proposal 48 + Joint 24, train/val=1500/300,
                                  real-val=30. Forces diagnostic, never formal.
  --keep-swap-highk72           Opt-in KeepMask exchange-supervision + high-K diagnostic:
                                  source K=4..56, Kc=72, Proposal 64 + Joint 32,
                                  train/val=1500/400, real-val=40. Forces diagnostic.
  --keep-state-recovery         Full r2 warm start, learned Keep states and global warp:
                                  source K=4..56, Kc=56, Proposal 8 + Joint 24,
                                  train/val=600/160, real-val=20, batch=32.
                                  Forces quick diagnostic; adds a paired 68-case check.
  --small-medium-k24            Train from scratch for source K=4..20, Kc=24
                                  (full cubic knot vector=32), numerical baselines cap=24.
                                  Proposal 24 + Joint 24, train/val=600/160,
                                  real-val=20, batch=32; no high-K strata.
                                  Diagnostic only, even with --benchmark-profile full.
                                  Initializers are rejected; --resume-run is supported.
  --prepare-real-data            Prepare missing UJI, Natural Earth, USGS and industrial-offset data
  --benchmark-profile quick|full Quick smoke benchmark or full comparison (default: full)
  --diagnostic                   Continue after a structural-integrity audit failure
  --dry-run                      Print every command without executing Python
  -h, --help                     Show this help

Evaluation-size options (override the selected profile):
  --synthetic-samples-per-k N    Cases per source K (4..20 for K24; otherwise 4..56)
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
    --selector-warmup-epochs) need_value "$@"; SELECTOR_WARMUP_EPOCHS="$2"; shift 2 ;;
    --selector-lr) need_value "$@"; SELECTOR_LR="$2"; shift 2 ;;
    --lr) need_value "$@"; PROPOSAL_LR="$2"; shift 2 ;;
    --proposal-joint-lr) need_value "$@"; PROPOSAL_JOINT_LR="$2"; shift 2 ;;
    --parameter-joint-lr) need_value "$@"; PARAMETER_JOINT_LR="$2"; shift 2 ;;
    --decoder-joint-lr) need_value "$@"; DECODER_JOINT_LR="$2"; shift 2 ;;
    --train-size) need_value "$@"; TRAIN_SIZE="$2"; shift 2 ;;
    --val-size) need_value "$@"; VAL_SIZE="$2"; shift 2 ;;
    --real-val-size) need_value "$@"; REAL_VAL_SIZE="$2"; shift 2 ;;
    --proposal-high-k-fraction) need_value "$@"; PROPOSAL_HIGH_K_FRACTION="$2"; shift 2 ;;
    --proposal-high-k-min-knots) need_value "$@"; PROPOSAL_HIGH_K_MIN_KNOTS="$2"; shift 2 ;;
    --joint-high-k-fraction) need_value "$@"; JOINT_HIGH_K_FRACTION_OVERRIDE="$2"; shift 2 ;;
    --synthetic-high-k-val-size) need_value "$@"; SYNTHETIC_HIGH_K_VAL_SIZE_OVERRIDE="$2"; shift 2 ;;
    --batch-size) need_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) need_value "$@"; NUM_WORKERS="$2"; shift 2 ;;
    --synthetic-samples-per-k) need_value "$@"; SYNTHETIC_SAMPLES_PER_K="$2"; SYNTHETIC_SAMPLES_EXPLICIT=1; shift 2 ;;
    --real-samples-per-dataset) need_value "$@"; REAL_SAMPLES_PER_DATASET="$2"; REAL_SAMPLES_EXPLICIT=1; shift 2 ;;
    --visual-samples-per-dataset) need_value "$@"; VISUAL_SAMPLES_PER_DATASET="$2"; shift 2 ;;
    --kang-admm-iterations) need_value "$@"; KANG_ADMM_ITERATIONS="$2"; KANG_ITERATIONS_EXPLICIT=1; shift 2 ;;
    --luo-de-population) need_value "$@"; LUO_DE_POPULATION="$2"; shift 2 ;;
    --luo-de-iterations) need_value "$@"; LUO_DE_ITERATIONS="$2"; LUO_ITERATIONS_EXPLICIT=1; shift 2 ;;
    --network-repeats) need_value "$@"; NETWORK_REPEATS="$2"; NETWORK_REPEATS_EXPLICIT=1; shift 2 ;;
    --end-to-end-repeats) need_value "$@"; END_TO_END_REPEATS="$2"; END_TO_END_REPEATS_EXPLICIT=1; shift 2 ;;
    --output-root) need_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --init-checkpoint) need_value "$@"; INIT_CHECKPOINT="$2"; shift 2 ;;
    --init-full-checkpoint) need_value "$@"; INIT_FULL_CHECKPOINT="$2"; INIT_FULL_EXPLICIT=1; shift 2 ;;
    --paired-reference-checkpoint) need_value "$@"; PAIRED_REFERENCE_CHECKPOINT="$2"; shift 2 ;;
    --no-init-checkpoint) INIT_CHECKPOINT=""; INIT_FULL_CHECKPOINT=""; INIT_FULL_EXPLICIT=0; shift ;;
    --resume-run) RESUME_RUN=1; shift ;;
    --feasible-teacher-batch-size) need_value "$@"; FEASIBLE_TEACHER_BATCH_SIZE="$2"; shift 2 ;;
    --pilot-kc56) shift ;;
    --keep-swap-highk72) shift ;;
    --keep-state-recovery) shift ;;
    --small-medium-k24) shift ;;
    --prepare-real-data) PREPARE_REAL_DATA=1; shift ;;
    --benchmark-profile) need_value "$@"; BENCHMARK_PROFILE="$2"; shift 2 ;;
    --diagnostic) DIAGNOSTIC=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

((PILOT_KC56 + KEEP_SWAP_HIGHK72 + KEEP_STATE_RECOVERY + SMALL_MEDIUM_K24 <= 1)) || \
  die "--pilot-kc56, --keep-swap-highk72, --keep-state-recovery and --small-medium-k24 are mutually exclusive"
if ((KEEP_STATE_RECOVERY)); then
  BENCHMARK_PROFILE=quick
  DIAGNOSTIC=1
  [[ -n "$INIT_FULL_CHECKPOINT" || "$RESUME_RUN" == 1 ]] || \
    die "--keep-state-recovery requires a full initializer or --resume-run"
fi
if ((SMALL_MEDIUM_K24)); then
  DIAGNOSTIC=1
  [[ -z "$INIT_CHECKPOINT" && -z "$INIT_FULL_CHECKPOINT" ]] || \
    die "--small-medium-k24 rejects --init-checkpoint and --init-full-checkpoint; train from scratch or --resume-run"
  [[ "$PROPOSAL_HIGH_K_FRACTION" =~ ^0([.]0+)?$ ]] || \
    die "--small-medium-k24 requires --proposal-high-k-fraction 0"
  [[ -z "$JOINT_HIGH_K_FRACTION_OVERRIDE" || "$JOINT_HIGH_K_FRACTION_OVERRIDE" =~ ^0([.]0+)?$ ]] || \
    die "--small-medium-k24 requires --joint-high-k-fraction 0"
  [[ -z "$SYNTHETIC_HIGH_K_VAL_SIZE_OVERRIDE" || "$SYNTHETIC_HIGH_K_VAL_SIZE_OVERRIDE" =~ ^0+$ ]] || \
    die "--small-medium-k24 requires --synthetic-high-k-val-size 0"
fi

[[ "$BENCHMARK_PROFILE" == "quick" || "$BENCHMARK_PROFILE" == "full" ]] || \
  die "benchmark profile must be quick or full"
if [[ "$BENCHMARK_PROFILE" == "quick" ]]; then
  ((SYNTHETIC_SAMPLES_EXPLICIT)) || SYNTHETIC_SAMPLES_PER_K=1
  ((REAL_SAMPLES_EXPLICIT)) || REAL_SAMPLES_PER_DATASET=5
  ((KANG_ITERATIONS_EXPLICIT)) || KANG_ADMM_ITERATIONS=300
  ((LUO_ITERATIONS_EXPLICIT)) || LUO_DE_ITERATIONS=30
  ((NETWORK_REPEATS_EXPLICIT)) || NETWORK_REPEATS=30
  ((END_TO_END_REPEATS_EXPLICIT)) || END_TO_END_REPEATS=1
fi

positive_integer() {
  local name="$1" value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer"
}

nonnegative_integer() {
  local name="$1" value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer"
}

positive_number() {
  local name="$1" value="$2" mantissa
  [[ "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || \
    die "$name must be a finite positive number"
  mantissa="${value%%[eE]*}"
  [[ "$mantissa" =~ [1-9] ]] || die "$name must be greater than zero"
}

nonnegative_number() {
  local name="$1" value="$2"
  [[ "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || \
    die "$name must be a finite non-negative number"
}

for item in \
  "EPOCHS:$EPOCHS" \
  "PROPOSAL_EPOCHS:$PROPOSAL_EPOCHS" \
  "TRAIN_SIZE:$TRAIN_SIZE" \
  "VAL_SIZE:$VAL_SIZE" \
  "REAL_VAL_SIZE:$REAL_VAL_SIZE" \
  "PROPOSAL_HIGH_K_MIN_KNOTS:$PROPOSAL_HIGH_K_MIN_KNOTS" \
  "BATCH_SIZE:$BATCH_SIZE" \
  "FEASIBLE_TEACHER_BATCH_SIZE:$FEASIBLE_TEACHER_BATCH_SIZE" \
  "MAX_CONTROL_POINTS:$MAX_CONTROL_POINTS" \
  "CANDIDATE_KNOTS:$CANDIDATE_KNOTS" \
  "BASELINE_CAP:$BASELINE_CAP" \
  "SYNTHETIC_BOUNDARY_VAL_SIZE:$SYNTHETIC_BOUNDARY_VAL_SIZE" \
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
nonnegative_integer "SELECTOR_WARMUP_EPOCHS" "$SELECTOR_WARMUP_EPOCHS"
if [[ -n "$SYNTHETIC_HIGH_K_VAL_SIZE_OVERRIDE" ]]; then
  nonnegative_integer "SYNTHETIC_HIGH_K_VAL_SIZE_OVERRIDE" "$SYNTHETIC_HIGH_K_VAL_SIZE_OVERRIDE"
fi
if [[ -n "$JOINT_HIGH_K_FRACTION_OVERRIDE" ]]; then
  [[ "$JOINT_HIGH_K_FRACTION_OVERRIDE" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] || \
    die "joint high-K fraction must lie in [0,1]"
fi
positive_number "SELECTOR_LR" "$SELECTOR_LR"
positive_number "PROPOSAL_LR" "$PROPOSAL_LR"
nonnegative_number "PROPOSAL_JOINT_LR" "$PROPOSAL_JOINT_LR"
nonnegative_number "PARAMETER_JOINT_LR" "$PARAMETER_JOINT_LR"
positive_number "DECODER_JOINT_LR" "$DECODER_JOINT_LR"
((PROPOSAL_EPOCHS < EPOCHS)) || die "proposal epochs must be smaller than total epochs"
((SELECTOR_WARMUP_EPOCHS < EPOCHS - PROPOSAL_EPOCHS)) || \
  die "selector warmup must be smaller than the Joint epoch budget"
((LUO_DE_POPULATION >= 5)) || die "Luo DE population must be at least 5"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "run name contains unsupported characters"
[[ "$DEVICE" == "auto" || "$DEVICE" == "cpu" || "$DEVICE" == "cuda" ]] || \
  die "device must be auto, cpu or cuda"
[[ "$PROPOSAL_HIGH_K_FRACTION" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] || \
  die "proposal high-K fraction must lie in [0,1]"
SOURCE_MAX_KNOTS=$((MAX_CONTROL_POINTS - 4))
((SOURCE_MAX_KNOTS >= 4 && SOURCE_MAX_KNOTS <= 56)) || \
  die "source maximum internal knots must lie in 4..56"
((CANDIDATE_KNOTS >= SOURCE_MAX_KNOTS)) || \
  die "candidate slots must cover the source maximum internal knots"
((PROPOSAL_HIGH_K_MIN_KNOTS >= 4 && PROPOSAL_HIGH_K_MIN_KNOTS <= SOURCE_MAX_KNOTS)) || \
  die "proposal high-K minimum must lie inside the training source range"
if ((RESUME_RUN)) && [[ -n "$INIT_CHECKPOINT" ]]; then
  die "--resume-run cannot be combined with --init-checkpoint"
fi
if [[ -n "$INIT_CHECKPOINT" && -n "$INIT_FULL_CHECKPOINT" ]]; then
  die "--init-checkpoint and --init-full-checkpoint are mutually exclusive"
fi
if ((RESUME_RUN && INIT_FULL_EXPLICIT)); then
  die "--resume-run cannot be combined with --init-full-checkpoint"
fi
if [[ -n "$PAIRED_REFERENCE_CHECKPOINT" ]] && ((KEEP_STATE_RECOVERY == 0)); then
  die "--paired-reference-checkpoint requires --keep-state-recovery"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd -- "$REPOSITORY_ROOT"

if [[ -n "$INIT_FULL_CHECKPOINT" ]]; then
  [[ "$INIT_FULL_CHECKPOINT" == /* ]] || INIT_FULL_CHECKPOINT="$REPOSITORY_ROOT/$INIT_FULL_CHECKPOINT"
  INIT_FULL_CHECKPOINT="$(readlink -m -- "$INIT_FULL_CHECKPOINT")"
  if ((RESUME_RUN == 0)); then
    [[ -f "$INIT_FULL_CHECKPOINT" ]] || die "full-model initializer is missing: $INIT_FULL_CHECKPOINT"
  fi
fi
if ((KEEP_STATE_RECOVERY)); then
  if [[ -z "$PAIRED_REFERENCE_CHECKPOINT" ]]; then
    PAIRED_REFERENCE_CHECKPOINT="$INIT_FULL_CHECKPOINT"
  fi
  [[ -n "$PAIRED_REFERENCE_CHECKPOINT" ]] || die "recovery resume needs --paired-reference-checkpoint"
  [[ "$PAIRED_REFERENCE_CHECKPOINT" == /* ]] || \
    PAIRED_REFERENCE_CHECKPOINT="$REPOSITORY_ROOT/$PAIRED_REFERENCE_CHECKPOINT"
  PAIRED_REFERENCE_CHECKPOINT="$(readlink -m -- "$PAIRED_REFERENCE_CHECKPOINT")"
  [[ -f "$PAIRED_REFERENCE_CHECKPOINT" ]] || die "paired reference is missing: $PAIRED_REFERENCE_CHECKPOINT"
fi

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
PROPOSAL_FINAL_PATH="$CHECKPOINT_DIRECTORY/$RUN_NAME.proposal.final.pt"
HISTORY_PATH="$CHECKPOINT_DIRECTORY/$RUN_NAME.history.json"
TEACHER_DIRECTORY="$OUTPUT_ROOT/teachers/$RUN_NAME"

if ((RESUME_RUN)); then
  [[ -f "$LAST_PATH" ]] || die "--resume-run requires the existing last checkpoint: $LAST_PATH"
  [[ -f "$PROPOSAL_PATH" ]] || die "--resume-run requires the existing best Proposal: $PROPOSAL_PATH"
else
  for owned in \
    "$CHECKPOINT_PATH" "$LAST_PATH" "$PROPOSAL_PATH" "$PROPOSAL_FINAL_PATH" "$HISTORY_PATH" \
    "$LOG_DIRECTORY" "$COMPARISON_ROOT" "$FIGURE_ROOT" "$TEACHER_DIRECTORY"; do
    [[ ! -e "$owned" ]] || die "refusing to overwrite existing run artifact: $owned"
  done
fi

UJI_MANIFEST="$REPOSITORY_ROOT/data/splits/uji_pen_v2.jsonl"
NATURAL_EARTH_MANIFEST="$REPOSITORY_ROOT/data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl"
USGS_MANIFEST="$REPOSITORY_ROOT/data/processed/usgs_contours/large_scale/manifest.jsonl"
INDUSTRIAL_OFFSET_MANIFEST="$REPOSITORY_ROOT/data/processed/industrial_offsets/v1/manifest.jsonl"

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
  if [[ ! -f "$INDUSTRIAL_OFFSET_MANIFEST" ]]; then
    printf '\n[prepare_industrial_offsets] generating CAD-driven semi-synthetic offset curves\n'
    if ((DRY_RUN)); then
      printf ' %q' "$PYTHON_BIN" scripts/prepare_industrial_offsets.py \
        --output-dir data/processed/industrial_offsets/v1 \
        --variants-per-family 12 --source-points 384 --reference-points 768
      printf '\n'
    else
      "$PYTHON_BIN" scripts/prepare_industrial_offsets.py \
        --output-dir data/processed/industrial_offsets/v1 \
        --variants-per-family 12 --source-points 384 --reference-points 768
    fi
  fi
}

if ((PREPARE_REAL_DATA)); then
  prepare_missing_real_data
fi
if ((DRY_RUN == 0)); then
  for manifest in \
    "$UJI_MANIFEST" "$NATURAL_EARTH_MANIFEST" "$USGS_MANIFEST" \
    "$INDUSTRIAL_OFFSET_MANIFEST"; do
    [[ -f "$manifest" ]] || die \
      "required real-data manifest does not exist: $manifest; rerun with --prepare-real-data"
  done
  if [[ "$DEVICE" == "cuda" ]]; then
    "$PYTHON_BIN" -c \
      'import torch; assert torch.cuda.is_available(), "CUDA requested but unavailable"; print(torch.cuda.get_device_name(0))'
  fi
fi

if ((DRY_RUN == 0)); then
  mkdir -p -- "$CHECKPOINT_DIRECTORY" "$LOG_DIRECTORY"
fi
CURRENT_PHASE="initialization"
trap 'code=$?; printf "\nPipeline failed during %s (exit %d). Logs: %s\n" "$CURRENT_PHASE" "$code" "$LOG_DIRECTORY" >&2; exit "$code"' ERR

run_logged() {
  local phase="$1"
  shift
  local log_path="$LOG_DIRECTORY/$phase.log"
  if [[ -e "$log_path" ]]; then
    # A resumed attempt must not truncate logs from a previous attempt.
    log_path="$LOG_DIRECTORY/${phase}_$(date -u +%Y%m%dT%H%M%S)_$$.log"
    [[ ! -e "$log_path" ]] || die "refusing to overwrite existing log: $log_path"
  fi
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

run_stage_logged() {
  local phase="$1" kind="$2"
  shift 2
  if ((DRY_RUN)); then
    # Keep dry-run commands directly executable and do not inspect checkpoints.
    run_logged "$phase" "$@"
  else
    local recovery_args=()
    ((RESUME_RUN == 0)) || recovery_args+=(--resume-run)
    run_logged "$phase" "$PYTHON_BIN" scripts/v16_pipeline_resume.py \
      "$kind" "${recovery_args[@]}" -- "$@"
  fi
}

printf '%s v16 Linux %s profile: MSE=1e-4, Kc=%s, training source K=4..%s, train/val=%s/%s, batch=%s.\n' \
  "$([[ $RESUME_RUN -eq 1 ]] && printf Resume || printf Fresh)" \
  "$RUN_PROFILE" \
  "$CANDIDATE_KNOTS" "$SOURCE_MAX_KNOTS" "$TRAIN_SIZE" "$VAL_SIZE" "$BATCH_SIZE"
printf 'Simplification contract: %s\n' "$SIMPLIFICATION_CONTRACT"
printf 'Checkpoint selection is empirical: inspect the trained Proposal, feasible Teacher and one-shot deployment separately; no pass rate is assumed achieved.\n'
printf 'Training supervision: certified Synthetic geometry labels plus an offline feasible-subset Teacher on a fixed Proposal frame; real data is validation/test only.\n'
printf 'Joint samples are fixed across epochs so teacher cache labels remain sample-aligned. Teacher cache: %s\n' "$TEACHER_DIRECTORY"
printf 'Benchmark profile=%s: synthetic=%s per K, real=%s per dataset. Quick results are diagnostic, not publication estimates.\n' \
  "$BENCHMARK_PROFILE" "$SYNTHETIC_SAMPLES_PER_K" "$REAL_SAMPLES_PER_DATASET"
if ((KEEP_SWAP_HIGHK72)); then
  printf 'Proposal synthetic high-K share=%s at K>=%s; Joint samples K>=45 at 0.50 within source K=4..%s.\n' \
    "$PROPOSAL_HIGH_K_FRACTION" "$PROPOSAL_HIGH_K_MIN_KNOTS" "$SOURCE_MAX_KNOTS"
elif ((KEEP_STATE_RECOVERY)); then
  printf 'Proposal synthetic high-K share=%s at K>=%s; Joint samples K>=45 at 0.40 within source K=4..%s.\n' \
    "$PROPOSAL_HIGH_K_FRACTION" "$PROPOSAL_HIGH_K_MIN_KNOTS" "$SOURCE_MAX_KNOTS"
else
  printf 'Proposal synthetic high-K share=%s at K>=%s; Joint restores training source K=4..%s.\n' \
    "$PROPOSAL_HIGH_K_FRACTION" "$PROPOSAL_HIGH_K_MIN_KNOTS" "$SOURCE_MAX_KNOTS"
fi
printf 'Candidate redundancy: %s proposal slots for at most %s labelled source knots. Synthetic evaluation K=4..%s.\n' \
  "$CANDIDATE_KNOTS" "$SOURCE_MAX_KNOTS" "$BENCHMARK_MAX_KNOTS"
if ((PILOT_KC56)); then
  printf 'PILOT ONLY: trained up to source K=%s but benchmark includes K=45..56 out-of-training-range stress cases; all outputs are diagnostic.\n' \
    "$SOURCE_MAX_KNOTS"
  printf 'PILOT offline Teacher: synthetic anchors + numerical MSE add-back; unmatched anchors fall back to full greedy. Count loss also updates candidate structure. Neither speed nor feasibility is assumed.\n'
fi
if ((KEEP_SWAP_HIGHK72)); then
  printf 'HIGH-K DIAGNOSTIC: train and test source K=4..56; Proposal high-K share=0.65 at K>=40 and Joint high-K share=0.50 at K>=45.\n'
  printf 'Offline teacher uses bounded numerical exchange supervision; Proposal remains fixed in Joint. Raw one-shot and numerical repair must be reported separately.\n'
  printf 'Capacity disclosure: network and numerical baselines each start/cap at %s internal knots; quick sample size is diagnostic only.\n' "$BASELINE_CAP"
fi
if ((KEEP_STATE_RECOVERY)); then
  printf 'KEEP-STATE RECOVERY: preserve all r2 learned weights; enable zero-initialized Keep embeddings and interval warp. Mixed-source warm-start provenance remains diagnostic.\n'
  printf 'Proposal LR=%s; Joint high-K share=0.40 at K>=45. Keep safety stays sigma=0.03, knots=0; no inactive schedule delays checkpoint selection.\n' "$PROPOSAL_LR"
  printf 'Paired check: identical 53 Synthetic + 3 real datasets x 5 curves; native one-shot deployment, no numerical repair.\n'
fi
if ((SMALL_MEDIUM_K24)); then
  printf 'SMALL/MEDIUM DIAGNOSTIC: source K=4..20; 24 internal candidates, full cubic knot vector=32; baseline cap=24.\n'
  printf 'Fresh initialization; Proposal and Joint high-K strata disabled, no dedicated high-K validation. The 16 boundary validation curves have source K=20.\n'
  printf 'Keep-state interaction, global warp and boundary ranking remain enabled. Narrowing the range does not establish that Keep selection has been fixed or that pass rates improve.\n'
fi
printf 'Joint optimization: fixed Proposal/encoder/ParameterHead; %s selector/decoder warmup epochs; grouped LR selector=%s, proposal=%s, parameter=%s, decoder=%s.\n' \
  "$SELECTOR_WARMUP_EPOCHS" "$SELECTOR_LR" "$PROPOSAL_JOINT_LR" \
  "$PARAMETER_JOINT_LR" "$DECODER_JOINT_LR"

TRAIN_ARGS=(
  "$PYTHON_BIN" scripts/train_v16.py
  --epochs "$EPOCHS"
  --proposal-epochs "$PROPOSAL_EPOCHS"
  --selector-warmup-epochs "$SELECTOR_WARMUP_EPOCHS"
  --selector-lr "$SELECTOR_LR"
  --lr "$PROPOSAL_LR"
  --proposal-joint-lr "$PROPOSAL_JOINT_LR"
  --parameter-joint-lr "$PARAMETER_JOINT_LR"
  --decoder-joint-lr "$DECODER_JOINT_LR"
  --train-size "$TRAIN_SIZE"
  --val-size "$VAL_SIZE"
  --synthetic-boundary-val-size "$SYNTHETIC_BOUNDARY_VAL_SIZE"
  --real-val-size "$REAL_VAL_SIZE"
  --proposal-high-k-fraction "$PROPOSAL_HIGH_K_FRACTION"
  --proposal-high-k-min-knots "$PROPOSAL_HIGH_K_MIN_KNOTS"
  --batch-size "$BATCH_SIZE"
  --feasible-teacher-batch-size "$FEASIBLE_TEACHER_BATCH_SIZE"
  --num-points 192
  --min-control-points 8
  --max-control-points "$MAX_CONTROL_POINTS"
  --candidate-knots "$CANDIDATE_KNOTS"
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
  --proposal-multiscale-recall-weight 0.25
  --keep-dice-weight 0.5
  --keep-cdf-weight 0.25
  --parameter-gap-weight 0.05
  --parameter-bias-weight 0.1
  --fine-teacher-weight 0.5
  --fine-teacher-ranking-weight 0.25
  --fine-teacher-temperature 0.5
  --keep-fuzzy-negative-radius 0.01
  --keep-fuzzy-negative-floor 0.1
  --proposal-parameter-warp-gradient-scale 0
  --joint-parameter-warp-gradient-scale 0.1
  --one-shot-selection-policy mass_topk
  "${INITIAL_COUNT_ARGS[@]}"
  --joint-supervision offline_feasible_teacher
  --feasible-teacher-cache-dir "$TEACHER_DIRECTORY"
  --synthetic-count-role reference_only
  --no-synthetic-geometry-oracle-teacher
  --one-shot-coverage-bins 0
  --min-selected-knots 4
  --one-shot-safety-sigma "$SAFETY_SIGMA"
  --one-shot-safety-knots 0
  --final-safety-sigma "$SAFETY_SIGMA"
  --final-safety-knots 0
  --safety-anneal-epochs "$SAFETY_ANNEAL_EPOCHS"
  --complexity-ramp-epochs "$COMPLEXITY_RAMP_EPOCHS"
  --complexity-max-scale 1.0
  --complexity-weight 0
  --real-fraction 0
  --real-manifest "$UJI_MANIFEST"
  --real-manifest "$NATURAL_EARTH_MANIFEST"
  --real-manifest "$USGS_MANIFEST"
  --real-manifest "$INDUSTRIAL_OFFSET_MANIFEST"
  --no-resample-train-each-epoch
  --num-workers "$NUM_WORKERS"
  --torch-num-threads 4
  --device "$DEVICE"
  --output "$CHECKPOINT_PATH"
)
if ((PILOT_KC56)); then
  TRAIN_ARGS+=(
    --feasible-teacher-strategy synthetic_anchor_first
    --feasible-teacher-anchor-match-tolerance 0.02
    --count-structure-coupling
  )
fi
if ((KEEP_SWAP_HIGHK72)); then
  TRAIN_ARGS+=(
    --joint-high-k-fraction 0.50
    --joint-high-k-min-knots 45
    --synthetic-high-k-val-size 32
    --synthetic-high-k-val-min-knots 45
    --feasible-teacher-strategy synthetic_anchor_counterfactual
    --feasible-teacher-anchor-match-tolerance 0.02
    --feasible-teacher-counterfactual-max-probes 16
    --counterfactual-or-weight 1.0
    --count-structure-coupling
  )
fi
if ((KEEP_STATE_RECOVERY)); then
  TRAIN_ARGS+=(
    --joint-high-k-fraction 0.40
    --joint-high-k-min-knots 45
    --synthetic-high-k-val-size 32
    --synthetic-high-k-val-min-knots 45
    --feasible-teacher-strategy synthetic_anchor_counterfactual
    --feasible-teacher-anchor-match-tolerance 0.02
    --feasible-teacher-counterfactual-max-probes 8
    --counterfactual-or-weight 1.0
    --keep-state-interaction
    --proposal-global-warp-limit 1.0
    --count-structure-coupling
    --keep-boundary-ranking-weight 1.0
  )
fi
if ((SMALL_MEDIUM_K24)); then
  TRAIN_ARGS+=(
    --study-scope small_medium_k24
    --joint-high-k-fraction 0
    --joint-high-k-min-knots 20
    --synthetic-high-k-val-size 0
    --synthetic-high-k-val-min-knots 20
    --feasible-teacher-strategy synthetic_anchor_counterfactual
    --feasible-teacher-anchor-match-tolerance 0.02
    --feasible-teacher-counterfactual-max-probes 8
    --counterfactual-or-weight 1.0
    --keep-state-interaction
    --proposal-global-warp-limit 1.0
    --count-structure-coupling
    --keep-boundary-ranking-weight 1.0
  )
fi
if [[ -n "$JOINT_HIGH_K_FRACTION_OVERRIDE" ]]; then
  TRAIN_ARGS+=(--joint-high-k-fraction "$JOINT_HIGH_K_FRACTION_OVERRIDE")
fi
if [[ -n "$SYNTHETIC_HIGH_K_VAL_SIZE_OVERRIDE" ]]; then
  TRAIN_ARGS+=(--synthetic-high-k-val-size "$SYNTHETIC_HIGH_K_VAL_SIZE_OVERRIDE")
fi
TRAIN_PHASE=train_fresh
if ((RESUME_RUN)); then
  printf 'Resuming the existing training run: %s\n' "$LAST_PATH"
  TRAIN_ARGS+=(--resume "$LAST_PATH")
  TRAIN_PHASE=train_resume
elif [[ -n "$INIT_FULL_CHECKPOINT" ]]; then
  printf 'Using full-model warm start: %s\n' "$INIT_FULL_CHECKPOINT"
  TRAIN_ARGS+=(--init-full-checkpoint "$INIT_FULL_CHECKPOINT")
elif [[ -n "$INIT_CHECKPOINT" ]]; then
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
TRAINING_STATUS=resume
if ((RESUME_RUN && DRY_RUN == 0)); then
  CURRENT_PHASE=validate_training_resume
  TRAINING_STATUS="$("$PYTHON_BIN" scripts/v16_pipeline_resume.py training-status \
    -- "${TRAIN_ARGS[@]:2}")"
fi
if [[ "$TRAINING_STATUS" == complete ]]; then
  printf '\n[train_already_complete] Validated completed training; reusing %s\n' "$CHECKPOINT_PATH"
else
  run_logged "$TRAIN_PHASE" "${TRAIN_ARGS[@]}"
fi

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

PASS_REFERENCE_OK=0
if ((DRY_RUN == 0 && CHECKPOINT_INTEGRITY_OK == 1)); then
  PASS_REFERENCE_OK="$("$PYTHON_BIN" -c '
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print(1 if checkpoint.get("qualification", {}).get("pass_rate_reference_met") is True else 0)
' "$CHECKPOINT_PATH")"
fi

if ((DRY_RUN)); then
  CHECKPOINT_HASH="0000000000000000000000000000000000000000000000000000000000000000"
else
  CHECKPOINT_HASH="$(sha256sum -- "$CHECKPOINT_PATH" | awk '{print $1}')"
fi
if ((PILOT_KC56 || DIAGNOSTIC || CHECKPOINT_INTEGRITY_OK == 0 || PASS_REFERENCE_OK == 0)) || [[ "$BENCHMARK_PROFILE" == "quick" ]]; then
  TAG="diagnostic_${CHECKPOINT_HASH:0:12}"
  DIAGNOSTIC_ARGS=(--allow-unqualified-diagnostic)
  FORCE_DIAGNOSTIC_ARGS=(--force-diagnostic)
else
  TAG="formal_${CHECKPOINT_HASH:0:12}"
  DIAGNOSTIC_ARGS=()
  FORCE_DIAGNOSTIC_ARGS=()
fi

COMPARISON_DIRECTORY="$COMPARISON_ROOT/$TAG"
METRIC_FIGURE_DIRECTORY="$FIGURE_ROOT/$TAG/four_metrics"
REAL_FIGURE_DIRECTORY="$FIGURE_ROOT/$TAG/six_method_real_cases"
for directory in "$COMPARISON_DIRECTORY" "$METRIC_FIGURE_DIRECTORY" "$REAL_FIGURE_DIRECTORY"; do
  if ((RESUME_RUN == 0)); then
    [[ ! -e "$directory" ]] || die "refusing to overwrite evaluation output: $directory"
  fi
  if ((DRY_RUN == 0)); then
    mkdir -p -- "$directory"
  fi
done

MANIFEST_ARGS=(
  --manifest "UJI=$UJI_MANIFEST"
  --manifest "NaturalEarth=$NATURAL_EARTH_MANIFEST"
  --manifest "USGS=$USGS_MANIFEST"
  --manifest "IndustrialOffset=$INDUSTRIAL_OFFSET_MANIFEST"
)
BASELINE_ARGS=(
  --max-internal-knots "$BASELINE_CAP"
  --gradient-steps 12
  --paper-initial-knots "$BASELINE_CAP"
  --paper-admm-iterations "$KANG_ADMM_ITERATIONS"
  --paper-lambda-bisections 10
  --paper-relocation-iterations 12
  --liang-dense-knots "$BASELINE_CAP"
  --liang-feature-samples 1025
  --dung-scan-intervals 10
  --dung-optimization-iterations 10
  --luo-eta 0.5
  --luo-de-population "$LUO_DE_POPULATION"
  --luo-de-iterations "$LUO_DE_ITERATIONS"
)
STRESS_ARGS=()
if ((PILOT_KC56)); then
  # Test-only generator extension; the checkpoint's training range stays K=4..44.
  STRESS_ARGS=(--synthetic-test-max-control-points 60)
fi

run_stage_logged benchmark_six_methods_plus_verified benchmark \
  "$PYTHON_BIN" scripts/benchmark_v16_datasets.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --output-dir "$COMPARISON_DIRECTORY" \
  --method-set published \
  --include-verified-ours \
  --samples-per-knot-count "$SYNTHETIC_SAMPLES_PER_K" \
  --min-knot-count 4 --max-knot-count "$BENCHMARK_MAX_KNOTS" \
  "${STRESS_ARGS[@]}" \
  --real-samples-per-dataset "$REAL_SAMPLES_PER_DATASET" \
  --mse-tolerance 1e-4 \
  "${MANIFEST_ARGS[@]}" "${BASELINE_ARGS[@]}" \
  --network-warmups 10 --network-repeats "$NETWORK_REPEATS" \
  --end-to-end-repeats "$END_TO_END_REPEATS" \
  --torch-num-threads 4 --device "$DEVICE" \
  "${DIAGNOSTIC_ARGS[@]}" "${FORCE_DIAGNOSTIC_ARGS[@]}"

run_stage_logged plot_four_metrics plot \
  "$PYTHON_BIN" scripts/plot_v16_method_comparison.py \
  --input "$COMPARISON_DIRECTORY/comparison.json" \
  --output-dir "$METRIC_FIGURE_DIRECTORY" \
  --method-set published --dpi 300 --reference \
  "${DIAGNOSTIC_ARGS[@]}"

run_stage_logged visualize_six_method_real_cases visualize \
  "$PYTHON_BIN" scripts/visualize_v16_real_deployments.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --output-dir "$REAL_FIGURE_DIRECTORY" \
  --real-samples-per-dataset "$VISUAL_SAMPLES_PER_DATASET" \
  --selection-seed 20260910 --mse-tolerance 1e-4 \
  "${MANIFEST_ARGS[@]}" "${BASELINE_ARGS[@]}" \
  --network-warmups 10 --network-repeats "$NETWORK_REPEATS" \
  --end-to-end-repeats "$END_TO_END_REPEATS" \
  --torch-num-threads 4 --device "$DEVICE" --dpi 300 \
  "${DIAGNOSTIC_ARGS[@]}" "${FORCE_DIAGNOSTIC_ARGS[@]}"

if ((KEEP_STATE_RECOVERY)); then
  run_stage_logged paired_native_deployment paired \
    "$PYTHON_BIN" scripts/quick_compare_v16_checkpoints.py \
    --old-checkpoint "$PAIRED_REFERENCE_CHECKPOINT" \
    --new-checkpoint "$CHECKPOINT_PATH" \
    --output-json "$COMPARISON_DIRECTORY/paired_native_stage/paired_native.json" \
    --min-source-k 4 --max-source-k 56 --samples-per-k 1 \
    --real-samples-per-dataset 5 \
    --manifest "UJI=$UJI_MANIFEST" \
    --manifest "NaturalEarth=$NATURAL_EARTH_MANIFEST" \
    --manifest "USGS=$USGS_MANIFEST" \
    --mse-tolerance 1e-4 --device "$DEVICE"
fi

CURRENT_PHASE="completed"
trap - ERR
printf '\nv16 Linux pipeline completed.\n'
printf '  checkpoint: %s\n' "$CHECKPOINT_PATH"
printf '  teacher:    %s\n' "$TEACHER_DIRECTORY"
printf '  diagnostic: %s\n' "$([[ ${#DIAGNOSTIC_ARGS[@]} -gt 0 ]] && printf true || printf false)"
printf '  comparison: %s\n' "$COMPARISON_DIRECTORY"
printf '  figures:    %s\n' "$FIGURE_ROOT/$TAG"
printf '  logs:       %s\n' "$LOG_DIRECTORY"
