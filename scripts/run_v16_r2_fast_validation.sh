#!/usr/bin/env bash
# Revisit the last committed r2 K=4..56 training code without changing HEAD.
set -Eeuo pipefail

LEGACY_COMMIT=48d8b4e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
LEGACY_ROOT="$REPO_ROOT/outputs/rollback_v16_r2_source"
RUN_NAME=v16_r2_rollback_fast_k4_56_r1
DEVICE=cuda
EPOCHS=13
PROPOSAL_EPOCHS=1
TRAIN_SIZE=600
VAL_SIZE=160
REAL_VAL_SIZE=20
BATCH_SIZE=128
WARMSTART="$REPO_ROOT/outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_ordered_highk_softcost_linux_r2_joint_fast.proposal.pt"
OLD_CHECKPOINT="$REPO_ROOT/outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_ordered_highk_softcost_linux_r2_joint_fast.pt"
USE_WARMSTART=1
RUN_PAIRED=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_v16_r2_fast_validation.sh [options]

Runs the historical r2 training code at commit 48d8b4e in an isolated
worktree. This is a quick diagnostic, not a claim of formal reproducibility.
  --run-name NAME          New output name; existing artifacts are never overwritten
  --device cpu|cuda|auto  Training device (default: cuda)
  --epochs N               Total epochs (default: 13)
  --proposal-epochs N      Proposal epochs (default: 1)
  --train-size N           Draws per epoch (default: 600)
  --val-size N             Synthetic validation curves (default: 160)
  --real-val-size N        Validation curves per real source (default: 20)
  --batch-size N           Batch size (default: 128)
  --warmstart PATH         Existing proposal checkpoint (default: historical r2)
  --no-warmstart           Train Proposal from scratch
  --old-checkpoint PATH    Historical r2 model used for fixed-case comparison
  --skip-paired            Skip the paired deployment diagnostic after training
  --dry-run                Check and print commands without writing
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
need_value() { (($# >= 2)) || die "$1 requires a value"; }
positive_int() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer"; }

while (($#)); do
  case "$1" in
    --run-name) need_value "$@"; RUN_NAME="$2"; shift 2 ;;
    --device) need_value "$@"; DEVICE="$2"; shift 2 ;;
    --epochs) need_value "$@"; EPOCHS="$2"; shift 2 ;;
    --proposal-epochs) need_value "$@"; PROPOSAL_EPOCHS="$2"; shift 2 ;;
    --train-size) need_value "$@"; TRAIN_SIZE="$2"; shift 2 ;;
    --val-size) need_value "$@"; VAL_SIZE="$2"; shift 2 ;;
    --real-val-size) need_value "$@"; REAL_VAL_SIZE="$2"; shift 2 ;;
    --batch-size) need_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --warmstart) need_value "$@"; WARMSTART="$2"; USE_WARMSTART=1; shift 2 ;;
    --no-warmstart) USE_WARMSTART=0; shift ;;
    --old-checkpoint) need_value "$@"; OLD_CHECKPOINT="$2"; shift 2 ;;
    --skip-paired) RUN_PAIRED=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ "$RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || die "run-name must be a simple filename"
[[ "$DEVICE" == cpu || "$DEVICE" == cuda || "$DEVICE" == auto ]] || die "device must be cpu, cuda or auto"
for pair in "epochs:$EPOCHS" "proposal-epochs:$PROPOSAL_EPOCHS" \
            "train-size:$TRAIN_SIZE" "val-size:$VAL_SIZE" \
            "real-val-size:$REAL_VAL_SIZE" "batch-size:$BATCH_SIZE"; do
  positive_int "${pair%%:*}" "${pair#*:}"
done
((PROPOSAL_EPOCHS < EPOCHS)) || die "proposal-epochs must be less than epochs"
((VAL_SIZE >= 16)) || die "val-size must include at least 16 K=56 boundary rows"
git -C "$REPO_ROOT" cat-file -e "${LEGACY_COMMIT}^{commit}" || die "historical commit is unavailable"
LEGACY_OBJECT="$(git -C "$REPO_ROOT" rev-parse "${LEGACY_COMMIT}^{commit}")"

CHECKPOINT="$REPO_ROOT/outputs/checkpoints/$RUN_NAME.pt"
LOG_DIR="$REPO_ROOT/outputs/logs/$RUN_NAME"
for artifact in "$CHECKPOINT" "${CHECKPOINT%.pt}.last.pt" \
                "${CHECKPOINT%.pt}.proposal.pt" "${CHECKPOINT%.pt}.history.json" \
                "$LOG_DIR"; do
  [[ ! -e "$artifact" && ! -L "$artifact" ]] || die "refusing to overwrite existing run artifact: $artifact"
done
if ((USE_WARMSTART)); then
  [[ "$WARMSTART" == /* ]] || WARMSTART="$REPO_ROOT/$WARMSTART"
  [[ -f "$WARMSTART" ]] || die "proposal warmstart is missing: $WARMSTART; use --no-warmstart for a fresh Proposal"
fi
if ((RUN_PAIRED)); then
  [[ "$OLD_CHECKPOINT" == /* ]] || OLD_CHECKPOINT="$REPO_ROOT/$OLD_CHECKPOINT"
  [[ -f "$OLD_CHECKPOINT" ]] || die "historical comparison checkpoint is missing: $OLD_CHECKPOINT; use --skip-paired to run training only"
fi

MANIFESTS=(
  "$REPO_ROOT/data/splits/uji_pen_v2.jsonl"
  "$REPO_ROOT/data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl"
  "$REPO_ROOT/data/processed/usgs_contours/large_scale/manifest.jsonl"
)
for manifest in "${MANIFESTS[@]}"; do
  [[ -f "$manifest" ]] || die "required real-data manifest is missing: $manifest"
done

if [[ -e "$LEGACY_ROOT" || -L "$LEGACY_ROOT" ]]; then
  [[ ! -L "$LEGACY_ROOT" ]] || die "legacy worktree path must not be a symlink: $LEGACY_ROOT"
  [[ -d "$LEGACY_ROOT" ]] || die "legacy target is not a directory: $LEGACY_ROOT"
  actual_commit="$(git -C "$LEGACY_ROOT" rev-parse HEAD 2>/dev/null || true)"
  [[ "$actual_commit" == "$LEGACY_OBJECT" ]] || die "legacy worktree exists at a different commit: $LEGACY_ROOT"
elif ((DRY_RUN)); then
  printf 'DRY-RUN: git -C %q worktree add --detach %q %q\n' "$REPO_ROOT" "$LEGACY_ROOT" "$LEGACY_COMMIT"
else
  git -C "$REPO_ROOT" worktree add --detach "$LEGACY_ROOT" "$LEGACY_COMMIT"
fi

TRAIN=(
  python -B scripts/train_v16.py
  --epochs "$EPOCHS" --proposal-epochs "$PROPOSAL_EPOCHS"
  --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE"
  --synthetic-boundary-val-size 16 --real-val-size "$REAL_VAL_SIZE"
  --batch-size "$BATCH_SIZE" --num-points 192
  --min-control-points 8 --max-control-points 60 --candidate-knots 56
  --knot-min-span 0.01 --mse-tolerance 1e-4 --knot-match-tolerance 0.01
  --certified-minimal-source --minimality-margin 0.2
  --minimality-max-attempts 16 --minimality-audit-points 512
  --oscillation-amplitude 0.3 --proposal-high-k-fraction 0.50
  --proposal-high-k-min-knots 40 --proposal-knot-assignment-weight 1.0
  --one-shot-selection-policy mass_topk --initial-keep-fraction 0.5357142857142857
  --teacher-low-count-sweep 16 --synthetic-count-role upper_bound
  --synthetic-geometry-oracle-teacher --oracle-teacher-extra-knots 2
  --one-shot-coverage-bins 0 --min-selected-knots 4
  --one-shot-safety-sigma 0.20 --one-shot-safety-knots 2
  --final-safety-sigma 0.03 --final-safety-knots 0
  --safety-anneal-epochs 12 --complexity-ramp-epochs 12
  --complexity-max-scale 6.0 --policy-samples 2 --counterfactual-edits 4
  --teacher-prefix-search-steps 7 --real-fraction 0.35
  --real-manifest "${MANIFESTS[0]}" --real-manifest "${MANIFESTS[1]}"
  --real-manifest "${MANIFESTS[2]}" --resample-train-each-epoch
  --num-workers 4 --torch-num-threads 4 --device "$DEVICE"
  --output "$CHECKPOINT"
)
((USE_WARMSTART == 0)) || TRAIN+=(--init-checkpoint "$WARMSTART")
INSPECT=(python -B scripts/inspect_v16_checkpoint.py --checkpoint "$CHECKPOINT" --mse-tolerance 1e-4)
PAIRED=(
  python -B "$REPO_ROOT/scripts/quick_compare_v16_checkpoints.py"
  --old-checkpoint "$OLD_CHECKPOINT" --new-checkpoint "$CHECKPOINT"
  --output-json "$LOG_DIR/paired.json" --mse-tolerance 1e-4
  --min-source-k 4 --max-source-k 56 --samples-per-k 1
  --max-generate-attempts 16 --real-samples-per-dataset 5
  --manifest "UJI=${MANIFESTS[0]}"
  --manifest "NaturalEarth=${MANIFESTS[1]}"
  --manifest "USGS=${MANIFESTS[2]}" --device "$DEVICE"
)

printf 'Historical code=%s; source K=4..56, Kc=56, MSE<=1e-4.\n' "$LEGACY_COMMIT"
printf 'Quick run: Proposal=%s, Joint=%s, train=%s, synthetic val=%s (+3x%s real), batch=%s.\n' \
  "$PROPOSAL_EPOCHS" "$((EPOCHS-PROPOSAL_EPOCHS))" "$TRAIN_SIZE" "$VAL_SIZE" "$REAL_VAL_SIZE" "$BATCH_SIZE"
printf 'Warmstart: %s\n' "$([[ $USE_WARMSTART == 1 ]] && printf '%s' "$WARMSTART" || printf none)"
printf 'Checkpoint: %s\n' "$CHECKPOINT"

if ((DRY_RUN)); then
  printf 'DRY-RUN in %q:' "$LEGACY_ROOT"; printf ' %q' "${TRAIN[@]}"; printf '\n'
  printf 'DRY-RUN in %q:' "$LEGACY_ROOT"; printf ' %q' "${INSPECT[@]}"; printf '\n'
  if ((RUN_PAIRED)); then
    printf 'DRY-RUN in %q:' "$LEGACY_ROOT"; printf ' %q' "${PAIRED[@]}"; printf '\n'
  fi
  exit 0
fi

mkdir -p -- "$LOG_DIR"
cd "$LEGACY_ROOT"
"${TRAIN[@]}" 2>&1 | tee "$LOG_DIR/train.log"
[[ -f "$CHECKPOINT" ]] || die "training finished without a best checkpoint"
set +e
"${INSPECT[@]}" 2>&1 | tee "$LOG_DIR/inspect.log"
inspect_status=${PIPESTATUS[0]}
set -e
printf 'Inspection exit=%s; quick-run pass rates are diagnostic and may not meet formal qualification.\n' "$inspect_status"
if ((RUN_PAIRED)); then
  "${PAIRED[@]}" 2>&1 | tee "$LOG_DIR/paired.log"
  printf 'Paired one-shot diagnostic: %s\n' "$LOG_DIR/paired.json"
fi
printf 'Validation history: %s\n' "${CHECKPOINT%.pt}.history.json"
