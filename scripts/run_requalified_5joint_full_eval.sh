#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="/home/feng/HouCode/SplineSurpervisedFitting"
cd "$ROOT"

# The old experiment cannot be resumed with a different validation contract.
# Start a new formal experiment and transfer its latest proposal-producing weights.
SOURCE="$ROOT/outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_ordered_highk_softcost_linux_r2_joint_fast.last.pt"
RUN="candidate_selection_v16_r2_requalified_boundary32_5joint"
CKPT="$ROOT/outputs/checkpoints/$RUN.pt"
UJI="$ROOT/data/splits/uji_pen_v2.jsonl"
NE="$ROOT/data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl"
USGS="$ROOT/data/processed/usgs_contours/large_scale/manifest.jsonl"
CMP="$ROOT/outputs/comparisons/$RUN/full_budget_10each"
METRICS="$ROOT/outputs/figures/$RUN/full_budget_10each/four_metrics"
CASES="$ROOT/outputs/figures/$RUN/full_budget_10each/six_method_real_cases"
LOG="$ROOT/outputs/logs/$RUN"

for required in "$SOURCE" "$UJI" "$NE" "$USGS"; do
    [[ -f "$required" ]] || { echo "ERROR: missing $required" >&2; exit 2; }
done
for target in "$CKPT" "${CKPT%.pt}.last.pt" "${CKPT%.pt}.proposal.pt" \
              "${CKPT%.pt}.history.json" "$CMP" "$METRICS" "$CASES"; do
    [[ ! -e "$target" ]] || {
        echo "ERROR: refusing to overwrite existing output: $target" >&2
        echo "Rename RUN inside this script if this experiment was already started." >&2
        exit 2
    }
done
mkdir -p "$LOG"

echo "[1/5] New formal-boundary run: 1 Proposal epoch + 5 Joint epochs"
python scripts/train_v16.py \
    --epochs 6 \
    --proposal-epochs 1 \
    --train-size 3000 \
    --val-size 300 \
    --synthetic-boundary-val-size 32 \
    --real-val-size 50 \
    --real-fraction 0.35 \
    --real-manifest "$UJI" \
    --real-manifest "$NE" \
    --real-manifest "$USGS" \
    --batch-size 128 \
    --candidate-knots 56 \
    --mse-tolerance 1e-4 \
    --policy-samples 2 \
    --synthetic-count-role upper_bound \
    --synthetic-geometry-oracle-teacher \
    --one-shot-safety-sigma 0.20 \
    --one-shot-safety-knots 2 \
    --final-safety-sigma 0.03 \
    --final-safety-knots 0 \
    --safety-anneal-epochs 2 \
    --one-shot-coverage-bins 0 \
    --complexity-ramp-epochs 2 \
    --complexity-max-scale 6 \
    --joint-lr 6e-5 \
    --num-workers 8 \
    --log-every-batches 20 \
    --device cuda \
    --init-checkpoint "$SOURCE" \
    --output "$CKPT" \
    2>&1 | tee "$LOG/train.log"

echo "[2/5] Mandatory formal audit (the script stops here if it fails)"
python scripts/inspect_v16_checkpoint.py \
    --checkpoint "$CKPT" \
    --mse-tolerance 1e-4 \
    2>&1 | tee "$LOG/audit.log"

echo "[3/5] Full-budget six-method benchmark: 10 curves per dataset"
python scripts/benchmark_v16_datasets.py \
    --checkpoint "$CKPT" \
    --output-dir "$CMP" \
    --method-set published \
    --samples-per-knot-count 1 \
    --min-knot-count 4 \
    --max-knot-count 13 \
    --scan-size 4096 \
    --seed 30000 \
    --selection-seed 20260911 \
    --real-samples-per-dataset 10 \
    --manifest "UJI=$UJI" \
    --manifest "NaturalEarth=$NE" \
    --manifest "USGS=$USGS" \
    --mse-tolerance 1e-4 \
    --max-internal-knots 56 \
    --gradient-steps 12 \
    --paper-initial-knots 56 \
    --paper-admm-iterations 1000 \
    --paper-lambda-bisections 10 \
    --paper-relocation-iterations 12 \
    --liang-dense-knots 56 \
    --liang-feature-samples 1025 \
    --dung-scan-intervals 10 \
    --dung-optimization-iterations 10 \
    --luo-eta 0.5 \
    --luo-de-population 20 \
    --luo-de-iterations 100 \
    --network-warmups 10 \
    --network-repeats 100 \
    --end-to-end-repeats 3 \
    --torch-num-threads 4 \
    --device cuda \
    2>&1 | tee "$LOG/benchmark.log"

[[ -f "$CMP/comparison.json" ]] || {
    echo "ERROR: benchmark did not produce comparison.json" >&2
    exit 3
}

echo "[4/5] Unwatermarked four-metric figures"
python scripts/plot_v16_method_comparison.py \
    --input "$CMP/comparison.json" \
    --output-dir "$METRICS" \
    --method-set published \
    --reference \
    --dpi 300 \
    2>&1 | tee "$LOG/four_metrics.log"

echo "[5/5] Unwatermarked full-budget six-method case figures"
python scripts/visualize_v16_real_deployments.py \
    --checkpoint "$CKPT" \
    --output-dir "$CASES" \
    --real-samples-per-dataset 10 \
    --selection-seed 20260911 \
    --manifest "UJI=$UJI" \
    --manifest "NaturalEarth=$NE" \
    --manifest "USGS=$USGS" \
    --mse-tolerance 1e-4 \
    --max-internal-knots 56 \
    --gradient-steps 12 \
    --paper-initial-knots 56 \
    --paper-admm-iterations 1000 \
    --paper-lambda-bisections 10 \
    --paper-relocation-iterations 12 \
    --liang-dense-knots 56 \
    --liang-feature-samples 1025 \
    --dung-scan-intervals 10 \
    --dung-optimization-iterations 10 \
    --luo-eta 0.5 \
    --luo-de-population 20 \
    --luo-de-iterations 100 \
    --network-warmups 10 \
    --network-repeats 100 \
    --end-to-end-repeats 3 \
    --torch-num-threads 4 \
    --device cuda \
    --dpi 300 \
    2>&1 | tee "$LOG/six_method_real_cases.log"

echo "DONE"
echo "checkpoint: $CKPT"
echo "comparison: $CMP"
echo "metrics: $METRICS"
echo "case figures: $CASES"
echo "logs: $LOG"
