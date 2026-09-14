#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="/home/feng/HouCode/SplineSurpervisedFitting"
cd "$ROOT"

SOURCE="$ROOT/outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_ordered_highk_softcost_linux_r2_joint_fast.last.pt"
RUN="candidate_selection_v16_r2_full_resume_e86_boundary32"
CKPT="$ROOT/outputs/checkpoints/$RUN.pt"
LAST="$ROOT/outputs/checkpoints/$RUN.last.pt"
UJI="$ROOT/data/splits/uji_pen_v2.jsonl"
NE="$ROOT/data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl"
USGS="$ROOT/data/processed/usgs_contours/large_scale/manifest.jsonl"
LOG="$ROOT/outputs/logs/$RUN"
CMP="$ROOT/outputs/comparisons/$RUN/full_budget_10each"
METRICS="$ROOT/outputs/figures/$RUN/full_budget_10each/four_metrics"
CASES="$ROOT/outputs/figures/$RUN/full_budget_10each/six_method_real_cases"

for path in "$SOURCE" "$UJI" "$NE" "$USGS"; do
    [[ -f "$path" ]] || { echo "ERROR: missing $path" >&2; exit 2; }
done
mkdir -p "$LOG"

echo "[1/6] Forking the complete epoch-81 state (including Selector and optimizer)"
python scripts/fork_v16_resume_checkpoint.py \
    --source-last "$SOURCE" \
    --output "$CKPT" \
    --epochs 86 \
    --synthetic-boundary-val-size 32 \
    2>&1 | tee "$LOG/fork.log"

echo "[2/6] Full-state resume: epochs 82..86"
python scripts/train_v16.py \
    --epochs 86 \
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
    --one-shot-safety-sigma 0.2 \
    --final-safety-sigma 0.03 \
    --safety-anneal-epochs 12 \
    --one-shot-coverage-bins 0 \
    --complexity-ramp-epochs 12 \
    --complexity-max-scale 6 \
    --joint-lr 6e-5 \
    --num-workers 8 \
    --log-every-batches 20 \
    --device cuda \
    --output "$CKPT" \
    --resume "$LAST" \
    2>&1 | tee "$LOG/train_82_86.log"

echo "[3/6] Formal audit; failure stops the pipeline"
python scripts/inspect_v16_checkpoint.py \
    --checkpoint "$CKPT" \
    --mse-tolerance 1e-4 \
    2>&1 | tee "$LOG/audit.log"

echo "[4/6] Full-budget six-method benchmark"
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

echo "[5/6] Unwatermarked four-metric plots"
python scripts/plot_v16_method_comparison.py \
    --input "$CMP/comparison.json" \
    --output-dir "$METRICS" \
    --method-set published \
    --reference \
    --dpi 300 \
    2>&1 | tee "$LOG/four_metrics.log"

echo "[6/6] Unwatermarked full-budget six-method case plots"
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
    2>&1 | tee "$LOG/six_method_cases.log"

echo "DONE"
echo "checkpoint: $CKPT"
echo "comparison: $CMP"
echo "metrics:    $METRICS"
echo "case PNGs:  $CASES"
