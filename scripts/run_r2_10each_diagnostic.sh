#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="/home/feng/HouCode/SplineSurpervisedFitting"
cd "$ROOT"

CKPT="$ROOT/outputs/checkpoints/candidate_selection_v16_mse1e-4_k56_ordered_highk_softcost_linux_r2_joint_fast.pt"
UJI="$ROOT/data/splits/uji_pen_v2.jsonl"
NE="$ROOT/data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl"
USGS="$ROOT/data/processed/usgs_contours/large_scale/manifest.jsonl"

for required in "$CKPT" "$UJI" "$NE" "$USGS"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: required file does not exist: $required" >&2
        exit 2
    fi
done

TAG="r2_joint_fast_10each_$(date +%Y%m%d_%H%M%S)"
CMP="$ROOT/outputs/comparisons/$TAG"
METRICS="$ROOT/outputs/figures/$TAG/four_metrics"
CASES="$ROOT/outputs/figures/$TAG/six_method_real_cases"
LOG="$ROOT/outputs/logs/$TAG"

mkdir -p "$LOG"

echo "[1/3] Benchmark: 10 synthetic + 10 UJI + 10 Natural Earth + 10 USGS"
python scripts/benchmark_v16_datasets.py \
    --checkpoint "$CKPT" \
    --output-dir "$CMP" \
    --method-set published \
    --samples-per-knot-count 1 \
    --min-knot-count 4 \
    --max-knot-count 13 \
    --scan-size 4096 \
    --seed 20000 \
    --selection-seed 20260911 \
    --real-samples-per-dataset 10 \
    --manifest "UJI=$UJI" \
    --manifest "NaturalEarth=$NE" \
    --manifest "USGS=$USGS" \
    --mse-tolerance 1e-4 \
    --max-internal-knots 56 \
    --gradient-steps 6 \
    --paper-initial-knots 56 \
    --paper-admm-iterations 80 \
    --paper-lambda-bisections 4 \
    --paper-relocation-iterations 5 \
    --liang-dense-knots 56 \
    --liang-feature-samples 257 \
    --dung-scan-intervals 5 \
    --dung-optimization-iterations 5 \
    --luo-eta 0.5 \
    --luo-de-population 6 \
    --luo-de-iterations 12 \
    --network-warmups 2 \
    --network-repeats 10 \
    --end-to-end-repeats 1 \
    --torch-num-threads 4 \
    --device cuda \
    --allow-unqualified-diagnostic \
    2>&1 | tee "$LOG/benchmark.log"

if [[ ! -f "$CMP/comparison.json" ]]; then
    echo "ERROR: benchmark completed without comparison.json" >&2
    exit 3
fi

echo "[2/3] Plotting four aggregate metrics"
python scripts/plot_v16_method_comparison.py \
    --input "$CMP/comparison.json" \
    --output-dir "$METRICS" \
    --method-set published \
    --reference \
    --dpi 300 \
    --allow-unqualified-diagnostic \
    2>&1 | tee "$LOG/four_metrics.log"

echo "[3/3] Plotting 10 six-method cases from each real dataset"
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
    --gradient-steps 6 \
    --paper-initial-knots 56 \
    --paper-admm-iterations 80 \
    --paper-lambda-bisections 4 \
    --paper-relocation-iterations 5 \
    --liang-dense-knots 56 \
    --liang-feature-samples 257 \
    --dung-scan-intervals 5 \
    --dung-optimization-iterations 5 \
    --luo-eta 0.5 \
    --luo-de-population 6 \
    --luo-de-iterations 12 \
    --network-warmups 2 \
    --network-repeats 5 \
    --end-to-end-repeats 1 \
    --torch-num-threads 4 \
    --device cuda \
    --dpi 300 \
    --allow-unqualified-diagnostic \
    2>&1 | tee "$LOG/six_method_real_cases.log"

echo
echo "DONE"
echo "comparison: $CMP/comparison.json"
echo "report:     $CMP/report.md"
echo "metrics:    $METRICS"
echo "case PNGs:  $CASES"
echo "logs:       $LOG"
