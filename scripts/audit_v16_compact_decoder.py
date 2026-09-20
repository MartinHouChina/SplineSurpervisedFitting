"""Read-only input-point diagnostic: deletion at fixed geometry vs re-decoding.

Uses exact normalized inputs saved by the six-method visualizer. Greedy search
is diagnostic work, NOT part of Ours deployment. No training, checkpoint edits,
or reference-curve accuracy claim is made. An optional report is a new file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import build_model_from_checkpoint  # noqa: E402
from spline_fitting.losses.v16_subset_loss import V16SubsetLoss  # noqa: E402


@torch.no_grad()
def audit_case(model, objective, record):
    weight = next(model.parameters())
    points = torch.as_tensor(record["network_input_points_normalized"],
                             dtype=weight.dtype, device=weight.device).unsqueeze(0)
    context = model.encode_candidates(points, mse_tolerance=objective.mse_tolerance)
    mask = model.select_mask(context)
    output = model.decode_subset(context, mask)
    tolerance = points.new_full((1,), objective.mse_tolerance)
    mse = objective._fit(output["params"], output["internal_knots"], mask, points, model.degree)
    compact, fixed_mse, scans, checks = objective._fixed_geometry_greedy(
        model, context, output["params"], output["internal_knots"], mask,
        points, model.degree, tolerance, model.min_selected_knots,
    )
    decoded_mse = objective._decode_mse(model, context, compact, points, model.degree)
    return dict(
        dataset=record["dataset"], sample_id=record["sample_id"],
        deployment_k=int(mask.sum()), deployment_mse=float(mse[0]),
        deployment_pass=bool(mse[0] <= tolerance[0]),
        fixed_geometry_k=int(compact.sum()), fixed_geometry_mse=float(fixed_mse[0]),
        fixed_geometry_pass=bool(fixed_mse[0] <= tolerance[0]),
        redecoded_mse=float(decoded_mse[0]),
        redecoded_pass=bool(decoded_mse[0] <= tolerance[0]),
        deleted_knots=int(mask.sum() - compact.sum()),
        deployment_mask=mask[0].tolist(), compact_mask=compact[0].tolist(),
        candidate_refits=scans, verification_refits=checks,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases-json", type=Path, required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=2)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--greedy-steps", type=int, default=32)
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--output", type=Path,
                        help="Optional NEW JSON path; existing files are never overwritten.")
    args = parser.parse_args(argv)
    if min(args.samples_per_dataset, args.greedy_steps, args.torch_num_threads) < 1:
        parser.error("sample, step and thread counts must be positive")
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}; choose a new name")
    torch.set_num_threads(args.torch_num_threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, config, _ = build_model_from_checkpoint(checkpoint)
    if not hasattr(model, "decode_subset"):
        parser.error("a native v16 subset-decoder checkpoint is required")
    model.eval()
    tolerance = float(config["mse_tolerance"])
    objective = V16SubsetLoss(mse_tolerance=tolerance, teacher_greedy_steps=args.greedy_steps)
    cases = json.loads(args.cases_json.read_text(encoding="utf-8-sig"))["records"]
    available = {record["dataset"] for record in cases}
    missing = set(args.dataset) - available
    if missing:
        parser.error(f"datasets absent from cases JSON: {sorted(missing)}")
    counts, results = {}, []
    for record in cases:
        dataset = record["dataset"]
        if args.dataset and dataset not in args.dataset:
            continue
        if counts.get(dataset, 0) >= args.samples_per_dataset:
            continue
        counts[dataset] = counts.get(dataset, 0) + 1
        result = audit_case(model, objective, record)
        results.append(result)
        print(f"{dataset}/{record['sample_id']}: "
              f"K {result['deployment_k']} -> {result['fixed_geometry_k']}, "
              f"MSE deployment/fixed/redecoded="
              f"{result['deployment_mse']:.6e}/{result['fixed_geometry_mse']:.6e}/"
              f"{result['redecoded_mse']:.6e}", flush=True)
    if not results:
        parser.error("no cases selected")
    report = dict(
        role="diagnostic_only_not_deployment_or_generalization_benchmark",
        checkpoint=str(args.checkpoint.resolve()), checkpoint_epoch=checkpoint.get("epoch"),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        cases_sha256=hashlib.sha256(args.cases_json.read_bytes()).hexdigest(),
        model_config=config, mse_tolerance=tolerance,
        error_scope="normalized network INPUT points; Euclidean MSE, no square root",
        solver="training-compatible float64 endpoint-constrained standard B-spline refit; jitter=1e-10",
        limitations=("First saved cases per requested dataset, NOT a random population sample. "
                     "Infeasible seeds are retained as failures, not omitted. "
                     "Reference/high-density errors are NOT measured here. "
                     "Greedy respects model coverage/min-count constraints and a bounded step budget."),
        greedy_steps=args.greedy_steps, records=results,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
        print(f"Saved diagnostic: {args.output}")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
