from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.synthetic import SyntheticCubicBSplineDataset
from spline_fitting.checkpointing import (
    CURRENT_OBJECTIVE_VERSION,
    build_model_from_checkpoint,
    migrate_loss_config,
)
from spline_fitting.evaluation.knot_diagnostics import (
    activity_statistics,
    match_internal_knots,
)
from spline_fitting.evaluation.bspline_inference import (
    refit_model_output_as_bsplines,
)
from spline_fitting.losses.total_loss import LossWeights, SplineFittingLoss


def _histogram(values: torch.Tensor, maximum: int) -> dict[str, int]:
    counts = torch.bincount(values.to(torch.long), minlength=maximum + 1)
    return {str(index): int(count) for index, count in enumerate(counts) if count}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the hard-gated surrogate and deployed standard B-spline "
            "for a checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "best.pt")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--activity-threshold", type=float, default=None)
    parser.add_argument(
        "--threshold-sweep",
        type=float,
        nargs="*",
        default=[0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9],
    )
    parser.add_argument("--knot-tolerance", type=float, default=0.05)
    parser.add_argument("--smoothness-weight", type=float, default=1e-6)
    parser.add_argument("--control-ridge", type=float, default=0.0)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    threshold = (
        args.activity_threshold
        if args.activity_threshold is not None
        else float(checkpoint.get("activity_threshold", 0.5))
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_config, legacy_checkpoint = build_model_from_checkpoint(checkpoint)
    model.set_activity_threshold(threshold)
    model.to(device).eval()

    dataset_config = dict(checkpoint.get("dataset_config", {}))
    dataset_config.setdefault("num_points", 64)
    dataset_config.setdefault("point_dim", model_config.get("point_dim", 2))
    dataset_config["return_ground_truth"] = True
    dataset = SyntheticCubicBSplineDataset(
        size=args.num_samples,
        seed=args.seed,
        **dataset_config,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)

    loss_config, assumed_loss_config = migrate_loss_config(
        checkpoint,
        legacy=legacy_checkpoint,
    )
    loss_fn = SplineFittingLoss(
        LossWeights(**loss_config["weights"]),
        min_knot_gap=loss_config.get("min_knot_gap", 1e-3),
        knot_position_beta=loss_config.get("knot_position_beta", 0.02),
    ).to(device)

    loss_sums: dict[str, float] = defaultdict(float)
    retained_counts: list[torch.Tensor] = []
    activity_masses: list[torch.Tensor] = []
    activity_ranges: list[torch.Tensor] = []
    importance_ranges: list[torch.Tensor] = []
    bspline_fit_losses: list[float] = []
    bspline_coordinate_losses: list[float] = []
    bspline_augmented_objectives: list[float] = []
    bspline_control_counts: list[int] = []
    bspline_rank_deficient: list[bool] = []
    sweep_counts: dict[float, list[torch.Tensor]] = {
        value: [] for value in sorted(set(args.threshold_sweep + [threshold]))
    }
    total_samples = 0
    activity_min = float("inf")
    activity_max = float("-inf")
    true_counts: list[int] = []
    total_matched = 0
    total_predicted = 0
    total_true = 0
    matched_error_sum = 0.0
    true_parameter_squared_error = 0.0
    true_parameter_values = 0

    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device)
            chord_params = batch["chord_params"].to(device)
            true_params = batch.get("true_params")
            if true_params is not None:
                true_params = true_params.to(device)
            true_internal_knots = batch.get("true_internal_knots")
            true_internal_knot_mask = batch.get("true_internal_knot_mask")
            if true_internal_knots is not None:
                true_internal_knots = true_internal_knots.to(device)
            if true_internal_knot_mask is not None:
                true_internal_knot_mask = true_internal_knot_mask.to(device)
            output = model(points)
            losses = loss_fn(
                output,
                points,
                chord_params=chord_params,
                true_params=true_params,
                true_internal_knots=true_internal_knots,
                true_internal_knot_mask=true_internal_knot_mask,
            )
            batch_size = points.shape[0]
            total_samples += batch_size
            for name, value in losses.items():
                loss_sums[name] += float(value) * batch_size

            summary = activity_statistics(output["activity"], threshold)
            activity_masses.append(summary["activity_mass"].cpu())
            activity_ranges.append(
                (
                    output["activity"].amax(dim=-1) - output["activity"].amin(dim=-1)
                ).cpu()
            )
            importance_ranges.append(
                (
                    output["normalized_knot_importance"].amax(dim=-1)
                    - output["normalized_knot_importance"].amin(dim=-1)
                ).cpu()
            )
            activity_min = min(activity_min, float(output["activity"].min()))
            activity_max = max(activity_max, float(output["activity"].max()))
            for sweep_threshold in sweep_counts:
                sweep_counts[sweep_threshold].append(
                    (output["activity"] >= sweep_threshold)
                    .sum(dim=-1)
                    .to(torch.long)
                    .cpu()
                )

            if legacy_checkpoint:
                # Old checkpoints did not produce deterministic binary gates.
                # Threshold them explicitly for backward-compatible diagnosis;
                # A-scheme checkpoints use activity_gate directly.
                deployment_output = dict(output)
                deployment_output["activity_gate"] = (
                    output["activity"] >= threshold
                ).to(output["activity"].dtype)
            else:
                deployment_output = output
            deployed = refit_model_output_as_bsplines(
                deployment_output,
                points,
                degree=model.degree,
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
            )
            retained_counts.append(
                torch.tensor(
                    [item.retained_count for item in deployed],
                    dtype=torch.long,
                )
            )
            bspline_fit_losses.extend(float(item.fit_mse) for item in deployed)
            bspline_coordinate_losses.extend(
                float(item.spline.coordinate_mse) for item in deployed
            )
            bspline_augmented_objectives.extend(
                float(item.spline.augmented_objective) for item in deployed
            )
            for item in deployed:
                control_count = int(item.control_points.shape[0])
                bspline_control_counts.append(control_count)
                if item.spline.solver_rank is not None:
                    bspline_rank_deficient.append(
                        item.spline.solver_rank < control_count
                    )

            if "true_params" in batch:
                difference = output["params"].cpu() - batch["true_params"]
                true_parameter_squared_error += float(difference.pow(2).sum())
                true_parameter_values += difference.numel()

            if "true_internal_knots" in batch:
                for index, item in enumerate(deployed):
                    target = batch["true_internal_knots"][index][
                        batch["true_internal_knot_mask"][index]
                    ]
                    matching = match_internal_knots(
                        item.retained_internal_knots,
                        target,
                        tolerance=args.knot_tolerance,
                    )
                    true_counts.append(matching.true_count)
                    total_matched += matching.matched_count
                    total_predicted += matching.predicted_count
                    total_true += matching.true_count
                    if matching.matched_count:
                        matched_error_sum += (
                            matching.matched_mae * matching.matched_count
                        )

    retained = torch.cat(retained_counts).to(torch.long)
    masses = torch.cat(activity_masses)
    ranges = torch.cat(activity_ranges)
    pilot_ranges = torch.cat(importance_ranges)
    candidate_count = model_config["max_internal_knots"]
    bspline_fit_mean = sum(bspline_fit_losses) / max(len(bspline_fit_losses), 1)
    bspline_coordinate_mean = sum(bspline_coordinate_losses) / max(
        len(bspline_coordinate_losses), 1
    )
    mean_losses = {name: value / total_samples for name, value in loss_sums.items()}
    weighted_components = {
        "fit": loss_fn.weights.fit * mean_losses["fit_loss"],
        "l0": loss_fn.weights.l0 * mean_losses["l0_loss"],
        "activity": loss_fn.weights.activity * mean_losses["activity_loss"],
        "binary": loss_fn.weights.binary * mean_losses["binary_loss"],
        "gap": loss_fn.weights.gap * mean_losses["gap_loss"],
        "parameter_prior": (
            loss_fn.weights.parameter_prior * mean_losses["parameter_prior_loss"]
        ),
        "true_parameter": (
            loss_fn.weights.true_parameter * mean_losses["true_parameter_loss"]
        ),
        "existence": loss_fn.weights.existence * mean_losses["existence_loss"],
        "knot_position": (
            loss_fn.weights.knot_position * mean_losses["knot_position_loss"]
        ),
        "count": loss_fn.weights.count * mean_losses["count_loss"],
    }
    if "orthogonal_loss" in mean_losses:
        weighted_components["legacy_orthogonal"] = (
            loss_fn.weights.orthogonal * mean_losses["orthogonal_loss"]
        )
    current_no_orthogonal_objective = mean_losses["loss"] - weighted_components.get(
        "legacy_orthogonal", 0.0
    )
    hard_histogram = _histogram(retained, candidate_count)
    true_histogram = (
        _histogram(torch.tensor(true_counts), max(true_counts)) if true_counts else {}
    )
    true_count_mean = (
        sum(true_counts) / len(true_counts) if true_counts else float("nan")
    )
    precision = total_matched / total_predicted if total_predicted else 0.0
    recall = total_matched / total_true if total_true else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    knot_mae = matched_error_sum / total_matched if total_matched else float("nan")
    parameter_rmse = math.sqrt(
        true_parameter_squared_error / max(true_parameter_values, 1)
    )

    report = {
        "schema_version": 3,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_val": checkpoint.get("best_val"),
        "objective_version": checkpoint.get("objective_version", "historical"),
        "dataset_seed": args.seed,
        "num_samples": total_samples,
        "dataset_config": dataset_config,
        "model_config": model_config,
        "gate_mode": model_config["gate_mode"],
        "activity_use_pilot_importance": model_config.get(
            "activity_use_pilot_importance", False
        ),
        "legacy_checkpoint_migration": legacy_checkpoint,
        "loss_config": loss_config,
        "loss_config_assumed": assumed_loss_config,
        "activity_threshold": threshold,
        "candidate_internal_knots": candidate_count,
        "expected_active_count_mean": float(masses.mean()),
        "activity_min": activity_min,
        "activity_max": activity_max,
        "mean_within_curve_activity_range": float(ranges.mean()),
        "mean_within_curve_pilot_importance_range": (
            float(pilot_ranges.mean())
            if model_config.get("activity_use_pilot_importance", False)
            else None
        ),
        "hard_retained_mean": float(retained.float().mean()),
        "hard_retained_min": int(retained.min()),
        "hard_retained_max": int(retained.max()),
        "zero_knot_fraction": float((retained == 0).float().mean()),
        "hard_count_histogram": hard_histogram,
        "threshold_sweep": {
            str(value): {
                "mean": float(torch.cat(chunks).float().mean()),
                "histogram": _histogram(torch.cat(chunks), candidate_count),
            }
            for value, chunks in sweep_counts.items()
        },
        "network_total_objective": mean_losses["loss"],
        "network_current_no_orthogonal_objective": (current_no_orthogonal_objective),
        "network_forward_fit_loss": mean_losses["fit_loss"],
        "network_forward_rms_euclidean": math.sqrt(mean_losses["fit_loss"]),
        "network_forward_coordinate_rmse": math.sqrt(
            mean_losses["fit_loss"] / dataset_config["point_dim"]
        ),
        "standard_bspline_refit_loss": bspline_fit_mean,
        "standard_bspline_refit_rms_euclidean": math.sqrt(bspline_fit_mean),
        "standard_bspline_coordinate_rmse": math.sqrt(bspline_coordinate_mean),
        "standard_bspline_control_count_mean": (
            sum(bspline_control_counts) / max(len(bspline_control_counts), 1)
        ),
        "standard_bspline_augmented_objective_mean": (
            sum(bspline_augmented_objectives)
            / max(len(bspline_augmented_objectives), 1)
        ),
        "standard_bspline_rank_deficient_fraction": (
            sum(bspline_rank_deficient) / len(bspline_rank_deficient)
            if bspline_rank_deficient
            else None
        ),
        "standard_bspline_smoothness_weight": args.smoothness_weight,
        "standard_bspline_control_ridge": args.control_ridge,
        "raw_loss_components": mean_losses,
        "weighted_loss_components": weighted_components,
        "true_parameter_rmse": parameter_rmse,
        "true_knot_count_histogram": true_histogram,
        "true_knot_count_mean": true_count_mean,
        "knot_match_tolerance": args.knot_tolerance,
        "knot_match_precision": precision,
        "knot_match_recall": recall,
        "knot_match_f1": f1,
        "matched_knot_mae": knot_mae if math.isfinite(knot_mae) else None,
    }

    print("Checkpoint knot-reduction evaluation")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  recorded best epoch: {checkpoint.get('epoch', 'not recorded')}")
    print(
        "  objective version: "
        f"{checkpoint.get('objective_version', 'historical (pre-removal)')}"
    )
    print(f"  samples / seed: {total_samples} / {args.seed}")
    print(f"  recorded noise_std: {dataset_config.get('noise_std', 'not recorded')}")
    print(f"  gate mode: {model_config['gate_mode']}")
    print(
        "  local cross-attention KnotHead: "
        f"{model_config.get('knot_use_local_cross_attention', False)}"
    )
    print(
        "  pilot knot importance (historical compatibility): "
        f"{model_config.get('activity_use_pilot_importance', False)}"
    )
    if legacy_checkpoint:
        print("  WARNING: pre-A checkpoint migrated to legacy_soft diagnostics.")
    max_true_internal = dataset_config.get("max_control_points", 4) - model.degree - 1
    if candidate_count < max_true_internal:
        print(
            f"  WARNING: data can contain {max_true_internal} true internal knots, "
            f"but the model has only {candidate_count} candidates."
        )
    if assumed_loss_config:
        print("  WARNING: loss_config was absent; compatible defaults were assumed.")
    if checkpoint.get("objective_version") != CURRENT_OBJECTIVE_VERSION:
        print(
            "  NOTE: historical objective loaded with its recorded architecture "
            "and loss configuration."
        )
    print("\nNetwork forward model")
    print(f"  total objective: {mean_losses['loss']:.9e}")
    if "legacy_orthogonal" in weighted_components:
        print(
            "  current no-orthogonal objective (same saved weights): "
            f"{current_no_orthogonal_objective:.9e}"
        )
    print(f"  fit loss (mean squared Euclidean): {mean_losses['fit_loss']:.9e}")
    print(f"  RMS Euclidean distance: {math.sqrt(mean_losses['fit_loss']):.9e}")
    print("  weighted objective components:")
    for name, value in weighted_components.items():
        print(f"    {name}: {value:.9e}")
    print(f"  E[K] / activity probability mass: {masses.mean().item():.6f}")
    print(f"  activity range over all values: [{activity_min:.7f}, {activity_max:.7f}]")
    print(f"  mean within-curve activity range: {ranges.mean().item():.7e}")
    if model_config.get("activity_use_pilot_importance", False):
        print(
            "  mean within-curve pilot importance range: "
            f"{pilot_ranges.mean().item():.7e}"
        )
    print("\nHard-Concrete deployment: hard pruning + standard B-spline refit")
    print(f"  threshold: {threshold:.4f}")
    print(
        f"  retained knots mean/min/max: {retained.float().mean().item():.3f}/"
        f"{int(retained.min())}/{int(retained.max())} out of {candidate_count}"
    )
    print(f"  retained-count histogram: {hard_histogram}")
    print(f"  zero-knot fraction: {(retained == 0).float().mean().item():.3f}")
    print(f"  standard B-spline refit loss: {bspline_fit_mean:.9e}")
    print(f"  standard B-spline RMS distance: {math.sqrt(bspline_fit_mean):.9e}")
    print(
        f"  standard B-spline coordinate RMSE: {math.sqrt(bspline_coordinate_mean):.9e}"
    )
    print(
        "  control points mean: "
        f"{sum(bspline_control_counts) / max(len(bspline_control_counts), 1):.3f}"
    )
    if bspline_rank_deficient:
        print(
            "  rank-deficient fraction: "
            f"{sum(bspline_rank_deficient) / len(bspline_rank_deficient):.3f}"
        )
    else:
        print("  rank-deficient fraction: not reported by this solver backend")
    print(f"  control-polygon smoothness weight: {args.smoothness_weight:.3e}")
    print("\nThreshold sweep (mean retained count and histogram)")
    for value, chunks in sweep_counts.items():
        counts = torch.cat(chunks)
        print(
            f"  {value:.2f}: mean={counts.float().mean().item():.3f}, "
            f"hist={_histogram(counts, candidate_count)}"
        )
    if true_counts:
        print("\nGround-truth diagnostics")
        print(f"  mean true internal knots: {true_count_mean:.3f}")
        print(f"  true knot-count histogram: {true_histogram}")
        print(f"  true parameter RMSE: {parameter_rmse:.9e}")
        print("  knot matching is meaningful only under a shared parameterization.")
        print(
            f"  match@{args.knot_tolerance:.3f}: precision={precision:.3f}, "
            f"recall={recall:.3f}, F1={f1:.3f}, matched MAE={knot_mae:.6f}"
        )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(f"\nSaved JSON report to: {args.json_output}")


if __name__ == "__main__":
    main()
