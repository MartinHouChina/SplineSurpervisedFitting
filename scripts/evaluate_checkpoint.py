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

from spline_fitting.checkpointing import (
    CURRENT_OBJECTIVE_VERSION,
    build_model_from_checkpoint,
    migrate_loss_config,
)
from spline_fitting.data.synthetic import SyntheticCubicBSplineDataset
from spline_fitting.evaluation.bspline_inference import (
    refit_model_output_as_bsplines,
    select_count_conditioned_output_by_bic,
)
from spline_fitting.evaluation.knot_diagnostics import match_internal_knots
from spline_fitting.losses.total_loss import LossWeights, SplineFittingLoss


def _histogram(values: torch.Tensor, maximum: int) -> dict[str, int]:
    counts = torch.bincount(values.to(torch.long), minlength=maximum + 1)
    return {str(index): int(count) for index, count in enumerate(counts) if count}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted knot structure and the deployed B-spline."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument(
        "--count-selection",
        choices=("auto", "network", "bic"),
        default="auto",
        help="Select the network argmax branch or compare complete branches by BIC.",
    )
    parser.add_argument("--count-prior-weight", type=float, default=1.0)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0:
        parser.error("sample and batch sizes must be positive")
    if args.knot_tolerance < 0.0:
        parser.error("--knot-tolerance must be non-negative")

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
    structure_mode = model_config.get("structure_mode", "hard_concrete")
    count_conditioned = structure_mode == "count_conditioned"
    count_selection = args.count_selection
    if count_selection == "auto":
        count_selection = (
            "bic"
            if checkpoint.get("objective_version") == CURRENT_OBJECTIVE_VERSION
            else "network"
        )

    dataset_config = dict(checkpoint.get("dataset_config", {}))
    dataset_config.setdefault("num_points", 64)
    dataset_config.setdefault("point_dim", model_config.get("point_dim", 2))
    dataset_config.setdefault(
        "canonical_knot_tolerance",
        5e-3
        if checkpoint.get("objective_version") == CURRENT_OBJECTIVE_VERSION
        else 0.0,
    )
    dataset_config["return_ground_truth"] = True
    dataset = SyntheticCubicBSplineDataset(
        size=args.num_samples,
        seed=args.seed,
        **dataset_config,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)

    loss_config, assumed_loss_config = migrate_loss_config(
        checkpoint, legacy=legacy_checkpoint
    )
    loss_fn = SplineFittingLoss(
        LossWeights(**loss_config["weights"]),
        min_knot_gap=loss_config.get("min_knot_gap", 1e-3),
        knot_position_beta=loss_config.get("knot_position_beta", 0.02),
    ).to(device)

    candidate_count = int(model_config["max_internal_knots"])
    loss_sums: dict[str, float] = defaultdict(float)
    retained_counts: list[torch.Tensor] = []
    predicted_counts: list[torch.Tensor] = []
    expected_counts: list[torch.Tensor] = []
    activity_values: list[torch.Tensor] = []
    activity_ranges: list[torch.Tensor] = []
    sweep_counts: dict[float, list[torch.Tensor]] = (
        {
            value: []
            for value in sorted(set(args.threshold_sweep + [threshold]))
        }
        if not count_conditioned
        else {}
    )
    count_confusion = torch.zeros(
        candidate_count + 1, candidate_count + 1, dtype=torch.long
    )
    true_counts: list[int] = []
    bspline_fit_losses: list[float] = []
    bspline_coordinate_losses: list[float] = []
    bspline_augmented_objectives: list[float] = []
    bspline_control_counts: list[int] = []
    bspline_rank_deficient: list[bool] = []
    total_samples = 0
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
            true_params = batch["true_params"].to(device)
            true_knots = batch["true_internal_knots"].to(device)
            true_mask = batch["true_internal_knot_mask"].to(device)
            output = model(points)
            losses = loss_fn(
                output,
                points,
                chord_params=chord_params,
                true_params=true_params,
                true_internal_knots=true_knots,
                true_internal_knot_mask=true_mask,
                activity_threshold=threshold,
            )
            batch_size = points.shape[0]
            total_samples += batch_size
            for name, value in losses.items():
                loss_sums[name] += float(value) * batch_size

            if count_conditioned:
                batch_predicted = output["predicted_knot_count"].cpu()
                batch_expected = output["expected_knot_count"].cpu()
                batch_true = true_mask.sum(dim=-1).to(torch.long).cpu()
                predicted_counts.append(batch_predicted)
                expected_counts.append(batch_expected)
                for target, predicted in zip(batch_true.tolist(), batch_predicted.tolist()):
                    count_confusion[target, predicted] += 1
                if count_selection == "bic":
                    deployment_output, _ = select_count_conditioned_output_by_bic(
                        output,
                        points,
                        degree=model.degree,
                        smoothness_weight=args.smoothness_weight,
                        prior_weight=args.count_prior_weight,
                    )
                else:
                    deployment_output = output
            else:
                activity = output["activity"]
                activity_values.append(activity.cpu())
                activity_ranges.append(
                    (activity.amax(dim=-1) - activity.amin(dim=-1)).cpu()
                )
                for sweep_threshold in sweep_counts:
                    sweep_counts[sweep_threshold].append(
                        (activity >= sweep_threshold).sum(dim=-1).cpu()
                    )
                if legacy_checkpoint:
                    deployment_output = dict(output)
                    deployment_output["activity_gate"] = (
                        activity >= threshold
                    ).to(activity.dtype)
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
                torch.tensor([item.retained_count for item in deployed])
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
                    bspline_rank_deficient.append(item.spline.solver_rank < control_count)

            parameter_difference = output["params"].cpu() - batch["true_params"]
            true_parameter_squared_error += float(parameter_difference.pow(2).sum())
            true_parameter_values += parameter_difference.numel()
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
                    matched_error_sum += matching.matched_mae * matching.matched_count

    retained = torch.cat(retained_counts).to(torch.long)
    true_count_tensor = torch.tensor(true_counts, dtype=torch.long)
    mean_losses = {name: value / total_samples for name, value in loss_sums.items()}
    weighted_components = {
        "fit": loss_fn.weights.fit * mean_losses["fit_loss"],
        "l0": loss_fn.weights.l0 * mean_losses["l0_loss"],
        "activity": loss_fn.weights.activity * mean_losses["activity_loss"],
        "binary": loss_fn.weights.binary * mean_losses["binary_loss"],
        "gap": loss_fn.weights.gap * mean_losses["gap_loss"],
        "parameter_prior": loss_fn.weights.parameter_prior
        * mean_losses["parameter_prior_loss"],
        "true_parameter": loss_fn.weights.true_parameter
        * mean_losses["true_parameter_loss"],
        "existence": loss_fn.weights.existence * mean_losses["existence_loss"],
        "knot_position": loss_fn.weights.knot_position
        * mean_losses["knot_position_loss"],
        "count": loss_fn.weights.count * mean_losses["count_loss"],
        "over_count": loss_fn.weights.over_count * mean_losses["over_count_loss"],
    }
    precision = total_matched / total_predicted if total_predicted else 0.0
    recall = total_matched / total_true if total_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    knot_mae = matched_error_sum / total_matched if total_matched else float("nan")
    bspline_fit_mean = sum(bspline_fit_losses) / len(bspline_fit_losses)
    bspline_coordinate_mean = sum(bspline_coordinate_losses) / len(
        bspline_coordinate_losses
    )
    parameter_rmse = math.sqrt(
        true_parameter_squared_error / max(true_parameter_values, 1)
    )
    hard_histogram = _histogram(retained, candidate_count)
    true_histogram = _histogram(true_count_tensor, candidate_count)

    if count_conditioned:
        predicted_count_tensor = torch.cat(predicted_counts).to(torch.long)
        expected_count_tensor = torch.cat(expected_counts)
        count_accuracy = float((predicted_count_tensor == true_count_tensor).float().mean())
        count_mae = float(
            (predicted_count_tensor - true_count_tensor).abs().float().mean()
        )
        expected_count_mean = float(expected_count_tensor.mean())
        network_histogram = _histogram(predicted_count_tensor, candidate_count)
        deployment_count_accuracy = float(
            (retained == true_count_tensor).float().mean()
        )
        deployment_count_mae = float(
            (retained - true_count_tensor).abs().float().mean()
        )
        activity_report = None
        threshold_report = None
    else:
        all_activity = torch.cat(activity_values)
        all_ranges = torch.cat(activity_ranges)
        predicted_count_tensor = retained
        count_accuracy = float((retained == true_count_tensor).float().mean())
        count_mae = float((retained - true_count_tensor).abs().float().mean())
        expected_count_mean = float(all_activity.sum(dim=-1).mean())
        network_histogram = hard_histogram
        deployment_count_accuracy = count_accuracy
        deployment_count_mae = count_mae
        activity_report = {
            "minimum": float(all_activity.min()),
            "maximum": float(all_activity.max()),
            "mean_within_curve_range": float(all_ranges.mean()),
        }
        threshold_report = {
            str(value): {
                "mean": float(torch.cat(chunks).float().mean()),
                "histogram": _histogram(torch.cat(chunks), candidate_count),
            }
            for value, chunks in sweep_counts.items()
        }

    report = {
        "schema_version": 7,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_selection_metric": checkpoint.get("selection_metric"),
        "checkpoint_selection_value": checkpoint.get("selection_value"),
        "objective_version": checkpoint.get("objective_version", "historical"),
        "structure_mode": structure_mode,
        "dataset_seed": args.seed,
        "num_samples": total_samples,
        "dataset_config": dataset_config,
        "model_config": model_config,
        "loss_config": loss_config,
        "loss_config_assumed": assumed_loss_config,
        "candidate_internal_knots": candidate_count,
        "count_selection": count_selection if count_conditioned else "threshold",
        "count_prior_weight": args.count_prior_weight if count_conditioned else None,
        "predicted_knot_count_mean": float(retained.float().mean()),
        "expected_knot_count_mean": expected_count_mean,
        "predicted_knot_count_histogram": hard_histogram,
        "network_predicted_knot_count_histogram": network_histogram,
        "true_knot_count_mean": float(true_count_tensor.float().mean()),
        "true_knot_count_histogram": true_histogram,
        "knot_count_accuracy": count_accuracy,
        "knot_count_mae": count_mae,
        "deployment_knot_count_accuracy": deployment_count_accuracy,
        "deployment_knot_count_mae": deployment_count_mae,
        "count_confusion_matrix": count_confusion.tolist()
        if count_conditioned
        else None,
        "activity_diagnostics": activity_report,
        "threshold_sweep": threshold_report,
        "zero_knot_fraction": float((retained == 0).float().mean()),
        "all_knot_fraction": float((retained == candidate_count).float().mean()),
        "network_total_objective": mean_losses["loss"],
        "network_forward_fit_loss": mean_losses["fit_loss"],
        "network_forward_rms_euclidean": math.sqrt(mean_losses["fit_loss"]),
        "network_forward_coordinate_rmse": math.sqrt(
            mean_losses["fit_loss"] / dataset_config["point_dim"]
        ),
        "standard_bspline_refit_loss": bspline_fit_mean,
        "standard_bspline_refit_rms_euclidean": math.sqrt(bspline_fit_mean),
        "standard_bspline_coordinate_rmse": math.sqrt(bspline_coordinate_mean),
        "standard_bspline_control_count_mean": sum(bspline_control_counts)
        / len(bspline_control_counts),
        "standard_bspline_augmented_objective_mean": sum(
            bspline_augmented_objectives
        )
        / len(bspline_augmented_objectives),
        "standard_bspline_rank_deficient_fraction": (
            sum(bspline_rank_deficient) / len(bspline_rank_deficient)
            if bspline_rank_deficient
            else None
        ),
        "raw_loss_components": mean_losses,
        "weighted_loss_components": weighted_components,
        "true_parameter_rmse": parameter_rmse,
        "knot_match_tolerance": args.knot_tolerance,
        "knot_match_precision": precision,
        "knot_match_recall": recall,
        "knot_match_f1": f1,
        "matched_knot_mae": knot_mae if math.isfinite(knot_mae) else None,
    }

    print("Checkpoint structured-knot evaluation")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  objective version: {report['objective_version']}")
    print(f"  structure mode: {structure_mode}")
    print(f"  recorded best epoch: {checkpoint.get('epoch', 'not recorded')}")
    print(
        "  checkpoint selection: "
        f"{checkpoint.get('selection_metric', 'historical')} = "
        f"{checkpoint.get('selection_value', checkpoint.get('best_val', 'not recorded'))}"
    )
    if checkpoint.get("objective_version") != CURRENT_OBJECTIVE_VERSION:
        print("  NOTE: historical objective loaded with compatibility semantics.")
    if assumed_loss_config:
        print("  WARNING: loss_config was absent; compatible defaults were assumed.")

    print("\nNetwork forward model")
    print(f"  total objective: {mean_losses['loss']:.9e}")
    print(f"  fit loss (mean squared Euclidean): {mean_losses['fit_loss']:.9e}")
    print(f"  RMS Euclidean distance: {math.sqrt(mean_losses['fit_loss']):.9e}")
    print("  weighted objective components:")
    for name, value in weighted_components.items():
        print(f"    {name}: {value:.9e}")

    if count_conditioned:
        print("\nSupervised knot count")
        print(f"  network count accuracy: {count_accuracy:.3f}")
        print(f"  network count MAE: {count_mae:.3f}")
        print(f"  expected count mean: {expected_count_mean:.3f}")
        print(f"  network count histogram: {network_histogram}")
        print(f"  deployment selection: {count_selection}")
        print(f"  deployment count accuracy: {deployment_count_accuracy:.3f}")
        print(f"  deployment count MAE: {deployment_count_mae:.3f}")
        print(f"  deployment count histogram: {hard_histogram}")
    else:
        print("\nHistorical threshold-gated structure")
        print(f"  threshold: {threshold:.3f}")
        print(f"  activity probability mass mean: {expected_count_mean:.3f}")
        print(f"  predicted count histogram: {hard_histogram}")
        print("  threshold sweep:")
        for value, item in threshold_report.items():
            print(f"    {value}: mean={item['mean']:.3f}, hist={item['histogram']}")

    print("\nStandard B-spline deployment")
    print(
        f"  internal knots mean/min/max: {retained.float().mean():.3f}/"
        f"{int(retained.min())}/{int(retained.max())}"
    )
    print(f"  zero-knot fraction: {report['zero_knot_fraction']:.3f}")
    print(f"  all-knot fraction: {report['all_knot_fraction']:.3f}")
    print(f"  refit loss: {bspline_fit_mean:.9e}")
    print(f"  refit RMS distance: {math.sqrt(bspline_fit_mean):.9e}")

    print("\nGround-truth diagnostics")
    print(f"  true knot-count mean: {true_count_tensor.float().mean():.3f}")
    print(f"  true knot-count histogram: {true_histogram}")
    print(f"  true parameter RMSE: {parameter_rmse:.9e}")
    print(
        f"  match@{args.knot_tolerance:.3f}: precision={precision:.3f}, "
        f"recall={recall:.3f}, F1={f1:.3f}, matched MAE={knot_mae:.6f}"
    )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nSaved JSON report to: {args.json_output}")


if __name__ == "__main__":
    main()
