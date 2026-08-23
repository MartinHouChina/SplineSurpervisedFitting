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
    CANDIDATE_PRUNING_OBJECTIVE_VERSION,
    COUNT_CONDITIONED_V5_OBJECTIVE_VERSION,
    CURRENT_OBJECTIVE_VERSION,
    build_model_from_checkpoint,
    migrate_loss_config,
)
from spline_fitting.data.synthetic import SyntheticCubicBSplineDataset
from spline_fitting.evaluation.bspline_inference import (
    HardGatedBSplineFit,
    refit_model_output_as_bsplines,
    select_count_conditioned_output_by_bic,
)
from spline_fitting.evaluation.knot_diagnostics import match_internal_knots
from spline_fitting.evaluation.minimal_knot_pruning import (
    MinimalKnotPruningResult,
    prune_knots_to_rms_tolerance,
)
from spline_fitting.losses.candidate_pruning_loss import (
    CandidatePruningLoss,
    CandidatePruningLossWeights,
)
from spline_fitting.losses.total_loss import LossWeights, SplineFittingLoss


def _histogram(values: torch.Tensor, maximum: int) -> dict[str, int]:
    counts = torch.bincount(values.to(torch.long), minlength=maximum + 1)
    return {str(index): int(count) for index, count in enumerate(counts) if count}


def resolve_fit_tolerance(
    checkpoint: dict[str, object], explicit_tolerance: float | None
) -> float:
    """Resolve the normalized RMS bound stored with a v7 checkpoint."""
    if explicit_tolerance is not None:
        value = explicit_tolerance
    else:
        deployment = checkpoint.get("deployment_config", {})
        dataset = checkpoint.get("dataset_config", {})
        if isinstance(deployment, dict) and "error_tolerance" in deployment:
            value = deployment["error_tolerance"]
        elif isinstance(dataset, dict) and "canonical_knot_tolerance" in dataset:
            value = dataset["canonical_knot_tolerance"]
        else:
            value = 5e-3
    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("fit tolerance must be finite and non-negative")
    return tolerance


def _pruning_result_as_deployed_fit(
    result: MinimalKnotPruningResult,
    sample_index: int,
) -> HardGatedBSplineFit:
    retained_indices = list(range(result.initial_count))
    for step in result.accepted_steps:
        retained_indices.pop(step.removed_index)
    retained_mask = torch.zeros(
        result.initial_count,
        dtype=torch.bool,
        device=result.initial_internal_knots.device,
    )
    if retained_indices:
        retained_mask[retained_indices] = True
    return HardGatedBSplineFit(
        sample_index=sample_index,
        candidate_count=result.initial_count,
        retained_count=result.final_count,
        retained_mask=retained_mask,
        hard_gate=retained_mask,
        spline=result.final_fit,
    )


def prune_candidate_output_batch(
    output: dict[str, torch.Tensor],
    points: torch.Tensor,
    *,
    error_tolerance: float,
    degree: int,
    smoothness_weight: float,
    control_ridge: float,
) -> tuple[list[HardGatedBSplineFit], list[MinimalKnotPruningResult]]:
    """Hard-delete from all v7 candidates using measured B-spline RMS only."""
    for key in ("params", "internal_knots"):
        if key not in output:
            raise KeyError(f"candidate-pruning output is missing {key}")
    results: list[MinimalKnotPruningResult] = []
    deployed: list[HardGatedBSplineFit] = []
    for index in range(points.shape[0]):
        result = prune_knots_to_rms_tolerance(
            output["params"][index],
            points[index],
            output["internal_knots"][index],
            error_tolerance=error_tolerance,
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
            interpolate_endpoints=True,
        )
        results.append(result)
        deployed.append(_pruning_result_as_deployed_fit(result, index))
    return deployed, results


def candidate_loss_from_checkpoint(checkpoint: dict[str, object]) -> CandidatePruningLoss:
    config = checkpoint.get("loss_config", {})
    config = config if isinstance(config, dict) else {}
    raw_weights = config.get("weights", {})
    raw_weights = raw_weights if isinstance(raw_weights, dict) else {}
    allowed = CandidatePruningLossWeights.__dataclass_fields__
    weights = CandidatePruningLossWeights(
        **{key: value for key, value in raw_weights.items() if key in allowed}
    )
    return CandidatePruningLoss(
        weights,
        knot_position_beta=float(config.get("knot_position_beta", 0.01)),
        candidate_match_tolerance=float(
            config.get("candidate_match_tolerance", 0.02)
        ),
        fit_tolerance=float(config.get("fit_tolerance", 5e-3)),
        repulsion_distance=config.get("repulsion_distance"),
        positive_keep_weight=float(config.get("positive_keep_weight", 2.0)),
        exact_deletion_supervision=bool(
            config.get("exact_deletion_supervision", True)
        ),
        deletion_smoothness_weight=float(
            config.get("deletion_smoothness_weight", 1e-6)
        ),
        deletion_control_ridge=float(
            config.get("deletion_control_ridge", 0.0)
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted knot structure and the deployed B-spline."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--seed",
        type=int,
        default=20000,
        help="Independent synthetic test seed (training defaults: train=42, val=10000).",
    )
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
        "--fit-tolerance",
        type=float,
        default=None,
        help=(
            "Normalized RMS bound for candidate_pruning. Defaults to the "
            "checkpoint deployment error tolerance, then canonical label "
            "tolerance."
        ),
    )
    parser.add_argument(
        "--count-selection",
        choices=("auto", "network", "bic"),
        default="auto",
        help="Use the network count decision or compare legacy complete branches by BIC.",
    )
    parser.add_argument("--count-prior-weight", type=float, default=1.0)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0:
        parser.error("sample and batch sizes must be positive")
    if args.knot_tolerance < 0.0:
        parser.error("--knot-tolerance must be non-negative")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    try:
        fit_tolerance = resolve_fit_tolerance(checkpoint, args.fit_tolerance)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
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
    candidate_pruning = structure_mode == "candidate_pruning"
    count_conditioned = structure_mode in {
        "count_conditioned",
        "interactive_dynamic",
    }
    count_selection = args.count_selection
    if count_selection == "auto":
        count_selection = (
            "bic"
            if checkpoint.get("objective_version")
            == COUNT_CONDITIONED_V5_OBJECTIVE_VERSION
            else "network"
        )
    if structure_mode == "interactive_dynamic" and count_selection == "bic":
        parser.error(
            "v6 decodes only the selected count and does not support exhaustive BIC"
        )

    dataset_config = dict(checkpoint.get("dataset_config", {}))
    dataset_config.setdefault("num_points", 64)
    dataset_config.setdefault("point_dim", model_config.get("point_dim", 2))
    dataset_config.setdefault(
        "canonical_knot_tolerance",
        5e-3
        if checkpoint.get("objective_version")
        in {
            CANDIDATE_PRUNING_OBJECTIVE_VERSION,
            CURRENT_OBJECTIVE_VERSION,
            COUNT_CONDITIONED_V5_OBJECTIVE_VERSION,
        }
        else 0.0,
    )
    dataset_config["return_ground_truth"] = True
    dataset = SyntheticCubicBSplineDataset(
        size=args.num_samples,
        seed=args.seed,
        **dataset_config,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)

    if candidate_pruning:
        raw_loss_config = checkpoint.get("loss_config", {})
        loss_config = raw_loss_config if isinstance(raw_loss_config, dict) else {}
        assumed_loss_config = not bool(loss_config)
        loss_fn = candidate_loss_from_checkpoint(checkpoint).to(device)
    else:
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
    mode_counts: list[torch.Tensor] = []
    expected_counts: list[torch.Tensor] = []
    count_entropies: list[torch.Tensor] = []
    count_max_probabilities: list[torch.Tensor] = []
    activity_values: list[torch.Tensor] = []
    activity_ranges: list[torch.Tensor] = []
    sweep_counts: dict[float, list[torch.Tensor]] = (
        {
            value: []
            for value in sorted(set(args.threshold_sweep + [threshold]))
        }
        if not count_conditioned and not candidate_pruning
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
    bspline_start_endpoint_distances: list[float] = []
    bspline_end_endpoint_distances: list[float] = []
    total_samples = 0
    total_matched = 0
    total_predicted = 0
    total_true = 0
    matched_error_sum = 0.0
    true_parameter_squared_error = 0.0
    true_parameter_values = 0
    pruning_initial_rms: list[float] = []
    pruning_accepted_deletions: list[int] = []
    pruning_threshold_satisfied: list[bool] = []
    pruning_first_rejected_rms: list[float] = []
    candidate_recall_tolerances = (0.005, 0.01, 0.02)
    candidate_proposal_matched = {
        tolerance: 0 for tolerance in candidate_recall_tolerances
    }
    candidate_proposal_true = {
        tolerance: 0 for tolerance in candidate_recall_tolerances
    }
    candidate_proposal_error_sum = {
        tolerance: 0.0 for tolerance in candidate_recall_tolerances
    }
    keep_probability_values: list[torch.Tensor] = []
    keep_probability_ranges: list[torch.Tensor] = []

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
                batch_mode = output.get(
                    "count_mode_knot_count", output["predicted_knot_count"]
                ).cpu()
                batch_expected = output["expected_knot_count"].cpu()
                batch_probabilities = output["count_probabilities"].cpu()
                batch_true = true_mask.sum(dim=-1).to(torch.long).cpu()
                predicted_counts.append(batch_predicted)
                mode_counts.append(batch_mode)
                expected_counts.append(batch_expected)
                count_entropies.append(
                    -(
                        batch_probabilities
                        * batch_probabilities.clamp_min(1e-12).log()
                    ).sum(dim=-1)
                )
                count_max_probabilities.append(
                    batch_probabilities.amax(dim=-1)
                )
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
            elif candidate_pruning:
                keep_probability = output["keep_probability"]
                keep_probability_values.append(keep_probability.cpu())
                keep_probability_ranges.append(
                    (
                        keep_probability.amax(dim=-1)
                        - keep_probability.amin(dim=-1)
                    ).cpu()
                )
                deployment_output = output
                for index in range(batch_size):
                    target = true_knots[index, true_mask[index]].cpu()
                    # ``internal_knots`` is the pruning head's refined full
                    # candidate set.  These proposal diagnostics deliberately
                    # precede any hard deletion.
                    refined_candidates = output["internal_knots"][
                        index
                    ].detach().cpu()
                    for proposal_tolerance in candidate_recall_tolerances:
                        proposal_matching = match_internal_knots(
                            refined_candidates,
                            target,
                            tolerance=proposal_tolerance,
                        )
                        candidate_proposal_matched[proposal_tolerance] += (
                            proposal_matching.matched_count
                        )
                        candidate_proposal_true[proposal_tolerance] += (
                            proposal_matching.true_count
                        )
                        if proposal_matching.matched_count:
                            candidate_proposal_error_sum[proposal_tolerance] += (
                                proposal_matching.matched_mae
                                * proposal_matching.matched_count
                            )
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

            if candidate_pruning:
                deployed, pruning_batch = prune_candidate_output_batch(
                    output,
                    points,
                    error_tolerance=fit_tolerance,
                    degree=model.degree,
                    smoothness_weight=args.smoothness_weight,
                    control_ridge=args.control_ridge,
                )
                pruning_initial_rms.extend(
                    float(item.initial_fit.fit_rmse) for item in pruning_batch
                )
                pruning_accepted_deletions.extend(
                    len(item.accepted_steps) for item in pruning_batch
                )
                pruning_threshold_satisfied.extend(
                    item.threshold_satisfied for item in pruning_batch
                )
                for item in pruning_batch:
                    rejected = next(
                        (step for step in item.steps if not step.accepted), None
                    )
                    if rejected is not None:
                        pruning_first_rejected_rms.append(
                            float(rejected.candidate_rmse)
                        )
            else:
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
            for index, item in enumerate(deployed):
                control_count = int(item.control_points.shape[0])
                bspline_control_counts.append(control_count)
                if item.spline.solver_rank is not None:
                    bspline_rank_deficient.append(item.spline.solver_rank < control_count)
                bspline_start_endpoint_distances.append(
                    float(
                        (
                            item.reconstructed_points[0] - points[index, 0]
                        ).norm()
                    )
                )
                bspline_end_endpoint_distances.append(
                    float(
                        (
                            item.reconstructed_points[-1] - points[index, -1]
                        ).norm()
                    )
                )

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
    if candidate_pruning:
        weighted_components = {
            "fit": loss_fn.weights.fit * mean_losses["normalized_fit_loss"],
            "threshold_violation": loss_fn.weights.threshold_violation
            * mean_losses["threshold_violation_loss"],
            "true_parameter": loss_fn.weights.true_parameter
            * mean_losses["true_parameter_loss"],
            "candidate_coverage": loss_fn.weights.candidate_coverage
            * mean_losses["candidate_coverage_loss"],
            "candidate_repulsion": loss_fn.weights.candidate_repulsion
            * mean_losses["candidate_repulsion_loss"],
            "keep": loss_fn.weights.keep * mean_losses["keep_loss"],
            "remove_action": loss_fn.weights.remove_action
            * mean_losses["remove_action_loss"],
            "knot_position": loss_fn.weights.knot_position
            * mean_losses["knot_position_loss"],
            "count_consistency": loss_fn.weights.count_consistency
            * mean_losses["count_consistency_loss"],
            "deletion_cost": loss_fn.weights.deletion_cost
            * mean_losses["deletion_cost_loss"],
        }
    else:
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
            "existence": loss_fn.weights.existence
            * mean_losses["existence_loss"],
            "knot_position": loss_fn.weights.knot_position
            * mean_losses["knot_position_loss"],
            "count": loss_fn.weights.count * mean_losses["count_loss"],
            "over_count": loss_fn.weights.over_count
            * mean_losses["over_count_loss"],
        }
    precision = total_matched / total_predicted if total_predicted else 0.0
    recall = total_matched / total_true if total_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    knot_mae = matched_error_sum / total_matched if total_matched else float("nan")
    bspline_fit_mean = sum(bspline_fit_losses) / len(bspline_fit_losses)
    bspline_rms_values = torch.tensor(bspline_fit_losses).sqrt()
    bspline_rms_p95 = float(torch.quantile(bspline_rms_values, 0.95))
    bspline_rms_max = float(bspline_rms_values.max())
    bspline_coordinate_mean = sum(bspline_coordinate_losses) / len(
        bspline_coordinate_losses
    )
    parameter_rmse = math.sqrt(
        true_parameter_squared_error / max(true_parameter_values, 1)
    )
    endpoint_distances = (
        bspline_start_endpoint_distances + bspline_end_endpoint_distances
    )
    endpoint_rmse = math.sqrt(
        sum(value * value for value in endpoint_distances)
        / max(len(endpoint_distances), 1)
    )
    endpoint_max = max(endpoint_distances, default=0.0)
    hard_histogram = _histogram(retained, candidate_count)
    true_histogram = _histogram(true_count_tensor, candidate_count)

    if count_conditioned:
        predicted_count_tensor = torch.cat(predicted_counts).to(torch.long)
        mode_count_tensor = torch.cat(mode_counts).to(torch.long)
        expected_count_tensor = torch.cat(expected_counts)
        count_entropy_tensor = torch.cat(count_entropies)
        count_max_probability_tensor = torch.cat(count_max_probabilities)
        count_accuracy = float((predicted_count_tensor == true_count_tensor).float().mean())
        count_mae = float(
            (predicted_count_tensor - true_count_tensor).abs().float().mean()
        )
        expected_count_mean = float(expected_count_tensor.mean())
        network_histogram = _histogram(predicted_count_tensor, candidate_count)
        mode_histogram = _histogram(mode_count_tensor, candidate_count)
        mode_count_accuracy = float(
            (mode_count_tensor == true_count_tensor).float().mean()
        )
        mode_count_mae = float(
            (mode_count_tensor - true_count_tensor).abs().float().mean()
        )
        count_entropy_mean = float(count_entropy_tensor.mean())
        count_max_probability_mean = float(count_max_probability_tensor.mean())
        deployment_count_accuracy = float(
            (retained == true_count_tensor).float().mean()
        )
        deployment_count_mae = float(
            (retained - true_count_tensor).abs().float().mean()
        )
        activity_report = None
        threshold_report = None
    elif candidate_pruning:
        all_keep_probability = torch.cat(keep_probability_values)
        all_keep_ranges = torch.cat(keep_probability_ranges)
        network_keep_count = (
            all_keep_probability >= threshold
        ).sum(dim=-1).to(torch.long)
        predicted_count_tensor = retained
        count_accuracy = float((retained == true_count_tensor).float().mean())
        count_mae = float(
            (retained - true_count_tensor).abs().float().mean()
        )
        expected_count_mean = float(all_keep_probability.sum(dim=-1).mean())
        network_histogram = _histogram(network_keep_count, candidate_count)
        mode_histogram = None
        mode_count_accuracy = None
        mode_count_mae = None
        count_entropy_mean = None
        count_max_probability_mean = None
        deployment_count_accuracy = count_accuracy
        deployment_count_mae = count_mae
        activity_report = {
            "role": "diagnostic_only_not_used_for_deployment",
            "threshold": threshold,
            "minimum": float(all_keep_probability.min()),
            "maximum": float(all_keep_probability.max()),
            "mean_within_curve_range": float(all_keep_ranges.mean()),
            "mean_probability_mass": expected_count_mean,
            "thresholded_count_histogram": network_histogram,
        }
        threshold_report = None
    else:
        all_activity = torch.cat(activity_values)
        all_ranges = torch.cat(activity_ranges)
        predicted_count_tensor = retained
        count_accuracy = float((retained == true_count_tensor).float().mean())
        count_mae = float((retained - true_count_tensor).abs().float().mean())
        expected_count_mean = float(all_activity.sum(dim=-1).mean())
        network_histogram = hard_histogram
        mode_histogram = None
        mode_count_accuracy = None
        mode_count_mae = None
        count_entropy_mean = None
        count_max_probability_mean = None
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

    candidate_proposal_metrics = {
        f"{tolerance:.3f}": {
            "recall": (
                candidate_proposal_matched[tolerance]
                / candidate_proposal_true[tolerance]
                if candidate_proposal_true[tolerance]
                else 0.0
            ),
            "matched_mae": (
                candidate_proposal_error_sum[tolerance]
                / candidate_proposal_matched[tolerance]
                if candidate_proposal_matched[tolerance]
                else None
            ),
            "matched_count": candidate_proposal_matched[tolerance],
            "true_count": candidate_proposal_true[tolerance],
        }
        for tolerance in candidate_recall_tolerances
    }
    candidate_proposal_recall = candidate_proposal_metrics["0.020"]["recall"]
    candidate_proposal_mae = candidate_proposal_metrics["0.020"]["matched_mae"]
    pruning_report = (
        {
            "method": "greedy_single_deletion_standard_bspline_refit",
            "fit_tolerance_normalized_rms": fit_tolerance,
            "threshold_satisfied_fraction": (
                sum(pruning_threshold_satisfied)
                / max(len(pruning_threshold_satisfied), 1)
            ),
            "initial_candidate_fit_rms_mean": (
                sum(pruning_initial_rms) / max(len(pruning_initial_rms), 1)
            ),
            "accepted_deletions_mean": (
                sum(pruning_accepted_deletions)
                / max(len(pruning_accepted_deletions), 1)
            ),
            "first_rejected_deletion_rms_mean": (
                sum(pruning_first_rejected_rms)
                / len(pruning_first_rejected_rms)
                if pruning_first_rejected_rms
                else None
            ),
            "learned_keep_probability_role": "diagnostic_only",
        }
        if candidate_pruning
        else None
    )

    report = {
        "schema_version": 10,
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
        "count_selection": (
            count_selection
            if count_conditioned
            else "hard_rms_pruning"
            if candidate_pruning
            else "threshold"
        ),
        "count_prior_weight": args.count_prior_weight if count_conditioned else None,
        "predicted_knot_count_mean": float(retained.float().mean()),
        "expected_knot_count_mean": expected_count_mean,
        "predicted_knot_count_histogram": hard_histogram,
        "network_predicted_knot_count_histogram": network_histogram,
        "network_mode_knot_count_histogram": mode_histogram,
        "true_knot_count_mean": float(true_count_tensor.float().mean()),
        "true_knot_count_histogram": true_histogram,
        "knot_count_accuracy": count_accuracy,
        "knot_count_mae": count_mae,
        "mode_knot_count_accuracy": mode_count_accuracy,
        "mode_knot_count_mae": mode_count_mae,
        "count_distribution_entropy_mean": count_entropy_mean,
        "count_distribution_max_probability_mean": count_max_probability_mean,
        "deployment_knot_count_accuracy": deployment_count_accuracy,
        "deployment_knot_count_mae": deployment_count_mae,
        "count_confusion_matrix": count_confusion.tolist()
        if count_conditioned
        else None,
        "activity_diagnostics": activity_report,
        "threshold_sweep": threshold_report,
        "minimal_knot_pruning": pruning_report,
        "candidate_proposal_definition": (
            "refined output['internal_knots'] before hard deletion"
            if candidate_pruning
            else None
        ),
        "candidate_proposal_metrics": (
            candidate_proposal_metrics if candidate_pruning else None
        ),
        "candidate_proposal_match_tolerance": 0.02 if candidate_pruning else None,
        "candidate_proposal_recall": (
            candidate_proposal_recall if candidate_pruning else None
        ),
        "candidate_proposal_matched_mae": (
            candidate_proposal_mae if candidate_pruning else None
        ),
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
        "standard_bspline_refit_rms_euclidean_p95": bspline_rms_p95,
        "standard_bspline_refit_rms_euclidean_max": bspline_rms_max,
        "standard_bspline_coordinate_rmse": math.sqrt(bspline_coordinate_mean),
        "standard_bspline_endpoint_rmse": endpoint_rmse,
        "standard_bspline_endpoint_max_distance": endpoint_max,
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
    if (
        checkpoint.get("objective_version") != CURRENT_OBJECTIVE_VERSION
        and not candidate_pruning
    ):
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
        print(
            "  count distribution / legal range: "
            f"{model_config.get('structure_count_mode', model_config.get('count_head_mode', 'historical'))} / "
            f"{model_config.get('min_internal_knots', 0)}..{candidate_count}"
        )
        print(f"  network count accuracy: {count_accuracy:.3f}")
        print(f"  network count MAE: {count_mae:.3f}")
        print(f"  expected count mean: {expected_count_mean:.3f}")
        print(f"  network count histogram: {network_histogram}")
        print("  network decision rule: posterior median (minimum absolute error)")
        print(f"  categorical/hazard mode accuracy: {mode_count_accuracy:.3f}")
        print(f"  categorical/hazard mode MAE: {mode_count_mae:.3f}")
        print(f"  categorical/hazard mode histogram: {mode_histogram}")
        print(f"  mean posterior entropy: {count_entropy_mean:.3f}")
        print(f"  mean maximum class probability: {count_max_probability_mean:.3f}")
        print(f"  deployment selection: {count_selection}")
        print(f"  deployment count accuracy: {deployment_count_accuracy:.3f}")
        print(f"  deployment count MAE: {deployment_count_mae:.3f}")
        print(f"  deployment count histogram: {hard_histogram}")
    elif candidate_pruning:
        print("\nHigh-recall candidates and learned pruning diagnostics")
        print("  proposal set: refined candidates before hard deletion")
        for tolerance in candidate_recall_tolerances:
            metric = candidate_proposal_metrics[f"{tolerance:.3f}"]
            mae_text = (
                f"{metric['matched_mae']:.6f}"
                if metric["matched_mae"] is not None
                else "n/a"
            )
            print(
                f"  candidate recall@{tolerance:.3f}: "
                f"{metric['recall']:.3f}, matched MAE={mae_text}"
            )
        print(
            "  learned keep probability mass mean (diagnostic only): "
            f"{expected_count_mean:.3f}"
        )
        print(f"  thresholded keep diagnostic histogram: {network_histogram}")
        print("\nHard minimum-complexity deployment")
        print(f"  normalized RMS threshold: {fit_tolerance:.9e}")
        print(
            "  threshold-satisfied fraction: "
            f"{pruning_report['threshold_satisfied_fraction']:.3f}"
        )
        print(
            "  candidate initial RMS mean: "
            f"{pruning_report['initial_candidate_fit_rms_mean']:.9e}"
        )
        print(
            "  accepted deletions mean: "
            f"{pruning_report['accepted_deletions_mean']:.3f}"
        )
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
    print(f"  refit RMS distance P95/max: {bspline_rms_p95:.9e}/{bspline_rms_max:.9e}")
    print(f"  endpoint RMS distance: {endpoint_rmse:.9e}")
    print(f"  endpoint max distance: {endpoint_max:.9e}")

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
