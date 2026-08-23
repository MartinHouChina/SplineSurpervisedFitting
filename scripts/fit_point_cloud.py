from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import (
    COUNT_CONDITIONED_V5_OBJECTIVE_VERSION,
    build_model_from_checkpoint,
)
from spline_fitting.data.point_cloud_io import (
    interpolate_parameters_by_chord,
    load_ordered_point_cloud,
    normalize_ordered_point_cloud,
    resample_ordered_point_cloud,
)
from spline_fitting.evaluation.bspline_inference import (
    HardGatedBSplineFit,
    refit_model_output_as_bsplines,
    select_count_conditioned_output_by_bic,
)
from spline_fitting.evaluation.minimal_knot_pruning import (
    MinimalKnotPruningResult,
    prune_knots_to_rms_tolerance,
)


def resolve_fit_tolerance(
    checkpoint: dict[str, object], explicit_tolerance: float | None
) -> float:
    """Resolve the normalized geometric RMS limit used by v7 deployment."""
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


def pruning_result_as_deployed_fit(
    result: MinimalKnotPruningResult,
    *,
    sample_index: int = 0,
) -> HardGatedBSplineFit:
    """Adapt an auditable hard-pruning result to the legacy fit interface."""
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


def _plot_result(
    source_points: torch.Tensor,
    dense_curve: torch.Tensor,
    control_points: torch.Tensor,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_points.shape[1] == 3:
        figure = plt.figure(figsize=(8, 6))
        axis = figure.add_subplot(111, projection="3d")
        axis.scatter(*source_points.T, s=14, label="input points")
        axis.scatter(
            *source_points[[0, -1]].T,
            s=55,
            marker="x",
            label="input endpoints",
        )
        axis.plot(*dense_curve.T, label="fitted B-spline")
        axis.plot(*control_points.T, "o-", alpha=0.5, label="control polygon")
    else:
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.scatter(source_points[:, 0], source_points[:, 1], s=14, label="input points")
        axis.scatter(
            source_points[[0, -1], 0],
            source_points[[0, -1], 1],
            s=55,
            marker="x",
            label="input endpoints",
        )
        axis.plot(dense_curve[:, 0], dense_curve[:, 1], label="fitted B-spline")
        axis.plot(
            control_points[:, 0],
            control_points[:, 1],
            "o-",
            alpha=0.5,
            label="control polygon",
        )
        axis.set_aspect("equal", adjustable="box")
    axis.legend()
    axis.set_title("Predicted variable-knot B-spline")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit one user-provided ordered point cloud with a checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--point-cloud", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--figure-output", type=Path, default=None)
    parser.add_argument("--reverse-points", action="store_true")
    parser.add_argument(
        "--num-points",
        type=int,
        default=None,
        help="Model input length; defaults to the checkpoint training length.",
    )
    parser.add_argument("--activity-threshold", type=float, default=None)
    parser.add_argument("--smoothness-weight", type=float, default=1e-6)
    parser.add_argument("--control-ridge", type=float, default=0.0)
    parser.add_argument(
        "--fit-tolerance",
        type=float,
        default=None,
        help=(
            "Normalized RMS limit for candidate_pruning. Defaults to "
            "deployment_config.error_tolerance, then the dataset canonical "
            "knot tolerance stored in the checkpoint."
        ),
    )
    parser.add_argument(
        "--count-selection", choices=("auto", "network", "bic"), default="auto"
    )
    parser.add_argument("--count-prior-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.smoothness_weight < 0.0 or args.control_ridge < 0.0:
        parser.error("refit regularization weights must be non-negative")
    if args.num_points is not None and args.num_points < 4:
        parser.error("--num-points must be at least four")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    try:
        fit_tolerance = resolve_fit_tolerance(checkpoint, args.fit_tolerance)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    model, model_config, legacy_checkpoint = build_model_from_checkpoint(checkpoint)
    point_dim = int(model_config.get("point_dim", 2))
    source_points = load_ordered_point_cloud(args.point_cloud, point_dim=point_dim)
    if args.reverse_points:
        source_points = source_points.flip(0)
    model_point_count = (
        args.num_points
        if args.num_points is not None
        else int(checkpoint.get("dataset_config", {}).get("num_points", 64))
    )
    minimum_recommended_points = max(16, model_point_count // 2)
    sparse_input_warning = source_points.shape[0] < minimum_recommended_points
    if sparse_input_warning:
        print(
            "WARNING: the input contains only "
            f"{source_points.shape[0]} points, while this checkpoint was trained "
            f"with {model_point_count}. Linear resampling does not create new "
            "geometry; knot-count predictions may not generalize.",
            flush=True,
        )
    model_points = resample_ordered_point_cloud(source_points, model_point_count)
    normalized = normalize_ordered_point_cloud(model_points)
    source_chord = normalize_ordered_point_cloud(source_points)["chord_params"]
    source_normalized_points = (
        source_points - normalized["center"]
    ) / normalized["scale"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = normalized["points"].unsqueeze(0).to(device)
    model.to(device).eval()

    threshold = (
        args.activity_threshold
        if args.activity_threshold is not None
        else float(checkpoint.get("activity_threshold", 0.5))
    )
    model.set_activity_threshold(threshold)
    structure_mode = model_config.get("structure_mode", "hard_concrete")
    count_selection = args.count_selection
    if count_selection == "auto":
        count_selection = (
            "bic"
            if checkpoint.get("objective_version")
            == COUNT_CONDITIONED_V5_OBJECTIVE_VERSION
            else "network"
        )
    if structure_mode == "interactive_dynamic" and count_selection == "bic":
        parser.error("v6 performs one network count decision and has no BIC branches")

    with torch.no_grad():
        output = model(points)
        if structure_mode == "count_conditioned" and count_selection == "bic":
            deployment_output, _ = select_count_conditioned_output_by_bic(
                output,
                points,
                degree=model.degree,
                smoothness_weight=args.smoothness_weight,
                prior_weight=args.count_prior_weight,
            )
        elif legacy_checkpoint:
            deployment_output = dict(output)
            deployment_output["activity_gate"] = (
                output["activity"] >= threshold
            ).to(output["activity"].dtype)
        else:
            deployment_output = output
        source_parameters = interpolate_parameters_by_chord(
            normalized["chord_params"].to(device),
            output["params"][0],
            source_chord.to(device),
        )
        full_resolution_points = source_normalized_points.unsqueeze(0).to(device)
        pruning_result: MinimalKnotPruningResult | None = None
        if structure_mode == "candidate_pruning":
            # The learned keep probability is diagnostic only.  Actual
            # deployment starts from every proposed knot, refits on the
            # original-resolution point cloud, and verifies every deletion.
            pruning_result = prune_knots_to_rms_tolerance(
                source_parameters,
                full_resolution_points[0],
                output["internal_knots"][0],
                error_tolerance=fit_tolerance,
                degree=model.degree,
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
                interpolate_endpoints=True,
            )
            deployed = pruning_result_as_deployed_fit(pruning_result)
        else:
            full_resolution_output = dict(deployment_output)
            full_resolution_output["params"] = source_parameters.unsqueeze(0)
            deployed = refit_model_output_as_bsplines(
                full_resolution_output,
                full_resolution_points,
                degree=model.degree,
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
            )[0]

    scale = normalized["scale"].cpu()
    center = normalized["center"].cpu()
    dense_params = torch.linspace(0.0, 1.0, 400, dtype=points.dtype, device=device)
    dense_normalized = deployed.spline.evaluate(dense_params).cpu()
    controls_normalized = deployed.control_points.cpu()
    dense_curve = dense_normalized * scale + center
    control_points = controls_normalized * scale + center
    fit_rmse_original = float(deployed.fit_rmse.cpu() * scale)
    underdetermined_warning = deployed.control_points.shape[0] > source_points.shape[0]
    if underdetermined_warning:
        print(
            "WARNING: predicted control-point count exceeds the number of original "
            "observations; the refit is data-underdetermined and should not be "
            "treated as a reliable reconstruction.",
            flush=True,
        )
    normalized_start_distance = float(
        (
            deployed.reconstructed_points[0] - full_resolution_points[0, 0]
        ).norm().cpu()
    )
    normalized_end_distance = float(
        (
            deployed.reconstructed_points[-1] - full_resolution_points[0, -1]
        ).norm().cpu()
    )

    report = {
        "schema_version": 3,
        "checkpoint": str(args.checkpoint),
        "point_cloud": str(args.point_cloud),
        "point_count": int(source_points.shape[0]),
        "model_point_count": int(model_points.shape[0]),
        "point_dim": int(source_points.shape[1]),
        "reversed": bool(args.reverse_points),
        "structure_mode": structure_mode,
        "count_selection": count_selection,
        "deployment_method": (
            "hard_standard_bspline_rms_pruning"
            if structure_mode == "candidate_pruning"
            else "legacy_structure_selection"
        ),
        "degree": int(model.degree),
        "predicted_internal_knot_count": int(deployed.retained_count),
        "predicted_internal_knots": deployed.retained_internal_knots.cpu().tolist(),
        "open_knot_vector": deployed.spline.knot_vector.cpu().tolist(),
        "predicted_parameters": output["params"][0].detach().cpu().tolist(),
        "model_predicted_parameters": output["params"][0].detach().cpu().tolist(),
        "full_resolution_refit_parameters": source_parameters.detach().cpu().tolist(),
        "full_resolution_refit_point_count": int(source_points.shape[0]),
        "sparse_input_warning": bool(sparse_input_warning),
        "data_underdetermined_warning": bool(underdetermined_warning),
        "control_points": control_points.tolist(),
        "fitted_curve": dense_curve.tolist(),
        "normalization_center": center.tolist(),
        "normalization_scale": float(scale),
        "normalized_fit_rmse": float(deployed.fit_rmse.cpu()),
        "original_scale_fit_rmse": fit_rmse_original,
        "normalized_start_endpoint_distance": normalized_start_distance,
        "normalized_end_endpoint_distance": normalized_end_distance,
        "original_scale_start_endpoint_distance": (
            normalized_start_distance * float(scale)
        ),
        "original_scale_end_endpoint_distance": (
            normalized_end_distance * float(scale)
        ),
    }
    if pruning_result is not None:
        report["fit_tolerance_normalized"] = fit_tolerance
        report["fit_tolerance_original_scale"] = fit_tolerance * float(scale)
        report["candidate_internal_knot_count"] = pruning_result.initial_count
        report["candidate_internal_knots"] = (
            pruning_result.initial_internal_knots.cpu().tolist()
        )
        report["candidate_fit_rmse_normalized"] = float(
            pruning_result.initial_fit.fit_rmse.cpu()
        )
        report["fit_tolerance_satisfied"] = pruning_result.threshold_satisfied
        report["accepted_deletion_count"] = len(pruning_result.accepted_steps)
        report["removed_knots_in_order"] = pruning_result.removed_knots.cpu().tolist()
        report["rms_trajectory_normalized"] = (
            pruning_result.rms_trajectory.cpu().tolist()
        )
        rejected_step = next(
            (step for step in pruning_result.steps if not step.accepted), None
        )
        report["first_rejected_deletion"] = (
            {
                "knot": float(rejected_step.removed_knot.cpu()),
                "candidate_rmse_normalized": float(
                    rejected_step.candidate_rmse.cpu()
                ),
            }
            if rejected_step is not None
            else None
        )
    if "count_probabilities" in output:
        report["count_probabilities"] = (
            output["count_probabilities"][0].detach().cpu().tolist()
        )
        report["posterior_mode_internal_knot_count"] = int(
            output.get(
                "count_mode_knot_count", output["predicted_knot_count"]
            )[0]
        )
    if "activity" in output:
        report["activity"] = output["activity"][0].detach().cpu().tolist()
    if "keep_probability" in output:
        report["keep_probability_diagnostic"] = (
            output["keep_probability"][0].detach().cpu().tolist()
        )
    if "analytic_drop_objective_delta" in output:
        report["analytic_drop_objective_delta_diagnostic"] = (
            output["analytic_drop_objective_delta"][0].detach().cpu().tolist()
        )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if args.figure_output is not None:
        _plot_result(source_points, dense_curve, control_points, args.figure_output)

    print("User point-cloud fitting")
    print(f"  point cloud: {args.point_cloud}")
    print(f"  points / dimension: {source_points.shape[0]} / {source_points.shape[1]}")
    print(f"  resampled model points: {model_points.shape[0]}")
    print(f"  predicted internal knots: {deployed.retained_count}")
    if pruning_result is not None:
        print(
            "  hard RMS pruning: "
            f"{pruning_result.initial_count} candidates -> "
            f"{pruning_result.final_count} retained"
        )
        print(f"  normalized fit threshold: {fit_tolerance:.9e}")
        print(
            "  threshold satisfied: "
            f"{pruning_result.threshold_satisfied}"
        )
        print(
            "  normalized RMS trajectory: "
            f"{pruning_result.rms_trajectory.cpu().tolist()}"
        )
    if "count_probabilities" in output:
        print(
            "  posterior mode internal knots (diagnostic): "
            f"{int(output.get('count_mode_knot_count', output['predicted_knot_count'])[0])}"
        )
    print(f"  knot values: {deployed.retained_internal_knots.cpu().tolist()}")
    print(f"  normalized RMS: {float(deployed.fit_rmse.cpu()):.9e}")
    print(f"  original-scale RMS: {fit_rmse_original:.9e}")
    print(
        "  endpoint distances (original scale): "
        f"start={normalized_start_distance * float(scale):.9e}, "
        f"end={normalized_end_distance * float(scale):.9e}"
    )
    if args.json_output is not None:
        print(f"  saved JSON: {args.json_output}")
    if args.figure_output is not None:
        print(f"  saved figure: {args.figure_output}")


if __name__ == "__main__":
    main()
