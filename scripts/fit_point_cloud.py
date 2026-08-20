from __future__ import annotations

import argparse
import json
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
    load_ordered_point_cloud,
    normalize_ordered_point_cloud,
    resample_ordered_point_cloud,
)
from spline_fitting.evaluation.bspline_inference import (
    refit_model_output_as_bsplines,
    select_count_conditioned_output_by_bic,
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
        axis.plot(*dense_curve.T, label="fitted B-spline")
        axis.plot(*control_points.T, "o-", alpha=0.5, label="control polygon")
    else:
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.scatter(source_points[:, 0], source_points[:, 1], s=14, label="input points")
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
        "--count-selection", choices=("auto", "network", "bic"), default="auto"
    )
    parser.add_argument("--count-prior-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.smoothness_weight < 0.0 or args.control_ridge < 0.0:
        parser.error("refit regularization weights must be non-negative")
    if args.num_points is not None and args.num_points < 4:
        parser.error("--num-points must be at least four")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
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
    model_points = resample_ordered_point_cloud(source_points, model_point_count)
    normalized = normalize_ordered_point_cloud(model_points)
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
        deployed = refit_model_output_as_bsplines(
            deployment_output,
            points,
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

    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "point_cloud": str(args.point_cloud),
        "point_count": int(source_points.shape[0]),
        "model_point_count": int(model_points.shape[0]),
        "point_dim": int(source_points.shape[1]),
        "reversed": bool(args.reverse_points),
        "structure_mode": structure_mode,
        "count_selection": count_selection,
        "degree": int(model.degree),
        "predicted_internal_knot_count": int(deployed.retained_count),
        "predicted_internal_knots": deployed.retained_internal_knots.cpu().tolist(),
        "open_knot_vector": deployed.spline.knot_vector.cpu().tolist(),
        "predicted_parameters": output["params"][0].detach().cpu().tolist(),
        "control_points": control_points.tolist(),
        "fitted_curve": dense_curve.tolist(),
        "normalization_center": center.tolist(),
        "normalization_scale": float(scale),
        "normalized_fit_rmse": float(deployed.fit_rmse.cpu()),
        "original_scale_fit_rmse": fit_rmse_original,
    }
    if "count_probabilities" in output:
        report["count_probabilities"] = (
            output["count_probabilities"][0].detach().cpu().tolist()
        )
    if "activity" in output:
        report["activity"] = output["activity"][0].detach().cpu().tolist()

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
    print(f"  knot values: {deployed.retained_internal_knots.cpu().tolist()}")
    print(f"  normalized RMS: {float(deployed.fit_rmse.cpu()):.9e}")
    print(f"  original-scale RMS: {fit_rmse_original:.9e}")
    if args.json_output is not None:
        print(f"  saved JSON: {args.json_output}")
    if args.figure_output is not None:
        print(f"  saved figure: {args.figure_output}")


if __name__ == "__main__":
    main()
