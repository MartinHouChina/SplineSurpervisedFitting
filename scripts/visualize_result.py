from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.synthetic import (
    SyntheticCubicBSplineDataset,
    evaluate_bspline_curve,
)
from spline_fitting.checkpointing import (
    build_model_from_checkpoint,
    migrate_loss_config,
)
from spline_fitting.evaluation.bspline_inference import (
    refit_model_output_as_bsplines,
)
from spline_fitting.evaluation.knot_diagnostics import (
    activity_statistics,
    knot_contribution_rms,
    match_internal_knots,
    point_fit_statistics,
)
from spline_fitting.losses.total_loss import LossWeights, SplineFittingLoss
from spline_fitting.spline.curve_evaluation import sample_curve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize one fitted cubic B-spline sample."
    )
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "result.png")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=None,
        help="Hard knot threshold; defaults to the checkpoint value or 0.5.",
    )
    parser.add_argument("--smoothness-weight", type=float, default=1e-6)
    parser.add_argument("--control-ridge", type=float, default=0.0)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    activity_threshold = (
        args.activity_threshold
        if args.activity_threshold is not None
        else float(checkpoint.get("activity_threshold", 0.5))
    )
    model, model_config, legacy_checkpoint = build_model_from_checkpoint(checkpoint)
    model.set_activity_threshold(activity_threshold)
    model.eval()

    dataset_config = dict(checkpoint.get("dataset_config", {}))
    dataset_config.setdefault("num_points", 64)
    dataset_config.setdefault("point_dim", model_config.get("point_dim", 2))
    dataset_config["return_ground_truth"] = True
    sample = SyntheticCubicBSplineDataset(
        size=max(args.sample_index + 1, 1),
        seed=2026,
        **dataset_config,
    )[args.sample_index]
    points = sample["points"].unsqueeze(0)

    loss_config, assumed_loss_config = migrate_loss_config(
        checkpoint,
        legacy=legacy_checkpoint,
    )
    if assumed_loss_config:
        print("Warning: checkpoint has no loss_config; using compatible defaults.")
    loss_fn = SplineFittingLoss(
        LossWeights(**loss_config["weights"]),
        min_knot_gap=loss_config.get("min_knot_gap", 1e-3),
        knot_position_beta=loss_config.get("knot_position_beta", 0.02),
    )

    with torch.no_grad():
        output = model(points)
        losses = loss_fn(
            output,
            points,
            chord_params=sample["chord_params"].unsqueeze(0),
            true_params=(
                sample["true_params"].unsqueeze(0) if "true_params" in sample else None
            ),
            true_internal_knots=(
                sample["true_internal_knots"].unsqueeze(0)
                if "true_internal_knots" in sample
                else None
            ),
            true_internal_knot_mask=(
                sample["true_internal_knot_mask"].unsqueeze(0)
                if "true_internal_knot_mask" in sample
                else None
            ),
        )
        forward_fit = point_fit_statistics(output["reconstructed_points"], points)
        activity_summary = activity_statistics(output["activity"], activity_threshold)
        if legacy_checkpoint:
            deployment_output = dict(output)
            deployment_output["activity_gate"] = (
                output["activity"] >= activity_threshold
            ).to(output["activity"].dtype)
        else:
            deployment_output = output
        deployed_fit = refit_model_output_as_bsplines(
            deployment_output,
            points,
            degree=model.degree,
            smoothness_weight=args.smoothness_weight,
            control_ridge=args.control_ridge,
        )[0]
        contribution = knot_contribution_rms(
            output["weighted_increment_basis"],
            output["increment_coefficients"],
        )[0]
        _, dense_curve = sample_curve(
            internal_knots=output["internal_knots"],
            activity=output["activity_gate"],
            coefficients=output["coefficients"],
            degree=model.degree,
            num_samples=300,
            eps=0.0,
            gate_transform="direct",
        )
        dense_parameters = torch.linspace(
            0.0,
            1.0,
            300,
            dtype=points.dtype,
            device=points.device,
        )
        dense_bspline_curve = deployed_fit.spline.evaluate(dense_parameters)

    p = points[0].numpy()
    c = dense_curve[0].numpy()
    activity = output["activity"][0].numpy()
    knots = output["internal_knots"][0].numpy()
    actual_gate = deployment_output["activity_gate"][0].numpy()
    pilot_delta = output["pilot_drop_objective_delta"][0].numpy()
    normalized_importance = output["normalized_knot_importance"][0].numpy()
    retained_mask = deployed_fit.retained_mask.numpy()
    bspline_curve = dense_bspline_curve.numpy()

    true_curve = None
    true_internal_knots = None
    true_internal_knots_tensor = None
    if "true_control_points" in sample:
        control = sample["true_control_points"][sample["true_control_mask"]]
        knot_vector = sample["true_knot_vector"][sample["true_knot_mask"]]
        dense_t = torch.linspace(0.0, 1.0, 300)
        true_curve = evaluate_bspline_curve(
            dense_t, control, knot_vector, degree=3
        ).numpy()
        true_internal_knots_tensor = sample["true_internal_knots"][
            sample["true_internal_knot_mask"]
        ]
        true_internal_knots = true_internal_knots_tensor.numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 4.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(p[:, 0], p[:, 1], s=14, label="samples")
    if true_curve is not None:
        ax1.plot(
            true_curve[:, 0],
            true_curve[:, 1],
            linestyle="--",
            linewidth=1.5,
            label="ground-truth cubic B-spline",
        )
    ax1.plot(c[:, 0], c[:, 1], linewidth=2, label="network forward fit")
    ax1.plot(
        bspline_curve[:, 0],
        bspline_curve[:, 1],
        linestyle=":",
        linewidth=2,
        label=f"standard B-spline refit (K={deployed_fit.retained_count})",
    )
    ax1.axis("equal")
    ax1.legend()
    ax1.set_title(
        "Cubic spline fitting\n"
        f"network fit={forward_fit['fit_mse'][0].item():.2e}, "
        f"B-spline refit={deployed_fit.fit_mse.item():.2e}"
    )

    ax2 = fig.add_subplot(1, 2, 2)
    colors = ["tab:blue" if keep else "lightgray" for keep in retained_mask]
    ax2.bar(range(len(activity)), activity, color=colors)
    ax2.axhline(
        activity_threshold,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label=f"threshold={activity_threshold:.2f}",
    )
    ax2.set_ylim(0.0, 1.0)
    ax2.scatter(
        range(len(actual_gate)),
        actual_gate,
        marker="_",
        s=90,
        color="black",
        label="deterministic gate",
        zorder=3,
    )
    ax2.set_xticks(range(len(activity)), [f"{u:.4f}" for u in knots], rotation=45)
    ax2.set_xlabel("predicted candidate knot value")
    ax2.set_ylabel("keep probability / gate")
    title = (
        f"Candidate knots: {len(activity)} -> {deployed_fit.retained_count} retained"
        f"\nE[K]: {activity_summary['activity_mass'][0].item():.3f}"
    )
    if true_internal_knots is not None:
        title += "\ntrue internal knots: " + ", ".join(
            f"{u:.2f}" for u in true_internal_knots
        )
    ax2.set_title(title)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(args.output, dpi=180)

    print("\nKnot reduction report")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  checkpoint best epoch: {checkpoint.get('epoch', 'not recorded')}")
    print(f"  gate mode: {model_config['gate_mode']}")
    print(
        "  local cross-attention KnotHead: "
        f"{model_config.get('knot_use_local_cross_attention', False)}"
    )
    if model_config.get("activity_use_pilot_importance", False):
        print("  historical pilot compatibility path: enabled")
    if legacy_checkpoint:
        print("  legacy checkpoint: threshold converted to a deployment hard mask")
    print(f"  dataset noise_std: {dataset_config.get('noise_std', 'not recorded')}")
    print(f"  candidate internal knots: {len(activity)}")
    print(f"  hard threshold: {activity_threshold:.4f}")
    print(f"  retained internal knots: {deployed_fit.retained_count}")
    print(
        f"  expected active count E[K]: "
        f"{activity_summary['activity_mass'][0].item():.6f}"
    )
    print(
        "  retained knot values: "
        + (
            ", ".join(
                f"{value:.6f}"
                for value in deployed_fit.retained_internal_knots.tolist()
            )
            if deployed_fit.retained_count
            else "[]"
        )
    )
    print(
        "  open-clamped knot vector after pruning: "
        + ", ".join(f"{value:.6f}" for value in deployed_fit.knot_vector.tolist())
    )
    print(f"  standard B-spline control points: {deployed_fit.control_points.shape[0]}")
    print(f"  standard B-spline solver rank: {deployed_fit.spline.solver_rank}")
    print("  threshold sweep counts:")
    sweep = sorted(
        {0.10, 0.30, 0.40, 0.50, 0.60, 0.70, 0.90, float(activity_threshold)}
    )
    for threshold in sweep:
        count = int((output["activity"][0] >= threshold).sum().item())
        print(f"    activity >= {threshold:.2f}: {count}")

    print("\nLoss report for this sample")
    print(f"  total objective: {losses['loss'].item():.9e}")
    print(
        f"  network fit loss (mean squared Euclidean): {forward_fit['fit_mse'][0].item():.9e}"
    )
    print(f"  network RMS Euclidean distance: {forward_fit['fit_rmse'][0].item():.9e}")
    print(f"  network coordinate RMSE: {forward_fit['coordinate_rmse'][0].item():.9e}")
    print(f"  standard B-spline refit loss: {deployed_fit.fit_mse.item():.9e}")
    print(
        f"  standard B-spline RMS Euclidean distance: {deployed_fit.fit_rmse.item():.9e}"
    )
    print(
        "  standard B-spline coordinate RMSE: "
        f"{deployed_fit.spline.coordinate_rmse.item():.9e}"
    )
    print(
        f"  B-spline augmented objective: {deployed_fit.spline.augmented_objective.item():.9e}"
    )
    print(
        f"  B-spline data squared error: {deployed_fit.spline.data_squared_error.item():.9e}"
    )
    print(
        f"  B-spline D2 squared norm: {deployed_fit.spline.smoothness_squared.item():.9e}"
    )
    print("  raw loss components:")
    component_names = [
        "fit_loss",
        "l0_loss",
        "activity_loss",
        "binary_loss",
        "gap_loss",
        "parameter_prior_loss",
        "true_parameter_loss",
    ]
    if "orthogonal_loss" in losses:
        component_names.insert(4, "orthogonal_loss")
    for name in component_names:
        print(f"    {name}: {losses[name].item():.9e}")
    weighted_components = {
        "fit": loss_fn.weights.fit * losses["fit_loss"],
        "l0": loss_fn.weights.l0 * losses["l0_loss"],
        "activity": loss_fn.weights.activity * losses["activity_loss"],
        "binary": loss_fn.weights.binary * losses["binary_loss"],
        "gap": loss_fn.weights.gap * losses["gap_loss"],
        "parameter_prior": (
            loss_fn.weights.parameter_prior * losses["parameter_prior_loss"]
        ),
        "true_parameter": (
            loss_fn.weights.true_parameter * losses["true_parameter_loss"]
        ),
    }
    if "orthogonal_loss" in losses:
        weighted_components["legacy_orthogonal"] = (
            loss_fn.weights.orthogonal * losses["orthogonal_loss"]
        )
    print("  weighted contributions to total objective:")
    for name, value in weighted_components.items():
        print(f"    {name}: {value.item():.9e}")

    print("\nPer-candidate diagnostics")
    uses_pilot = model_config.get("activity_use_pilot_importance", False)
    suffix = "       pilot_delta  importance_z" if uses_pilot else ""
    print("  index      knot    activity    keep    contribution_rms" + suffix)
    for index, (knot, value, keep, score, delta, importance) in enumerate(
        zip(
            knots,
            activity,
            retained_mask,
            contribution.tolist(),
            pilot_delta,
            normalized_importance,
        )
    ):
        line = (
            f"  {index:5d}  {knot:8.5f}  {value:10.7f}  "
            f"{str(bool(keep)):>5s}  {score:16.9e}"
        )
        if uses_pilot:
            line += f"  {delta:16.9e}  {importance:9.4f}"
        print(line)

    if true_internal_knots_tensor is not None:
        matching = match_internal_knots(
            deployed_fit.retained_internal_knots,
            true_internal_knots_tensor,
            tolerance=0.05,
        )
        print("\nGround-truth comparison (valid only under a shared parameterization)")
        print(
            "  true internal knots: "
            + ", ".join(f"{value:.6f}" for value in true_internal_knots_tensor.tolist())
        )
        print(
            f"  matched within 0.05: {matching.matched_count}/{matching.true_count} | "
            f"precision={matching.precision:.3f} | recall={matching.recall:.3f}"
        )
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
