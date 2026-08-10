from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one fitted B-spline sample.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--activity-threshold", type=float, default=None)
    parser.add_argument("--smoothness-weight", type=float, default=1e-6)
    parser.add_argument("--control-ridge", type=float, default=0.0)
    parser.add_argument(
        "--count-selection", choices=("auto", "network", "bic"), default="auto"
    )
    parser.add_argument("--count-prior-weight", type=float, default=1.0)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    threshold = (
        args.activity_threshold
        if args.activity_threshold is not None
        else float(checkpoint.get("activity_threshold", 0.5))
    )
    model, model_config, legacy_checkpoint = build_model_from_checkpoint(checkpoint)
    model.set_activity_threshold(threshold)
    model.eval()
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
        size=args.sample_index + 1,
        seed=10000,
        **dataset_config,
    )
    sample = dataset[args.sample_index]
    points = sample["points"].unsqueeze(0)
    with torch.no_grad():
        output = model(points)
        if count_conditioned and count_selection == "bic":
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

    loss_config, _ = migrate_loss_config(checkpoint, legacy=legacy_checkpoint)
    loss_fn = SplineFittingLoss(
        LossWeights(**loss_config["weights"]),
        min_knot_gap=loss_config.get("min_knot_gap", 1e-3),
        knot_position_beta=loss_config.get("knot_position_beta", 0.02),
    )
    losses = loss_fn(
        output,
        points,
        chord_params=sample["chord_params"].unsqueeze(0),
        true_params=sample["true_params"].unsqueeze(0),
        true_internal_knots=sample["true_internal_knots"].unsqueeze(0),
        true_internal_knot_mask=sample["true_internal_knot_mask"].unsqueeze(0),
        activity_threshold=threshold,
    )

    observed = sample["points"].numpy()
    forward_curve = output["reconstructed_points"][0].detach().numpy()
    dense_params = torch.linspace(0.0, 1.0, 400, dtype=points.dtype)
    dense_curve = deployed.spline.evaluate(dense_params).numpy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_curve, ax_structure = axes
    ax_curve.scatter(observed[:, 0], observed[:, 1], s=13, label="samples")
    ax_curve.plot(
        forward_curve[:, 0], forward_curve[:, 1], "--", label="network surrogate"
    )
    ax_curve.plot(dense_curve[:, 0], dense_curve[:, 1], label="deployed B-spline")
    controls = deployed.control_points.numpy()
    ax_curve.plot(
        controls[:, 0], controls[:, 1], "o-", alpha=0.45, label="control polygon"
    )
    ax_curve.set_aspect("equal", adjustable="box")
    ax_curve.set_title(
        f"Internal knots: {deployed.retained_count} | "
        f"RMS: {float(deployed.fit_rmse):.4e}"
    )
    ax_curve.legend()

    if count_conditioned:
        probabilities = output["count_probabilities"][0].detach().numpy()
        counts = list(range(len(probabilities)))
        predicted_count = int(output["predicted_knot_count"][0])
        deployed_count = deployed.retained_count
        colors = [
            "tab:orange"
            if count == deployed_count
            else "tab:green"
            if count == predicted_count
            else "tab:blue"
            for count in counts
        ]
        ax_structure.bar(counts, probabilities, color=colors)
        ax_structure.set_xlabel("internal-knot count")
        ax_structure.set_ylabel("count probability")
        ax_structure.set_title(
            f"CountHead K={predicted_count} | deployed K={deployed_count} "
            f"({count_selection})"
        )
        ax_structure.set_xticks(counts)
    else:
        activity = output["activity"][0].detach().numpy()
        knots = output["internal_knots"][0].detach().numpy()
        kept = deployed.retained_mask.numpy()
        colors = ["tab:orange" if value else "tab:blue" for value in kept]
        ax_structure.bar(range(len(activity)), activity, color=colors)
        ax_structure.axhline(threshold, color="black", linestyle="--")
        ax_structure.set_xticks(
            range(len(knots)), [f"{value:.3f}" for value in knots], rotation=45
        )
        ax_structure.set_xlabel("candidate knot")
        ax_structure.set_ylabel("existence probability")
        ax_structure.set_title("Historical threshold-gated structure")

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)

    true_knots = sample["true_internal_knots"][sample["true_internal_knot_mask"]]
    matching = match_internal_knots(
        deployed.retained_internal_knots,
        true_knots,
        tolerance=0.05,
    )
    print("Spline structure report")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  structure mode: {structure_mode}")
    print(f"  checkpoint best epoch: {checkpoint.get('epoch', 'not recorded')}")
    if count_conditioned:
        print(f"  predicted knot count: {int(output['predicted_knot_count'][0])}")
        print(f"  expected knot count: {float(output['expected_knot_count'][0]):.6f}")
        print(f"  count probabilities: {output['count_probabilities'][0].tolist()}")
        print(f"  deployment count selection: {count_selection}")
    else:
        print(f"  activity threshold: {threshold:.4f}")
    print(f"  retained internal knots: {deployed.retained_count}")
    print(
        "  retained knot values: "
        + ", ".join(
            f"{value:.6f}" for value in deployed.retained_internal_knots.tolist()
        )
    )
    print(f"  true internal knots: {true_knots.tolist()}")
    print(
        f"  match@0.05: precision={matching.precision:.3f}, "
        f"recall={matching.recall:.3f}, F1={matching.f1:.3f}"
    )
    print(f"  total objective: {float(losses['loss']):.9e}")
    print(f"  network fit loss: {float(losses['fit_loss']):.9e}")
    print(f"  standard B-spline refit loss: {float(deployed.fit_mse):.9e}")
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
