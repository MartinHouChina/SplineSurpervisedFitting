from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

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
    refit_bspline_control_points,
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


def resolve_fit_tolerance(
    checkpoint: dict[str, object], explicit_tolerance: float | None
) -> float:
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


def candidate_mode_flags(
    checkpoint: dict[str, object], model_config: dict[str, object]
) -> tuple[bool, bool]:
    structure = str(model_config.get("structure_mode", "")).lower()
    objective = str(checkpoint.get("objective_version", "")).lower()
    one_shot = (
        structure == "candidate_pruning_one_shot"
        or "candidate_pruning_one_shot" in objective
    )
    return one_shot or structure == "candidate_pruning", one_shot


def one_shot_selection(
    output: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, str]:
    probabilities = output["keep_probability"]
    if probabilities.shape != output["internal_knots"].shape:
        raise ValueError("keep_probability and internal_knots must share [B,K]")
    if "learned_keep_mask" in output:
        mask = output["learned_keep_mask"]
        if mask.shape != probabilities.shape:
            raise ValueError("learned_keep_mask must share shape [B,K]")
        if mask.dtype != torch.bool:
            if mask.is_floating_point() and not torch.isfinite(mask).all():
                raise ValueError("learned_keep_mask must contain only finite values")
            if not torch.all((mask == 0) | (mask == 1)):
                raise ValueError("learned_keep_mask must be Boolean or strictly 0/1")
        mask = mask.to(torch.bool)
        source = "learned_keep_mask"
    else:
        mask = probabilities >= 0.5
        source = "keep_probability>=0.5_fallback"
    adaptive = output.get("adaptive_keep_threshold")
    if adaptive is None:
        adaptive = output.get("adaptive_keep_logit_threshold")
    if adaptive is None:
        adaptive = probabilities.new_full((probabilities.shape[0],), float("nan"))
    else:
        adaptive = adaptive.reshape(probabilities.shape[0], -1)
        if adaptive.shape[1] != 1:
            raise ValueError("adaptive_keep_threshold must have one value per curve")
        adaptive = adaptive[:, 0]
    return mask, adaptive, source


def pruning_result_as_deployed_fit(
    result: MinimalKnotPruningResult,
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
        sample_index=0,
        candidate_count=result.initial_count,
        retained_count=result.final_count,
        retained_mask=retained_mask,
        hard_gate=retained_mask,
        spline=result.final_fit,
    )


def refit_candidate_mask_as_deployed_fit(
    parameters: torch.Tensor,
    points: torch.Tensor,
    candidate_knots: torch.Tensor,
    retained_mask: torch.Tensor,
    *,
    degree: int,
    smoothness_weight: float,
    control_ridge: float,
) -> HardGatedBSplineFit:
    """Refit one explicit candidate subset as a standard open B-spline."""
    if candidate_knots.ndim != 1:
        raise ValueError("candidate_knots must have shape [Kc]")
    if retained_mask.shape != candidate_knots.shape:
        raise ValueError("retained_mask must have the same shape as candidate_knots")
    if retained_mask.dtype != torch.bool:
        raise ValueError("retained_mask must be boolean")
    selected_knots = candidate_knots[retained_mask]
    fit = refit_bspline_control_points(
        parameters,
        points,
        selected_knots,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        interpolate_endpoints=True,
    )
    return HardGatedBSplineFit(
        sample_index=0,
        candidate_count=int(candidate_knots.numel()),
        retained_count=int(retained_mask.sum().item()),
        retained_mask=retained_mask,
        hard_gate=retained_mask,
        spline=fit,
    )


def candidate_loss_from_checkpoint(
    checkpoint: dict[str, object], *, disable_exact_teacher: bool = False
) -> CandidatePruningLoss:
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
        candidate_match_tolerance=float(config.get("candidate_match_tolerance", 0.02)),
        fit_tolerance=float(config.get("fit_tolerance", 5e-3)),
        repulsion_distance=config.get("repulsion_distance"),
        positive_keep_weight=float(config.get("positive_keep_weight", 2.0)),
        exact_deletion_supervision=(
            bool(config.get("exact_deletion_supervision", True))
            and not disable_exact_teacher
        ),
        deletion_smoothness_weight=float(
            config.get("deletion_smoothness_weight", 1e-6)
        ),
        deletion_control_ridge=float(config.get("deletion_control_ridge", 0.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize one fitted B-spline sample."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=20000,
        help="Synthetic test seed; use 10000 only to inspect validation samples.",
    )
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=None,
        help=(
            "Historical activity threshold. For v8 it is ignored by deployment; "
            "the learned mask (or centered keep probability >= 0.5 fallback) is used."
        ),
    )
    parser.add_argument("--smoothness-weight", type=float, default=1e-6)
    parser.add_argument("--control-ridge", type=float, default=0.0)
    parser.add_argument(
        "--fit-tolerance",
        type=float,
        default=None,
        help=(
            "Normalized RMS reference for one-shot satisfaction reporting "
            "and optional offline hard diagnostics. It never changes the v8 "
            "learned mask."
        ),
    )
    parser.add_argument(
        "--pruning-view",
        choices=("all", "learned", "hard", "comparison"),
        default=None,
        help=(
            "Candidate-pruning visualization: all candidates, learned "
            "one-shot mask, offline hard RMS teacher, or comparison. "
            "Defaults to learned for v8 and hard for v7."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output resolution (default: 300).",
    )
    parser.add_argument(
        "--count-selection", choices=("auto", "network", "bic"), default="auto"
    )
    parser.add_argument("--count-prior-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

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
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        parser.error("--activity-threshold must lie in [0, 1]")
    model, model_config, legacy_checkpoint = build_model_from_checkpoint(checkpoint)
    model.set_activity_threshold(threshold)
    model.eval()
    structure_mode = model_config.get("structure_mode", "hard_concrete")
    candidate_pruning, candidate_one_shot = candidate_mode_flags(
        checkpoint, model_config
    )
    if args.pruning_view is None:
        args.pruning_view = "learned" if candidate_one_shot else "hard"
    if not candidate_pruning and args.pruning_view != "hard":
        parser.error(
            "--pruning-view all/learned/comparison requires a "
            "candidate-pruning checkpoint"
        )
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
        if candidate_pruning
        or checkpoint.get("objective_version")
        in {
            CANDIDATE_PRUNING_OBJECTIVE_VERSION,
            CURRENT_OBJECTIVE_VERSION,
            COUNT_CONDITIONED_V5_OBJECTIVE_VERSION,
        }
        else 0.0,
    )
    dataset_config["return_ground_truth"] = True
    dataset = SyntheticCubicBSplineDataset(
        size=args.sample_index + 1,
        seed=args.seed,
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
            deployment_output["activity_gate"] = (output["activity"] >= threshold).to(
                output["activity"].dtype
            )
        else:
            deployment_output = output
        pruning_result: MinimalKnotPruningResult | None = None
        candidate_fits: dict[str, HardGatedBSplineFit] = {}
        adaptive_keep_threshold: torch.Tensor | None = None
        learned_selection_source: str | None = None
        if candidate_pruning:
            candidate_knots = output["internal_knots"][0]
            all_mask = torch.ones_like(candidate_knots, dtype=torch.bool)
            if candidate_one_shot:
                learned_masks, adaptive_thresholds, learned_selection_source = (
                    one_shot_selection(output)
                )
                learned_mask = learned_masks[0]
                adaptive_keep_threshold = adaptive_thresholds[0]
            else:
                learned_mask = output["keep_probability"][0] >= threshold
            requested_views = (
                {"all", "learned", "hard"}
                if args.pruning_view == "comparison"
                else {args.pruning_view}
            )
            for name, mask in (("all", all_mask), ("learned", learned_mask)):
                if name not in requested_views:
                    continue
                candidate_fits[name] = refit_candidate_mask_as_deployed_fit(
                    output["params"][0],
                    points[0],
                    candidate_knots,
                    mask,
                    degree=model.degree,
                    smoothness_weight=args.smoothness_weight,
                    control_ridge=args.control_ridge,
                )
            if "hard" in requested_views:
                pruning_result = prune_knots_to_rms_tolerance(
                    output["params"][0],
                    points[0],
                    candidate_knots,
                    error_tolerance=fit_tolerance,
                    degree=model.degree,
                    smoothness_weight=args.smoothness_weight,
                    control_ridge=args.control_ridge,
                    interpolate_endpoints=True,
                )
                candidate_fits["hard"] = pruning_result_as_deployed_fit(pruning_result)
            primary_view = args.pruning_view
            if primary_view == "comparison":
                primary_view = "learned" if candidate_one_shot else "hard"
            deployed = candidate_fits[primary_view]
        else:
            deployed = refit_model_output_as_bsplines(
                deployment_output,
                points,
                degree=model.degree,
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
            )[0]

    if candidate_pruning:
        loss_fn = candidate_loss_from_checkpoint(
            checkpoint,
            disable_exact_teacher=candidate_one_shot,
        )
    else:
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

    def plot_curve(
        axis,
        fit: HardGatedBSplineFit,
        *,
        title: str,
        spline_label: str,
        show_network: bool,
    ) -> None:
        dense_curve = fit.spline.evaluate(dense_params).numpy()
        controls = fit.control_points.numpy()
        axis.scatter(observed[:, 0], observed[:, 1], s=13, label="samples")
        axis.scatter(
            observed[[0, -1], 0],
            observed[[0, -1], 1],
            s=55,
            marker="x",
            label="sample endpoints",
        )
        if show_network:
            axis.plot(
                forward_curve[:, 0],
                forward_curve[:, 1],
                "--",
                label="network surrogate",
            )
        axis.plot(dense_curve[:, 0], dense_curve[:, 1], label=spline_label)
        axis.plot(
            controls[:, 0],
            controls[:, 1],
            "o-",
            alpha=0.45,
            label="control polygon",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            f"{title}\nK={fit.retained_count} | RMS={float(fit.fit_rmse):.4e}"
        )
        axis.legend(fontsize="small")

    comparison_view = candidate_pruning and args.pruning_view == "comparison"
    if comparison_view:
        figure, axes = plt.subplots(2, 2, figsize=(13, 10))
        ax_all, ax_learned, ax_hard, ax_structure = axes.ravel()
        plot_curve(
            ax_all,
            candidate_fits["all"],
            title="(a) All candidates",
            spline_label="all-candidate B-spline",
            show_network=True,
        )
        plot_curve(
            ax_learned,
            candidate_fits["learned"],
            title=(
                "(b) One-shot learned deployment"
                if candidate_one_shot
                else f"(b) Learned keep (p >= {threshold:.2f})"
            ),
            spline_label="one-shot learned B-spline",
            show_network=False,
        )
        plot_curve(
            ax_hard,
            candidate_fits["hard"],
            title=(
                f"(c) Offline hard teacher (epsilon={fit_tolerance:.2e})"
                if candidate_one_shot
                else f"(c) Hard RMS pruning (epsilon={fit_tolerance:.2e})"
            ),
            spline_label=(
                "offline teacher B-spline"
                if candidate_one_shot
                else "hard-pruned B-spline"
            ),
            show_network=False,
        )
        curve_axes = (ax_all, ax_learned, ax_hard)
        x_limits = [axis.get_xlim() for axis in curve_axes]
        y_limits = [axis.get_ylim() for axis in curve_axes]
        common_xlim = (
            min(limit[0] for limit in x_limits),
            max(limit[1] for limit in x_limits),
        )
        common_ylim = (
            min(limit[0] for limit in y_limits),
            max(limit[1] for limit in y_limits),
        )
        for axis in curve_axes:
            axis.set_xlim(common_xlim)
            axis.set_ylim(common_ylim)
    else:
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_curve, ax_structure = axes
        if candidate_pruning:
            view_titles = {
                "all": "All candidates",
                "learned": (
                    "One-shot learned deployment"
                    if candidate_one_shot
                    else f"Learned keep (p >= {threshold:.2f})"
                ),
                "hard": (
                    f"Offline hard teacher (epsilon={fit_tolerance:.2e})"
                    if candidate_one_shot
                    else f"Hard RMS pruning (epsilon={fit_tolerance:.2e})"
                ),
            }
            spline_labels = {
                "all": "all-candidate B-spline",
                "learned": "learned-pruned B-spline",
                "hard": "hard-pruned B-spline",
            }
            plot_curve(
                ax_curve,
                deployed,
                title=view_titles[args.pruning_view],
                spline_label=spline_labels[args.pruning_view],
                show_network=True,
            )
        else:
            plot_curve(
                ax_curve,
                deployed,
                title="Deployed standard B-spline",
                spline_label="deployed B-spline",
                show_network=True,
            )

    if count_conditioned:
        probabilities = output["count_probabilities"][0].detach().numpy()
        counts = list(range(len(probabilities)))
        mode_count = int(
            output.get("count_mode_knot_count", output["predicted_knot_count"])[0]
        )
        deployed_count = deployed.retained_count
        colors = [
            "tab:orange"
            if count == deployed_count
            else "tab:green"
            if count == mode_count
            else "tab:blue"
            for count in counts
        ]
        ax_structure.bar(counts, probabilities, color=colors)
        ax_structure.set_xlabel("internal-knot count")
        ax_structure.set_ylabel("count probability")
        head_name = (
            "StructureHead" if structure_mode == "interactive_dynamic" else "CountHead"
        )
        ax_structure.set_title(
            f"{head_name} mode={mode_count} | median/deployed K={deployed_count} "
            f"({count_selection})"
        )
        ax_structure.set_xticks(counts)
    elif candidate_pruning:
        keep_probability = output["keep_probability"][0].detach().numpy()
        knots = output["internal_knots"][0].detach().numpy()
        if args.pruning_view == "all":
            displayed_mask = candidate_fits["all"].retained_mask
            selection_description = "selected in all-candidate fit"
        elif args.pruning_view == "learned":
            displayed_mask = candidate_fits["learned"].retained_mask
            selection_description = (
                "learned adaptive one-shot mask"
                if candidate_one_shot
                else f"learned p >= {threshold:.2f}"
            )
        else:
            displayed_mask = candidate_fits["hard"].retained_mask
            selection_description = (
                "offline hard teacher"
                if candidate_one_shot
                else "retained by hard RMS pruning"
            )
        kept = displayed_mask.cpu().numpy()
        colors = ["tab:orange" if value else "tab:blue" for value in kept]
        raw_importance_tensor = output.get(
            "keep_importance_logits",
            output.get("raw_keep_importance", output.get("raw_importance")),
        )
        if candidate_one_shot and raw_importance_tensor is not None:
            structure_values = raw_importance_tensor[0].detach().numpy()
            adaptive_value = float(adaptive_keep_threshold)
            ax_structure.bar(
                range(len(structure_values)), structure_values, color=colors
            )
            if math.isfinite(adaptive_value):
                ax_structure.axhline(
                    adaptive_value,
                    color="black",
                    linestyle="--",
                    label="adaptive logit beta",
                )
            structure_ylabel = "raw learned keep-importance logit"
        else:
            structure_values = keep_probability
            ax_structure.bar(
                range(len(structure_values)), structure_values, color=colors
            )
            decision_cutoff = 0.5 if candidate_one_shot else threshold
            ax_structure.axhline(decision_cutoff, color="black", linestyle="--")
            structure_ylabel = (
                "threshold-centered keep probability"
                if candidate_one_shot
                else "learned keep probability (diagnostic only)"
            )
        if comparison_view:
            learned_indices = (
                torch.nonzero(candidate_fits["learned"].retained_mask, as_tuple=False)
                .squeeze(-1)
                .cpu()
                .numpy()
            )
            ax_structure.scatter(
                learned_indices,
                structure_values[learned_indices],
                marker="D",
                facecolors="none",
                edgecolors="tab:green",
                label="learned keep",
                zorder=3,
            )
            ax_structure.legend(fontsize="small")
        ax_structure.set_xticks(
            range(len(knots)), [f"{value:.3f}" for value in knots], rotation=45
        )
        ax_structure.set_xlabel(f"candidate knot (orange = {selection_description})")
        ax_structure.set_ylabel(structure_ylabel)
        if comparison_view:
            ax_structure.set_title(
                "(d) Candidate decisions\n"
                f"learned K={candidate_fits['learned'].retained_count} | "
                f"hard K={candidate_fits['hard'].retained_count}"
                + (" (offline diagnostic)" if candidate_one_shot else "")
            )
        else:
            ax_structure.set_title(
                f"{args.pruning_view.capitalize()} selection "
                f"{len(knots)} -> {deployed.retained_count}"
            )
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
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    true_knots = sample["true_internal_knots"][sample["true_internal_knot_mask"]]
    matching = match_internal_knots(
        deployed.retained_internal_knots,
        true_knots,
        tolerance=0.05,
    )
    selected_fit_is_deployment = (
        not candidate_pruning
        or (candidate_one_shot and args.pruning_view in {"learned", "comparison"})
        or (not candidate_one_shot and args.pruning_view in {"hard", "comparison"})
    )
    selected_fit_role = (
        "deployment" if selected_fit_is_deployment else "selected offline diagnostic"
    )
    print("Spline structure report")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  structure mode: {structure_mode}")
    print(f"  checkpoint best epoch: {checkpoint.get('epoch', 'not recorded')}")
    if count_conditioned:
        print(f"  predicted knot count: {int(output['predicted_knot_count'][0])}")
        print(
            "  posterior mode knot count: "
            f"{int(output.get('count_mode_knot_count', output['predicted_knot_count'])[0])}"
        )
        print(f"  expected knot count: {float(output['expected_knot_count'][0]):.6f}")
        print(f"  count probabilities: {output['count_probabilities'][0].tolist()}")
        print(f"  deployment count selection: {count_selection}")
    elif candidate_pruning:
        print(f"  pruning view: {args.pruning_view}")
        if candidate_one_shot:
            assert adaptive_keep_threshold is not None
            if selected_fit_is_deployment:
                print("  selected result role: v8 learned one-shot deployment")
            else:
                print(
                    f"  selected result role: offline {args.pruning_view} diagnostic; "
                    "not the v8 deployment result"
                )
                print(
                    "  v8 deployment policy: learned one-shot mask followed by one "
                    "standard B-spline refit (not executed in this selected view)"
                )
            print(f"  learned selection source: {learned_selection_source}")
            print("  final threshold-centered keep-probability cutoff: 0.5")
            print(
                "  adaptive raw-importance logit beta: "
                f"{float(adaptive_keep_threshold):.6f}"
            )
            print(
                "  fit tolerance is reporting-only for learned deployment; "
                "it does not change the mask"
            )
            if selected_fit_is_deployment:
                print(
                    "  deployment standard B-spline refits: 1 "
                    "(network-forward proxy solves are excluded)"
                )
        else:
            print(f"  learned keep threshold: {threshold:.4f}")
        for name, fit in candidate_fits.items():
            role = (
                " (offline diagnostic only)"
                if candidate_one_shot and name in {"all", "hard"}
                else " (deployment)"
                if candidate_one_shot and name == "learned"
                else ""
            )
            print(
                f"  {name} standard B-spline{role}: K={fit.retained_count}, "
                f"RMS={float(fit.fit_rmse):.9e}"
            )
        if pruning_result is not None:
            print(
                "  hard-pruning RMS threshold"
                + (" (offline teacher only)" if candidate_one_shot else "")
                + f": {fit_tolerance:.9e}"
            )
            print(
                "  hard candidates -> retained: "
                f"{pruning_result.initial_count} -> {pruning_result.final_count}"
            )
            print(f"  threshold satisfied: {pruning_result.threshold_satisfied}")
            print(f"  RMS trajectory: {pruning_result.rms_trajectory.cpu().tolist()}")
        print(
            (
                "  threshold-centered keep probabilities "
                "(v8 learned deployment selector): "
                if candidate_one_shot
                else "  learned keep probabilities (diagnostic only): "
            )
            + f"{output['keep_probability'][0].detach().cpu().tolist()}"
        )
    else:
        print(f"  activity threshold: {threshold:.4f}")
    print(f"  {selected_fit_role} retained internal knots: {deployed.retained_count}")
    print(
        f"  {selected_fit_role} retained knot values: "
        + ", ".join(
            f"{value:.6f}" for value in deployed.retained_internal_knots.tolist()
        )
    )
    print(f"  true internal knots: {true_knots.tolist()}")
    print(
        f"  {selected_fit_role} match@0.05: precision={matching.precision:.3f}, "
        f"recall={matching.recall:.3f}, F1={matching.f1:.3f}"
    )
    if candidate_one_shot:
        print(
            "  inference proxy objective (teacher terms unavailable): "
            f"{float(losses['loss']):.9e}"
        )
    else:
        print(f"  total objective: {float(losses['loss']):.9e}")
    print(f"  network fit loss: {float(losses['fit_loss']):.9e}")
    print(
        f"  {selected_fit_role} standard B-spline refit loss: "
        f"{float(deployed.fit_mse):.9e}"
    )
    print(
        "  endpoint distances: "
        f"start={float((deployed.reconstructed_points[0] - points[0, 0]).norm()):.9e}, "
        f"end={float((deployed.reconstructed_points[-1] - points[0, -1]).norm()):.9e}"
    )
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
