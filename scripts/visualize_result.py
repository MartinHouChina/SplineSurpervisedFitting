from __future__ import annotations

import argparse
import math
import statistics
import sys
import textwrap
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, TypeVar

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
from spline_fitting.data.synthetic import (
    SyntheticCubicBSplineDataset,
    bspline_basis_matrix,
)
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


T = TypeVar("T")


class CurveComparisonPanel(NamedTuple):
    """Everything needed for one paper-ready curve comparison panel."""

    dense_curve: torch.Tensor
    reconstructed_points: torch.Tensor
    control_points: torch.Tensor
    internal_knots: torch.Tensor
    knot_vector: torch.Tensor
    fit_rmse: float
    elapsed_ms: float
    postprocess_ms: float
    label: str

    @property
    def internal_knot_count(self) -> int:
        return int(self.internal_knots.numel())


def timed_call(function: Callable[[], T], *, repeats: int) -> tuple[T, float]:
    """Return the final result and median wall time in milliseconds."""
    if repeats <= 0:
        raise ValueError("timing repeats must be positive")
    durations: list[float] = []
    result: T | None = None
    for _ in range(repeats):
        started_at = time.perf_counter()
        result = function()
        durations.append(1e3 * (time.perf_counter() - started_at))
    assert result is not None
    return result, float(statistics.median(durations))


def fit_as_comparison_panel(
    fit: HardGatedBSplineFit,
    dense_params: torch.Tensor,
    *,
    elapsed_ms: float,
    postprocess_ms: float,
    label: str,
) -> CurveComparisonPanel:
    return CurveComparisonPanel(
        dense_curve=fit.spline.evaluate(dense_params).detach().cpu(),
        reconstructed_points=fit.reconstructed_points.detach().cpu(),
        control_points=fit.control_points.detach().cpu(),
        internal_knots=fit.retained_internal_knots.detach().cpu(),
        knot_vector=fit.spline.knot_vector.detach().cpu(),
        fit_rmse=float(fit.fit_rmse),
        elapsed_ms=float(elapsed_ms),
        postprocess_ms=float(postprocess_ms),
        label=label,
    )


def compact_knot_vector(knot_vector: torch.Tensor, degree: int) -> str:
    """Format a full open knot vector without hiding repeated endpoints."""
    endpoint_repeat = degree + 1
    internal = knot_vector[endpoint_repeat:-endpoint_repeat]
    internal_text = ", ".join(f"{float(value):.3f}" for value in internal)
    middle = f", {internal_text}, " if internal_text else ", "
    return f"U=[0x{endpoint_repeat}{middle}1x{endpoint_repeat}]"


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
        teacher_ranking_margin=float(config.get("teacher_ranking_margin", 1.0)),
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
            "Historical activity threshold. For v8-v10 it is ignored by deployment; "
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
            "and optional offline hard diagnostics. It never changes the v8-v10 "
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
            "Defaults to learned for v8-v10 and hard for v7."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output resolution (default: 300).",
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=1,
        help=(
            "Repeat forward/refit/pruning timing and report the median. Use 3-5 "
            "for paper timing; default 1 keeps multi-sample plotting fast."
        ),
    )
    parser.add_argument(
        "--one-shot-selection-policy",
        choices=("checkpoint", "threshold", "mass_topk"),
        default="checkpoint",
    )
    parser.add_argument("--one-shot-safety-sigma", type=float, default=None)
    parser.add_argument("--one-shot-coverage-bins", type=int, default=None)
    parser.add_argument(
        "--count-selection", choices=("auto", "network", "bic"), default="auto"
    )
    parser.add_argument("--count-prior-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.timing_repeats <= 0:
        parser.error("--timing-repeats must be positive")
    if args.one_shot_safety_sigma is not None and args.one_shot_safety_sigma < 0.0:
        parser.error("--one-shot-safety-sigma must be non-negative")
    if args.one_shot_coverage_bins is not None and args.one_shot_coverage_bins < 0:
        parser.error("--one-shot-coverage-bins must be non-negative")

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
    if candidate_one_shot:
        if args.one_shot_selection_policy != "checkpoint":
            model.pruning_head.one_shot_selection_policy = (
                args.one_shot_selection_policy
            )
        if args.one_shot_safety_sigma is not None:
            model.pruning_head.one_shot_safety_sigma = args.one_shot_safety_sigma
        if args.one_shot_coverage_bins is not None:
            model.pruning_head.one_shot_coverage_bins = args.one_shot_coverage_bins
    elif (
        args.one_shot_selection_policy != "checkpoint"
        or args.one_shot_safety_sigma is not None
        or args.one_shot_coverage_bins is not None
    ):
        parser.error("one-shot selection overrides require a one-shot checkpoint")
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

    def run_model_forward() -> dict[str, torch.Tensor]:
        with torch.no_grad():
            return model(points)

    output, network_forward_ms = timed_call(
        run_model_forward,
        repeats=args.timing_repeats,
    )
    with torch.no_grad():
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
        candidate_postprocess_ms: dict[str, float] = {}
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
                candidate_fits[name], candidate_postprocess_ms[name] = timed_call(
                    lambda mask=mask: refit_candidate_mask_as_deployed_fit(
                        output["params"][0],
                        points[0],
                        candidate_knots,
                        mask,
                        degree=model.degree,
                        smoothness_weight=args.smoothness_weight,
                        control_ridge=args.control_ridge,
                    ),
                    repeats=args.timing_repeats,
                )
            if "hard" in requested_views:
                pruning_result, candidate_postprocess_ms["hard"] = timed_call(
                    lambda: prune_knots_to_rms_tolerance(
                        output["params"][0],
                        points[0],
                        candidate_knots,
                        error_tolerance=fit_tolerance,
                        degree=model.degree,
                        smoothness_weight=args.smoothness_weight,
                        control_ridge=args.control_ridge,
                        interpolate_endpoints=True,
                    ),
                    repeats=args.timing_repeats,
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

    source_control_key = (
        "source_control_points"
        if "source_control_points" in sample
        else "true_control_points"
    )
    source_control_mask_key = (
        "source_control_mask"
        if "source_control_mask" in sample
        else "true_control_mask"
    )
    source_knot_key = (
        "source_knot_vector" if "source_knot_vector" in sample else "true_knot_vector"
    )
    source_knot_mask_key = (
        "source_knot_mask" if "source_knot_mask" in sample else "true_knot_mask"
    )
    reference_controls = sample[source_control_key][
        sample[source_control_mask_key].to(torch.bool)
    ]
    reference_knot_vector = sample[source_knot_key][
        sample[source_knot_mask_key].to(torch.bool)
    ]
    if "source_knot_vector" in sample:
        endpoint_multiplicity = model.degree + 1
        reference_internal_knots = reference_knot_vector[
            endpoint_multiplicity:-endpoint_multiplicity
        ]
    else:
        reference_internal_knots = sample["true_internal_knots"][
            sample["true_internal_knot_mask"].to(torch.bool)
        ]

    def evaluate_reference_curve() -> tuple[torch.Tensor, torch.Tensor]:
        dense_basis = bspline_basis_matrix(
            dense_params,
            reference_knot_vector,
            model.degree,
            num_control_points=reference_controls.shape[0],
        )
        sample_basis = bspline_basis_matrix(
            sample["true_params"],
            reference_knot_vector,
            model.degree,
            num_control_points=reference_controls.shape[0],
        )
        return dense_basis @ reference_controls, sample_basis @ reference_controls

    (reference_dense_curve, reference_reconstruction), reference_time_ms = timed_call(
        evaluate_reference_curve,
        repeats=args.timing_repeats,
    )
    reference_rmse = float(
        (reference_reconstruction - points[0]).square().sum(dim=-1).mean().sqrt()
    )
    reference_panel = CurveComparisonPanel(
        dense_curve=reference_dense_curve.detach().cpu(),
        reconstructed_points=reference_reconstruction.detach().cpu(),
        control_points=reference_controls.detach().cpu(),
        internal_knots=reference_internal_knots.detach().cpu(),
        knot_vector=reference_knot_vector.detach().cpu(),
        fit_rmse=reference_rmse,
        elapsed_ms=reference_time_ms,
        postprocess_ms=reference_time_ms,
        label="reference B-spline",
    )

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

    def plot_comparison_panel(
        axis,
        panel: CurveComparisonPanel,
        *,
        title: str,
        time_description: str,
    ) -> None:
        dense_curve = panel.dense_curve.numpy()
        controls = panel.control_points.numpy()
        axis.scatter(
            observed[:, 0],
            observed[:, 1],
            s=11,
            alpha=0.72,
            label="sample points",
            zorder=2,
        )
        axis.scatter(
            observed[[0, -1], 0],
            observed[[0, -1], 1],
            s=52,
            marker="x",
            linewidths=1.8,
            label="endpoints",
            zorder=5,
        )
        axis.plot(
            dense_curve[:, 0],
            dense_curve[:, 1],
            linewidth=1.8,
            label=panel.label,
            zorder=3,
        )
        axis.plot(
            controls[:, 0],
            controls[:, 1],
            "o-",
            markersize=4.5,
            linewidth=1.0,
            alpha=0.58,
            label="control polygon",
            zorder=1,
        )
        if panel.internal_knots.numel():
            knot_basis = bspline_basis_matrix(
                panel.internal_knots,
                panel.knot_vector,
                model.degree,
                num_control_points=panel.control_points.shape[0],
            )
            knot_locations = (knot_basis @ panel.control_points).numpy()
            axis.scatter(
                knot_locations[:, 0],
                knot_locations[:, 1],
                s=30,
                marker="D",
                facecolors="none",
                linewidths=1.1,
                label="internal knots C(u)",
                zorder=4,
            )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(
            f"{title}\nK={panel.internal_knot_count} | "
            f"RMS={panel.fit_rmse:.4e} | {time_description}"
        )
        knot_text = compact_knot_vector(panel.knot_vector, model.degree)
        axis.text(
            0.5,
            -0.16,
            textwrap.fill(knot_text, width=92),
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
        )
        axis.legend(fontsize=7.5, loc="best")

    comparison_view = candidate_pruning and args.pruning_view == "comparison"
    if comparison_view:
        comparison_panels = {
            "reference": reference_panel,
            "all": fit_as_comparison_panel(
                candidate_fits["all"],
                dense_params,
                elapsed_ms=network_forward_ms + candidate_postprocess_ms["all"],
                postprocess_ms=candidate_postprocess_ms["all"],
                label="redundant all-candidate B-spline",
            ),
            "learned": fit_as_comparison_panel(
                candidate_fits["learned"],
                dense_params,
                elapsed_ms=network_forward_ms + candidate_postprocess_ms["learned"],
                postprocess_ms=candidate_postprocess_ms["learned"],
                label="learned deployment B-spline",
            ),
            "hard": fit_as_comparison_panel(
                candidate_fits["hard"],
                dense_params,
                elapsed_ms=network_forward_ms + candidate_postprocess_ms["hard"],
                postprocess_ms=candidate_postprocess_ms["hard"],
                label="Hard-RMS B-spline",
            ),
        }
        figure, axes = plt.subplots(2, 2, figsize=(16, 12))
        ax_reference, ax_all, ax_learned, ax_hard = axes.ravel()
        plot_comparison_panel(
            ax_reference,
            comparison_panels["reference"],
            title="(a) Original/source data",
            time_description=f"CPU reference eval={reference_time_ms:.2f} ms",
        )
        plot_comparison_panel(
            ax_all,
            comparison_panels["all"],
            title="(b) Redundant all-candidate fit",
            time_description=(
                f"CPU total={comparison_panels['all'].elapsed_ms:.2f} ms "
                f"(net={network_forward_ms:.2f}, refit="
                f"{comparison_panels['all'].postprocess_ms:.2f})"
            ),
        )
        plot_comparison_panel(
            ax_learned,
            comparison_panels["learned"],
            title=(
                "(c) Learned one-shot deployment"
                if candidate_one_shot
                else f"(c) Learned keep (p >= {threshold:.2f})"
            ),
            time_description=(
                f"CPU total={comparison_panels['learned'].elapsed_ms:.2f} ms "
                f"(net={network_forward_ms:.2f}, refit="
                f"{comparison_panels['learned'].postprocess_ms:.2f})"
            ),
        )
        plot_comparison_panel(
            ax_hard,
            comparison_panels["hard"],
            title=(
                f"(d) Offline Hard-RMS deletion (epsilon={fit_tolerance:.2e})"
                if candidate_one_shot
                else f"(d) Hard-RMS pruning (epsilon={fit_tolerance:.2e})"
            ),
            time_description=(
                f"CPU total={comparison_panels['hard'].elapsed_ms:.2f} ms "
                f"(net={network_forward_ms:.2f}, prune="
                f"{comparison_panels['hard'].postprocess_ms:.2f})"
            ),
        )
        curve_axes = (ax_reference, ax_all, ax_learned, ax_hard)
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
    elif candidate_pruning and not comparison_view:
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
    elif not comparison_view:
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

    if comparison_view:
        figure.subplots_adjust(hspace=0.42, wspace=0.24, bottom=0.08, top=0.94)
    else:
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
                print("  selected result role: learned one-shot deployment")
            else:
                print(
                    f"  selected result role: offline {args.pruning_view} diagnostic; "
                    "not the one-shot deployment result"
                )
                print(
                    "  one-shot deployment policy: learned mask followed by one "
                    "standard B-spline refit (not executed in this selected view)"
                )
            print(f"  learned selection source: {learned_selection_source}")
            print(
                "  selection policy: "
                f"{getattr(model.pruning_head, 'one_shot_selection_policy', 'threshold')}"
                " | safety sigma="
                f"{getattr(model.pruning_head, 'one_shot_safety_sigma', 0.0):.3f}"
                " | coverage bins="
                f"{getattr(model.pruning_head, 'one_shot_coverage_bins', 0)}"
            )
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
            if name in candidate_postprocess_ms:
                postprocess_name = "prune" if name == "hard" else "refit"
                print(
                    f"    CPU time: total="
                    f"{network_forward_ms + candidate_postprocess_ms[name]:.3f} ms "
                    f"(network={network_forward_ms:.3f} ms, "
                    f"{postprocess_name}={candidate_postprocess_ms[name]:.3f} ms)"
                )
                print(
                    "    knot vector: "
                    + compact_knot_vector(fit.spline.knot_vector, model.degree)
                )
        if comparison_view:
            print(
                "  reference data: "
                f"K={reference_panel.internal_knot_count}, "
                f"RMS={reference_panel.fit_rmse:.9e}, "
                f"evaluation={reference_panel.elapsed_ms:.3f} ms"
            )
            print(
                "    knot vector: "
                + compact_knot_vector(reference_panel.knot_vector, model.degree)
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
                "(learned one-shot deployment selector): "
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
    print(f"  canonical supervision internal knots: {true_knots.tolist()}")
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
