from __future__ import annotations

# Script entry points intentionally add ``src`` to sys.path before importing
# the local package so they also run from an unpacked repository.
# ruff: noqa: E402

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import (
    COUNT_CONDITIONED_V5_OBJECTIVE_VERSION,
    V12_COUPLED_RELOCATION_OBJECTIVE_VERSION,
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
from spline_fitting.evaluation.hybrid_knot_search import (
    HybridKnotSearchResult,
    hybrid_minimal_knot_search,
)
from spline_fitting.evaluation.verified_knot_repair import (
    VerifiedKnotRepairResult,
    verified_confidence_repair,
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


def resolve_verified_refit_device(
    requested: str,
    *,
    model_device: torch.device,
) -> torch.device:
    """Resolve the exact verified-refit device independently of the network."""

    if requested not in {"auto", "cpu", "model"}:
        raise ValueError("verified refit device must be one of auto, cpu, model")
    if requested == "model":
        return torch.device(model_device)
    return torch.device("cpu")


def scale_normalized_fit_errors(
    normalized_mse: float,
    normalized_rmse: float,
    scale: float,
) -> tuple[float, float]:
    """Map normalized MSE/RMS back to the source coordinate scale."""

    if not all(
        math.isfinite(value) for value in (normalized_mse, normalized_rmse, scale)
    ):
        raise ValueError("fit errors and scale must be finite")
    if normalized_mse < 0.0 or normalized_rmse < 0.0 or scale <= 0.0:
        raise ValueError("fit errors must be non-negative and scale must be positive")
    return normalized_mse * scale * scale, normalized_rmse * scale


def candidate_mode_flags(
    checkpoint: dict[str, object], model_config: dict[str, object]
) -> tuple[bool, bool]:
    structure = str(model_config.get("structure_mode", "")).lower()
    objective = str(checkpoint.get("objective_version", "")).lower()
    one_shot = (
        structure == "candidate_pruning_one_shot"
        or "candidate_pruning_one_shot" in objective
        or objective == V12_COUPLED_RELOCATION_OBJECTIVE_VERSION
    )
    return one_shot or structure == "candidate_pruning", one_shot


def resolve_deployment_mode(
    requested: str,
    *,
    candidate_pruning: bool,
    candidate_one_shot: bool,
) -> str:
    """Resolve a checkpoint-compatible deployment mode.

    Historical one-shot checkpoints retain their learned single-refit default,
    while v7 candidate checkpoints retain exhaustive greedy hard pruning.
    Other historical structures continue through their existing deployment
    path under the internal ``legacy`` mode.
    """

    if requested == "checkpoint":
        if candidate_one_shot:
            return "learned"
        if candidate_pruning:
            return "hard"
        return "legacy"
    if requested in {"learned", "hybrid", "verified"} and not candidate_one_shot:
        raise ValueError(f"{requested} deployment requires a one-shot checkpoint")
    if requested == "hard" and not candidate_pruning:
        raise ValueError("hard deployment requires a candidate-pruning checkpoint")
    return requested


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


def refit_one_shot_full_resolution(
    output: dict[str, torch.Tensor],
    parameters: torch.Tensor,
    points: torch.Tensor,
    *,
    degree: int,
    smoothness_weight: float,
    control_ridge: float,
) -> tuple[HardGatedBSplineFit, torch.Tensor, torch.Tensor, str]:
    """Apply the learned one-shot mask and refit once at source resolution."""
    mask, adaptive_threshold, source = one_shot_selection(output)
    selected_output = {
        "params": parameters.unsqueeze(0),
        "internal_knots": output["internal_knots"],
        "knot_mask": mask,
    }
    deployed = refit_model_output_as_bsplines(
        selected_output,
        points,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        interpolate_endpoints=True,
    )[0]
    return deployed, mask[0], adaptive_threshold[0], source


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


def hybrid_result_as_deployed_fit(
    result: HybridKnotSearchResult,
    *,
    sample_index: int = 0,
) -> HardGatedBSplineFit:
    """Adapt a slow hybrid-search result to the common deployment interface."""

    return HardGatedBSplineFit(
        sample_index=sample_index,
        candidate_count=result.proposal_count,
        retained_count=result.final_count,
        retained_mask=result.retained_proposal_mask,
        hard_gate=result.retained_proposal_mask,
        spline=result.final_fit,
    )


def verified_result_as_deployed_fit(
    result: VerifiedKnotRepairResult,
    *,
    sample_index: int = 0,
) -> HardGatedBSplineFit:
    """Adapt a verified confidence-repair result to the common fit interface."""

    return HardGatedBSplineFit(
        sample_index=sample_index,
        candidate_count=result.deployment_candidate_count,
        retained_count=result.final_count,
        retained_mask=result.deployment_retained_mask,
        hard_gate=result.deployment_retained_mask,
        spline=result.final_fit,
    )


def run_verified_repair_full_resolution(
    output: dict[str, torch.Tensor],
    source_parameters: torch.Tensor,
    full_resolution_points: torch.Tensor,
    learned_mask: torch.Tensor,
    *,
    source_chord_parameters: torch.Tensor | None = None,
    fit_tolerance_rms: float,
    degree: int,
    smoothness_weight: float,
    control_ridge: float,
    min_internal_knots: int = 0,
    compact: bool = True,
    hard_fallback: bool = True,
    residual_fallback: bool = True,
    max_residual_insertions: int = 8,
    residual_min_gap: float = 1e-3,
    refit_device: torch.device | str | None = None,
    parameterization_policy: str = "network",
) -> VerifiedKnotRepairResult:
    """Verify and, only when needed, repair a one-shot full-resolution fit."""

    if full_resolution_points.ndim != 3 or full_resolution_points.shape[0] != 1:
        raise ValueError("full_resolution_points must have shape [1,M,D]")
    if source_parameters.ndim != 1:
        raise ValueError("source_parameters must have shape [M]")
    if source_parameters.shape[0] != full_resolution_points.shape[1]:
        raise ValueError("source parameters and full-resolution points must align")
    if source_chord_parameters is not None and (
        source_chord_parameters.ndim != 1
        or source_chord_parameters.shape != source_parameters.shape
    ):
        raise ValueError("source chord parameters must share source shape [M]")
    for key in ("internal_knots", "keep_probability"):
        if key not in output:
            raise KeyError(f"one-shot output is missing {key}")
    proposal_knots = output.get("proposal_internal_knots", output["internal_knots"])[0]
    deployment_knots = output.get(
        "deployment_internal_knots", output["internal_knots"]
    )[0]
    keep_scores = output["keep_probability"][0]
    target_device = (
        source_parameters.device if refit_device is None else torch.device(refit_device)
    )
    return verified_confidence_repair(
        source_parameters.detach().to(target_device),
        full_resolution_points[0].detach().to(target_device),
        proposal_knots.detach().to(target_device),
        deployment_knots.detach().to(target_device),
        learned_mask.detach().to(target_device),
        keep_scores.detach().to(target_device),
        fit_tolerance_rms=fit_tolerance_rms,
        min_internal_knots=min_internal_knots,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        interpolate_endpoints=True,
        compact=compact,
        hard_fallback=hard_fallback,
        residual_fallback=residual_fallback,
        max_residual_insertions=max_residual_insertions,
        residual_min_gap=residual_min_gap,
        alternate_parameters=(
            source_chord_parameters.detach().to(target_device)
            if source_chord_parameters is not None
            else None
        ),
        parameterization_policy=parameterization_policy,
    )


def run_hybrid_search_full_resolution(
    output: dict[str, torch.Tensor],
    source_parameters: torch.Tensor,
    full_resolution_points: torch.Tensor,
    learned_mask: torch.Tensor,
    *,
    fit_tolerance_rms: float,
    degree: int,
    smoothness_weight: float,
    control_ridge: float,
    beam_width: int,
    branch_factor: int,
    position_sweeps: int,
    position_grid_size: int,
    position_restarts: int,
    position_refine_count_margin: int = 1,
    position_refine_candidate_multiplier: int = 4,
    min_gap: float = 1e-3,
) -> HybridKnotSearchResult:
    """Run slow search on source-resolution data with an MSE stopping bound."""

    if full_resolution_points.ndim != 3 or full_resolution_points.shape[0] != 1:
        raise ValueError("full_resolution_points must have shape [1,M,D]")
    if source_parameters.ndim != 1:
        raise ValueError("source_parameters must have shape [M]")
    if source_parameters.shape[0] != full_resolution_points.shape[1]:
        raise ValueError("source parameters and full-resolution points must align")
    proposal_knots = output.get("proposal_internal_knots", output["internal_knots"])[0]
    deployment_knots = output.get(
        "deployment_internal_knots", output["internal_knots"]
    )[0]
    return hybrid_minimal_knot_search(
        source_parameters,
        full_resolution_points[0],
        proposal_knots,
        deployment_knots=deployment_knots,
        learned_mask=learned_mask,
        # Checkpoints store a geometric RMS tolerance.  The hybrid core
        # deliberately ranks and stops on unrooted squared-Euclidean MSE.
        mse_tolerance=fit_tolerance_rms * fit_tolerance_rms,
        min_internal_knots=0,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        interpolate_endpoints=True,
        beam_width=beam_width,
        branch_factor=branch_factor,
        position_sweeps=position_sweeps,
        position_grid_size=position_grid_size,
        position_restarts=position_restarts,
        position_refine_count_margin=position_refine_count_margin,
        position_refine_candidate_multiplier=position_refine_candidate_multiplier,
        min_gap=min_gap,
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
        axis.scatter(
            source_points[:, 0], source_points[:, 1], s=14, label="input points"
        )
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
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=None,
        help=(
            "Historical activity threshold. For v8-v12 it is ignored by deployment; "
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
            "Normalized RMS fit bound. It is reporting-only for learned one-shot "
            "deployment, but is the exact stopping bound for verified/hybrid/hard "
            "deployment. The hybrid MSE core receives its square. Defaults to "
            "deployment_config.error_tolerance, then the dataset canonical knot "
            "tolerance."
        ),
    )
    parser.add_argument(
        "--deployment-mode",
        choices=("checkpoint", "learned", "verified", "hybrid", "hard"),
        default="checkpoint",
        help=(
            "Deployment selector. checkpoint preserves the checkpoint-family "
            "default (learned for v8-v12, hard for v7). verified checks the learned "
            "fit and repairs only failures with confidence add-back and optional "
            "compaction. hybrid runs a slower learned-initialized beam/position "
            "search under the exact fit bound."
        ),
    )
    parser.add_argument(
        "--verified-compact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Greedily compact a feasible verified confidence prefix. Disable with "
            "--no-verified-compact for the fastest verified repair."
        ),
    )
    parser.add_argument(
        "--verified-hard-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fall back to exhaustive proposal hard pruning when no confidence "
            "prefix meets the fit bound. Disable with --no-verified-hard-fallback."
        ),
    )
    parser.add_argument(
        "--verified-residual-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When the complete proposal set still misses the bound, insert knots "
            "at the largest current point residuals before traditional hard fallback."
        ),
    )
    parser.add_argument(
        "--verified-max-residual-insertions",
        type=int,
        default=8,
        help="Maximum exact-refit residual insertions for an extreme failed sample.",
    )
    parser.add_argument(
        "--verified-residual-min-gap",
        type=float,
        default=None,
        help=(
            "Minimum parameter gap for residual insertions. Defaults to the "
            "checkpoint model min_knot_gap."
        ),
    )
    parser.add_argument(
        "--verified-refit-device",
        choices=("auto", "cpu", "model"),
        default="auto",
        help=(
            "Device for verified exact standard-B-spline refits. auto uses CPU "
            "for small least-squares systems even when the model runs on CUDA; "
            "model keeps refits on the network device."
        ),
    )
    parser.add_argument(
        "--verified-parameterization",
        choices=("network", "chord-fallback", "chord"),
        default="chord-fallback",
        help=(
            "Exact-refit parameterization. chord-fallback preserves a feasible "
            "network-domain fit and retries failed curves in deterministic chord "
            "length; chord uses chord length for the full verified path."
        ),
    )
    parser.add_argument("--hybrid-beam-width", type=int, default=4)
    parser.add_argument(
        "--hybrid-branch-factor",
        type=int,
        default=4,
        help="Deletion children per beam state; zero evaluates every deletion.",
    )
    parser.add_argument("--hybrid-position-sweeps", type=int, default=2)
    parser.add_argument("--hybrid-position-grid-size", type=int, default=7)
    parser.add_argument("--hybrid-position-restarts", type=int, default=2)
    parser.add_argument("--hybrid-position-refine-count-margin", type=int, default=1)
    parser.add_argument(
        "--hybrid-position-refine-candidate-multiplier", type=int, default=4
    )
    parser.add_argument(
        "--hybrid-min-gap",
        type=float,
        default=None,
        help="Minimum hybrid knot gap; defaults to the checkpoint min_knot_gap.",
    )
    parser.add_argument(
        "--count-selection", choices=("auto", "network", "bic"), default="auto"
    )
    parser.add_argument("--count-prior-weight", type=float, default=1.0)
    parser.add_argument(
        "--one-shot-selection-policy",
        choices=("checkpoint", "threshold", "mass_topk"),
        default="checkpoint",
    )
    parser.add_argument("--one-shot-safety-sigma", type=float, default=None)
    parser.add_argument("--one-shot-coverage-bins", type=int, default=None)
    args = parser.parse_args()
    if args.smoothness_weight < 0.0 or args.control_ridge < 0.0:
        parser.error("refit regularization weights must be non-negative")
    if args.num_points is not None and args.num_points < 4:
        parser.error("--num-points must be at least four")
    if args.one_shot_safety_sigma is not None and args.one_shot_safety_sigma < 0.0:
        parser.error("--one-shot-safety-sigma must be non-negative")
    if args.one_shot_coverage_bins is not None and args.one_shot_coverage_bins < 0:
        parser.error("--one-shot-coverage-bins must be non-negative")
    if args.hybrid_beam_width <= 0:
        parser.error("--hybrid-beam-width must be positive")
    if args.hybrid_branch_factor < 0:
        parser.error("--hybrid-branch-factor must be non-negative")
    if args.hybrid_position_sweeps < 0:
        parser.error("--hybrid-position-sweeps must be non-negative")
    if args.hybrid_position_grid_size < 3 or args.hybrid_position_grid_size % 2 == 0:
        parser.error("--hybrid-position-grid-size must be odd and at least 3")
    if args.hybrid_position_restarts <= 0:
        parser.error("--hybrid-position-restarts must be positive")
    if args.hybrid_position_refine_count_margin < 0:
        parser.error("--hybrid-position-refine-count-margin must be non-negative")
    if args.hybrid_position_refine_candidate_multiplier <= 0:
        parser.error("--hybrid-position-refine-candidate-multiplier must be positive")
    if args.hybrid_min_gap is not None and (
        not math.isfinite(args.hybrid_min_gap) or args.hybrid_min_gap < 0.0
    ):
        parser.error("--hybrid-min-gap must be finite and non-negative")
    if args.verified_max_residual_insertions < 0:
        parser.error("--verified-max-residual-insertions must be non-negative")
    if args.verified_residual_min_gap is not None and (
        not math.isfinite(args.verified_residual_min_gap)
        or args.verified_residual_min_gap < 0.0
    ):
        parser.error("--verified-residual-min-gap must be finite and non-negative")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    try:
        fit_tolerance = resolve_fit_tolerance(checkpoint, args.fit_tolerance)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    model, model_config, legacy_checkpoint = build_model_from_checkpoint(checkpoint)
    hybrid_min_gap = (
        float(args.hybrid_min_gap)
        if args.hybrid_min_gap is not None
        else float(model_config.get("min_knot_gap", 1e-3))
    )
    verified_residual_min_gap = (
        float(args.verified_residual_min_gap)
        if args.verified_residual_min_gap is not None
        else float(model_config.get("min_knot_gap", 1e-3))
    )
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
    source_normalized_points = (source_points - normalized["center"]) / normalized[
        "scale"
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verified_refit_device = resolve_verified_refit_device(
        args.verified_refit_device,
        model_device=device,
    )
    points = normalized["points"].unsqueeze(0).to(device)
    model.to(device).eval()

    threshold = (
        args.activity_threshold
        if args.activity_threshold is not None
        else float(checkpoint.get("activity_threshold", 0.5))
    )
    model.set_activity_threshold(threshold)
    structure_mode = model_config.get("structure_mode", "hard_concrete")
    candidate_pruning, candidate_one_shot = candidate_mode_flags(
        checkpoint, model_config
    )
    try:
        deployment_mode = resolve_deployment_mode(
            args.deployment_mode,
            candidate_pruning=candidate_pruning,
            candidate_one_shot=candidate_one_shot,
        )
    except ValueError as error:
        parser.error(str(error))
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
        # One-shot deployment immediately performs a standard B-spline refit;
        # its final truncated-power surrogate is training-only.  Omitting that
        # solve preserves params/proposals/KeepMask/relocation exactly.
        output = (
            model.forward_deployment(points)
            if candidate_one_shot and hasattr(model, "forward_deployment")
            else model(points)
        )
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
            deployment_output["activity_gate"] = (output["activity"] >= threshold).to(
                output["activity"].dtype
            )
        else:
            deployment_output = output
        source_parameters = interpolate_parameters_by_chord(
            normalized["chord_params"].to(device),
            output["params"][0],
            source_chord.to(device),
        )
        full_resolution_points = source_normalized_points.unsqueeze(0).to(device)
        pruning_result: MinimalKnotPruningResult | None = None
        hybrid_result: HybridKnotSearchResult | None = None
        verified_result: VerifiedKnotRepairResult | None = None
        initial_learned_fit: HardGatedBSplineFit | None = None
        initial_learned_refit_time_ms: float | None = None
        hybrid_search_time_ms: float | None = None
        verified_repair_time_ms: float | None = None
        one_shot_mask: torch.Tensor | None = None
        adaptive_keep_threshold: torch.Tensor | None = None
        one_shot_selection_source: str | None = None
        if candidate_one_shot:
            masks, adaptive_thresholds, one_shot_selection_source = one_shot_selection(
                output
            )
            one_shot_mask = masks[0]
            adaptive_keep_threshold = adaptive_thresholds[0]

        if deployment_mode == "learned":
            (
                deployed,
                one_shot_mask,
                adaptive_keep_threshold,
                one_shot_selection_source,
            ) = refit_one_shot_full_resolution(
                output,
                source_parameters,
                full_resolution_points,
                degree=model.degree,
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
            )
        elif deployment_mode == "verified":
            assert one_shot_mask is not None
            raw_deployment_config = checkpoint.get("deployment_config", {})
            deployment_config = (
                raw_deployment_config if isinstance(raw_deployment_config, dict) else {}
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started_at = time.perf_counter()
            verified_result = run_verified_repair_full_resolution(
                output,
                source_parameters,
                full_resolution_points,
                one_shot_mask,
                source_chord_parameters=source_chord.to(device),
                fit_tolerance_rms=fit_tolerance,
                degree=model.degree,
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
                min_internal_knots=int(deployment_config.get("min_internal_knots", 0)),
                compact=args.verified_compact,
                hard_fallback=args.verified_hard_fallback,
                residual_fallback=args.verified_residual_fallback,
                max_residual_insertions=args.verified_max_residual_insertions,
                residual_min_gap=verified_residual_min_gap,
                refit_device=verified_refit_device,
                parameterization_policy=args.verified_parameterization,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            verified_repair_time_ms = 1e3 * (time.perf_counter() - started_at)
            deployed = verified_result_as_deployed_fit(verified_result)
        elif deployment_mode == "hybrid":
            # Keep a separately measured one-shot baseline for the deployment
            # report.  The actual slow search below independently refits every
            # state and uses the learned mask only as one deterministic start.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started_at = time.perf_counter()
            (
                initial_learned_fit,
                one_shot_mask,
                adaptive_keep_threshold,
                one_shot_selection_source,
            ) = refit_one_shot_full_resolution(
                output,
                source_parameters,
                full_resolution_points,
                degree=model.degree,
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            initial_learned_refit_time_ms = 1e3 * (time.perf_counter() - started_at)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started_at = time.perf_counter()
            hybrid_result = run_hybrid_search_full_resolution(
                output,
                source_parameters,
                full_resolution_points,
                one_shot_mask,
                fit_tolerance_rms=fit_tolerance,
                degree=model.degree,
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
                beam_width=args.hybrid_beam_width,
                branch_factor=args.hybrid_branch_factor,
                position_sweeps=args.hybrid_position_sweeps,
                position_grid_size=args.hybrid_position_grid_size,
                position_restarts=args.hybrid_position_restarts,
                position_refine_count_margin=(args.hybrid_position_refine_count_margin),
                position_refine_candidate_multiplier=(
                    args.hybrid_position_refine_candidate_multiplier
                ),
                min_gap=hybrid_min_gap,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            hybrid_search_time_ms = 1e3 * (time.perf_counter() - started_at)
            deployed = hybrid_result_as_deployed_fit(hybrid_result)
        elif deployment_mode == "hard":
            # The learned keep probability is diagnostic only.  Actual
            # deployment starts from every proposed knot, refits on the
            # original-resolution point cloud, and verifies every deletion.
            proposal_knots = output.get(
                "proposal_internal_knots", output["internal_knots"]
            )[0]
            pruning_result = prune_knots_to_rms_tolerance(
                source_parameters,
                full_resolution_points[0],
                proposal_knots,
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
    deployed_parameters = (
        verified_result.final_parameters
        if verified_result is not None
        else source_parameters
    )
    deployed_device = deployed.control_points.device
    dense_params = torch.linspace(
        0.0,
        1.0,
        400,
        dtype=deployed.control_points.dtype,
        device=deployed_device,
    )
    dense_normalized = deployed.spline.evaluate(dense_params).cpu()
    controls_normalized = deployed.control_points.cpu()
    dense_curve = dense_normalized * scale + center
    control_points = controls_normalized * scale + center
    scale_value = float(scale)
    fit_mse_normalized = float(deployed.fit_mse.cpu())
    fit_rmse_normalized = float(deployed.fit_rmse.cpu())
    fit_mse_original, fit_rmse_original = scale_normalized_fit_errors(
        fit_mse_normalized,
        fit_rmse_normalized,
        scale_value,
    )
    underdetermined_warning = deployed.control_points.shape[0] > source_points.shape[0]
    if underdetermined_warning:
        print(
            "WARNING: predicted control-point count exceeds the number of original "
            "observations; the refit is data-underdetermined and should not be "
            "treated as a reliable reconstruction.",
            flush=True,
        )
    refit_observations = full_resolution_points[0].to(deployed_device)
    normalized_start_distance = float(
        (deployed.reconstructed_points[0] - refit_observations[0]).norm().cpu()
    )
    normalized_end_distance = float(
        (deployed.reconstructed_points[-1] - refit_observations[-1]).norm().cpu()
    )
    checkpoint_count_selection = (
        "learned_one_shot_mask" if candidate_one_shot else count_selection
    )
    learned_relocation_shifts = torch.empty(0, dtype=points.dtype)
    learned_total_shifts = torch.empty(0, dtype=points.dtype)
    if candidate_one_shot:
        assert one_shot_mask is not None
        pre_relocation = output.get(
            "pre_relocation_candidate_positions",
            output.get("proposal_internal_knots", output["internal_knots"]),
        )[0]
        post_relocation = output.get(
            "deployment_internal_knots", output["internal_knots"]
        )[0]
        learned_relocation_shifts = (
            (post_relocation[one_shot_mask] - pre_relocation[one_shot_mask])
            .abs()
            .detach()
            .cpu()
        )
        proposal_positions = output.get(
            "proposal_internal_knots", output["internal_knots"]
        )[0]
        learned_total_shifts = (
            (post_relocation[one_shot_mask] - proposal_positions[one_shot_mask])
            .abs()
            .detach()
            .cpu()
        )

    report = {
        "schema_version": 6,
        "checkpoint": str(args.checkpoint),
        "point_cloud": str(args.point_cloud),
        "point_count": int(source_points.shape[0]),
        "model_point_count": int(model_points.shape[0]),
        "point_dim": int(source_points.shape[1]),
        "reversed": bool(args.reverse_points),
        "structure_mode": structure_mode,
        "deployment_mode_requested": args.deployment_mode,
        "deployment_mode_resolved": deployment_mode,
        "model_device": str(device),
        "verified_refit_device_requested": args.verified_refit_device,
        "verified_refit_device": str(verified_refit_device),
        "verified_refit_device_used": deployment_mode == "verified",
        "verified_parameterization_requested": args.verified_parameterization,
        "count_selection": (
            checkpoint_count_selection
            if deployment_mode in {"learned", "legacy"}
            else "verified_adaptive_exact_repair"
            if deployment_mode == "verified"
            else "learned_initialized_hybrid_mse_search"
            if deployment_mode == "hybrid"
            else "hard_rms_pruning"
            if deployment_mode == "hard"
            else count_selection
        ),
        "deployment_method": (
            "learned_one_shot_mask_single_standard_bspline_refit"
            if deployment_mode == "learned"
            else "verified_adaptive_repair_with_exact_standard_bspline_checks"
            if deployment_mode == "verified"
            else "learned_initialized_hybrid_mse_beam_position_search"
            if deployment_mode == "hybrid"
            else "hard_standard_bspline_rms_pruning"
            if deployment_mode == "hard"
            else "legacy_structure_selection"
        ),
        "degree": int(model.degree),
        "predicted_internal_knot_count": int(deployed.retained_count),
        "predicted_internal_knots": deployed.retained_internal_knots.cpu().tolist(),
        "open_knot_vector": deployed.spline.knot_vector.cpu().tolist(),
        "predicted_parameters": output["params"][0].detach().cpu().tolist(),
        "model_predicted_parameters": output["params"][0].detach().cpu().tolist(),
        "full_resolution_refit_parameters": (
            deployed_parameters.detach().cpu().tolist()
        ),
        "full_resolution_refit_parameterization": (
            verified_result.final_parameterization
            if verified_result is not None
            else "network_predicted"
        ),
        "full_resolution_refit_point_count": int(source_points.shape[0]),
        "sparse_input_warning": bool(sparse_input_warning),
        "data_underdetermined_warning": bool(underdetermined_warning),
        "control_points": control_points.tolist(),
        "fitted_curve": dense_curve.tolist(),
        "normalization_center": center.tolist(),
        "normalization_scale": scale_value,
        "normalized_fit_mse": fit_mse_normalized,
        "normalized_fit_rmse": fit_rmse_normalized,
        "original_scale_fit_mse": fit_mse_original,
        "original_scale_fit_rmse": fit_rmse_original,
        "normalized_start_endpoint_distance": normalized_start_distance,
        "normalized_end_endpoint_distance": normalized_end_distance,
        "original_scale_start_endpoint_distance": (
            normalized_start_distance * scale_value
        ),
        "original_scale_end_endpoint_distance": (normalized_end_distance * scale_value),
        "verified_deployment": None,
        "hybrid_deployment": None,
    }
    if candidate_one_shot:
        assert one_shot_mask is not None
        assert adaptive_keep_threshold is not None
        adaptive_value = float(adaptive_keep_threshold.detach().cpu())
        report["candidate_internal_knot_count"] = int(
            output["internal_knots"].shape[-1]
        )
        report["candidate_internal_knots"] = (
            output["internal_knots"][0].detach().cpu().tolist()
        )
        report["proposal_internal_knots"] = (
            output.get("proposal_internal_knots", output["internal_knots"])[0]
            .detach()
            .cpu()
            .tolist()
        )
        report["deployment_candidate_internal_knots"] = (
            output.get("deployment_internal_knots", output["internal_knots"])[0]
            .detach()
            .cpu()
            .tolist()
        )
        report["pre_relocation_candidate_internal_knots"] = (
            output.get(
                "pre_relocation_candidate_positions",
                output.get("proposal_internal_knots", output["internal_knots"]),
            )[0]
            .detach()
            .cpu()
            .tolist()
        )
        report["network_survivor_relocation"] = {
            "enabled": bool(
                getattr(model.pruning_head, "one_shot_survivor_relocation", False)
            ),
            "reference": "selected pre-relocation positions",
            "selected_knot_mean_absolute_shift": (
                float(learned_relocation_shifts.mean())
                if learned_relocation_shifts.numel()
                else 0.0
            ),
            "selected_knot_max_absolute_shift": (
                float(learned_relocation_shifts.max())
                if learned_relocation_shifts.numel()
                else 0.0
            ),
            "selected_knot_moved_fraction_at_1e-6": (
                float((learned_relocation_shifts > 1e-6).float().mean())
                if learned_relocation_shifts.numel()
                else 0.0
            ),
            "selected_knot_proposal_to_final_mean_absolute_shift": (
                float(learned_total_shifts.mean())
                if learned_total_shifts.numel()
                else 0.0
            ),
            "selected_knot_proposal_to_final_max_absolute_shift": (
                float(learned_total_shifts.max())
                if learned_total_shifts.numel()
                else 0.0
            ),
        }
        report["learned_keep_mask"] = one_shot_mask.detach().cpu().tolist()
        report["one_shot_selection_source"] = one_shot_selection_source
        report["one_shot_selection_policy"] = getattr(
            model.pruning_head, "one_shot_selection_policy", "threshold"
        )
        report["one_shot_safety_sigma"] = float(
            getattr(model.pruning_head, "one_shot_safety_sigma", 0.0)
        )
        report["one_shot_coverage_bins"] = int(
            getattr(model.pruning_head, "one_shot_coverage_bins", 0)
        )
        report["keep_probability_cutoff"] = 0.5
        report["activity_threshold_cli_used_for_selection"] = False
        report["fit_tolerance_used_for_selection"] = deployment_mode in {
            "verified",
            "hybrid",
            "hard",
        }
        report["adaptive_keep_logit_threshold"] = (
            adaptive_value if math.isfinite(adaptive_value) else None
        )
        report["fit_tolerance_normalized"] = fit_tolerance
        report["fit_tolerance_normalized_rms"] = fit_tolerance
        report["fit_tolerance_normalized_mse"] = fit_tolerance * fit_tolerance
        report["fit_tolerance_original_scale"] = fit_tolerance * scale_value
        report["fit_tolerance_original_scale_rms"] = fit_tolerance * scale_value
        report["fit_tolerance_original_scale_mse"] = (
            fit_tolerance * fit_tolerance * scale_value * scale_value
        )
        report["fit_tolerance_satisfied"] = (
            fit_mse_normalized <= fit_tolerance * fit_tolerance
        )
        report["standard_refit_count"] = (
            1
            if deployment_mode == "learned"
            else verified_result.direct_refit_count
            if verified_result is not None
            else None
        )
        report["deployment_standard_refit_count"] = report["standard_refit_count"]
        report["model_forward_internal_proxy_solves_counted_as_deployment_refits"] = (
            False
        )
        report["hard_pruning_used"] = deployment_mode in {"hybrid", "hard"} or (
            verified_result is not None
            and (verified_result.cleanup_used or verified_result.hard_fallback_used)
        )
    if verified_result is not None:
        assert verified_repair_time_ms is not None
        report["verified_deployment"] = {
            "method": "adaptive_exact_verify_add_back_residual_rescue",
            "fit_tolerance_normalized_rms": fit_tolerance,
            "fit_tolerance_normalized_mse": fit_tolerance * fit_tolerance,
            "selection_rule": (
                "learned_fast_path_else_confidence_prefix_exact_checks_then_"
                "optional_greedy_compaction"
            ),
            "final_source": verified_result.final_source,
            "threshold_satisfied": verified_result.threshold_satisfied,
            "learned_threshold_satisfied": (
                verified_result.learned_threshold_satisfied
            ),
            "fallback_used": verified_result.fallback_used,
            "hard_fallback_used": verified_result.hard_fallback_used,
            "residual_fallback_requested": bool(args.verified_residual_fallback),
            "residual_fallback_used": verified_result.residual_fallback_used,
            "max_residual_insertions": args.verified_max_residual_insertions,
            "residual_min_gap": verified_residual_min_gap,
            "parameterization_policy": verified_result.parameterization_policy,
            "learned_parameterization": verified_result.learned_parameterization,
            "final_parameterization": verified_result.final_parameterization,
            "parameterization_fallback_attempted": (
                verified_result.parameterization_fallback_attempted
            ),
            "parameterization_fallback_used": (
                verified_result.parameterization_fallback_used
            ),
            "parameterization_fallback_full_fit_mse": (
                verified_result.parameterization_fallback_full_fit_mse
            ),
            "parameterization_fallback_full_threshold_satisfied": (
                verified_result.parameterization_fallback_full_threshold_satisfied
            ),
            "final_refit_parameters": (
                verified_result.final_parameters.detach().cpu().tolist()
            ),
            "inserted_internal_knots": (
                verified_result.inserted_internal_knots.detach().cpu().tolist()
            ),
            "inserted_internal_knot_count": verified_result.inserted_count,
            "residual_insertion_fraction": float(verified_result.inserted_count > 0),
            "residual_inserted_knot_count_mean": float(verified_result.inserted_count),
            "residual_inserted_knot_count_max": verified_result.inserted_count,
            "residual_fallback_refit_count": (
                verified_result.residual_fallback_refit_count
            ),
            "compact_requested": bool(args.verified_compact),
            "cleanup_used": verified_result.cleanup_used,
            "hard_fallback_requested": bool(args.verified_hard_fallback),
            "model_device": str(device),
            "refit_device_requested": args.verified_refit_device,
            "refit_device": str(verified_refit_device),
            "learned_knot_count": int(one_shot_mask.sum().item()),
            "final_knot_count": verified_result.final_count,
            "learned_mse_normalized": float(verified_result.learned_fit.fit_mse),
            "learned_rms_normalized": float(verified_result.learned_fit.fit_rmse),
            "final_mse_normalized": float(verified_result.final_fit.fit_mse),
            "final_rms_normalized": float(verified_result.final_fit.fit_rmse),
            "retained_proposal_indices": (
                verified_result.retained_proposal_indices.detach().cpu().tolist()
            ),
            "retained_proposal_mask": (
                verified_result.retained_proposal_mask.detach().cpu().tolist()
            ),
            "deployment_candidate_count": (verified_result.deployment_candidate_count),
            "deployment_retained_mask": (
                verified_result.deployment_retained_mask.detach().cpu().tolist()
            ),
            "prefix_counts_evaluated": list(verified_result.prefix_counts_evaluated),
            "prefix_fit_count": len(verified_result.prefix_counts_evaluated),
            "direct_standard_refit_count": verified_result.direct_refit_count,
            "exact_fit_evaluation_count": verified_result.fit_evaluation_count,
            "fit_evaluation_count_semantics": (
                "direct prefix/materialization refits plus every exact batched "
                "single-deletion candidate state"
            ),
            "postprocess_time_ms": verified_repair_time_ms,
            "global_minimum_guaranteed": False,
        }
    if hybrid_result is not None:
        assert initial_learned_fit is not None
        assert initial_learned_refit_time_ms is not None
        assert hybrid_search_time_ms is not None
        initial_learned_mse = float(initial_learned_fit.fit_mse.cpu())
        initial_learned_rms = float(initial_learned_fit.fit_rmse.cpu())
        greedy_mse = float(hybrid_result.greedy_fit.fit_mse.cpu())
        greedy_rms = float(hybrid_result.greedy_fit.fit_rmse.cpu())
        final_mse = float(hybrid_result.final_fit.fit_mse.cpu())
        final_rms = float(hybrid_result.final_fit.fit_rmse.cpu())
        mse_tolerance = fit_tolerance * fit_tolerance
        report["hybrid_deployment"] = {
            "method": "learned_initialized_beam_deletion_with_position_refinement",
            "selection_rule": "feasible_first_then_minimum_k_then_minimum_mse",
            "mse_definition": "mean_i ||C(t_i)-Q_i||_2^2",
            "fit_tolerance_normalized_mse": mse_tolerance,
            "fit_tolerance_normalized_rms": fit_tolerance,
            "fit_tolerance_original_scale_mse": (
                mse_tolerance * scale_value * scale_value
            ),
            "fit_tolerance_original_scale_rms": fit_tolerance * scale_value,
            "initial_learned_knot_count": initial_learned_fit.retained_count,
            "initial_learned_mse_normalized": initial_learned_mse,
            "initial_learned_rms_normalized": initial_learned_rms,
            "initial_learned_threshold_satisfied": (
                initial_learned_mse <= mse_tolerance
            ),
            "greedy_knot_count": hybrid_result.greedy_count,
            "greedy_mse_normalized": greedy_mse,
            "greedy_rms_normalized": greedy_rms,
            "greedy_threshold_satisfied": (hybrid_result.greedy_threshold_satisfied),
            "final_knot_count": hybrid_result.final_count,
            "final_mse_normalized": final_mse,
            "final_rms_normalized": final_rms,
            "final_mse_original_scale": final_mse * scale_value * scale_value,
            "final_rms_original_scale": final_rms * scale_value,
            "threshold_satisfied": hybrid_result.threshold_satisfied,
            "retained_proposal_indices": (
                hybrid_result.retained_proposal_indices.detach().cpu().tolist()
            ),
            "retained_proposal_mask": (
                hybrid_result.retained_proposal_mask.detach().cpu().tolist()
            ),
            "greedy_retained_proposal_indices": (
                hybrid_result.greedy_retained_proposal_indices.detach().cpu().tolist()
            ),
            "greedy_retained_proposal_mask": (
                hybrid_result.greedy_retained_proposal_mask.detach().cpu().tolist()
            ),
            "visited_state_count": hybrid_result.visited_state_count,
            "search_refit_count": hybrid_result.refit_count,
            "search_fit_evaluation_count": hybrid_result.refit_count,
            "search_refit_count_semantics": (
                "exact candidate fit evaluations; batched deletion states are "
                "counted individually"
            ),
            "initial_learned_diagnostic_refit_count": 1,
            "search_time_ms": hybrid_search_time_ms,
            "initial_learned_diagnostic_refit_time_ms": (initial_learned_refit_time_ms),
            "total_postprocess_time_ms": (
                initial_learned_refit_time_ms + hybrid_search_time_ms
            ),
            "levels_explored": list(hybrid_result.levels_explored),
            "position_refined_counts": list(hybrid_result.position_refined_counts),
            "final_source": hybrid_result.final_source,
            "mean_absolute_proposal_to_final_position_shift": (
                hybrid_result.mean_absolute_position_shift
            ),
            "max_absolute_proposal_to_final_position_shift": (
                hybrid_result.max_absolute_position_shift
            ),
            "start_sources": list(hybrid_result.start_sources),
            "search_options": {
                "beam_width": args.hybrid_beam_width,
                "branch_factor": args.hybrid_branch_factor,
                "position_sweeps": args.hybrid_position_sweeps,
                "position_grid_size": args.hybrid_position_grid_size,
                "position_restarts": args.hybrid_position_restarts,
                "position_refine_count_margin": (
                    args.hybrid_position_refine_count_margin
                ),
                "position_refine_candidate_multiplier": (
                    args.hybrid_position_refine_candidate_multiplier
                ),
                "min_gap": hybrid_min_gap,
            },
            "global_minimum_guaranteed": False,
        }
    if pruning_result is not None:
        report["fit_tolerance_normalized"] = fit_tolerance
        report["fit_tolerance_normalized_rms"] = fit_tolerance
        report["fit_tolerance_normalized_mse"] = fit_tolerance * fit_tolerance
        report["fit_tolerance_original_scale"] = fit_tolerance * scale_value
        report["fit_tolerance_original_scale_rms"] = fit_tolerance * scale_value
        report["fit_tolerance_original_scale_mse"] = (
            fit_tolerance * fit_tolerance * scale_value * scale_value
        )
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
                "candidate_rmse_normalized": float(rejected_step.candidate_rmse.cpu()),
            }
            if rejected_step is not None
            else None
        )
    if "count_probabilities" in output:
        report["count_probabilities"] = (
            output["count_probabilities"][0].detach().cpu().tolist()
        )
        report["posterior_mode_internal_knot_count"] = int(
            output.get("count_mode_knot_count", output["predicted_knot_count"])[0]
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
    if candidate_one_shot:
        assert one_shot_mask is not None
        assert adaptive_keep_threshold is not None
        print(
            "  network one-shot selection: "
            f"{int(output['internal_knots'].shape[-1])} candidates -> "
            f"{int(one_shot_mask.sum())} retained"
        )
        print(f"  selection source: {one_shot_selection_source}")
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
            f"{float(adaptive_keep_threshold.detach().cpu()):.9e}"
        )
        print(
            "  selected pre->post |shift| mean/max/moved@1e-6: "
            f"{(float(learned_relocation_shifts.mean()) if learned_relocation_shifts.numel() else 0.0):.6e}/"
            f"{(float(learned_relocation_shifts.max()) if learned_relocation_shifts.numel() else 0.0):.6e}/"
            f"{(float((learned_relocation_shifts > 1e-6).float().mean()) if learned_relocation_shifts.numel() else 0.0):.3f}"
        )
        print(
            "  selected proposal->final |shift| mean/max: "
            f"{(float(learned_total_shifts.mean()) if learned_total_shifts.numel() else 0.0):.6e}/"
            f"{(float(learned_total_shifts.max()) if learned_total_shifts.numel() else 0.0):.6e}"
        )
        print(f"  normalized fit tolerance: {fit_tolerance:.9e}")
        print(
            f"  tolerance satisfied: {float(deployed.fit_rmse.cpu()) <= fit_tolerance}"
        )
        if deployment_mode == "learned":
            print(
                "  deployment standard B-spline refits: 1; hard pruning: False "
                "(network-forward proxy solves are not counted as deployment refits)"
            )
    if verified_result is not None:
        assert verified_repair_time_ms is not None
        assert one_shot_mask is not None
        print(
            "  model / verified refit device: "
            f"{device}/{verified_refit_device} "
            f"(requested={args.verified_refit_device})"
        )
        print(
            "  verified confidence repair: learned/final K="
            f"{int(one_shot_mask.sum().item())}/{verified_result.final_count}"
        )
        print(f"  verified final source: {verified_result.final_source}")
        print(
            "  verified parameterization policy/final/fallback-used: "
            f"{verified_result.parameterization_policy}/"
            f"{verified_result.final_parameterization}/"
            f"{verified_result.parameterization_fallback_used}"
        )
        print(
            "  learned/final threshold satisfied: "
            f"{verified_result.learned_threshold_satisfied}/"
            f"{verified_result.threshold_satisfied}"
        )
        print(
            "  fallback/cleanup/residual/hard-fallback used: "
            f"{verified_result.fallback_used}/"
            f"{verified_result.cleanup_used}/"
            f"{verified_result.residual_fallback_used}/"
            f"{verified_result.hard_fallback_used}"
        )
        print(
            "  residual inserted K/refits/max/min-gap: "
            f"{verified_result.inserted_count}/"
            f"{verified_result.residual_fallback_refit_count}/"
            f"{args.verified_max_residual_insertions}/"
            f"{verified_residual_min_gap:.6g}"
        )
        print(
            "  confidence prefix counts evaluated: "
            f"{list(verified_result.prefix_counts_evaluated)}"
        )
        print(
            "  direct refits/exact fit evaluations/postprocess time: "
            f"{verified_result.direct_refit_count}/"
            f"{verified_result.fit_evaluation_count}/"
            f"{verified_repair_time_ms:.2f} ms"
        )
        print("  global minimum guaranteed: False")
    if hybrid_result is not None:
        assert initial_learned_fit is not None
        assert hybrid_search_time_ms is not None
        print(
            "  slow hybrid MSE search: learned/greedy/final K="
            f"{initial_learned_fit.retained_count}/"
            f"{hybrid_result.greedy_count}/{hybrid_result.final_count}"
        )
        print(
            "  normalized MSE threshold/final: "
            f"{fit_tolerance * fit_tolerance:.9e}/"
            f"{float(hybrid_result.final_fit.fit_mse.cpu()):.9e}"
        )
        print(f"  threshold satisfied: {hybrid_result.threshold_satisfied}")
        print(
            "  proposal -> final knot-position |shift| mean/max: "
            f"{hybrid_result.mean_absolute_position_shift:.6e}/"
            f"{hybrid_result.max_absolute_position_shift:.6e}"
        )
        print(
            "  search visited/fit-evaluations/time: "
            f"{hybrid_result.visited_state_count}/"
            f"{hybrid_result.refit_count}/{hybrid_search_time_ms:.2f} ms"
        )
        print("  global minimum guaranteed: False")
    if pruning_result is not None:
        print(
            "  hard RMS pruning: "
            f"{pruning_result.initial_count} candidates -> "
            f"{pruning_result.final_count} retained"
        )
        print(f"  normalized fit threshold: {fit_tolerance:.9e}")
        print(f"  threshold satisfied: {pruning_result.threshold_satisfied}")
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
    print(f"  normalized MSE: {fit_mse_normalized:.9e}")
    print(f"  original-scale MSE: {fit_mse_original:.9e}")
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
