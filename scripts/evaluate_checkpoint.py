from __future__ import annotations

# Script entry points intentionally add ``src`` to sys.path before importing
# the local package so they also run from an unpacked repository.
# ruff: noqa: E402

import argparse
import json
import math
import statistics
import sys
import time
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
    V12_COUPLED_RELOCATION_OBJECTIVE_VERSION,
    build_model_from_checkpoint,
    migrate_loss_config,
)
from spline_fitting.data.synthetic import SyntheticCubicBSplineDataset
from spline_fitting.evaluation.bspline_inference import (
    HardGatedBSplineFit,
    refit_model_output_as_bsplines,
    select_count_conditioned_output_by_bic,
)
from spline_fitting.evaluation.knot_diagnostics import (
    match_internal_knots,
    warp_internal_knots_to_parameterization,
)
from spline_fitting.evaluation.hybrid_knot_search import (
    HybridKnotSearchResult,
    hybrid_minimal_knot_search,
)
from spline_fitting.evaluation.minimal_knot_pruning import (
    MinimalKnotPruningResult,
    prune_knots_to_rms_tolerance,
)
from spline_fitting.evaluation.verified_knot_repair import (
    VerifiedKnotRepairResult,
    verified_confidence_repair,
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
    # Small rank-revealing least-squares systems are more robust and usually
    # faster on CPU.  ``auto`` therefore selects CPU for both CPU and CUDA
    # model execution; callers can explicitly request ``model`` when desired.
    return torch.device("cpu")


def candidate_mode_flags(
    checkpoint: dict[str, object], model_config: dict[str, object]
) -> tuple[bool, bool]:
    """Return ``(candidate_family, one_shot)`` using stable features."""
    structure = str(model_config.get("structure_mode", "")).lower()
    objective = str(checkpoint.get("objective_version", "")).lower()
    one_shot = (
        structure == "candidate_pruning_one_shot"
        or "candidate_pruning_one_shot" in objective
        or objective == V12_COUPLED_RELOCATION_OBJECTIVE_VERSION
    )
    candidate_family = one_shot or structure == "candidate_pruning"
    return candidate_family, one_shot


def resolve_deployment_mode(requested: str, *, candidate_one_shot: bool) -> str:
    """Resolve the quality-deployment policy without changing legacy modes."""

    choices = {"checkpoint", "learned", "verified", "hybrid", "hard"}
    if requested not in choices:
        raise ValueError(
            "deployment mode must be one of checkpoint, learned, verified, hybrid, hard"
        )
    if requested == "checkpoint":
        return "learned" if candidate_one_shot else "checkpoint"
    if not candidate_one_shot:
        raise ValueError(
            f"deployment mode '{requested}' requires a one-shot candidate checkpoint"
        )
    return requested


def one_shot_selection(
    output: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Read the learned one-shot decision without applying a second threshold.

    ``keep_probability`` is already centered by the learned adaptive logit
    threshold, so its neutral deployment cutoff is 0.5.  The separately
    reported ``adaptive_keep_threshold`` lives in raw-importance space.
    """
    required = ("internal_knots", "keep_probability")
    missing = [name for name in required if name not in output]
    if missing:
        raise KeyError("one-shot output is missing: " + ", ".join(missing))
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


def refit_one_shot_output_batch(
    output: dict[str, torch.Tensor],
    points: torch.Tensor,
    *,
    degree: int,
    smoothness_weight: float,
    control_ridge: float,
) -> tuple[list[HardGatedBSplineFit], torch.Tensor, torch.Tensor, str]:
    """Apply the learned one-shot mask, then perform one standard B-spline refit."""
    mask, adaptive_threshold, source = one_shot_selection(output)
    selected_output = {
        "params": output["params"],
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
    )
    return deployed, mask, adaptive_threshold, source


def hybrid_deploy_output_batch(
    output: dict[str, torch.Tensor],
    points: torch.Tensor,
    *,
    fit_tolerance_rms: float,
    degree: int,
    smoothness_weight: float,
    control_ridge: float,
    min_internal_knots: int = 0,
    min_gap: float = 1e-4,
    beam_width: int = 4,
    branch_factor: int = 4,
    position_sweeps: int = 1,
    position_grid_size: int = 5,
    position_restarts: int = 1,
    position_refine_count_margin: int = 1,
    position_refine_candidate_multiplier: int = 4,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> tuple[
    list[HardGatedBSplineFit],
    list[HybridKnotSearchResult],
    list[float],
]:
    """Run learned-seeded hybrid MSE search for every curve in a batch."""

    for key in ("params", "internal_knots"):
        if key not in output:
            raise KeyError(f"one-shot output is missing {key}")
    if not math.isfinite(fit_tolerance_rms) or fit_tolerance_rms < 0.0:
        raise ValueError("fit_tolerance_rms must be finite and non-negative")
    learned_mask, _, _ = one_shot_selection(output)
    proposal_knots = output.get("proposal_internal_knots", output["internal_knots"])
    deployment_knots = output.get("deployment_internal_knots", output["internal_knots"])
    if not (
        proposal_knots.shape
        == deployment_knots.shape
        == learned_mask.shape
        == output["internal_knots"].shape
    ):
        raise ValueError("one-shot proposal/deployment/mask tensors must share [B,K]")
    if points.shape[0] != proposal_knots.shape[0]:
        raise ValueError("points and one-shot output must share a batch size")

    mse_tolerance = fit_tolerance_rms * fit_tolerance_rms
    deployed: list[HardGatedBSplineFit] = []
    results: list[HybridKnotSearchResult] = []
    elapsed_ms: list[float] = []
    for index in range(points.shape[0]):
        started_at = time.perf_counter()
        result = hybrid_minimal_knot_search(
            output["params"][index],
            points[index],
            proposal_knots[index],
            deployment_knots=deployment_knots[index],
            learned_mask=learned_mask[index],
            mse_tolerance=mse_tolerance,
            min_internal_knots=min_internal_knots,
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
            position_refine_candidate_multiplier=(position_refine_candidate_multiplier),
            min_gap=min_gap,
        )
        elapsed_ms.append(1e3 * (time.perf_counter() - started_at))
        if progress_total is not None:
            print(
                "Hybrid quality search "
                f"[{progress_offset + index + 1}/{progress_total}] | "
                f"greedy K={result.greedy_count} -> final K={result.final_count} | "
                f"MSE={float(result.final_fit.fit_mse):.5e} | "
                f"pass={result.threshold_satisfied} | "
                f"time={elapsed_ms[-1] / 1e3:.2f}s",
                flush=True,
            )
        retained_mask = result.retained_proposal_mask
        deployed.append(
            HardGatedBSplineFit(
                sample_index=index,
                candidate_count=int(proposal_knots.shape[1]),
                retained_count=result.final_count,
                retained_mask=retained_mask,
                hard_gate=retained_mask,
                spline=result.final_fit,
            )
        )
        results.append(result)
    return deployed, results, elapsed_ms


def verified_deploy_output_batch(
    output: dict[str, torch.Tensor],
    points: torch.Tensor,
    *,
    chord_parameters: torch.Tensor | None = None,
    parameterization_policy: str = "network",
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
) -> tuple[
    list[HardGatedBSplineFit],
    list[VerifiedKnotRepairResult],
    list[float],
]:
    """Verify one-shot fits and repair only failed curves with exact refits."""

    learned_mask, _, _ = one_shot_selection(output)
    proposal_knots = output.get("proposal_internal_knots", output["internal_knots"])
    deployment_knots = output.get("deployment_internal_knots", output["internal_knots"])
    keep_scores = output["keep_probability"]
    expected_shape = output["internal_knots"].shape
    if not (
        proposal_knots.shape
        == deployment_knots.shape
        == learned_mask.shape
        == keep_scores.shape
        == expected_shape
    ):
        raise ValueError("one-shot proposal/deployment/mask/scores must share [B,K]")
    if points.shape[0] != proposal_knots.shape[0]:
        raise ValueError("points and one-shot output must share a batch size")
    if parameterization_policy not in {"network", "chord-fallback", "chord"}:
        raise ValueError(
            "parameterization_policy must be one of network, chord-fallback, chord"
        )
    if parameterization_policy != "network":
        if chord_parameters is None:
            raise ValueError("chord_parameters are required outside network policy")
        if chord_parameters.shape != output["params"].shape:
            raise ValueError("chord_parameters and params must share shape [B,M]")

    target_device = (
        points.device if refit_device is None else torch.device(refit_device)
    )

    deployed: list[HardGatedBSplineFit] = []
    results: list[VerifiedKnotRepairResult] = []
    elapsed_ms: list[float] = []
    for index in range(points.shape[0]):
        started_at = time.perf_counter()
        result = verified_confidence_repair(
            output["params"][index].detach().to(target_device),
            points[index].detach().to(target_device),
            proposal_knots[index].detach().to(target_device),
            deployment_knots[index].detach().to(target_device),
            learned_mask[index].detach().to(target_device),
            keep_scores[index].detach().to(target_device),
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
                chord_parameters[index].detach().to(target_device)
                if chord_parameters is not None
                else None
            ),
            parameterization_policy=parameterization_policy,
        )
        elapsed_ms.append(1e3 * (time.perf_counter() - started_at))
        deployed.append(
            HardGatedBSplineFit(
                sample_index=index,
                candidate_count=result.deployment_candidate_count,
                retained_count=result.final_count,
                retained_mask=result.deployment_retained_mask,
                hard_gate=result.deployment_retained_mask,
                spline=result.final_fit,
            )
        )
        results.append(result)
    return deployed, results, elapsed_ms


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
    proposal_knots = output.get("proposal_internal_knots", output["internal_knots"])
    for index in range(points.shape[0]):
        result = prune_knots_to_rms_tolerance(
            output["params"][index],
            points[index],
            proposal_knots[index],
            error_tolerance=error_tolerance,
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
            interpolate_endpoints=True,
        )
        results.append(result)
        deployed.append(_pruning_result_as_deployed_fit(result, index))
    return deployed, results


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
        candidate_coverage_tolerances=tuple(
            float(value) for value in config.get("candidate_coverage_tolerances", ())
        ),
        position_aware_distribution=bool(
            config.get("position_aware_distribution", False)
        ),
        joint_position_supervision=bool(
            config.get("joint_position_supervision", False)
        ),
        teacher_relocation_supervision=bool(
            config.get("teacher_relocation_supervision", False)
        ),
        teacher_survivor_spacing_weight=float(
            config.get("teacher_survivor_spacing_weight", 0.25)
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
        "--torch-num-threads",
        type=int,
        default=None,
        help=(
            "CPU intra-op threads. Small exact spline solves are often faster "
            "with 4 threads than with all host cores."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20000,
        help="Independent synthetic test seed (training defaults: train=42, val=10000).",
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
            "Normalized RMS bound. In learned mode it is reporting-only; verified, "
            "hybrid and hard modes use it as an exact standard-B-spline acceptance "
            "constraint. Defaults to the checkpoint deployment error tolerance, "
            "then canonical label tolerance."
        ),
    )
    parser.add_argument(
        "--run-hard-diagnostic",
        action="store_true",
        help=(
            "For v8-v12 only, additionally run the offline greedy hard-pruning "
            "teacher for comparison. It never replaces one-shot deployment."
        ),
    )
    parser.add_argument(
        "--deployment-mode",
        choices=("checkpoint", "learned", "verified", "hybrid", "hard"),
        default="checkpoint",
        help=(
            "Deployment policy. One-shot checkpoints resolve checkpoint to learned; "
            "verified exact-refit checks the fast path and repairs failed curves; "
            "hybrid performs learned-seeded MSE beam search; hard uses traditional "
            "proposal-only greedy pruning."
        ),
    )
    parser.add_argument(
        "--verified-compact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After confidence add-back reaches the fit threshold, greedily remove "
            "redundant knots from that smaller feasible set. Disable for the "
            "lowest-latency verified path."
        ),
    )
    parser.add_argument(
        "--verified-hard-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If even the complete proposal prefix misses the threshold, run the "
            "traditional proposal-only greedy fallback."
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
            "Exact-refit parameterization. network preserves predicted parameters; "
            "chord-fallback changes domain only after the complete network proposal "
            "fails; chord uses deterministic chord length throughout verified repair."
        ),
    )
    parser.add_argument("--hybrid-beam-width", type=int, default=4)
    parser.add_argument(
        "--hybrid-branch-factor",
        type=int,
        default=4,
        help="Deletions expanded per beam parent; zero expands all deletions.",
    )
    parser.add_argument("--hybrid-position-sweeps", type=int, default=2)
    parser.add_argument("--hybrid-position-grid-size", type=int, default=7)
    parser.add_argument("--hybrid-position-restarts", type=int, default=2)
    parser.add_argument("--hybrid-position-refine-count-margin", type=int, default=1)
    parser.add_argument(
        "--hybrid-position-refine-candidate-multiplier",
        type=int,
        default=4,
        help=(
            "Near the deletion boundary, refine this many beam-widths before "
            "truncating the beam. Larger values strengthen delete/move coupling."
        ),
    )
    parser.add_argument("--hybrid-min-gap", type=float, default=None)
    parser.add_argument(
        "--one-shot-selection-policy",
        choices=("checkpoint", "threshold", "mass_topk"),
        default="checkpoint",
        help=(
            "Override the LearnedKeep mask policy for a one-shot checkpoint. "
            "checkpoint preserves the policy recorded by the model."
        ),
    )
    parser.add_argument(
        "--one-shot-safety-sigma",
        type=float,
        default=None,
        help=(
            "Override the mass_topk Bernoulli uncertainty reserve. The model's "
            "checkpoint value is used when omitted."
        ),
    )
    parser.add_argument(
        "--one-shot-coverage-bins",
        type=int,
        default=None,
        help=(
            "Override the number of parameter-domain coverage anchors used by "
            "mass_topk. The checkpoint value is used when omitted."
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
    if args.torch_num_threads is not None:
        if args.torch_num_threads <= 0:
            parser.error("--torch-num-threads must be positive")
        torch.set_num_threads(args.torch_num_threads)
    if args.knot_tolerance < 0.0:
        parser.error("--knot-tolerance must be non-negative")
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
    threshold = (
        args.activity_threshold
        if args.activity_threshold is not None
        else float(checkpoint.get("activity_threshold", 0.5))
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verified_refit_device = resolve_verified_refit_device(
        args.verified_refit_device,
        model_device=device,
    )
    model, model_config, legacy_checkpoint = build_model_from_checkpoint(checkpoint)
    verified_residual_min_gap = (
        float(args.verified_residual_min_gap)
        if args.verified_residual_min_gap is not None
        else float(model_config.get("min_knot_gap", 1e-3))
    )
    model.set_activity_threshold(threshold)
    model.to(device).eval()
    structure_mode = model_config.get("structure_mode", "hard_concrete")
    candidate_pruning, candidate_one_shot = candidate_mode_flags(
        checkpoint, model_config
    )
    try:
        deployment_mode = resolve_deployment_mode(
            args.deployment_mode,
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
    candidate_hard_v7 = candidate_pruning and not candidate_one_shot
    if args.run_hard_diagnostic and not candidate_one_shot:
        parser.error("--run-hard-diagnostic requires a v8-v12 one-shot checkpoint")
    if args.run_hard_diagnostic and deployment_mode == "hard":
        parser.error("--run-hard-diagnostic is redundant when --deployment-mode hard")
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
        size=args.num_samples,
        seed=args.seed,
        **dataset_config,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)

    if candidate_pruning:
        raw_loss_config = checkpoint.get("loss_config", {})
        loss_config = raw_loss_config if isinstance(raw_loss_config, dict) else {}
        assumed_loss_config = not bool(loss_config)
        loss_fn = candidate_loss_from_checkpoint(
            checkpoint,
            disable_exact_teacher=candidate_one_shot,
        ).to(device)
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
        {value: [] for value in sorted(set(args.threshold_sweep + [threshold]))}
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
    parameter_feedback_squared_shift = 0.0
    parameter_feedback_values = 0
    parameter_feedback_gap_shift_absolute = 0.0
    parameter_feedback_gap_shift_values = 0
    parameter_feedback_chord_blend_sum = 0.0
    parameter_feedback_chord_blend_values = 0
    pruning_initial_rms: list[float] = []
    pruning_accepted_deletions: list[int] = []
    pruning_threshold_satisfied: list[bool] = []
    pruning_first_rejected_rms: list[float] = []
    candidate_recall_tolerances = (0.005, 0.01, 0.02, 0.05)
    candidate_proposal_matched = {
        tolerance: 0 for tolerance in candidate_recall_tolerances
    }
    candidate_proposal_true = {
        tolerance: 0 for tolerance in candidate_recall_tolerances
    }
    candidate_proposal_error_sum = {
        tolerance: 0.0 for tolerance in candidate_recall_tolerances
    }
    knot_stage_accumulators = {
        stage: {
            tolerance: {
                "matched_count": 0,
                "predicted_count": 0,
                "true_count": 0,
                "matched_error_sum": 0.0,
            }
            for tolerance in candidate_recall_tolerances
        }
        for stage in (
            "proposal_full",
            "selected_pre_update",
            "selected_pre_relocation",
            "deployment_post_update",
        )
    }

    def accumulate_knot_stage(
        stage: str,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        for stage_tolerance in candidate_recall_tolerances:
            stage_matching = match_internal_knots(
                predicted,
                target,
                tolerance=stage_tolerance,
            )
            accumulator = knot_stage_accumulators[stage][stage_tolerance]
            accumulator["matched_count"] += stage_matching.matched_count
            accumulator["predicted_count"] += stage_matching.predicted_count
            accumulator["true_count"] += stage_matching.true_count
            if stage_matching.matched_count:
                accumulator["matched_error_sum"] += (
                    stage_matching.matched_mae * stage_matching.matched_count
                )

    keep_probability_values: list[torch.Tensor] = []
    keep_probability_ranges: list[torch.Tensor] = []
    learned_keep_masks: list[torch.Tensor] = []
    one_shot_adaptive_thresholds: list[torch.Tensor] = []
    one_shot_selection_sources: set[str] = set()
    one_shot_threshold_satisfied: list[bool] = []
    one_shot_hard_diagnostic_counts: list[int] = []
    one_shot_hard_diagnostic_rms: list[float] = []
    one_shot_hard_diagnostic_satisfied: list[bool] = []
    learned_baseline_counts: list[int] = []
    learned_baseline_mse: list[float] = []
    learned_baseline_satisfied: list[bool] = []
    learned_relocation_abs_shifts: list[float] = []
    learned_total_abs_shifts: list[float] = []
    learned_relocation_sample_mean_shifts: list[float] = []
    learned_relocation_sample_max_shifts: list[float] = []
    hybrid_greedy_counts: list[int] = []
    hybrid_greedy_mse: list[float] = []
    hybrid_greedy_satisfied: list[bool] = []
    hybrid_final_counts: list[int] = []
    hybrid_final_mse: list[float] = []
    hybrid_final_satisfied: list[bool] = []
    hybrid_refit_counts: list[int] = []
    hybrid_visited_state_counts: list[int] = []
    hybrid_level_counts: list[int] = []
    hybrid_elapsed_ms: list[float] = []
    hybrid_mean_abs_position_shifts: list[float] = []
    hybrid_max_abs_position_shifts: list[float] = []
    hybrid_progress_completed = 0
    verified_final_counts: list[int] = []
    verified_final_mse: list[float] = []
    verified_final_satisfied: list[bool] = []
    verified_fallback_used: list[bool] = []
    verified_hard_fallback_used: list[bool] = []
    verified_cleanup_used: list[bool] = []
    verified_residual_fallback_used: list[bool] = []
    verified_inserted_counts: list[int] = []
    verified_residual_refit_counts: list[int] = []
    verified_direct_refit_counts: list[int] = []
    verified_fit_evaluation_counts: list[int] = []
    verified_prefix_counts: list[int] = []
    verified_elapsed_ms: list[float] = []
    verified_sources: dict[str, int] = defaultdict(int)
    verified_final_parameterizations: dict[str, int] = defaultdict(int)
    verified_parameterization_fallback_attempted: list[bool] = []
    verified_parameterization_fallback_used: list[bool] = []
    verified_parameterization_fallback_full_mse: list[float] = []

    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device)
            chord_params = batch["chord_params"].to(device)
            true_params = batch["true_params"].to(device)
            true_knots = batch["true_internal_knots"].to(device)
            true_mask = batch["true_internal_knot_mask"].to(device)
            output = model(points)
            batch_size = points.shape[0]
            proposal_params = output.get("proposal_params")
            if proposal_params is not None:
                parameter_shift = (output["params"] - proposal_params).detach()
                parameter_feedback_squared_shift += float(
                    parameter_shift.square().sum().cpu()
                )
                parameter_feedback_values += parameter_shift.numel()
            feedback_gap_shift = output.get("parameter_feedback_gap_logit_delta")
            if feedback_gap_shift is not None:
                detached_gap_shift = feedback_gap_shift.detach()
                parameter_feedback_gap_shift_absolute += float(
                    detached_gap_shift.abs().sum().cpu()
                )
                parameter_feedback_gap_shift_values += detached_gap_shift.numel()
            feedback_chord_blend = output.get("parameter_feedback_chord_blend_weight")
            if feedback_chord_blend is not None:
                detached_chord_blend = feedback_chord_blend.detach()
                parameter_feedback_chord_blend_sum += float(
                    detached_chord_blend.sum().cpu()
                )
                parameter_feedback_chord_blend_values += detached_chord_blend.numel()
            deployment_parameters = [
                output["params"][index].detach().cpu() for index in range(batch_size)
            ]
            losses = loss_fn(
                output,
                points,
                chord_params=chord_params,
                true_params=true_params,
                true_internal_knots=true_knots,
                true_internal_knot_mask=true_mask,
                activity_threshold=threshold,
            )
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
                        batch_probabilities * batch_probabilities.clamp_min(1e-12).log()
                    ).sum(dim=-1)
                )
                count_max_probabilities.append(batch_probabilities.amax(dim=-1))
                for target, predicted in zip(
                    batch_true.tolist(), batch_predicted.tolist()
                ):
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
                        keep_probability.amax(dim=-1) - keep_probability.amin(dim=-1)
                    ).cpu()
                )
                deployment_output = output
                for index in range(batch_size):
                    target = true_knots[index, true_mask[index]].cpu()
                    predicted_parameters = output["params"][index].detach().cpu()
                    proposal_parameters = (
                        output.get("proposal_params", output["params"])[index]
                        .detach()
                        .cpu()
                    )
                    true_parameters = true_params[index].detach().cpu()
                    # Joint-refinement checkpoints keep the immutable proposal
                    # set separate from the Keep-conditioned deployment locations.
                    refined_candidates = (
                        output.get("proposal_internal_knots", output["internal_knots"])[
                            index
                        ]
                        .detach()
                        .cpu()
                    )
                    refined_candidates_true_domain = (
                        warp_internal_knots_to_parameterization(
                            refined_candidates,
                            proposal_parameters,
                            true_parameters,
                        )
                    )
                    accumulate_knot_stage(
                        "proposal_full", refined_candidates_true_domain, target
                    )
                    for proposal_tolerance in candidate_recall_tolerances:
                        proposal_matching = match_internal_knots(
                            refined_candidates_true_domain,
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
                    if candidate_one_shot:
                        learned_stage_mask = (
                            output.get(
                                "final_hard_keep_mask",
                                output["keep_probability"] >= 0.5,
                            )[index]
                            .detach()
                            .cpu()
                            .to(torch.bool)
                        )
                        pre_relocation_candidates = (
                            output.get(
                                "pre_relocation_candidate_positions",
                                output.get(
                                    "proposal_internal_knots",
                                    output["internal_knots"],
                                ),
                            )[index]
                            .detach()
                            .cpu()
                        )
                        pre_relocation_true_domain = (
                            warp_internal_knots_to_parameterization(
                                pre_relocation_candidates[learned_stage_mask],
                                proposal_parameters,
                                true_parameters,
                            )
                        )
                        accumulate_knot_stage(
                            "selected_pre_update",
                            refined_candidates_true_domain[learned_stage_mask],
                            target,
                        )
                        accumulate_knot_stage(
                            "selected_pre_relocation",
                            pre_relocation_true_domain,
                            target,
                        )
                        deployment_candidates = (
                            output["internal_knots"][index].detach().cpu()
                        )
                        source_deployment_candidates = (
                            output.get(
                                "parameter_feedback_source_internal_knots",
                                output["internal_knots"],
                            )[index]
                            .detach()
                            .cpu()
                        )
                        deployment_true_domain = (
                            warp_internal_knots_to_parameterization(
                                deployment_candidates[learned_stage_mask],
                                predicted_parameters,
                                true_parameters,
                            )
                        )
                        accumulate_knot_stage(
                            "deployment_post_update",
                            deployment_true_domain,
                            target,
                        )
                        selected_shift = (
                            source_deployment_candidates[learned_stage_mask]
                            - pre_relocation_candidates[learned_stage_mask]
                        ).abs()
                        selected_total_shift = (
                            source_deployment_candidates[learned_stage_mask]
                            - refined_candidates[learned_stage_mask]
                        ).abs()
                        if selected_shift.numel():
                            learned_relocation_abs_shifts.extend(
                                float(value) for value in selected_shift
                            )
                            learned_relocation_sample_mean_shifts.append(
                                float(selected_shift.mean())
                            )
                            learned_relocation_sample_max_shifts.append(
                                float(selected_shift.max())
                            )
                            learned_total_abs_shifts.extend(
                                float(value) for value in selected_total_shift
                            )
                        else:
                            learned_relocation_sample_mean_shifts.append(0.0)
                            learned_relocation_sample_max_shifts.append(0.0)
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
                    deployment_output["activity_gate"] = (activity >= threshold).to(
                        activity.dtype
                    )
                else:
                    deployment_output = output

            if candidate_hard_v7:
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
            elif candidate_one_shot:
                learned_mask, adaptive_threshold, selection_source = one_shot_selection(
                    output
                )
                learned_keep_masks.append(learned_mask.cpu())
                one_shot_adaptive_thresholds.append(adaptive_threshold.cpu())
                one_shot_selection_sources.add(selection_source)
                learned_deployed: list[HardGatedBSplineFit] | None = None
                if deployment_mode != "verified":
                    learned_deployed, _, _, _ = refit_one_shot_output_batch(
                        output,
                        points,
                        degree=model.degree,
                        smoothness_weight=args.smoothness_weight,
                        control_ridge=args.control_ridge,
                    )
                    learned_baseline_counts.extend(
                        item.retained_count for item in learned_deployed
                    )
                    learned_baseline_mse.extend(
                        float(item.fit_mse) for item in learned_deployed
                    )
                    learned_baseline_satisfied.extend(
                        float(item.fit_mse) <= fit_tolerance * fit_tolerance
                        for item in learned_deployed
                    )
                if deployment_mode == "learned":
                    if learned_deployed is None:
                        raise RuntimeError(
                            "learned deployment refit was not materialized"
                        )
                    deployed = learned_deployed
                elif deployment_mode == "verified":
                    raw_deployment_config = checkpoint.get("deployment_config", {})
                    deployment_config = (
                        raw_deployment_config
                        if isinstance(raw_deployment_config, dict)
                        else {}
                    )
                    deployed, verified_batch, batch_elapsed_ms = (
                        verified_deploy_output_batch(
                            output,
                            points,
                            chord_parameters=chord_params,
                            parameterization_policy=args.verified_parameterization,
                            fit_tolerance_rms=fit_tolerance,
                            degree=model.degree,
                            smoothness_weight=args.smoothness_weight,
                            control_ridge=args.control_ridge,
                            min_internal_knots=int(
                                deployment_config.get("min_internal_knots", 0)
                            ),
                            compact=args.verified_compact,
                            hard_fallback=args.verified_hard_fallback,
                            residual_fallback=args.verified_residual_fallback,
                            max_residual_insertions=(
                                args.verified_max_residual_insertions
                            ),
                            residual_min_gap=verified_residual_min_gap,
                            refit_device=verified_refit_device,
                        )
                    )
                    # The verifier's first exact refit is exactly the learned
                    # KeepMask baseline.  Reuse it instead of performing an
                    # otherwise duplicate standard B-spline solve before the
                    # verified path.
                    learned_baseline_counts.extend(
                        int(value) for value in learned_mask.sum(dim=-1).tolist()
                    )
                    learned_baseline_mse.extend(
                        float(item.learned_fit.fit_mse) for item in verified_batch
                    )
                    learned_baseline_satisfied.extend(
                        float(item.learned_fit.fit_mse) <= fit_tolerance * fit_tolerance
                        for item in verified_batch
                    )
                    verified_final_counts.extend(
                        item.final_count for item in verified_batch
                    )
                    verified_final_mse.extend(
                        float(item.final_fit.fit_mse) for item in verified_batch
                    )
                    verified_final_satisfied.extend(
                        item.threshold_satisfied for item in verified_batch
                    )
                    verified_fallback_used.extend(
                        item.fallback_used for item in verified_batch
                    )
                    verified_hard_fallback_used.extend(
                        item.hard_fallback_used for item in verified_batch
                    )
                    verified_cleanup_used.extend(
                        item.cleanup_used for item in verified_batch
                    )
                    verified_residual_fallback_used.extend(
                        item.residual_fallback_used for item in verified_batch
                    )
                    verified_inserted_counts.extend(
                        item.inserted_count for item in verified_batch
                    )
                    verified_residual_refit_counts.extend(
                        item.residual_fallback_refit_count for item in verified_batch
                    )
                    verified_direct_refit_counts.extend(
                        item.direct_refit_count for item in verified_batch
                    )
                    verified_fit_evaluation_counts.extend(
                        item.fit_evaluation_count for item in verified_batch
                    )
                    verified_prefix_counts.extend(
                        len(item.prefix_counts_evaluated) for item in verified_batch
                    )
                    verified_elapsed_ms.extend(batch_elapsed_ms)
                    for item in verified_batch:
                        verified_sources[item.final_source] += 1
                        verified_final_parameterizations[
                            item.final_parameterization
                        ] += 1
                        verified_parameterization_fallback_attempted.append(
                            item.parameterization_fallback_attempted
                        )
                        verified_parameterization_fallback_used.append(
                            item.parameterization_fallback_used
                        )
                        if item.parameterization_fallback_full_fit_mse is not None:
                            verified_parameterization_fallback_full_mse.append(
                                item.parameterization_fallback_full_fit_mse
                            )
                    deployment_parameters = [
                        item.final_parameters.detach().cpu() for item in verified_batch
                    ]
                elif deployment_mode == "hard":
                    deployed, _ = prune_candidate_output_batch(
                        output,
                        points,
                        error_tolerance=fit_tolerance,
                        degree=model.degree,
                        smoothness_weight=args.smoothness_weight,
                        control_ridge=args.control_ridge,
                    )
                elif deployment_mode == "hybrid":
                    raw_deployment_config = checkpoint.get("deployment_config", {})
                    deployment_config = (
                        raw_deployment_config
                        if isinstance(raw_deployment_config, dict)
                        else {}
                    )
                    deployed, hybrid_batch, batch_elapsed_ms = (
                        hybrid_deploy_output_batch(
                            output,
                            points,
                            fit_tolerance_rms=fit_tolerance,
                            degree=model.degree,
                            smoothness_weight=args.smoothness_weight,
                            control_ridge=args.control_ridge,
                            min_internal_knots=int(
                                deployment_config.get("min_internal_knots", 0)
                            ),
                            min_gap=(
                                float(args.hybrid_min_gap)
                                if args.hybrid_min_gap is not None
                                else float(model_config.get("min_knot_gap", 1e-3))
                            ),
                            beam_width=args.hybrid_beam_width,
                            branch_factor=args.hybrid_branch_factor,
                            position_sweeps=args.hybrid_position_sweeps,
                            position_grid_size=args.hybrid_position_grid_size,
                            position_restarts=args.hybrid_position_restarts,
                            position_refine_count_margin=(
                                args.hybrid_position_refine_count_margin
                            ),
                            position_refine_candidate_multiplier=(
                                args.hybrid_position_refine_candidate_multiplier
                            ),
                            progress_offset=hybrid_progress_completed,
                            progress_total=args.num_samples,
                        )
                    )
                    hybrid_progress_completed += len(hybrid_batch)
                    hybrid_greedy_counts.extend(
                        item.greedy_count for item in hybrid_batch
                    )
                    hybrid_greedy_mse.extend(
                        float(item.greedy_fit.fit_mse) for item in hybrid_batch
                    )
                    hybrid_greedy_satisfied.extend(
                        item.greedy_threshold_satisfied for item in hybrid_batch
                    )
                    hybrid_final_counts.extend(
                        item.final_count for item in hybrid_batch
                    )
                    hybrid_final_mse.extend(
                        float(item.final_fit.fit_mse) for item in hybrid_batch
                    )
                    hybrid_final_satisfied.extend(
                        item.threshold_satisfied for item in hybrid_batch
                    )
                    hybrid_refit_counts.extend(
                        item.refit_count for item in hybrid_batch
                    )
                    hybrid_visited_state_counts.extend(
                        item.visited_state_count for item in hybrid_batch
                    )
                    hybrid_level_counts.extend(
                        len(item.levels_explored) for item in hybrid_batch
                    )
                    hybrid_elapsed_ms.extend(batch_elapsed_ms)
                    hybrid_mean_abs_position_shifts.extend(
                        item.mean_absolute_position_shift for item in hybrid_batch
                    )
                    hybrid_max_abs_position_shifts.extend(
                        item.max_absolute_position_shift for item in hybrid_batch
                    )
                else:
                    raise RuntimeError(
                        f"unexpected resolved one-shot deployment mode: {deployment_mode}"
                    )
                one_shot_threshold_satisfied.extend(
                    float(item.fit_mse) <= fit_tolerance * fit_tolerance
                    for item in deployed
                )
                if args.run_hard_diagnostic:
                    _, diagnostic_batch = prune_candidate_output_batch(
                        output,
                        points,
                        error_tolerance=fit_tolerance,
                        degree=model.degree,
                        smoothness_weight=args.smoothness_weight,
                        control_ridge=args.control_ridge,
                    )
                    one_shot_hard_diagnostic_counts.extend(
                        item.final_count for item in diagnostic_batch
                    )
                    one_shot_hard_diagnostic_rms.extend(
                        float(item.final_fit.fit_rmse) for item in diagnostic_batch
                    )
                    one_shot_hard_diagnostic_satisfied.extend(
                        item.threshold_satisfied for item in diagnostic_batch
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
                    bspline_rank_deficient.append(
                        item.spline.solver_rank < control_count
                    )
                observed_points = points[index].to(item.reconstructed_points.device)
                bspline_start_endpoint_distances.append(
                    float((item.reconstructed_points[0] - observed_points[0]).norm())
                )
                bspline_end_endpoint_distances.append(
                    float((item.reconstructed_points[-1] - observed_points[-1]).norm())
                )

            parameter_difference = output["params"].cpu() - batch["true_params"]
            true_parameter_squared_error += float(parameter_difference.pow(2).sum())
            true_parameter_values += parameter_difference.numel()
            for index, item in enumerate(deployed):
                target = batch["true_internal_knots"][index][
                    batch["true_internal_knot_mask"][index]
                ]
                deployed_knots_true_domain = warp_internal_knots_to_parameterization(
                    item.retained_internal_knots.detach().cpu(),
                    deployment_parameters[index],
                    batch["true_params"][index],
                )
                matching = match_internal_knots(
                    deployed_knots_true_domain,
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
            "teacher_risk": loss_fn.weights.teacher_risk
            * mean_losses["teacher_risk_loss"],
            "teacher_ranking": loss_fn.weights.teacher_ranking
            * mean_losses["teacher_ranking_loss"],
            "teacher_distribution": loss_fn.weights.teacher_distribution
            * mean_losses["teacher_distribution_loss"],
            "teacher_critical_recall": loss_fn.weights.teacher_critical_recall
            * mean_losses["teacher_critical_recall_loss"],
            "teacher_false_positive": loss_fn.weights.teacher_false_positive
            * mean_losses["teacher_false_positive_loss"],
            "teacher_count": loss_fn.weights.teacher_count
            * mean_losses["teacher_count_loss"],
            "policy_count": loss_fn.weights.policy_count
            * mean_losses["policy_count_loss"],
            "canonical_selection": loss_fn.weights.canonical_selection
            * mean_losses["canonical_selection_loss"],
            "complexity": loss_fn.weights.complexity * mean_losses["complexity_loss"],
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
    bspline_rms_values = torch.tensor(bspline_fit_losses).sqrt()
    bspline_rms_mean = float(bspline_rms_values.mean())
    bspline_rms_pooled = math.sqrt(bspline_fit_mean)
    bspline_rms_p95 = float(torch.quantile(bspline_rms_values, 0.95))
    bspline_rms_max = float(bspline_rms_values.max())
    bspline_coordinate_mean = sum(bspline_coordinate_losses) / len(
        bspline_coordinate_losses
    )
    parameter_rmse = math.sqrt(
        true_parameter_squared_error / max(true_parameter_values, 1)
    )
    parameter_feedback_report = {
        "enabled": bool(model_config.get("parameter_feedback_fusion", False)),
        "parameter_rmse_from_proposal": math.sqrt(
            parameter_feedback_squared_shift / max(parameter_feedback_values, 1)
        ),
        "gap_logit_delta_mean_absolute": (
            parameter_feedback_gap_shift_absolute
            / max(parameter_feedback_gap_shift_values, 1)
        ),
        "chord_blend_weight_mean": (
            parameter_feedback_chord_blend_sum
            / max(parameter_feedback_chord_blend_values, 1)
        ),
    }
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
        count_accuracy = float(
            (predicted_count_tensor == true_count_tensor).float().mean()
        )
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
        if candidate_one_shot:
            network_keep_count = (
                torch.cat(learned_keep_masks).sum(dim=-1).to(torch.long)
            )
        else:
            network_keep_count = (
                (all_keep_probability >= threshold).sum(dim=-1).to(torch.long)
            )
        predicted_count_tensor = retained
        count_accuracy = float((retained == true_count_tensor).float().mean())
        count_mae = float((retained - true_count_tensor).abs().float().mean())
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
            "role": (
                "one_shot_deployment_selector"
                if candidate_one_shot
                else "diagnostic_only_not_used_for_deployment"
            ),
            "threshold": 0.5 if candidate_one_shot else threshold,
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

    def finalize_stage_metric(values: dict[str, int | float]) -> dict[str, object]:
        matched = int(values["matched_count"])
        predicted = int(values["predicted_count"])
        true = int(values["true_count"])
        precision_value = matched / predicted if predicted else 0.0
        recall_value = matched / true if true else 0.0
        f1_value = (
            2.0 * precision_value * recall_value / (precision_value + recall_value)
            if precision_value + recall_value
            else 0.0
        )
        return {
            "matched_count": matched,
            "predicted_count": predicted,
            "true_count": true,
            "precision": precision_value,
            "recall": recall_value,
            "f1": f1_value,
            "matched_mae": (
                float(values["matched_error_sum"]) / matched if matched else None
            ),
        }

    knot_stage_metrics: dict[str, object] = {}
    for stage_tolerance in candidate_recall_tolerances:
        tolerance_key = f"{stage_tolerance:.3f}"
        stage_values = {
            stage: finalize_stage_metric(
                knot_stage_accumulators[stage][stage_tolerance]
            )
            for stage in knot_stage_accumulators
        }
        pre = stage_values["selected_pre_update"]
        pre_relocation = stage_values["selected_pre_relocation"]
        post = stage_values["deployment_post_update"]
        stage_values["position_recall_delta"] = float(post["recall"]) - float(
            pre["recall"]
        )
        stage_values["position_precision_delta"] = float(post["precision"]) - float(
            pre["precision"]
        )
        stage_values["relocation_recall_delta"] = float(post["recall"]) - float(
            pre_relocation["recall"]
        )
        stage_values["relocation_precision_delta"] = float(post["precision"]) - float(
            pre_relocation["precision"]
        )
        knot_stage_metrics[tolerance_key] = stage_values
    pruning_report = (
        {
            "method": "greedy_single_deletion_standard_bspline_refit",
            "fit_tolerance_normalized_rms": fit_tolerance,
            "retained_count_mean": float(retained.float().mean()),
            "retained_count_min": int(retained.min()),
            "retained_count_max": int(retained.max()),
            # Compatibility key: v7 historically used pooled RMS here.
            "refit_rms_mean": bspline_rms_pooled,
            "refit_rms_mean_curve": bspline_rms_mean,
            "refit_rms_pooled": bspline_rms_pooled,
            "refit_rms_mean_compatibility_semantics": "pooled_rms",
            "refit_rms_p95": bspline_rms_p95,
            "refit_rms_max": bspline_rms_max,
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
                sum(pruning_first_rejected_rms) / len(pruning_first_rejected_rms)
                if pruning_first_rejected_rms
                else None
            ),
            "learned_keep_probability_role": "diagnostic_only",
        }
        if candidate_hard_v7
        else None
    )
    adaptive_threshold_values = (
        torch.cat(one_shot_adaptive_thresholds)
        if one_shot_adaptive_thresholds
        else torch.empty(0)
    )
    finite_adaptive_thresholds = adaptive_threshold_values[
        torch.isfinite(adaptive_threshold_values)
    ]
    learned_baseline_report = (
        {
            "retained_count_mean": sum(learned_baseline_counts)
            / len(learned_baseline_counts),
            "mse_mean": sum(learned_baseline_mse) / len(learned_baseline_mse),
            "threshold_satisfied_fraction": sum(learned_baseline_satisfied)
            / len(learned_baseline_satisfied),
        }
        if learned_baseline_counts
        else None
    )
    hybrid_report = (
        {
            "method": "learned_seeded_beam_deletion_with_position_refinement",
            "mse_tolerance": fit_tolerance * fit_tolerance,
            "rms_tolerance_compatibility_input": fit_tolerance,
            "global_minimum_guaranteed": False,
            "selection_objective": "minimum_K_then_minimum_MSE_under_tolerance",
            "traditional_greedy_is_permanent_incumbent": True,
            "learned_baseline": learned_baseline_report,
            "traditional_greedy": {
                "retained_count_mean": sum(hybrid_greedy_counts)
                / len(hybrid_greedy_counts),
                "mse_mean": sum(hybrid_greedy_mse) / len(hybrid_greedy_mse),
                "threshold_satisfied_fraction": sum(hybrid_greedy_satisfied)
                / len(hybrid_greedy_satisfied),
            },
            "hybrid_final": {
                "retained_count_mean": sum(hybrid_final_counts)
                / len(hybrid_final_counts),
                "mse_mean": sum(hybrid_final_mse) / len(hybrid_final_mse),
                "threshold_satisfied_fraction": sum(hybrid_final_satisfied)
                / len(hybrid_final_satisfied),
                "mean_knot_reduction_vs_greedy": (
                    sum(hybrid_greedy_counts) - sum(hybrid_final_counts)
                )
                / len(hybrid_final_counts),
                "mean_absolute_proposal_to_final_position_shift": (
                    sum(hybrid_mean_abs_position_shifts)
                    / len(hybrid_mean_abs_position_shifts)
                ),
                "max_absolute_proposal_to_final_position_shift": max(
                    hybrid_max_abs_position_shifts
                ),
            },
            "search_refit_count_mean": sum(hybrid_refit_counts)
            / len(hybrid_refit_counts),
            "search_refit_count_total": sum(hybrid_refit_counts),
            "search_fit_evaluation_count_mean": sum(hybrid_refit_counts)
            / len(hybrid_refit_counts),
            "search_fit_evaluation_count_total": sum(hybrid_refit_counts),
            "search_refit_count_semantics": (
                "exact candidate fit evaluations; batched deletion states are "
                "counted individually"
            ),
            "visited_state_count_mean": sum(hybrid_visited_state_counts)
            / len(hybrid_visited_state_counts),
            "visited_state_count_total": sum(hybrid_visited_state_counts),
            "levels_explored_mean": sum(hybrid_level_counts) / len(hybrid_level_counts),
            "levels_explored_max": max(hybrid_level_counts),
            "search_time_ms_mean": sum(hybrid_elapsed_ms) / len(hybrid_elapsed_ms),
            "search_time_ms_total": sum(hybrid_elapsed_ms),
            "beam_width": args.hybrid_beam_width,
            "branch_factor": args.hybrid_branch_factor,
            "position_sweeps": args.hybrid_position_sweeps,
            "position_grid_size": args.hybrid_position_grid_size,
            "position_restarts": args.hybrid_position_restarts,
            "position_refine_count_margin": (args.hybrid_position_refine_count_margin),
            "position_refine_candidate_multiplier": (
                args.hybrid_position_refine_candidate_multiplier
            ),
            "min_gap": (
                float(args.hybrid_min_gap)
                if args.hybrid_min_gap is not None
                else float(model_config.get("min_knot_gap", 1e-3))
            ),
        }
        if hybrid_final_counts
        else None
    )
    verified_report = (
        {
            "method": "adaptive_exact_guard_add_back_residual_rescue",
            "mse_tolerance": fit_tolerance * fit_tolerance,
            "rms_tolerance_compatibility_input": fit_tolerance,
            "global_minimum_guaranteed": False,
            "prefix_monotonicity_assumed_for_acceptance": False,
            "accepted_states_are_exactly_refit": True,
            "compact": bool(args.verified_compact),
            "hard_fallback_enabled": bool(args.verified_hard_fallback),
            "residual_fallback_enabled": bool(args.verified_residual_fallback),
            "max_residual_insertions": args.verified_max_residual_insertions,
            "residual_min_gap": verified_residual_min_gap,
            "parameterization_policy": args.verified_parameterization,
            "parameterization_fallback_attempted_fraction": (
                sum(verified_parameterization_fallback_attempted)
                / len(verified_parameterization_fallback_attempted)
            ),
            "parameterization_fallback_used_fraction": (
                sum(verified_parameterization_fallback_used)
                / len(verified_parameterization_fallback_used)
            ),
            "parameterization_fallback_full_fit_mse_mean_when_attempted": (
                sum(verified_parameterization_fallback_full_mse)
                / len(verified_parameterization_fallback_full_mse)
                if verified_parameterization_fallback_full_mse
                else None
            ),
            "final_parameterization_histogram": dict(
                sorted(verified_final_parameterizations.items())
            ),
            "model_device": str(device),
            "refit_device_requested": args.verified_refit_device,
            "refit_device": str(verified_refit_device),
            "learned_baseline": learned_baseline_report,
            "verified_final": {
                "retained_count_mean": sum(verified_final_counts)
                / len(verified_final_counts),
                "mse_mean": sum(verified_final_mse) / len(verified_final_mse),
                "threshold_satisfied_fraction": sum(verified_final_satisfied)
                / len(verified_final_satisfied),
            },
            "learned_feasible_fraction": sum(learned_baseline_satisfied)
            / len(learned_baseline_satisfied),
            "fast_path_fraction": sum(
                not fallback and not cleanup
                for fallback, cleanup in zip(
                    verified_fallback_used,
                    verified_cleanup_used,
                    strict=True,
                )
            )
            / len(verified_fallback_used),
            "repair_fraction": sum(verified_fallback_used)
            / len(verified_fallback_used),
            "hard_fallback_fraction": sum(verified_hard_fallback_used)
            / len(verified_hard_fallback_used),
            "residual_fallback_fraction": sum(verified_residual_fallback_used)
            / len(verified_residual_fallback_used),
            "residual_insertion_fraction": sum(
                count > 0 for count in verified_inserted_counts
            )
            / len(verified_inserted_counts),
            "residual_inserted_knot_count_mean": sum(verified_inserted_counts)
            / len(verified_inserted_counts),
            "residual_inserted_knot_count_mean_when_used": (
                sum(verified_inserted_counts)
                / sum(count > 0 for count in verified_inserted_counts)
                if any(count > 0 for count in verified_inserted_counts)
                else 0.0
            ),
            "residual_inserted_knot_count_max": max(verified_inserted_counts),
            "residual_fallback_refit_count_mean": sum(verified_residual_refit_counts)
            / len(verified_residual_refit_counts),
            "residual_fallback_refit_count_max": max(verified_residual_refit_counts),
            "cleanup_fraction": sum(verified_cleanup_used) / len(verified_cleanup_used),
            "final_source_histogram": dict(sorted(verified_sources.items())),
            "direct_refit_count_mean": sum(verified_direct_refit_counts)
            / len(verified_direct_refit_counts),
            "fit_evaluation_count_mean": sum(verified_fit_evaluation_counts)
            / len(verified_fit_evaluation_counts),
            "prefix_count_evaluations_mean": sum(verified_prefix_counts)
            / len(verified_prefix_counts),
            "repair_time_ms_mean": sum(verified_elapsed_ms) / len(verified_elapsed_ms),
            "repair_time_ms_median": statistics.median(verified_elapsed_ms),
            "repair_time_ms_p95": float(
                torch.quantile(torch.tensor(verified_elapsed_ms), 0.95)
            ),
            "repair_time_scope": (
                "exact learned refit plus conditional proposal repair/cleanup; "
                "extreme failures may use residual insertions; network forward excluded"
            ),
        }
        if verified_final_counts
        else None
    )
    one_shot_report = (
        {
            "method": {
                "learned": "learned_mask_then_single_standard_bspline_refit",
                "verified": "verified_adaptive_exact_repair",
                "hybrid": "learned_seeded_hybrid_mse_search",
                "hard": "proposal_greedy_hard_pruning",
            }[deployment_mode],
            "deployment_mode_requested": args.deployment_mode,
            "deployment_mode_resolved": deployment_mode,
            "selection_sources": sorted(one_shot_selection_sources),
            "selection_policy": getattr(
                model.pruning_head, "one_shot_selection_policy", "threshold"
            ),
            "selection_safety_sigma": float(
                getattr(model.pruning_head, "one_shot_safety_sigma", 0.0)
            ),
            "selection_coverage_bins": int(
                getattr(model.pruning_head, "one_shot_coverage_bins", 0)
            ),
            "keep_probability_cutoff": 0.5,
            "activity_threshold_cli_used_for_selection": False,
            "fit_tolerance_used_for_selection": deployment_mode != "learned",
            "adaptive_keep_logit_threshold_mean": (
                float(finite_adaptive_thresholds.mean())
                if finite_adaptive_thresholds.numel()
                else None
            ),
            "adaptive_keep_logit_threshold_min": (
                float(finite_adaptive_thresholds.min())
                if finite_adaptive_thresholds.numel()
                else None
            ),
            "adaptive_keep_logit_threshold_max": (
                float(finite_adaptive_thresholds.max())
                if finite_adaptive_thresholds.numel()
                else None
            ),
            "fit_tolerance_normalized_rms": fit_tolerance,
            "retained_count_mean": float(retained.float().mean()),
            "retained_count_min": int(retained.min()),
            "retained_count_max": int(retained.max()),
            # Compatibility key: v8 introduced this as mean per-curve RMS.
            "refit_rms_mean": bspline_rms_mean,
            "refit_rms_mean_curve": bspline_rms_mean,
            "refit_rms_pooled": bspline_rms_pooled,
            "refit_rms_mean_compatibility_semantics": "mean_curve_rms",
            "refit_rms_p95": bspline_rms_p95,
            "refit_rms_max": bspline_rms_max,
            "threshold_satisfied_fraction": (
                sum(one_shot_threshold_satisfied)
                / max(len(one_shot_threshold_satisfied), 1)
            ),
            "standard_refits_per_sample": (1 if deployment_mode == "learned" else None),
            "deployment_standard_refits_per_sample": (
                1 if deployment_mode == "learned" else None
            ),
            "model_forward_internal_proxy_solves_counted_as_deployment_refits": False,
            "offline_diagnostic_refits_counted_as_deployment_refits": False,
            "hard_pruning_used_for_deployment": (
                deployment_mode in {"hard", "hybrid"}
                or (deployment_mode == "verified" and any(verified_hard_fallback_used))
            ),
            "loss_time_exact_teacher_executed": False,
            "objective_role": "inference_proxy_teacher_terms_unavailable",
            "network_survivor_relocation": {
                "enabled": bool(
                    getattr(
                        model.pruning_head,
                        "one_shot_survivor_relocation",
                        False,
                    )
                ),
                "reference": "selected pre-relocation positions",
                "selected_knot_mean_absolute_shift": (
                    sum(learned_relocation_abs_shifts)
                    / len(learned_relocation_abs_shifts)
                    if learned_relocation_abs_shifts
                    else 0.0
                ),
                "selected_knot_max_absolute_shift": (
                    max(learned_relocation_abs_shifts)
                    if learned_relocation_abs_shifts
                    else 0.0
                ),
                "sample_mean_absolute_shift_mean": (
                    sum(learned_relocation_sample_mean_shifts)
                    / len(learned_relocation_sample_mean_shifts)
                    if learned_relocation_sample_mean_shifts
                    else 0.0
                ),
                "sample_max_absolute_shift_mean": (
                    sum(learned_relocation_sample_max_shifts)
                    / len(learned_relocation_sample_max_shifts)
                    if learned_relocation_sample_max_shifts
                    else 0.0
                ),
                "selected_knot_moved_fraction_at_1e-6": (
                    sum(value > 1e-6 for value in learned_relocation_abs_shifts)
                    / len(learned_relocation_abs_shifts)
                    if learned_relocation_abs_shifts
                    else 0.0
                ),
                "selected_knot_proposal_to_final_mean_absolute_shift": (
                    sum(learned_total_abs_shifts) / len(learned_total_abs_shifts)
                    if learned_total_abs_shifts
                    else 0.0
                ),
                "selected_knot_proposal_to_final_max_absolute_shift": (
                    max(learned_total_abs_shifts) if learned_total_abs_shifts else 0.0
                ),
            },
            "offline_hard_diagnostic_requested": bool(args.run_hard_diagnostic),
            "learned_baseline": learned_baseline_report,
            "verified_repair": verified_report,
            "hybrid_search": hybrid_report,
            "offline_hard_diagnostic": (
                {
                    "role": "teacher_diagnostic_only",
                    "retained_count_mean": sum(one_shot_hard_diagnostic_counts)
                    / len(one_shot_hard_diagnostic_counts),
                    "rms_mean": sum(one_shot_hard_diagnostic_rms)
                    / len(one_shot_hard_diagnostic_rms),
                    "rms_mean_curve": sum(one_shot_hard_diagnostic_rms)
                    / len(one_shot_hard_diagnostic_rms),
                    "rms_pooled": math.sqrt(
                        sum(value * value for value in one_shot_hard_diagnostic_rms)
                        / len(one_shot_hard_diagnostic_rms)
                    ),
                    "threshold_satisfied_fraction": sum(
                        one_shot_hard_diagnostic_satisfied
                    )
                    / len(one_shot_hard_diagnostic_satisfied),
                }
                if one_shot_hard_diagnostic_counts
                else None
            ),
        }
        if candidate_one_shot
        else None
    )

    report = {
        "schema_version": 15,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_selection_metric": checkpoint.get("selection_metric"),
        "checkpoint_selection_value": checkpoint.get("selection_value"),
        "objective_version": checkpoint.get("objective_version", "historical"),
        "structure_mode": structure_mode,
        "deployment_mode_requested": args.deployment_mode,
        "deployment_mode_resolved": deployment_mode,
        "model_device": str(device),
        "verified_refit_device_requested": args.verified_refit_device,
        "verified_refit_device": str(verified_refit_device),
        "verified_refit_device_used": deployment_mode == "verified",
        "torch_num_threads": torch.get_num_threads(),
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
            else {
                "learned": "learned_one_shot_mask",
                "verified": "exact_refit_verified_adaptive_repair",
                "hybrid": "hybrid_mse_search",
                "hard": "hard_proposal_pruning",
            }[deployment_mode]
            if candidate_one_shot
            else "hard_rms_pruning"
            if candidate_hard_v7
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
        "one_shot_deployment": one_shot_report,
        "verified_quality_deployment": verified_report,
        "hybrid_quality_deployment": hybrid_report,
        "candidate_proposal_definition": (
            "output['proposal_internal_knots'] before LearnedKeep selection"
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
        "knot_stage_metrics": knot_stage_metrics if candidate_one_shot else None,
        "zero_knot_fraction": float((retained == 0).float().mean()),
        "all_knot_fraction": float((retained == candidate_count).float().mean()),
        "network_objective_role": (
            "inference_proxy_teacher_terms_unavailable"
            if candidate_one_shot
            else "evaluation_objective"
        ),
        "network_objective_comparable_to_training_validation": not candidate_one_shot,
        "network_inference_proxy_objective": (
            mean_losses["loss"] if candidate_one_shot else None
        ),
        # Compatibility key retained for existing report consumers. For v8 its
        # role is described by ``network_objective_role`` above.
        "network_total_objective": mean_losses["loss"],
        "network_forward_fit_loss": mean_losses["fit_loss"],
        "network_forward_rms_euclidean": math.sqrt(mean_losses["fit_loss"]),
        "network_forward_coordinate_rmse": math.sqrt(
            mean_losses["fit_loss"] / dataset_config["point_dim"]
        ),
        "standard_bspline_refit_loss": bspline_fit_mean,
        # Compatibility key: historically this was pooled RMS.
        "standard_bspline_refit_rms_euclidean": bspline_rms_pooled,
        "standard_bspline_refit_rms_euclidean_pooled": bspline_rms_pooled,
        "standard_bspline_refit_rms_euclidean_mean_curve": bspline_rms_mean,
        "standard_bspline_refit_rms_euclidean_p95": bspline_rms_p95,
        "standard_bspline_refit_rms_euclidean_max": bspline_rms_max,
        "standard_bspline_coordinate_rmse": math.sqrt(bspline_coordinate_mean),
        "standard_bspline_endpoint_rmse": endpoint_rmse,
        "standard_bspline_endpoint_max_distance": endpoint_max,
        "standard_bspline_control_count_mean": sum(bspline_control_counts)
        / len(bspline_control_counts),
        "standard_bspline_augmented_objective_mean": sum(bspline_augmented_objectives)
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
        "knot_match_parameterization": (
            "deployed knots warped through sample correspondence from each recorded "
            "deployment parameterization into the ground-truth parameter domain"
        ),
        "knot_match_precision": precision,
        "knot_match_recall": recall,
        "knot_match_f1": f1,
        "matched_knot_mae": knot_mae if math.isfinite(knot_mae) else None,
        "parameter_feedback": parameter_feedback_report,
    }

    print("Checkpoint structured-knot evaluation")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  objective version: {report['objective_version']}")
    print(f"  structure mode: {structure_mode}")
    print(
        "  deployment mode requested/resolved: "
        f"{args.deployment_mode}/{deployment_mode}"
    )
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
    if candidate_one_shot:
        print("  objective role: inference proxy (teacher terms unavailable)")
        print(f"  inference proxy objective: {mean_losses['loss']:.9e}")
    else:
        print(f"  total objective: {mean_losses['loss']:.9e}")
    print(f"  fit loss (mean squared Euclidean): {mean_losses['fit_loss']:.9e}")
    print(f"  RMS Euclidean distance: {math.sqrt(mean_losses['fit_loss']):.9e}")
    if parameter_feedback_report["enabled"]:
        print(
            "  parameter feedback t0->t1 RMSE: "
            f"{parameter_feedback_report['parameter_rmse_from_proposal']:.9e}"
        )
        print(
            "  parameter feedback mean |gap-logit delta|: "
            f"{parameter_feedback_report['gap_logit_delta_mean_absolute']:.9e}"
        )
        print(
            "  parameter feedback chord-gap blend mean: "
            f"{parameter_feedback_report['chord_blend_weight_mean']:.6f}"
        )
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
        print("\nHigh-recall candidate diagnostics")
        print("  proposal set: immutable proposal candidates before selection")
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
        if candidate_one_shot:
            print("  staged knot matching (proposal -> selected -> position-updated)")
            for tolerance in (0.01, 0.02, 0.05):
                stages = knot_stage_metrics[f"{tolerance:.3f}"]
                pre = stages["selected_pre_update"]
                post = stages["deployment_post_update"]
                print(
                    f"    @{tolerance:.3f}: pre P/R/F1="
                    f"{pre['precision']:.3f}/{pre['recall']:.3f}/{pre['f1']:.3f} "
                    f"-> post={post['precision']:.3f}/{post['recall']:.3f}/"
                    f"{post['f1']:.3f}"
                )
            print(
                "  learned keep probability mass mean (deployment): "
                f"{expected_count_mean:.3f}"
            )
            print(f"  learned one-shot count histogram: {network_histogram}")
            relocation_report = one_shot_report["network_survivor_relocation"]
            print(
                f"  network survivor relocation enabled: {relocation_report['enabled']}"
            )
            print(
                "  selected pre->post |shift| mean/max/moved@1e-6: "
                f"{relocation_report['selected_knot_mean_absolute_shift']:.6e}/"
                f"{relocation_report['selected_knot_max_absolute_shift']:.6e}/"
                f"{relocation_report['selected_knot_moved_fraction_at_1e-6']:.3f}"
            )
            print(
                "  selected proposal->final |shift| mean/max: "
                f"{relocation_report['selected_knot_proposal_to_final_mean_absolute_shift']:.6e}/"
                f"{relocation_report['selected_knot_proposal_to_final_max_absolute_shift']:.6e}"
            )
            print("\nQuality standard B-spline deployment")
            print(
                f"  selection: {deployment_mode} | learned policy="
                f"{one_shot_report['selection_policy']} | safety_sigma="
                f"{one_shot_report['selection_safety_sigma']:.3f} | coverage_bins="
                f"{one_shot_report['selection_coverage_bins']}"
            )
            print(
                "  adaptive raw-importance logit beta mean/min/max: "
                f"{one_shot_report['adaptive_keep_logit_threshold_mean']}/"
                f"{one_shot_report['adaptive_keep_logit_threshold_min']}/"
                f"{one_shot_report['adaptive_keep_logit_threshold_max']}"
            )
            print(f"  normalized RMS tolerance: {fit_tolerance:.9e}")
            if deployment_mode == "learned":
                print(
                    "  tolerance role: reporting only; it does not change the "
                    "learned mask"
                )
            else:
                print(
                    "  tolerance role: hard selection constraint; equivalent MSE="
                    f"{fit_tolerance * fit_tolerance:.9e}"
                )
            print(
                "  deployed K mean/min/max: "
                f"{one_shot_report['retained_count_mean']:.3f}/"
                f"{one_shot_report['retained_count_min']}/"
                f"{one_shot_report['retained_count_max']}"
            )
            print(
                "  deployed refit mean-curve RMS/P95/max: "
                f"{one_shot_report['refit_rms_mean']:.9e}/"
                f"{one_shot_report['refit_rms_p95']:.9e}/"
                f"{one_shot_report['refit_rms_max']:.9e}"
            )
            print(f"  deployed pooled RMS: {one_shot_report['refit_rms_pooled']:.9e}")
            print(
                "  threshold-satisfied fraction: "
                f"{one_shot_report['threshold_satisfied_fraction']:.3f}"
            )
            print(
                "  hard search used for deployment: "
                f"{one_shot_report['hard_pruning_used_for_deployment']}"
            )
            if deployment_mode == "learned":
                print(
                    "  deployment standard B-spline refits per sample: 1 "
                    "(network-forward proxy solves are excluded)"
                )
            elif deployment_mode == "verified":
                print(
                    "  model / verified refit device: "
                    f"{device}/{verified_refit_device} "
                    f"(requested={args.verified_refit_device})"
                )
                print(
                    "  learned-feasible / zero-cleanup fast / repair / hard-fallback: "
                    f"{verified_report['learned_feasible_fraction']:.3f}/"
                    f"{verified_report['fast_path_fraction']:.3f}/"
                    f"{verified_report['repair_fraction']:.3f}/"
                    f"{verified_report['hard_fallback_fraction']:.3f}"
                )
                print(
                    "  parameterization policy/fallback-used/final domains: "
                    f"{args.verified_parameterization}/"
                    f"{verified_report['parameterization_fallback_used_fraction']:.3f}/"
                    f"{verified_report['final_parameterization_histogram']}"
                )
                print(
                    "  verified residual fallback/insertion fractions: "
                    f"{verified_report['residual_fallback_fraction']:.3f}/"
                    f"{verified_report['residual_insertion_fraction']:.3f}"
                )
                print(
                    "  residual inserted K mean/used-mean/max; refits mean/max: "
                    f"{verified_report['residual_inserted_knot_count_mean']:.3f}/"
                    f"{verified_report['residual_inserted_knot_count_mean_when_used']:.3f}/"
                    f"{verified_report['residual_inserted_knot_count_max']}; "
                    f"{verified_report['residual_fallback_refit_count_mean']:.3f}/"
                    f"{verified_report['residual_fallback_refit_count_max']}"
                )
                print(
                    "  verified final K/MSE/pass: "
                    f"{verified_report['verified_final']['retained_count_mean']:.3f}/"
                    f"{verified_report['verified_final']['mse_mean']:.9e}/"
                    f"{verified_report['verified_final']['threshold_satisfied_fraction']:.3f}"
                )
                print(
                    "  verified exact fit-evaluations/direct-refits mean: "
                    f"{verified_report['fit_evaluation_count_mean']:.1f}/"
                    f"{verified_report['direct_refit_count_mean']:.1f}"
                )
                print(
                    "  verified repair time mean/median/P95 (forward excluded): "
                    f"{verified_report['repair_time_ms_mean']:.2f}/"
                    f"{verified_report['repair_time_ms_median']:.2f}/"
                    f"{verified_report['repair_time_ms_p95']:.2f} ms"
                )
                print(
                    "  verified final sources: "
                    f"{verified_report['final_source_histogram']}"
                )
            elif deployment_mode == "hybrid":
                print(
                    "  hybrid learned/greedy/final K mean: "
                    f"{hybrid_report['learned_baseline']['retained_count_mean']:.3f}/"
                    f"{hybrid_report['traditional_greedy']['retained_count_mean']:.3f}/"
                    f"{hybrid_report['hybrid_final']['retained_count_mean']:.3f}"
                )
                print(
                    "  hybrid final MSE/pass: "
                    f"{hybrid_report['hybrid_final']['mse_mean']:.9e}/"
                    f"{hybrid_report['hybrid_final']['threshold_satisfied_fraction']:.3f}"
                )
                print(
                    "  hybrid fit-evaluations/visited/time mean: "
                    f"{hybrid_report['search_refit_count_mean']:.1f}/"
                    f"{hybrid_report['visited_state_count_mean']:.1f}/"
                    f"{hybrid_report['search_time_ms_mean']:.2f} ms"
                )
                print("  global minimum guaranteed: False")
            if one_shot_report["offline_hard_diagnostic"] is not None:
                diagnostic = one_shot_report["offline_hard_diagnostic"]
                print(
                    "  offline hard teacher diagnostic only: "
                    f"K mean={diagnostic['retained_count_mean']:.3f}, "
                    f"RMS mean={diagnostic['rms_mean']:.9e}, "
                    "threshold-satisfied fraction="
                    f"{diagnostic['threshold_satisfied_fraction']:.3f}"
                )
        else:
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
    print(f"  refit pooled RMS distance: {bspline_rms_pooled:.9e}")
    print(f"  refit mean-curve RMS distance: {bspline_rms_mean:.9e}")
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
