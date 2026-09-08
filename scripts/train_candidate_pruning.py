from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import (  # noqa: E402
    V13_SET_RELOCATION_OBJECTIVE_VERSION,
    V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
    V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
)
from spline_fitting.data.synthetic import SyntheticCubicBSplineDataset  # noqa: E402
from spline_fitting.losses import (  # noqa: E402
    CandidatePruningLoss,
    CandidatePruningLossWeights,
    ParameterFeedbackLoss,
    ParameterFeedbackLossWeights,
    build_redundant_boehm_candidates,
)
from spline_fitting.models import SplineFittingNetwork  # noqa: E402
from spline_fitting.evaluation.bspline_inference import (  # noqa: E402
    refit_bspline_control_points,
)
from spline_fitting.evaluation.knot_diagnostics import (  # noqa: E402
    warp_internal_knots_to_parameterization,
)
from spline_fitting.training import (  # noqa: E402
    OneShotTeacherBatch,
    OneShotTeacherConfig,
    TeacherAugmentedDataset,
    Trainer,
    build_one_shot_teacher_batch,
    load_one_shot_teacher_cache,
    save_one_shot_teacher_cache,
)


_TEACHER_FEASIBILITY_POLICY = (
    "calibrated_all_candidate_standard_bspline_then_true_then_chord_v2"
)
_CANONICAL_BOEHM_TEACHER_POLICY = (
    "canonical_true_knots_largest_interval_boehm_then_hard_rms_v1"
)
_TEACHER_START_DOMAIN_METADATA_VERSION = 2
_TEACHER_START_MODES = ("canonical_boehm", "learned_proposal")


def _teacher_feasibility_policy(start_mode: str) -> str:
    if start_mode == "canonical_boehm":
        return _CANONICAL_BOEHM_TEACHER_POLICY
    if start_mode == "learned_proposal":
        return _TEACHER_FEASIBILITY_POLICY
    raise ValueError(f"unsupported teacher start mode: {start_mode!r}")

# These are defaults for newly trained candidate-pruning models.  The lower-level
# network constructors retain their historical defaults because checkpoint
# migration supplies legacy values for metadata that predates these fields.
_NEW_TRAINING_PARAMETER_RESIDUAL_LOGIT_LIMIT = 2.5
_NEW_TRAINING_CANDIDATE_INTERVAL_LOGIT_LIMIT = 1.75
_NEW_TRAINING_TRUE_PARAMETER_WEIGHT = 2.0
_NEW_TRAINING_REDUNDANT_CANDIDATE_WEIGHT = 2.5
_MAX_PROPOSAL_PARAMETER_WARMUP_EPOCHS = 15


def _resolve_proposal_parameter_warmup_epochs(
    candidate_pretrain_epochs: int,
    requested_epochs: int | None,
) -> int:
    """Resolve an optional warm-up while preserving the proposal epoch budget."""

    if candidate_pretrain_epochs < 0:
        raise ValueError("candidate pretraining epochs must be non-negative")
    if requested_epochs is None:
        # Keep most of the proposal budget for joint parameter/candidate
        # learning.  The former ``pretrain - 1`` rule made, for example, a
        # five-epoch smoke run spend four epochs warming up ParameterHead and
        # left only one epoch for CandidateKnotHead.
        return min(
            _MAX_PROPOSAL_PARAMETER_WARMUP_EPOCHS,
            candidate_pretrain_epochs // 3,
        )
    if requested_epochs < 0:
        raise ValueError("proposal parameter warm-up epochs must be non-negative")
    if candidate_pretrain_epochs == 0:
        if requested_epochs != 0:
            raise ValueError(
                "proposal parameter warm-up must be zero when candidate "
                "pretraining is disabled"
            )
        return 0
    if requested_epochs >= candidate_pretrain_epochs:
        raise ValueError(
            "proposal parameter warm-up must be smaller than candidate "
            "pretraining epochs"
        )
    return requested_epochs


def _sample_progress_line(
    label: str,
    completed: int,
    total: int,
    elapsed: float,
    *,
    width: int = 28,
) -> str:
    fraction = completed / max(total, 1)
    filled = min(width, max(0, int(round(width * fraction))))
    rate = completed / elapsed if elapsed > 0.0 else 0.0
    remaining = (total - completed) / rate if rate > 0.0 else float("inf")
    if remaining == float("inf"):
        eta = "--:--"
    else:
        total_seconds = int(round(max(remaining, 0.0)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        eta = (
            f"{hours:d}:{minutes:02d}:{seconds:02d}"
            if hours
            else f"{minutes:02d}:{seconds:02d}"
        )
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"  {label:<18} [{bar}] {completed:>5}/{total:<5} "
        f"{100.0 * fraction:6.2f}% | {rate:6.2f} sample/s | ETA {eta}"
    )


def _show_sample_progress(
    label: str,
    completed: int,
    total: int,
    started_at: float,
) -> None:
    interactive = sys.stdout.isatty()
    report_interval = 1 if interactive else max(total // 20, 1)
    if completed % report_interval and completed != total:
        return
    line = _sample_progress_line(
        label,
        completed,
        total,
        max(time.perf_counter() - started_at, 1e-9),
    )
    if interactive:
        print(f"\r{line:<130}", end="", flush=True)
        if completed == total:
            print(flush=True)
    else:
        print(line, flush=True)


def _checkpoint_with_metadata(
    checkpoint: dict[str, object],
    *,
    model_config: dict[str, object],
    dataset_config: dict[str, object],
    loss_config: dict[str, object],
    training_config: dict[str, object],
    deployment_config: dict[str, object],
    histories: dict[str, list[dict[str, float]]],
    objective_version: str,
) -> dict[str, object]:
    checkpoint.update(
        {
            "model_config": model_config,
            "dataset_config": dataset_config,
            "dataset_type": "synthetic_open_cubic_bspline",
            "objective_version": objective_version,
            "loss_config": loss_config,
            "training_config": training_config,
            "deployment_config": deployment_config,
            "selected_stage": checkpoint.get("stage", "one_shot_distillation"),
            "stage_histories": histories,
        }
    )
    return checkpoint


def _load_model_state(
    model: SplineFittingNetwork,
    state: Mapping[str, torch.Tensor],
    *,
    allow_new_parameter_feedback: bool,
    allow_new_joint_feedback: bool = False,
) -> None:
    """Load an old proposal while permitting only explicitly attached heads."""

    if not allow_new_parameter_feedback and not allow_new_joint_feedback:
        model.load_state_dict(state, strict=True)
        return
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = sorted(incompatible.unexpected_keys)
    missing = sorted(incompatible.missing_keys)
    allowed_prefixes: list[str] = []
    if allow_new_parameter_feedback:
        allowed_prefixes.append("parameter_feedback_head.")
    if allow_new_joint_feedback:
        allowed_prefixes.append("joint_parameter_structure_head.")
    disallowed_missing = [
        name
        for name in missing
        if not any(name.startswith(p) for p in allowed_prefixes)
    ]
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "proposal checkpoint is incompatible with the feedback model: "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )


def _model_fingerprint(model: torch.nn.Module) -> str:
    """Bind an offline teacher cache to the exact proposal-producing weights."""
    digest = hashlib.sha256()
    proposal_prefixes = (
        "encoder.",
        "parameter_head.",
        "candidate_head.",
        # The calibrated teacher parameterization consumes preliminary
        # structural tokens, so every pruning-head weight can affect t1 even
        # though the teacher knots themselves never use a Keep-dependent path.
        "pruning_head.",
        "parameter_feedback_head.",
    )
    feedback_head = getattr(model, "parameter_feedback_head", None)
    proposal_metadata = {
        "degree": int(model.degree),
        "lambda_poly": float(model.lambda_poly),
        "lambda_knot": float(model.lambda_knot),
        "pruning_residual_bandwidth": float(model.pruning_residual_bandwidth),
        "parameter_min_gap": float(model.parameter_head.min_gap),
        "parameter_gap_parameterization": str(
            model.parameter_head.gap_parameterization
        ),
        "parameter_gap_reference": str(model.parameter_head.gap_reference),
        "parameter_residual_logit_limit": float(
            model.parameter_head.residual_logit_limit
        ),
        "geometry_feature_mode": str(model.encoder.feature_mode),
        "candidate_count": int(model.candidate_head.num_candidates),
        "candidate_attention_heads": int(model.candidate_head.attention_heads),
        "candidate_min_gap": float(model.candidate_head.min_gap),
        "candidate_local_attention_bandwidth": float(
            model.candidate_head.local_attention_bandwidth
        ),
        "candidate_interval_logit_limit": float(
            model.candidate_head.interval_logit_limit
        ),
        "candidate_position_parameterization": str(
            model.candidate_head.position_parameterization
        ),
        "proposal_position_fraction": float(model.pruning_head.max_position_fraction),
        "pruning_attention_heads": int(model.pruning_head.self_attention.num_heads),
        "pruning_min_gap": float(model.pruning_head.min_gap),
        "fixed_proposal_geometry": bool(
            model.pruning_head.one_shot_fixed_proposal_geometry
        ),
        "one_shot_selection_policy": str(
            model.pruning_head.one_shot_selection_policy
        ),
        "one_shot_safety_sigma": float(model.pruning_head.one_shot_safety_sigma),
        "one_shot_coverage_bins": int(model.pruning_head.one_shot_coverage_bins),
        "one_shot_max_position_shift": float(
            model.pruning_head.one_shot_max_position_shift
        ),
        # This non-parameter flag changes the analytic features seen by the
        # fixed proposal head.  Include it in the fingerprint so a v12
        # float32-pilot teacher cache can never be silently reused by v13's
        # stable float64-pilot path even though the state_dict is identical.
        "stable_pilot_descriptors": bool(model.stable_pilot_descriptors),
        "parameter_feedback_fusion": bool(model.parameter_feedback_fusion),
        "parameter_feedback_fusion_mode": (
            str(feedback_head.fusion_mode) if feedback_head is not None else "disabled"
        ),
        "parameter_feedback_max_logit_shift": (
            float(feedback_head.max_logit_shift) if feedback_head is not None else 0.0
        ),
        "enforce_ordered_joint_candidates": bool(
            model.enforce_ordered_joint_candidates
        ),
    }
    digest.update(
        json.dumps(proposal_metadata, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith(proposal_prefixes):
            continue
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        if tensor.is_floating_point():
            # CPU linear algebra can differ by a few float32 ulps across
            # otherwise identical runs.  Such sub-micro changes do not alter
            # teacher geometry, so hash a stable 1e-6 quantisation instead of
            # making cache reuse depend on thread-level reduction order.
            tensor = torch.round(tensor.to(torch.float64) * 1e6).to(torch.int64)
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


_PROPOSAL_SEMANTIC_KEYS = (
    "point_dim",
    "degree",
    "hidden_dim",
    "encoder_layers",
    "max_internal_knots",
    "min_parameter_gap",
    "min_knot_gap",
    "gap_parameterization",
    "parameter_gap_reference",
    "parameter_residual_logit_limit",
    "lambda_poly",
    "lambda_knot",
    "structure_mode",
    "structure_attention_heads",
    "geometry_feature_mode",
    "pruning_residual_bandwidth",
    "candidate_local_attention_bandwidth",
    "candidate_interval_logit_limit",
    "candidate_position_parameterization",
    "one_shot_fixed_proposal_geometry",
    "stable_pilot_descriptors",
    "compute_first_derivative",
)


def _proposal_semantic_mismatches(
    requested: Mapping[str, object],
    saved: Mapping[str, object],
) -> list[str]:
    """Find non-weight settings that can change fixed proposal geometry."""

    mismatches: list[str] = []
    for key in _PROPOSAL_SEMANTIC_KEYS:
        if key not in saved or key not in requested:
            continue
        requested_value = requested[key]
        saved_value = saved[key]
        if isinstance(requested_value, float) or isinstance(saved_value, float):
            equal = abs(float(requested_value) - float(saved_value)) <= 1e-12
        else:
            equal = requested_value == saved_value
        if not equal:
            mismatches.append(
                f"{key}: requested={requested_value!r}, checkpoint={saved_value!r}"
            )
    return mismatches


def _dataset_fingerprint(
    dataset: SyntheticCubicBSplineDataset,
    *,
    split: str,
    seed: int,
    dataset_config: dict[str, object],
) -> str:
    """Hash the actual fixed samples, not only their reusable row indices."""
    digest = hashlib.sha256()
    metadata = {
        "split": split,
        "seed": int(seed),
        "size": len(dataset),
        "dataset_config": dataset_config,
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    identity_keys = (
        "sample_id",
        "points",
        "true_params",
        "true_internal_knots",
        "true_internal_knot_mask",
    )
    started_at = time.perf_counter()
    total = len(dataset)
    for index in range(total):
        sample = dataset[index]
        for key in identity_keys:
            value = torch.as_tensor(sample[key]).detach().cpu().contiguous()
            digest.update(key.encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.numpy().tobytes())
        _show_sample_progress(
            f"fingerprint {split}",
            index + 1,
            total,
            started_at,
        )
    return digest.hexdigest()


def _concatenate_teacher_batches(
    batches: list[OneShotTeacherBatch],
    config: OneShotTeacherConfig,
) -> OneShotTeacherBatch:
    if not batches:
        raise ValueError("cannot concatenate an empty teacher batch list")
    labels = {
        key: torch.cat([batch.as_loss_kwargs()[key] for batch in batches], dim=0)
        for key in batches[0].as_loss_kwargs()
    }
    sample_indices = torch.cat([batch.sample_indices for batch in batches], dim=0)
    first_shapes = batches[0].input_shapes
    total = int(sample_indices.numel())
    return OneShotTeacherBatch(
        sample_indices=sample_indices,
        input_shapes={
            "parameters": (total, *first_shapes["parameters"][1:]),
            "points": (total, *first_shapes["points"][1:]),
            "candidate_knots": (total, *first_shapes["candidate_knots"][1:]),
        },
        config=config,
        **labels,
    )


def _teacher_relocation_metadata(
    teacher: OneShotTeacherBatch,
) -> dict[str, float | int]:
    sample_count = max(int(teacher.teacher_count.numel()), 1)
    extra = teacher.teacher_extra_deleted_after_relocation.to(torch.float64)
    greedy_count = teacher.teacher_greedy_count.to(torch.float64)
    final_count = teacher.teacher_count.to(torch.float64)
    return {
        "sample_count": sample_count,
        "greedy_count_mean": float(greedy_count.mean()),
        "final_count_mean": float(final_count.mean()),
        "extra_deleted_total": int(extra.sum().item()),
        "extra_deleted_sample_fraction": float((extra > 0).sum()) / sample_count,
        "extra_deleted_survivor_fraction": float(extra.sum())
        / max(float(greedy_count.sum()), 1.0),
        "relocation_mean_abs": float(teacher.teacher_relocation_mean_abs.mean()),
        "relocation_max_abs": float(teacher.teacher_relocation_max_abs.max()),
        # Teacher positions are cached in fixed proposal t0 coordinates, so
        # these mapped shifts also quantify ordered-rank target reachability
        # from the current learned proposal slots.
        "ordered_rank_anchor_shift_mean": float(
            teacher.teacher_relocation_mean_abs.mean()
        ),
        "ordered_rank_anchor_shift_max": float(
            teacher.teacher_relocation_max_abs.max()
        ),
        "ordered_rank_anchor_shift_over_relocation_limit_fraction": float(
            (
                teacher.teacher_relocation_max_abs
                > teacher.config.relocation_max_shift + 1e-12
            )
            .to(torch.float64)
            .mean()
        ),
        "fit_mse_mean": float(teacher.teacher_fit_mse.mean()),
        "threshold_satisfied_rate": float(
            teacher.teacher_threshold_satisfied.to(torch.float64).mean()
        ),
    }


def _report_teacher_relocation(
    label: str,
    teacher: OneShotTeacherBatch,
) -> None:
    summary = _teacher_relocation_metadata(teacher)
    print(
        f"  {label} teacher relocation: "
        f"K greedy/final={summary['greedy_count_mean']:.3f}/"
        f"{summary['final_count_mean']:.3f} | "
        f"extra deleted={summary['extra_deleted_total']} "
        f"({100.0 * summary['extra_deleted_sample_fraction']:.1f}% samples, "
        f"{100.0 * summary['extra_deleted_survivor_fraction']:.2f}% "
        "of greedy survivors) | "
        f"mean|max |shift|={summary['relocation_mean_abs']:.6f}|"
        f"{summary['relocation_max_abs']:.6f} | "
        "rank-target over move-bound="
        f"{summary['ordered_rank_anchor_shift_over_relocation_limit_fraction']:.3%} | "
        f"MSE={summary['fit_mse_mean']:.6e} | "
        f"pass={summary['threshold_satisfied_rate']:.3f}",
        flush=True,
    )


def _teacher_start_domain_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.name + ".start_domains.json")


def _validate_teacher_start_domain_counts(
    counts: Mapping[str, object],
    *,
    sample_count: int,
    max_fallback_fraction: float,
    start_mode: str = "learned_proposal",
) -> dict[str, int | float]:
    """Validate start provenance without mixing supervised and fallback labels."""

    if start_mode not in _TEACHER_START_MODES:
        raise RuntimeError(f"unsupported teacher start mode: {start_mode!r}")
    expected_domains = (
        ("canonical_boehm",)
        if start_mode == "canonical_boehm"
        else ("calibrated", "true", "chord")
    )
    if set(counts) != set(expected_domains):
        raise RuntimeError(
            "teacher start-domain statistics do not match mode "
            f"{start_mode!r}: expected {expected_domains}"
        )
    normalized: dict[str, int] = {}
    for name in expected_domains:
        value = counts[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"invalid teacher start-domain count for {name}")
        normalized[name] = value
    if sum(normalized.values()) != sample_count:
        raise RuntimeError(
            "teacher start-domain counts do not match the cached sample count"
        )
    fallback_count = (
        0
        if start_mode == "canonical_boehm"
        else normalized["true"] + normalized["chord"]
    )
    fallback_fraction = fallback_count / max(sample_count, 1)
    if fallback_fraction > max_fallback_fraction + 1e-12:
        raise RuntimeError(
            "offline teacher fallback fraction exceeds the permitted limit: "
            f"{fallback_fraction:.3%} > {max_fallback_fraction:.3%}. "
            "The network-calibrated all-candidate proposal is not deployment-"
            "feasible often enough; improve/retrain the proposal instead of "
            "training the selector on true/chord-parameter fallbacks. For "
            "labelled synthetic training, prefer --teacher-start-mode "
            "canonical_boehm; do not loosen the fallback budget merely to "
            "silence this guard."
        )
    return {
        **normalized,
        "fallback_count": fallback_count,
        "fallback_fraction": fallback_fraction,
    }


def _full_candidate_standard_bspline_fit(
    parameters: torch.Tensor,
    points: torch.Tensor,
    candidate_knots: torch.Tensor,
    config: OneShotTeacherConfig,
):
    """Refit all proposal knots and reject numerically invalid teacher starts."""

    fit = refit_bspline_control_points(
        parameters,
        points,
        candidate_knots,
        degree=config.degree,
        smoothness_weight=config.smoothness_weight,
        control_ridge=config.control_ridge,
        interpolate_endpoints=True,
        rcond=config.rcond,
    )
    expected_rank = int(candidate_knots.numel()) + config.degree + 1
    rank_valid = fit.solver_rank is None or fit.solver_rank == expected_rank
    finite = bool(
        torch.isfinite(fit.fit_rmse)
        and torch.isfinite(fit.reconstructed_points).all()
        and torch.isfinite(fit.control_points).all()
    )
    feasible = (
        finite
        and rank_valid
        and float(fit.fit_rmse) <= config.error_tolerance
    )
    return fit, feasible, rank_valid


def _strict_unit_parameterization(values: torch.Tensor) -> torch.Tensor | None:
    """Return a pinned strict [0,1] row, or ``None`` when it cannot be warped."""

    if values.ndim != 1 or values.numel() < 2 or not values.is_floating_point():
        return None
    if not bool(torch.isfinite(values).all()):
        return None
    result = values.detach().clone()
    result[0], result[-1] = 0.0, 1.0
    if torch.any(result[1:] <= result[:-1]):
        return None
    return result


def _select_feasible_teacher_start(
    proposal_parameters: torch.Tensor,
    points: torch.Tensor,
    proposal_knots: torch.Tensor,
    config: OneShotTeacherConfig,
    *,
    calibrated_parameters: torch.Tensor | None = None,
    true_parameters: torch.Tensor | None,
    chord_parameters: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, str, float]:
    """Choose a feasible full-candidate teacher domain without changing slots.

    The calibrated ``t1`` domain is authoritative whenever its all-candidate
    standard B-spline fit passes.  Otherwise true parameters and then chord
    parameters are tested as warm starts. Candidate slot identity is preserved
    by warping the immutable ``t0`` proposal knots through corresponding points.
    """

    proposal_parameters = _strict_unit_parameterization(proposal_parameters)
    if proposal_parameters is None:
        raise RuntimeError("proposal parameters are not strictly increasing")

    calibrated = _strict_unit_parameterization(
        proposal_parameters if calibrated_parameters is None else calibrated_parameters
    )
    if calibrated is None:
        raise RuntimeError("calibrated parameters are not strictly increasing")
    calibrated_knots = warp_internal_knots_to_parameterization(
        proposal_knots,
        proposal_parameters,
        calibrated,
    )
    trials: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        ("calibrated", calibrated, calibrated_knots)
    ]
    for name, target in (
        ("true", true_parameters),
        ("chord", chord_parameters),
    ):
        if target is None:
            continue
        strict_target = _strict_unit_parameterization(target)
        if strict_target is None:
            continue
        warped_knots = warp_internal_knots_to_parameterization(
            proposal_knots,
            proposal_parameters,
            strict_target,
        )
        if warped_knots.numel() > 1 and torch.any(
            warped_knots[1:] <= warped_knots[:-1]
        ):
            continue
        trials.append((name, strict_target, warped_knots))

    diagnostics: list[str] = []
    feasible_fallbacks: list[tuple[float, str, torch.Tensor, torch.Tensor]] = []
    for name, parameters, knots in trials:
        try:
            fit, feasible, rank_valid = _full_candidate_standard_bspline_fit(
                parameters,
                points,
                knots,
                config,
            )
        except (RuntimeError, ValueError) as error:
            diagnostics.append(f"{name}=error({error})")
            continue
        rms = float(fit.fit_rmse)
        diagnostics.append(
            f"{name}=RMS({rms:.6g}),rank_ok({int(rank_valid)})"
        )
        if feasible:
            if name == "calibrated":
                return parameters, knots, name, rms
            feasible_fallbacks.append((rms, name, parameters, knots))

    if feasible_fallbacks:
        rms, name, parameters, knots = min(feasible_fallbacks, key=lambda item: item[0])
        return parameters, knots, name, rms
    raise RuntimeError(
        "all-candidate standard B-spline is infeasible in calibrated, true and "
        "chord parameterizations; refusing to cache an all-keep failure label: "
        + "; ".join(diagnostics)
    )


def _canonical_boehm_teacher_start(
    true_parameters: torch.Tensor,
    points: torch.Tensor,
    true_internal_knots: torch.Tensor,
    true_internal_knot_mask: torch.Tensor,
    candidate_count: int,
    config: OneShotTeacherConfig,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Create a proposal-independent, exactly supervised teacher start.

    The canonical labelled knots are deterministically refined to ``Kc`` in
    their generating parameter domain.  A full-candidate refit must pass before
    pruning, otherwise the dataset/tolerance contract itself is inconsistent
    and no all-keep failure label is cached.
    """

    parameters = _strict_unit_parameterization(true_parameters)
    if parameters is None:
        raise RuntimeError("true parameters are not strictly increasing")
    candidates = build_redundant_boehm_candidates(
        true_internal_knots.unsqueeze(0),
        true_internal_knot_mask.to(torch.bool).unsqueeze(0),
        candidate_count,
    )[0]
    if candidates.numel() > 1 and torch.any(candidates[1:] <= candidates[:-1]):
        raise RuntimeError("canonical Boehm candidates are not strictly increasing")
    fit, feasible, rank_valid = _full_candidate_standard_bspline_fit(
        parameters,
        points,
        candidates,
        config,
    )
    if not feasible:
        raise RuntimeError(
            "canonical Boehm all-candidate start is infeasible: "
            f"RMS={float(fit.fit_rmse):.6g}, rank_ok={int(rank_valid)}, "
            f"epsilon={config.error_tolerance:.6g}. The clean canonical labels, "
            "observation noise and requested tolerance are inconsistent."
        )
    return parameters, candidates, float(fit.fit_rmse)


def _map_teacher_positions_back_to_proposal_domain(
    teacher: OneShotTeacherBatch,
    *,
    teacher_parameters: torch.Tensor,
    proposal_parameters: torch.Tensor,
    proposal_knots: torch.Tensor,
) -> OneShotTeacherBatch:
    """Express fallback-domain relocated knots in the immutable proposal domain."""

    count = int(teacher.teacher_count[0])
    if count == 0:
        return teacher
    mapped = warp_internal_knots_to_parameterization(
        teacher.teacher_internal_knots[0, :count],
        teacher_parameters,
        proposal_parameters,
    )
    packed = teacher.teacher_internal_knots.clone()
    packed[0, :count] = mapped
    anchors = proposal_knots[teacher.teacher_retained_mask[0]]
    shifts = (mapped - anchors).abs()
    relocation_mean = teacher.teacher_relocation_mean_abs.clone()
    relocation_max = teacher.teacher_relocation_max_abs.clone()
    relocation_mean[0] = shifts.mean()
    relocation_max[0] = shifts.max()
    return replace(
        teacher,
        teacher_internal_knots=packed,
        teacher_relocation_mean_abs=relocation_mean,
        teacher_relocation_max_abs=relocation_max,
    )


def _build_or_load_teacher_cache(
    *,
    model: SplineFittingNetwork,
    dataset: SyntheticCubicBSplineDataset,
    cache_path: Path,
    config: OneShotTeacherConfig,
    batch_size: int,
    device: torch.device,
    reuse: bool,
    max_fallback_fraction: float,
    teacher_start_mode: str = "learned_proposal",
) -> OneShotTeacherBatch:
    if teacher_start_mode not in _TEACHER_START_MODES:
        raise ValueError(f"unsupported teacher start mode: {teacher_start_mode!r}")
    feasibility_policy = _teacher_feasibility_policy(teacher_start_mode)
    sample_count = len(dataset)
    expected_indices = torch.arange(sample_count, dtype=torch.long)
    sample = dataset[0]
    expected_shapes = {
        "parameters": (sample_count, int(sample["points"].shape[0])),
        "points": (sample_count, *tuple(sample["points"].shape)),
        "candidate_knots": (
            sample_count,
            int(model.candidate_head.num_candidates),
        ),
    }
    domain_metadata_path = _teacher_start_domain_metadata_path(cache_path)
    if reuse and cache_path.exists():
        print(f"Loading verified offline teacher cache: {cache_path}", flush=True)
        teacher = load_one_shot_teacher_cache(
            cache_path,
            expected_config=config,
            expected_sample_indices=expected_indices,
            expected_input_shapes=expected_shapes,
        )
        if not domain_metadata_path.exists():
            raise RuntimeError(
                "teacher cache predates start-domain certification; rerun with "
                "--no-reuse-teacher-cache to regenerate it"
            )
        try:
            start_metadata = json.loads(
                domain_metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("teacher start-domain metadata is unreadable") from error
        if not isinstance(start_metadata, Mapping):
            raise RuntimeError("teacher start-domain metadata must be an object")
        metadata_version = start_metadata.get("version")
        if metadata_version == 1:
            cached_start_mode = "learned_proposal"
        elif metadata_version == _TEACHER_START_DOMAIN_METADATA_VERSION:
            cached_start_mode = start_metadata.get("teacher_start_mode")
            if start_metadata.get("slot_alignment") != "ordered_rank":
                raise RuntimeError("teacher cache slot-alignment policy changed")
            if start_metadata.get("position_storage_domain") != "fixed_proposal_t0":
                raise RuntimeError("teacher cache position-storage domain changed")
            if not isinstance(
                start_metadata.get("ordered_rank_anchor_diagnostics"), Mapping
            ):
                raise RuntimeError(
                    "teacher cache lacks ordered-rank anchor diagnostics"
                )
        else:
            raise RuntimeError("unsupported teacher start-domain metadata version")
        if cached_start_mode != teacher_start_mode:
            raise RuntimeError(
                "teacher cache start mode differs: "
                f"cached={cached_start_mode!r}, requested={teacher_start_mode!r}"
            )
        if start_metadata.get("feasibility_policy") != feasibility_policy:
            raise RuntimeError("teacher start-domain feasibility policy changed")
        if start_metadata.get("config_fingerprint") != teacher.config_fingerprint:
            raise RuntimeError("teacher start-domain metadata is bound to another cache")
        raw_counts = start_metadata.get("counts")
        if not isinstance(raw_counts, Mapping):
            raise RuntimeError("teacher start-domain counts are missing")
        summary = _validate_teacher_start_domain_counts(
            raw_counts,
            sample_count=sample_count,
            max_fallback_fraction=max_fallback_fraction,
            start_mode=teacher_start_mode,
        )
        print(
            "  verified teacher feasible start domains: "
            + ", ".join(f"{name}={summary[name]}" for name in raw_counts)
            + f" | fallback={summary['fallback_fraction']:.3%}",
            flush=True,
        )
        _report_teacher_relocation(cache_path.stem, teacher)
        return teacher

    # A failed regeneration must not leave an older cache that looks current.
    cache_path.unlink(missing_ok=True)
    domain_metadata_path.unlink(missing_ok=True)
    print(f"Generating offline Hard-RMS teacher cache: {cache_path}", flush=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batches: list[OneShotTeacherBatch] = []
    model.eval()
    processed = 0
    started_at = time.perf_counter()
    fallback_counts = (
        {"canonical_boehm": 0}
        if teacher_start_mode == "canonical_boehm"
        else {"calibrated": 0, "true": 0, "chord": 0}
    )
    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device)
            output = model(points)
            proposal_parameters = (
                output.get("proposal_params", output["params"])
                .detach()
                .cpu()
                .to(torch.float64)
            )
            proposal_knots = (
                output.get("proposal_internal_knots", output["internal_knots"])
                .detach()
                .cpu()
                .to(torch.float64)
            )
            calibrated_parameters = output["params"].detach().cpu().to(torch.float64)
            teacher_points = batch["points"].detach().cpu().to(torch.float64)
            true_parameters = batch.get("true_params")
            chord_parameters = batch.get("chord_params")
            true_internal_knots = batch.get("true_internal_knots")
            true_internal_knot_mask = batch.get("true_internal_knot_mask")
            if true_parameters is not None:
                true_parameters = true_parameters.detach().cpu().to(torch.float64)
            if chord_parameters is not None:
                chord_parameters = chord_parameters.detach().cpu().to(torch.float64)
            if true_internal_knots is not None:
                true_internal_knots = (
                    true_internal_knots.detach().cpu().to(torch.float64)
                )
            if true_internal_knot_mask is not None:
                true_internal_knot_mask = (
                    true_internal_knot_mask.detach().cpu().to(torch.bool)
                )

            if teacher_start_mode == "canonical_boehm" and (
                true_parameters is None
                or true_internal_knots is None
                or true_internal_knot_mask is None
            ):
                raise RuntimeError(
                    "canonical_boehm teacher start requires true_params, "
                    "true_internal_knots and true_internal_knot_mask; it is a "
                    "synthetic supervised-training mode, never a deployment path"
                )

            for row in range(points.shape[0]):
                try:
                    if teacher_start_mode == "canonical_boehm":
                        assert true_parameters is not None
                        assert true_internal_knots is not None
                        assert true_internal_knot_mask is not None
                        start_parameters, start_knots, _ = (
                            _canonical_boehm_teacher_start(
                                true_parameters[row],
                                teacher_points[row],
                                true_internal_knots[row],
                                true_internal_knot_mask[row],
                                int(model.candidate_head.num_candidates),
                                config,
                            )
                        )
                        source = "canonical_boehm"
                    else:
                        start_parameters, start_knots, source, _ = _select_feasible_teacher_start(
                            proposal_parameters[row],
                            teacher_points[row],
                            proposal_knots[row],
                            config,
                            calibrated_parameters=calibrated_parameters[row],
                            true_parameters=(
                                true_parameters[row]
                                if true_parameters is not None
                                else None
                            ),
                            chord_parameters=(
                                chord_parameters[row]
                                if chord_parameters is not None
                                else None
                            ),
                        )
                except RuntimeError as error:
                    sample_id = int(batch["sample_id"][row])
                    raise RuntimeError(
                        f"offline teacher sample {sample_id} has no feasible "
                        f"{teacher_start_mode} all-candidate start. {error}"
                    ) from error
                teacher_row = build_one_shot_teacher_batch(
                    start_parameters.unsqueeze(0),
                    teacher_points[row : row + 1],
                    start_knots.unsqueeze(0),
                    sample_indices=batch["sample_id"][row : row + 1],
                    config=config,
                )
                if not bool(teacher_row.teacher_threshold_satisfied[0]):
                    raise RuntimeError(
                        "offline teacher became infeasible after starting from a "
                        "verified full-candidate fit"
                    )
                # Cache positions in immutable t0 coordinates. Candidate pruning
                # itself used t1/true/chord, while later feedback losses transport
                # this target from t0 to the current t1 exactly once.
                teacher_row = _map_teacher_positions_back_to_proposal_domain(
                    teacher_row,
                    teacher_parameters=start_parameters,
                    proposal_parameters=proposal_parameters[row],
                    proposal_knots=proposal_knots[row],
                )
                fallback_counts[source] += 1
                fallback_so_far = (
                    fallback_counts.get("true", 0)
                    + fallback_counts.get("chord", 0)
                )
                if (
                    teacher_start_mode == "learned_proposal"
                    and fallback_so_far
                    > max_fallback_fraction * sample_count + 1e-12
                ):
                    raise RuntimeError(
                        "offline teacher already requires more true/chord fallbacks "
                        "than --max-teacher-fallback-fraction permits; aborting "
                        "before writing a cache. For labelled synthetic training, "
                        "use --teacher-start-mode canonical_boehm instead of "
                        "loosening the fallback limit"
                    )
                batches.append(teacher_row)
            processed += points.shape[0]
            _show_sample_progress(
                "Hard-RMS teacher",
                processed,
                sample_count,
                started_at,
            )
    teacher = _concatenate_teacher_batches(batches, config)
    if not torch.equal(teacher.sample_indices, expected_indices):
        raise RuntimeError("offline teacher rows are not in dataset index order")
    start_summary = _validate_teacher_start_domain_counts(
        fallback_counts,
        sample_count=sample_count,
        max_fallback_fraction=max_fallback_fraction,
        start_mode=teacher_start_mode,
    )
    relocation_summary = _teacher_relocation_metadata(teacher)
    try:
        save_one_shot_teacher_cache(cache_path, teacher)
        domain_metadata_path.write_text(
            json.dumps(
                {
                    "version": _TEACHER_START_DOMAIN_METADATA_VERSION,
                    "teacher_start_mode": teacher_start_mode,
                    "feasibility_policy": feasibility_policy,
                    "slot_alignment": "ordered_rank",
                    "position_storage_domain": "fixed_proposal_t0",
                    "ordered_rank_anchor_diagnostics": {
                        "mean_abs_shift": relocation_summary[
                            "ordered_rank_anchor_shift_mean"
                        ],
                        "max_abs_shift": relocation_summary[
                            "ordered_rank_anchor_shift_max"
                        ],
                        "over_relocation_limit_fraction": relocation_summary[
                            "ordered_rank_anchor_shift_over_relocation_limit_fraction"
                        ],
                        "relocation_limit": config.relocation_max_shift,
                    },
                    "config_fingerprint": teacher.config_fingerprint,
                    "counts": dict(fallback_counts),
                    "fallback_fraction": start_summary["fallback_fraction"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError):
        cache_path.unlink(missing_ok=True)
        domain_metadata_path.unlink(missing_ok=True)
        raise
    print(
        "  teacher feasible start domains: "
        + ", ".join(f"{name}={count}" for name, count in fallback_counts.items())
        + f" | fallback={start_summary['fallback_fraction']:.3%}",
        flush=True,
    )
    _report_teacher_relocation(cache_path.stem, teacher)
    return teacher


def _set_one_shot_trainable(
    model: SplineFittingNetwork,
    *,
    calibrate_positions: bool,
) -> list[torch.nn.Parameter]:
    """Freeze proposal geometry and expose cache-compatible one-shot modules."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    module_names = ["keep_head", "adaptive_threshold_head"]
    fixed_geometry = bool(
        getattr(model.pruning_head, "one_shot_fixed_proposal_geometry", False)
    )
    joint_refinement = bool(
        getattr(
            model.pruning_head,
            "one_shot_joint_position_refinement",
            False,
        )
    )
    survivor_relocation = bool(
        getattr(model.pruning_head, "one_shot_survivor_relocation", False)
    )
    if fixed_geometry:
        module_names.extend(
            [
                "one_shot_selector_attention",
                "one_shot_selector_attention_norm",
                "one_shot_selector_feed_forward",
                "one_shot_selector_output_norm",
                "one_shot_selector_extra_attention",
                "one_shot_selector_extra_attention_norm",
                "one_shot_selector_extra_feed_forward",
                "one_shot_selector_extra_output_norm",
            ]
        )
    else:
        module_names.append("position_to_keep_feedback")
    if calibrate_positions and survivor_relocation:
        # v13 keeps the complete bidirectional path trainable:
        # preliminary position -> Keep feedback -> final selected-only moves.
        # Earlier v12 runs reset the preliminary/feedback terminal layers to
        # zero but omitted them here, silently disabling position-to-Keep
        # interaction for the whole calibration stage.
        module_names.extend(
            [
                "joint_preliminary_position_projection",
                "joint_preliminary_position_norm",
                "joint_preliminary_position_head",
                "joint_position_to_keep_feedback",
                "joint_position_to_keep_norm",
                "joint_final_position_projection",
                "joint_final_position_norm",
                "joint_final_position_head",
                "survivor_relative_projection",
                "survivor_relocation_input_norm",
                "survivor_relocation_attention",
                "survivor_relocation_attention_norm",
                "survivor_relocation_feed_forward",
                "survivor_relocation_output_norm",
                "survivor_relocation_head",
            ]
        )
    elif calibrate_positions and joint_refinement:
        module_names.extend(
            [
                "joint_preliminary_position_projection",
                "joint_preliminary_position_norm",
                "joint_preliminary_position_head",
                "joint_position_to_keep_feedback",
                "joint_position_to_keep_norm",
                "joint_final_position_projection",
                "joint_final_position_norm",
                "joint_final_position_head",
            ]
        )
    elif calibrate_positions and not fixed_geometry:
        module_names.extend(
            [
                "keep_context_projection",
                "keep_context_norm",
                "position_residual_head",
            ]
        )
    trainable: list[torch.nn.Parameter] = []
    for module_name in module_names:
        module = getattr(model.pruning_head, module_name, None)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("one-shot head exposes no trainable selection parameters")
    return trainable


def _set_parameter_feedback_trainable(
    model: SplineFittingNetwork,
) -> list[torch.nn.Parameter]:
    """Freeze proposal/pilot modules; calibrate the late joint feedback loop."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    feedback_head = getattr(model, "parameter_feedback_head", None)
    if feedback_head is None:
        raise RuntimeError("parameter-feedback calibration requires a v14/v15 head")
    if feedback_head.fusion_mode == "fast_global":
        modules = (
            feedback_head.global_input_projection,
            feedback_head.global_chord_adjust_head,
        )
        trainable = [feedback_head.chord_blend_weight]
        for module in modules:
            trainable.extend(module.parameters())
    elif feedback_head.fusion_mode == "fast_structural":
        modules = (
            feedback_head.fast_input_projection,
            feedback_head.fast_gap_logit_head,
            feedback_head.fast_chord_adjust_head,
        )
        trainable = [feedback_head.chord_blend_weight]
        for module in modules:
            trainable.extend(module.parameters())
    else:
        trainable = list(feedback_head.parameters())
    joint_head = getattr(model, "joint_parameter_structure_head", None)
    if joint_head is not None:
        trainable.extend(joint_head.parameters())
        # The second relocation pass must adapt to the newly selected final
        # mask.  These modules are shared with the preliminary pass but the
        # fixed proposal/pilot and teacher-cache fingerprint remain frozen.
        for module_name in (
            "survivor_relative_projection",
            "survivor_relocation_input_norm",
            "survivor_relocation_attention",
            "survivor_relocation_attention_norm",
            "survivor_relocation_feed_forward",
            "survivor_relocation_output_norm",
            "survivor_relocation_head",
        ):
            module = getattr(model.pruning_head, module_name, None)
            if module is not None:
                trainable.extend(module.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    if not trainable:
        raise RuntimeError("parameter-feedback head exposes no trainable parameters")
    return trainable


def _reset_one_shot_selector_to_neutral(
    model: SplineFittingNetwork,
    *,
    initial_probability: float,
) -> None:
    """Discard conservative proposal-time Keep logits before distillation.

    Candidate pretraining forces every gate open, so its selector weights carry
    no learned pruning information.  A deterministic neutral reset avoids
    starting the offline-teacher student at p=0.9 (the all-keep basin) and also
    makes proposal-checkpoint/cache reuse reproducible.
    """
    head = model.pruning_head
    if not getattr(head, "one_shot_adaptive", False):
        raise ValueError("neutral selector reset requires an adaptive one-shot head")
    if not 0.5 < initial_probability < 1.0:
        raise ValueError("initial_probability must lie strictly in (0.5, 1)")
    initial_logit = torch.logit(torch.tensor(initial_probability)).item()
    with torch.no_grad():
        head.keep_head.weight.zero_()
        head.keep_head.bias.fill_(initial_logit)
        for module_name in (
            "adaptive_threshold_head",
            "position_to_keep_feedback",
            "joint_position_to_keep_feedback",
            "joint_preliminary_position_head",
            "joint_final_position_head",
            "survivor_relocation_head",
        ):
            module = getattr(head, module_name, None)
            if module is None:
                continue
            final_linear = (
                module[-1] if isinstance(module, torch.nn.Sequential) else module
            )
            final_linear.weight.zero_()
            final_linear.bias.zero_()


def _reset_and_freeze_late_feedback_heads(
    model: SplineFittingNetwork,
) -> None:
    """Keep v14's late feedback loop out of proposal/teacher generation.

    Candidate pretraining is responsible only for ``ParameterHead`` and the
    redundant proposal geometry.  The parameter-feedback and joint
    parameter/structure heads are calibrated *after* the selector has learned
    from a fixed offline teacher.  Allowing their zero-initialised terminal
    layers to train during proposal pretraining makes the exact validation
    parameterisation depend on an untrained KeepMask and can turn a feasible
    proposal into a fallback-only teacher start.

    Resetting terminal projections also makes adapting an older v14 checkpoint
    safe: inherited late-feedback corrections cannot leak into the new teacher
    cache.  Non-terminal feature projections need not be reset because every
    output path below is gated by one of these zero terminal projections.
    """

    feedback_head = getattr(model, "parameter_feedback_head", None)
    joint_head = getattr(model, "joint_parameter_structure_head", None)
    with torch.no_grad():
        if feedback_head is not None:
            feedback_head.chord_blend_weight.zero_()
            for module_name in (
                "gap_logit_head",
                "fast_gap_logit_head",
                "fast_chord_adjust_head",
                "global_chord_adjust_head",
            ):
                module = getattr(feedback_head, module_name, None)
                if module is not None:
                    module.weight.zero_()
                    module.bias.zero_()
        if joint_head is not None:
            for module_name in (
                "token_delta_head",
                "keep_delta_head",
                "position_delta_head",
            ):
                module = getattr(joint_head, module_name, None)
                if module is not None:
                    module.weight.zero_()
                    module.bias.zero_()
            joint_head.relocation_scale.zero_()

    for head in (feedback_head, joint_head):
        if head is not None:
            for parameter in head.parameters():
                parameter.requires_grad_(False)


def _required_proposal_pass_rate(
    *,
    deployment_pass_rate_target: float,
    max_teacher_fallback_fraction: float,
    teacher_start_mode: str = "learned_proposal",
) -> float:
    """Return the exact-fit target used to select a proposal checkpoint."""

    if teacher_start_mode == "canonical_boehm":
        return float(deployment_pass_rate_target)
    if teacher_start_mode != "learned_proposal":
        raise ValueError(f"unsupported teacher start mode: {teacher_start_mode!r}")
    return max(
        float(deployment_pass_rate_target),
        1.0 - float(max_teacher_fallback_fraction),
    )


def _require_feasible_proposal_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    deployment_pass_rate_target: float,
    max_teacher_fallback_fraction: float,
    teacher_start_mode: str = "learned_proposal",
) -> float:
    """Check proposal feasibility only when teacher labels depend on it.

    Historical ``learned_proposal`` labels require the exact all-candidate pass
    rate to satisfy both deployment and fallback contracts.  Supervised
    ``canonical_boehm`` labels do not consume learned proposal geometry, so its
    pass rate is returned only as an optional diagnostic.
    """

    required_pass_rate = _required_proposal_pass_rate(
        deployment_pass_rate_target=deployment_pass_rate_target,
        max_teacher_fallback_fraction=max_teacher_fallback_fraction,
        teacher_start_mode=teacher_start_mode,
    )

    raw_pass_rate = checkpoint.get("best_threshold_satisfied_rate")
    if teacher_start_mode == "canonical_boehm":
        try:
            observed_pass_rate = float(raw_pass_rate)
        except (TypeError, ValueError):
            return float("nan")
        return observed_pass_rate if math.isfinite(observed_pass_rate) else float("nan")
    try:
        pass_rate = float(raw_pass_rate)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "proposal checkpoint has no exact all-candidate standard B-spline "
            "validation pass rate; rerun candidate pretraining with the current "
            "script before generating an offline teacher"
        ) from error
    if not math.isfinite(pass_rate):
        raise RuntimeError(
            "proposal checkpoint exact validation pass rate is not finite; "
            "retrain the proposal before generating an offline teacher"
        )
    if pass_rate + 1e-12 < required_pass_rate:
        raise RuntimeError(
            "selected proposal checkpoint did not reach the required exact "
            "all-candidate standard B-spline validation pass rate: "
            f"{pass_rate:.3%} < {required_pass_rate:.3%}. The requirement is "
            "max(--deployment-pass-rate-target="
            f"{deployment_pass_rate_target:.3%}, 1 - "
            "--max-teacher-fallback-fraction="
            f"{1.0 - max_teacher_fallback_fraction:.3%}). Offline teacher "
            "generation has not started. Improve/retrain the proposal; only "
            "relax the fallback budget explicitly for diagnostic runs."
        )
    return pass_rate


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train high-recall proposals and a one-shot adaptive LearnedKeep "
            "student distilled from an offline Hard-RMS teacher."
        )
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--candidate-pretrain-epochs", type=int, default=20)
    parser.add_argument(
        "--proposal-parameter-warmup-epochs",
        type=int,
        default=None,
        help=(
            "Epochs at the start of the fixed proposal budget that optimize "
            "only supervised ParameterHead output. Omit for an automatic "
            "min(15, candidate-pretrain-epochs // 3) warm-up; it becomes zero "
            "for fewer than three pretraining epochs. An explicit value must "
            "lie in [0, candidate-pretrain-epochs)."
        ),
    )
    parser.add_argument(
        "--proposal-checkpoint",
        type=Path,
        default=None,
        help=(
            "Load a previously selected proposal checkpoint. This is required "
            "when candidate pretraining epochs are zero and makes teacher-cache "
            "reuse reproducible."
        ),
    )
    parser.add_argument(
        "--selector-calibration-epochs",
        "--keep-position-calibration-epochs",
        "--joint-finetune-epochs",
        dest="keep_position_calibration_epochs",
        type=int,
        default=20,
        help=(
            "Final low-learning-rate joint Keep/position calibration epochs. "
            "The immutable proposal slots remain teacher-compatible while the "
            "separate deployment positions are updated."
        ),
    )
    parser.add_argument(
        "--parameter-feedback-epochs",
        type=int,
        default=10,
        help=(
            "Additional v14 epochs after selector/relocation training. In the "
            "default joint mode this trains parameter correction, final Keep/"
            "position refinement and final-mask survivor relocation together; "
            "these epochs are in addition to --epochs. Set zero for v13."
        ),
    )
    parser.add_argument(
        "--joint-parameter-structure-feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Close the v14 loop after parameter correction: locally revisit "
            "candidate positions and Keep logits, then select and relocate the "
            "final survivor set once. Disable for legacy v14 behavior."
        ),
    )
    parser.add_argument(
        "--enforce-ordered-joint-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Project the v14 joint all-candidate and final-survivor positions "
            "onto legal ordered knot sets."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--log-every-batches", type=int, default=20)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show live train/validation batch progress bars in an interactive terminal.",
    )
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--val-seed", type=int, default=10000)
    parser.add_argument("--num-points", type=int, default=192)
    parser.add_argument("--point-dim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--min-control-points", type=int, default=8)
    parser.add_argument("--max-control-points", type=int, default=24)
    parser.add_argument("--candidate-knots", type=int, default=28)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.001)
    parser.add_argument("--knot-nonuniformity", type=float, default=0.65)
    parser.add_argument("--sampling-nonuniformity", type=float, default=0.45)
    parser.add_argument("--turn-strength", type=float, default=0.45)
    parser.add_argument(
        "--certified-minimal-source",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Generate labels from a clean source whose knot set is certified "
            "minimal among subsets of the source knots at --fit-tolerance. "
            "Observation noise is then added only to the network input."
        ),
    )
    parser.add_argument(
        "--minimality-margin",
        type=float,
        default=0.2,
        help="Relative safety margin required of every one-knot deletion.",
    )
    parser.add_argument(
        "--minimality-max-attempts",
        type=int,
        default=16,
        help="Maximum deterministic rejection attempts per certified sample.",
    )
    parser.add_argument(
        "--minimality-audit-points",
        type=int,
        default=512,
        help=(
            "Clean dense-reference samples used by the certificate; zero reuses "
            "the network sampling locations."
        ),
    )
    parser.add_argument(
        "--oscillation-amplitude",
        type=float,
        default=0.3,
        help="Local-detail amplitude for certified complexity-aligned curves.",
    )
    parser.add_argument(
        "--fit-tolerance",
        type=float,
        default=5e-3,
        help=(
            "Normalized mean Euclidean RMS threshold. It defines canonical "
            "labels, offline teacher pruning and the one-shot training target."
        ),
    )
    parser.add_argument("--candidate-match-tolerance", type=float, default=0.01)
    parser.add_argument("--min-knot-gap", type=float, default=1e-3)
    parser.add_argument(
        "--parameter-gap-reference",
        choices=("learned", "chord_residual", "uniform_residual"),
        default="uniform_residual",
        help=(
            "ParameterHead base measure. New training defaults to a bounded "
            "residual around strictly uniform gaps; chord_residual instead "
            "uses deployable chord-length parameters."
        ),
    )
    parser.add_argument(
        "--parameter-residual-logit-limit",
        type=float,
        default=_NEW_TRAINING_PARAMETER_RESIDUAL_LOGIT_LIMIT,
        help=(
            "Bound on centred ParameterHead residual logits. New training uses "
            "2.5 for sufficient parameter-warp range; an unadapted proposal "
            "checkpoint keeps its recorded historical value."
        ),
    )
    parser.add_argument("--pruning-residual-bandwidth", type=float, default=0.05)
    parser.add_argument("--lambda-poly", type=float, default=1e-6)
    parser.add_argument("--lambda-knot", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--parameter-feedback-lr", type=float, default=1e-4)
    parser.add_argument(
        "--parameter-feedback-max-logit-shift",
        type=float,
        default=0.5,
        help="Maximum additive correction to each proposal parameter-gap logit.",
    )
    parser.add_argument(
        "--parameter-feedback-fusion-mode",
        choices=(
            "fast_global",
            "fast_structural",
            "gaussian_pool",
            "cross_attention",
        ),
        default=None,
        help=(
            "Structure-to-parameter fusion. v14_joint defaults to full "
            "cross-attention, while --no-joint-parameter-structure-feedback "
            "retains legacy fast_global unless this option is explicit."
        ),
    )
    parser.add_argument(
        "--joint-feedback-local-bandwidth",
        type=float,
        default=0.08,
        help="Gaussian locality scale for corrected-parameter candidate attention.",
    )
    parser.add_argument(
        "--joint-feedback-max-keep-logit-shift",
        type=float,
        default=2.0,
        help="Maximum residual change to each final Keep logit.",
    )
    parser.add_argument(
        "--parameter-feedback-initial-chord-blend",
        type=float,
        default=0.6,
        help=(
            "Chord-gap weight used to seed the v14 feedback candidate after the "
            "strict identity baseline has been saved; it remains learnable."
        ),
    )
    parser.add_argument(
        "--selector-lr",
        type=float,
        default=5e-4,
        help=(
            "Learning rate for the small frozen-proposal one-shot selector. "
            "It is intentionally higher than the proposal learning rate so "
            "the mask can leave its conservative initialization."
        ),
    )
    parser.add_argument(
        "--relocation-lr-scale",
        type=float,
        default=0.25,
        help=(
            "Calibration learning rate as a fraction of --selector-lr; the "
            "v12 interaction block needs more adaptation than the old linear head."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--lambda-fit",
        type=float,
        default=0.0,
        help=(
            "Proposal-stage truncated-power fit weight. The default is zero; "
            "exact all-candidate standard-B-spline validation selects proposals."
        ),
    )
    parser.add_argument(
        "--lambda-threshold-violation",
        type=float,
        default=0.0,
        help=(
            "Proposal-stage truncated-power threshold penalty. Keep zero to "
            "avoid ill-conditioned surrogate gradients."
        ),
    )
    parser.add_argument(
        "--lambda-one-shot-surrogate-fit",
        type=float,
        default=0.0,
        help=(
            "Truncated-power fit weight during one-shot distillation/calibration. "
            "Keep at zero: standard-B-spline Hard-RMS teacher labels supervise "
            "selection, while the surrogate remains diagnostic only."
        ),
    )
    parser.add_argument(
        "--lambda-one-shot-surrogate-threshold",
        type=float,
        default=0.0,
        help=(
            "Truncated-power threshold-violation weight during one-shot stages. "
            "Keep at zero to prevent the all-keep shortcut."
        ),
    )
    parser.add_argument(
        "--lambda-true-params",
        type=float,
        default=_NEW_TRAINING_TRUE_PARAMETER_WEIGHT,
        help=(
            "Proposal-stage normalized true-parameter weight. New training "
            "uses 2.0 so ParameterHead supervision remains comparable to the "
            "candidate coverage and redundant-vector objectives."
        ),
    )
    parser.add_argument(
        "--lambda-true-parameter-gap",
        type=float,
        default=1.0,
        help=(
            "Proposal-pretraining weight for robust log-space supervision of "
            "each adjacent parameter interval. This complements cumulative "
            "parameter-coordinate MSE and is disabled after proposal pretraining."
        ),
    )
    parser.add_argument(
        "--lambda-feedback-fit",
        type=float,
        default=0.0,
        help="Legacy truncated-power feedback fit weight; v15 defaults to zero.",
    )
    parser.add_argument(
        "--lambda-feedback-threshold-violation",
        type=float,
        default=0.0,
        help="Legacy truncated-power feedback threshold weight.",
    )
    parser.add_argument(
        "--lambda-feedback-deployment-fit",
        type=float,
        default=0.25,
        help=(
            "v15 normalized MSE weight from the differentiable standard "
            "B-spline deployment refit."
        ),
    )
    parser.add_argument(
        "--lambda-feedback-deployment-threshold",
        type=float,
        default=2.0,
        help=(
            "v15 hinge weight for exact deployment RMS above --fit-tolerance."
        ),
    )
    parser.add_argument("--lambda-feedback-true-params", type=float, default=1.0)
    parser.add_argument("--lambda-feedback-chord-prior", type=float, default=0.05)
    parser.add_argument("--lambda-feedback-identity", type=float, default=0.01)
    parser.add_argument("--lambda-feedback-joint-keep", type=float, default=1.0)
    parser.add_argument(
        "--lambda-feedback-joint-critical-recall", type=float, default=0.5
    )
    parser.add_argument("--lambda-feedback-joint-position", type=float, default=2.0)
    parser.add_argument(
        "--lambda-feedback-joint-set-position", type=float, default=1.0
    )
    parser.add_argument("--lambda-feedback-joint-spacing", type=float, default=0.5)
    parser.add_argument("--lambda-feedback-joint-count", type=float, default=2.0)
    parser.add_argument(
        "--feedback-parameter-error-scale",
        type=float,
        default=0.02,
        help=(
            "Normalisation scale for v14 parameter RMSE supervision. The old "
            "unnormalised parameter loss was too small relative to fit loss."
        ),
    )
    parser.add_argument("--lambda-candidate-coverage", type=float, default=5.0)
    parser.add_argument(
        "--lambda-redundant-candidate",
        type=float,
        default=_NEW_TRAINING_REDUNDANT_CANDIDATE_WEIGHT,
        help=(
            "Supervise all Kc ordered proposal slots with deterministic Boehm-"
            "style largest-interval midpoint insertion targets. Position and "
            "interval-gap errors are normalized by the strict match tolerance; "
            "the 2.5 default balances this full-vector term with nearest-knot "
            "coverage."
        ),
    )
    parser.add_argument(
        "--proposal-parameter-error-scale",
        type=float,
        default=0.02,
        help="RMSE scale used to normalize proposal true-parameter supervision.",
    )
    parser.add_argument("--lambda-candidate-repulsion", type=float, default=0.05)
    parser.add_argument("--lambda-keep", type=float, default=1.0)
    parser.add_argument(
        "--lambda-remove-action",
        type=float,
        default=0.0,
        help="Compatibility option; one-shot training does not use sequential actions.",
    )
    parser.add_argument("--lambda-knot-position", type=float, default=2.0)
    parser.add_argument(
        "--lambda-deletion-cost",
        type=float,
        default=0.0,
        help="Compatibility option; offline teacher risk replaces this v7 term.",
    )
    parser.add_argument("--lambda-teacher-risk", type=float, default=0.5)
    parser.add_argument("--lambda-teacher-ranking", type=float, default=1.0)
    parser.add_argument("--lambda-teacher-distribution", type=float, default=2.0)
    parser.add_argument("--lambda-teacher-critical-recall", type=float, default=1.0)
    parser.add_argument("--lambda-teacher-false-positive", type=float, default=1.0)
    parser.add_argument("--teacher-ranking-margin", type=float, default=1.0)
    parser.add_argument("--lambda-teacher-count", type=float, default=4.0)
    parser.add_argument("--lambda-policy-count", type=float, default=4.0)
    parser.add_argument("--lambda-canonical-selection", type=float, default=0.5)
    parser.add_argument("--lambda-joint-position", type=float, default=1.0)
    parser.add_argument(
        "--lambda-joint-fit",
        type=float,
        default=0.0,
        help=(
            "Optional truncated-power diagnostic-fit weight during joint "
            "calibration. The v13 default is zero because checkpoint selection "
            "uses the actual standard-B-spline deployment refit."
        ),
    )
    parser.add_argument(
        "--lambda-joint-threshold-violation",
        type=float,
        default=0.0,
        help=(
            "Optional truncated-power threshold penalty during calibration. "
            "The v13 default is zero; real standard-B-spline validation pass "
            "rate and RMS select the deployed checkpoint."
        ),
    )
    parser.add_argument("--lambda-complexity", type=float, default=0.1)
    parser.add_argument("--positive-keep-weight", type=float, default=1.5)
    parser.add_argument(
        "--one-shot-selection-policy",
        choices=("threshold", "mass_topk"),
        default="mass_topk",
        help=(
            "Discrete LearnedKeep construction. mass_topk derives cardinality "
            "from probability mass and selects a global top-K subset."
        ),
    )
    parser.add_argument(
        "--one-shot-safety-sigma",
        type=float,
        default=0.25,
        help=(
            "Bernoulli uncertainty reserve used by mass_topk. It adds sigma*std "
            "to probability mass before rounding the retained count."
        ),
    )
    parser.add_argument(
        "--one-shot-selector-layers",
        type=int,
        default=2,
        help="Number of position-aware candidate interaction blocks.",
    )
    parser.add_argument(
        "--one-shot-coverage-bins",
        type=int,
        default=0,
        help=(
            "Reserve the highest-probability candidate in each parameter-domain "
            "bin before filling the remaining mass_topk budget; zero disables it."
        ),
    )
    parser.add_argument(
        "--candidate-local-attention-bandwidth",
        type=float,
        default=0.08,
        help=(
            "Gaussian parameter-space bandwidth for anchor-centred candidate "
            "cross-attention; zero restores the historical global attention."
        ),
    )
    parser.add_argument(
        "--candidate-interval-logit-limit",
        type=float,
        default=_NEW_TRAINING_CANDIDATE_INTERVAL_LOGIT_LIMIT,
        help=(
            "Bound centred interval logits before softmax. New training uses "
            "1.75 to retain target coverage without permitting proposal "
            "collapse; zero restores the historical unbounded parameterization."
        ),
    )
    parser.add_argument(
        "--candidate-position-parameterization",
        choices=("interval_softmax", "bounded_anchor_residual"),
        default="interval_softmax",
        help=(
            "Ordered proposal parameterization. interval_softmax remains the "
            "default because it can represent the complete Boehm target."
        ),
    )
    parser.add_argument(
        "--candidate-coverage-tolerances",
        type=float,
        nargs="+",
        default=(0.005, 0.01, 0.02),
        help="Multi-scale proposal-recall margins used during pretraining.",
    )
    parser.add_argument(
        "--one-shot-max-position-shift",
        type=float,
        default=0.15,
        help=("Maximum absolute v12 post-KeepMask survivor relocation in [0,1]."),
    )
    parser.add_argument(
        "--initial-keep-probability",
        type=float,
        default=0.55,
        help=(
            "Neutral-but-conservative selector initialization. Values near 0.5 "
            "avoid both the historical p=0.9 all-keep basin and an immediate "
            "all-remove mask at the hard cutoff."
        ),
    )
    parser.add_argument(
        "--deployment-pass-rate-target",
        type=float,
        default=0.97,
        help=(
            "Validation feasibility target. Before reaching it v13 ranks by "
            "real standard-B-spline pass rate and mean/P95 RMS; afterwards it "
            "minimizes retained knots, with fit and localization as tie-breakers."
        ),
    )
    parser.add_argument(
        "--selector-checkpoint-warmup-epochs",
        type=int,
        default=10,
        help=(
            "Ignore the selector's conservative initialization when choosing "
            "the best distillation checkpoint. The effective value is clipped "
            "to leave at least one selectable epoch."
        ),
    )
    parser.add_argument("--knot-position-beta", type=float, default=0.01)
    parser.add_argument(
        "--exact-deletion-supervision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate exact standard-B-spline Hard-RMS labels offline. "
            "Disabling this is unsupported by the v13 one-shot objective."
        ),
    )
    parser.add_argument(
        "--teacher-start-mode",
        choices=_TEACHER_START_MODES,
        default="canonical_boehm",
        help=(
            "Source of the offline Hard-RMS candidate set. canonical_boehm "
            "deterministically inserts labelled canonical knots to Kc under "
            "true synthetic parameters, then maps the final teacher positions "
            "to fixed proposal t0; this clean supervised mode does not require "
            "the learned proposal to be feasible before distillation. "
            "learned_proposal preserves the historical calibrated/true/chord "
            "fallback policy. Deployment never consumes ground truth."
        ),
    )
    parser.add_argument("--teacher-risk-temperature", type=float, default=0.1)
    parser.add_argument("--teacher-batch-size", type=int, default=4)
    parser.add_argument(
        "--max-teacher-fallback-fraction",
        type=float,
        default=0.0,
        help=(
            "Maximum fraction of offline-teacher rows allowed to use true/chord "
            "parameters because the calibrated network proposal is infeasible "
            "in --teacher-start-mode learned_proposal. It is ignored by the "
            "clean canonical_boehm mode. "
            "Formal training defaults to zero: fallback-domain masks are useful "
            "diagnostics, but are not guaranteed feasible under learned deployment "
            "parameters. Before teacher generation, the selected proposal must "
            "therefore reach at least max(--deployment-pass-rate-target, 1 - this "
            "value) exact validation pass rate; zero requires 100%%."
        ),
    )
    parser.add_argument(
        "--teacher-survivor-relaxation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After greedy deletion, optimize surviving knot positions and retry "
            "deletion before caching the v13 teacher labels."
        ),
    )
    parser.add_argument(
        "--teacher-relaxation-rounds",
        type=int,
        default=2,
        help="Maximum delete-then-relax boundary rounds per teacher sample.",
    )
    parser.add_argument("--teacher-relaxation-sweeps", type=int, default=2)
    parser.add_argument("--teacher-relaxation-grid-size", type=int, default=7)
    parser.add_argument("--teacher-relaxation-restarts", type=int, default=1)
    parser.add_argument(
        "--teacher-relaxation-min-gap",
        type=float,
        default=None,
        help="Minimum teacher survivor gap; defaults to --min-knot-gap.",
    )
    parser.add_argument(
        "--teacher-survivor-spacing-weight",
        type=float,
        default=0.25,
        help=(
            "Relative weight of boundary/adjacent survivor-gap supervision "
            "inside the v13 position loss."
        ),
    )
    parser.add_argument(
        "--teacher-cache-dir",
        type=Path,
        default=None,
        help="Directory for train/validation offline teacher .pt caches.",
    )
    parser.add_argument(
        "--reuse-teacher-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse only a cache whose config, proposal fingerprint and shapes match.",
    )
    parser.add_argument(
        "--resample-train-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Regenerate curves each epoch. Unsupported with an index-aligned "
            "offline teacher cache; keep this disabled for v13."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "candidate_pruning_v14_joint.pt",
    )
    args = parser.parse_args()

    if args.epochs < 2:
        parser.error("--epochs must be at least 2")
    if not 0 <= args.candidate_pretrain_epochs < args.epochs:
        parser.error("--candidate-pretrain-epochs must lie in [0, epochs)")
    try:
        args.proposal_parameter_warmup_epochs = (
            _resolve_proposal_parameter_warmup_epochs(
                args.candidate_pretrain_epochs,
                args.proposal_parameter_warmup_epochs,
            )
        )
    except ValueError as error:
        parser.error(str(error))
    if args.parameter_feedback_epochs < 0:
        parser.error("--parameter-feedback-epochs must be non-negative")
    if args.candidate_pretrain_epochs == 0 and args.proposal_checkpoint is None:
        parser.error("zero candidate pretraining requires --proposal-checkpoint")
    if args.proposal_checkpoint is not None and not args.proposal_checkpoint.is_file():
        checkpoint_parent = args.proposal_checkpoint.parent
        nearby = (
            sorted(checkpoint_parent.glob("*.pt")) if checkpoint_parent.is_dir() else []
        )
        nearby_text = (
            "\nAvailable checkpoints in that directory:\n  "
            + "\n  ".join(str(path) for path in nearby)
            if nearby
            else ""
        )
        parser.error(
            f"proposal checkpoint does not exist: {args.proposal_checkpoint}"
            f"{nearby_text}"
        )
    if (
        not 0
        <= args.keep_position_calibration_epochs
        < (args.epochs - args.candidate_pretrain_epochs)
    ):
        parser.error(
            "--keep-position-calibration-epochs must leave at least one "
            "distillation epoch"
        )
    if args.train_size <= 0 or args.val_size <= 0 or args.batch_size <= 0:
        parser.error("dataset and batch sizes must be positive")
    if args.log_every_batches < 0:
        parser.error("--log-every-batches must be non-negative")
    if args.selector_checkpoint_warmup_epochs < 0:
        parser.error("--selector-checkpoint-warmup-epochs must be non-negative")
    if args.relocation_lr_scale <= 0.0:
        parser.error("--relocation-lr-scale must be positive")
    if args.parameter_feedback_lr <= 0.0:
        parser.error("--parameter-feedback-lr must be positive")
    if args.parameter_feedback_max_logit_shift <= 0.0:
        parser.error("--parameter-feedback-max-logit-shift must be positive")
    if args.joint_feedback_local_bandwidth <= 0.0:
        parser.error("--joint-feedback-local-bandwidth must be positive")
    if args.joint_feedback_max_keep_logit_shift <= 0.0:
        parser.error("--joint-feedback-max-keep-logit-shift must be positive")
    if not 0.0 <= args.parameter_feedback_initial_chord_blend <= 1.0:
        parser.error("--parameter-feedback-initial-chord-blend must lie in [0,1]")
    if args.feedback_parameter_error_scale <= 0.0:
        parser.error("--feedback-parameter-error-scale must be positive")
    if args.proposal_parameter_error_scale <= 0.0:
        parser.error("--proposal-parameter-error-scale must be positive")
    if args.parameter_residual_logit_limit <= 0.0:
        parser.error("--parameter-residual-logit-limit must be positive")
    if args.fit_tolerance <= 0.0:
        parser.error("--fit-tolerance must be positive")
    if args.minimality_margin < 0.0:
        parser.error("--minimality-margin must be non-negative")
    if args.minimality_max_attempts < 1:
        parser.error("--minimality-max-attempts must be positive")
    if args.minimality_audit_points not in (0,) and args.minimality_audit_points < 2:
        parser.error("--minimality-audit-points must be zero or at least 2")
    if args.oscillation_amplitude <= 0.0:
        parser.error("--oscillation-amplitude must be positive")
    if args.teacher_risk_temperature <= 0.0 or args.teacher_batch_size <= 0:
        parser.error("teacher temperature and batch size must be positive")
    if not 0.0 <= args.max_teacher_fallback_fraction <= 1.0:
        parser.error("--max-teacher-fallback-fraction must lie in [0,1]")
    if args.teacher_relaxation_rounds < 1:
        parser.error("--teacher-relaxation-rounds must be positive")
    if args.teacher_relaxation_sweeps < 1:
        parser.error("--teacher-relaxation-sweeps must be positive")
    if (
        args.teacher_relaxation_grid_size < 3
        or args.teacher_relaxation_grid_size % 2 == 0
    ):
        parser.error("--teacher-relaxation-grid-size must be odd and at least 3")
    if args.teacher_relaxation_restarts < 1:
        parser.error("--teacher-relaxation-restarts must be positive")
    if (
        args.teacher_relaxation_min_gap is not None
        and args.teacher_relaxation_min_gap < 0.0
    ):
        parser.error("--teacher-relaxation-min-gap must be non-negative")
    if (
        args.teacher_relaxation_min_gap is not None
        and args.teacher_relaxation_min_gap > args.min_knot_gap
    ):
        parser.error(
            "--teacher-relaxation-min-gap cannot exceed --min-knot-gap: "
            "the fixed proposal supplied to the teacher must already satisfy "
            "the requested gap; raise --min-knot-gap as well"
        )
    if args.teacher_survivor_spacing_weight < 0.0:
        parser.error("--teacher-survivor-spacing-weight must be non-negative")
    if not args.exact_deletion_supervision:
        parser.error("v13 requires --exact-deletion-supervision for offline labels")
    if args.resample_train_each_epoch:
        parser.error("offline teacher labels require --no-resample-train-each-epoch")
    if args.candidate_match_tolerance <= 0.0:
        parser.error("--candidate-match-tolerance must be positive")
    if args.attention_heads <= 0 or args.hidden_dim % args.attention_heads:
        parser.error("--attention-heads must divide --hidden-dim")
    max_source_knots = args.max_control_points - 4
    if args.candidate_knots < max_source_knots:
        parser.error(
            "--candidate-knots must cover the maximum source knot count "
            f"({max_source_knots})"
        )
    if args.min_knot_gap * (args.candidate_knots + 1) >= 1.0:
        parser.error("--min-knot-gap leaves no free parameter interval")
    nonnegative = (
        args.lambda_poly,
        args.lambda_knot,
        args.weight_decay,
        args.selector_lr,
        args.lambda_fit,
        args.lambda_threshold_violation,
        args.lambda_one_shot_surrogate_fit,
        args.lambda_one_shot_surrogate_threshold,
        args.lambda_true_params,
        args.lambda_true_parameter_gap,
        args.lambda_feedback_fit,
        args.lambda_feedback_threshold_violation,
        args.lambda_feedback_deployment_fit,
        args.lambda_feedback_deployment_threshold,
        args.lambda_feedback_true_params,
        args.lambda_feedback_chord_prior,
        args.lambda_feedback_identity,
        args.lambda_feedback_joint_keep,
        args.lambda_feedback_joint_critical_recall,
        args.lambda_feedback_joint_position,
        args.lambda_feedback_joint_set_position,
        args.lambda_feedback_joint_spacing,
        args.lambda_feedback_joint_count,
        args.lambda_candidate_coverage,
        args.lambda_redundant_candidate,
        args.lambda_candidate_repulsion,
        args.lambda_keep,
        args.lambda_remove_action,
        args.lambda_knot_position,
        args.lambda_deletion_cost,
        args.lambda_teacher_risk,
        args.lambda_teacher_ranking,
        args.lambda_teacher_distribution,
        args.lambda_teacher_critical_recall,
        args.lambda_teacher_false_positive,
        args.lambda_teacher_count,
        args.lambda_policy_count,
        args.lambda_canonical_selection,
        args.lambda_joint_position,
        args.lambda_joint_fit,
        args.lambda_joint_threshold_violation,
        args.lambda_complexity,
    )
    if any(value < 0.0 for value in nonnegative):
        parser.error("regularization and loss weights must be non-negative")
    if not 0.0 <= args.deployment_pass_rate_target <= 1.0:
        parser.error("--deployment-pass-rate-target must lie in [0, 1]")
    if not 0.5 < args.initial_keep_probability < 1.0:
        parser.error("--initial-keep-probability must lie strictly in (0.5, 1)")
    if args.teacher_ranking_margin < 0.0:
        parser.error("--teacher-ranking-margin must be non-negative")
    if args.one_shot_safety_sigma < 0.0:
        parser.error("--one-shot-safety-sigma must be non-negative")
    if args.one_shot_selector_layers < 1:
        parser.error("--one-shot-selector-layers must be positive")
    if args.one_shot_coverage_bins < 0:
        parser.error("--one-shot-coverage-bins must be non-negative")
    if args.candidate_local_attention_bandwidth < 0.0:
        parser.error("--candidate-local-attention-bandwidth must be non-negative")
    if args.candidate_interval_logit_limit < 0.0:
        parser.error("--candidate-interval-logit-limit must be non-negative")
    if any(value <= 0.0 for value in args.candidate_coverage_tolerances):
        parser.error("--candidate-coverage-tolerances must be positive")
    if not 0.0 < args.one_shot_max_position_shift < 0.5:
        parser.error("--one-shot-max-position-shift must lie in (0,0.5)")

    joint_feedback_enabled = bool(
        args.parameter_feedback_epochs > 0 and args.joint_parameter_structure_feedback
    )
    feedback_objective_version = (
        V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION
        if joint_feedback_enabled
        else V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION
    )
    effective_parameter_feedback_fusion_mode = (
        args.parameter_feedback_fusion_mode
        if args.parameter_feedback_fusion_mode is not None
        else ("cross_attention" if joint_feedback_enabled else "fast_global")
    )

    effective_lambda_policy_count = args.lambda_policy_count
    if args.one_shot_selection_policy == "threshold":
        effective_lambda_policy_count = 0.0
        if args.lambda_policy_count:
            print(
                "Threshold selection does not use probability-mass cardinality; "
                "disabling --lambda-policy-count.",
                flush=True,
            )

    torch.manual_seed(args.train_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_config: dict[str, object] = {
        "num_points": args.num_points,
        "point_dim": args.point_dim,
        "min_control_points": args.min_control_points,
        "max_control_points": args.max_control_points,
        "noise_std": args.noise_std,
        "knot_nonuniformity": args.knot_nonuniformity,
        "sampling_nonuniformity": args.sampling_nonuniformity,
        "turn_strength": args.turn_strength,
        "certified_minimal_source": args.certified_minimal_source,
        "minimality_margin": args.minimality_margin,
        "minimality_max_attempts": args.minimality_max_attempts,
        "minimality_audit_points": args.minimality_audit_points,
        "oscillation_amplitude": args.oscillation_amplitude,
        # Canonical proposal labels and the offline teacher share one RMS scale.
        "canonical_knot_tolerance": args.fit_tolerance,
        "normalize": True,
        "return_ground_truth": True,
    }
    train_set = SyntheticCubicBSplineDataset(
        size=args.train_size,
        seed=args.train_seed,
        resample_each_epoch=args.resample_train_each_epoch,
        **dataset_config,
    )
    val_set = SyntheticCubicBSplineDataset(
        size=args.val_size,
        seed=args.val_seed,
        resample_each_epoch=False,
        **dataset_config,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    proposal_initializer_checkpoint: dict[str, object] | None = None
    saved_proposal_model_config: Mapping[str, object] = {}
    saved_proposal_bandwidth = 0.0
    saved_candidate_interval_logit_limit = 0.0
    saved_candidate_position_parameterization = "interval_softmax"
    saved_parameter_gap_reference = "learned"
    saved_parameter_residual_logit_limit = 0.5
    if args.proposal_checkpoint is not None:
        proposal_initializer_checkpoint = torch.load(
            args.proposal_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        saved_model_config = proposal_initializer_checkpoint.get("model_config", {})
        if isinstance(saved_model_config, Mapping):
            saved_proposal_model_config = saved_model_config
            saved_proposal_bandwidth = float(
                saved_model_config.get("candidate_local_attention_bandwidth", 0.0)
            )
            saved_candidate_interval_logit_limit = float(
                saved_model_config.get("candidate_interval_logit_limit", 0.0)
            )
            saved_candidate_position_parameterization = str(
                saved_model_config.get(
                    "candidate_position_parameterization",
                    "interval_softmax",
                )
            )
            saved_parameter_gap_reference = str(
                saved_model_config.get("parameter_gap_reference", "learned")
            )
            saved_parameter_residual_logit_limit = float(
                saved_model_config.get("parameter_residual_logit_limit", 0.5)
            )

    effective_local_attention_bandwidth = args.candidate_local_attention_bandwidth
    effective_candidate_interval_logit_limit = args.candidate_interval_logit_limit
    effective_candidate_position_parameterization = (
        args.candidate_position_parameterization
    )
    effective_parameter_gap_reference = args.parameter_gap_reference
    effective_parameter_residual_logit_limit = args.parameter_residual_logit_limit
    if args.candidate_pretrain_epochs == 0:
        # No adaptation means the proposal forward must remain byte-for-byte
        # compatible with its source checkpoint. Applying the current Gaussian
        # bias to old global-attention weights silently degrades recall.
        effective_local_attention_bandwidth = saved_proposal_bandwidth
        if (
            effective_local_attention_bandwidth
            != args.candidate_local_attention_bandwidth
        ):
            print(
                "Proposal adaptation is disabled; preserving checkpoint local "
                "attention bandwidth "
                f"{effective_local_attention_bandwidth:g} instead of requested "
                f"{args.candidate_local_attention_bandwidth:g}.",
                flush=True,
            )
        effective_candidate_interval_logit_limit = (
            saved_candidate_interval_logit_limit
        )
        effective_candidate_position_parameterization = (
            saved_candidate_position_parameterization
        )
        effective_parameter_gap_reference = saved_parameter_gap_reference
        effective_parameter_residual_logit_limit = (
            saved_parameter_residual_logit_limit
        )
        if (
            effective_candidate_interval_logit_limit
            != args.candidate_interval_logit_limit
            or effective_candidate_position_parameterization
            != args.candidate_position_parameterization
        ):
            print(
                "Proposal adaptation is disabled; preserving checkpoint "
                "candidate parameterization "
                f"{effective_candidate_position_parameterization} with interval "
                f"logit limit {effective_candidate_interval_logit_limit:g}.",
                flush=True,
            )
        if (
            effective_parameter_gap_reference != args.parameter_gap_reference
            or effective_parameter_residual_logit_limit
            != args.parameter_residual_logit_limit
        ):
            print(
                "Proposal adaptation is disabled; preserving checkpoint "
                "ParameterHead reference "
                f"{effective_parameter_gap_reference} with residual logit limit "
                f"{effective_parameter_residual_logit_limit:g}.",
                flush=True,
            )

    model_config: dict[str, object] = {
        "point_dim": args.point_dim,
        "degree": 3,
        "hidden_dim": args.hidden_dim,
        "encoder_layers": args.encoder_layers,
        "max_internal_knots": args.candidate_knots,
        "min_parameter_gap": 1e-4,
        "min_knot_gap": args.min_knot_gap,
        "gap_parameterization": "strict",
        "parameter_gap_reference": effective_parameter_gap_reference,
        "parameter_residual_logit_limit": (
            effective_parameter_residual_logit_limit
        ),
        "lambda_poly": args.lambda_poly,
        "lambda_knot": args.lambda_knot,
        "structure_mode": "candidate_pruning_one_shot",
        "structure_attention_heads": args.attention_heads,
        "geometry_feature_mode": "chord_derivatives",
        "pruning_residual_bandwidth": args.pruning_residual_bandwidth,
        "pruning_initial_keep_probability": args.initial_keep_probability,
        "one_shot_fixed_proposal_geometry": True,
        "one_shot_selection_policy": args.one_shot_selection_policy,
        "one_shot_safety_sigma": args.one_shot_safety_sigma,
        "one_shot_selector_layers": args.one_shot_selector_layers,
        "one_shot_coverage_bins": args.one_shot_coverage_bins,
        "candidate_local_attention_bandwidth": effective_local_attention_bandwidth,
        "candidate_interval_logit_limit": (
            effective_candidate_interval_logit_limit
        ),
        "candidate_position_parameterization": (
            effective_candidate_position_parameterization
        ),
        "one_shot_joint_position_refinement": True,
        "one_shot_survivor_relocation": True,
        "one_shot_max_position_shift": args.one_shot_max_position_shift,
        "stable_pilot_descriptors": True,
        "parameter_feedback_fusion": args.parameter_feedback_epochs > 0,
        "parameter_feedback_fusion_mode": (effective_parameter_feedback_fusion_mode),
        "joint_parameter_structure_feedback": joint_feedback_enabled,
        "parameter_feedback_attention_heads": args.attention_heads,
        "parameter_feedback_max_logit_shift": (args.parameter_feedback_max_logit_shift),
        "joint_parameter_structure_local_bandwidth": (
            args.joint_feedback_local_bandwidth
        ),
        "joint_parameter_structure_max_keep_logit_shift": (
            args.joint_feedback_max_keep_logit_shift
        ),
        "enforce_ordered_joint_candidates": (
            joint_feedback_enabled and args.enforce_ordered_joint_candidates
        ),
        "compute_first_derivative": False,
    }
    if args.candidate_pretrain_epochs == 0:
        if saved_proposal_model_config:
            semantic_mismatches = _proposal_semantic_mismatches(
                model_config,
                saved_proposal_model_config,
            )
            if semantic_mismatches:
                parser.error(
                    "fixed proposal checkpoint semantics do not match the requested "
                    "model; use matching arguments or enable proposal adaptation:\n  "
                    + "\n  ".join(semantic_mismatches)
                )
        else:
            print(
                "Warning: proposal checkpoint has no model_config; assuming "
                "historical global attention and unable to verify other proposal "
                "semantics.",
                flush=True,
            )
    model = SplineFittingNetwork(**model_config)
    if proposal_initializer_checkpoint is not None and args.candidate_pretrain_epochs:
        # A v10/v11 checkpoint is a useful initializer, but local proposal
        # attention still needs a short adaptation stage.  Previously the
        # checkpoint argument was silently ignored whenever pretraining was
        # enabled, which made migration unexpectedly start from scratch.
        initializer_state = proposal_initializer_checkpoint.get(
            "model_state_dict", proposal_initializer_checkpoint
        )
        _load_model_state(
            model,
            initializer_state,
            allow_new_parameter_feedback=args.parameter_feedback_epochs > 0,
            allow_new_joint_feedback=joint_feedback_enabled,
        )
        print(
            f"Initialized v13 proposal adaptation from: {args.proposal_checkpoint}",
            flush=True,
        )
    _reset_and_freeze_late_feedback_heads(model)
    proposal_weights = CandidatePruningLossWeights(
        fit=args.lambda_fit,
        threshold_violation=args.lambda_threshold_violation,
        true_parameter=args.lambda_true_params,
        true_parameter_gap=args.lambda_true_parameter_gap,
        candidate_coverage=args.lambda_candidate_coverage,
        redundant_candidate=args.lambda_redundant_candidate,
        candidate_repulsion=args.lambda_candidate_repulsion,
        keep=args.lambda_keep,
        remove_action=0.0,
        knot_position=args.lambda_knot_position,
        count_consistency=0.0,
        deletion_cost=0.0,
        teacher_risk=args.lambda_teacher_risk,
        teacher_ranking=args.lambda_teacher_ranking,
        teacher_distribution=args.lambda_teacher_distribution,
        teacher_critical_recall=args.lambda_teacher_critical_recall,
        teacher_false_positive=args.lambda_teacher_false_positive,
        teacher_count=args.lambda_teacher_count,
        policy_count=effective_lambda_policy_count,
        canonical_selection=args.lambda_canonical_selection,
        complexity=args.lambda_complexity,
    )
    pretrain_weights = replace(
        proposal_weights,
        keep=0.0,
        remove_action=0.0,
        count_consistency=0.0,
        deletion_cost=0.0,
        teacher_risk=0.0,
        teacher_ranking=0.0,
        teacher_distribution=0.0,
        teacher_critical_recall=0.0,
        teacher_false_positive=0.0,
        teacher_count=0.0,
        policy_count=0.0,
        canonical_selection=0.0,
        complexity=0.0,
    )
    parameter_warmup_weights = CandidatePruningLossWeights(
        **{
            name: (
                value
                if name in {"true_parameter", "true_parameter_gap"}
                else 0.0
            )
            for name, value in asdict(pretrain_weights).items()
        }
    )
    one_shot_weights = replace(
        proposal_weights,
        # The selected mask is supervised by an offline teacher computed with
        # the exact deployment B-spline solver.  The truncated-power curve is
        # deliberately diagnostic-only here: its ST gradient previously made
        # retaining every candidate the easiest way to reduce violation loss.
        fit=args.lambda_one_shot_surrogate_fit,
        threshold_violation=args.lambda_one_shot_surrogate_threshold,
        true_parameter=0.0,
        true_parameter_gap=0.0,
        candidate_coverage=0.0,
        redundant_candidate=0.0,
        candidate_repulsion=0.0,
        knot_position=0.0,
    )
    calibration_weights = replace(
        one_shot_weights,
        fit=args.lambda_joint_fit,
        threshold_violation=args.lambda_joint_threshold_violation,
        knot_position=args.lambda_joint_position,
    )
    feedback_weights = ParameterFeedbackLossWeights(
        fit=args.lambda_feedback_fit,
        threshold_violation=args.lambda_feedback_threshold_violation,
        deployment_fit=args.lambda_feedback_deployment_fit,
        deployment_threshold_violation=(
            args.lambda_feedback_deployment_threshold
        ),
        true_parameter=args.lambda_feedback_true_params,
        chord_prior=args.lambda_feedback_chord_prior,
        identity=args.lambda_feedback_identity,
        joint_keep=(args.lambda_feedback_joint_keep if joint_feedback_enabled else 0.0),
        joint_critical_recall=(
            args.lambda_feedback_joint_critical_recall
            if joint_feedback_enabled
            else 0.0
        ),
        joint_position=(
            args.lambda_feedback_joint_position if joint_feedback_enabled else 0.0
        ),
        joint_set_position=(
            args.lambda_feedback_joint_set_position
            if joint_feedback_enabled
            else 0.0
        ),
        joint_spacing=(
            args.lambda_feedback_joint_spacing if joint_feedback_enabled else 0.0
        ),
        joint_count=(
            args.lambda_feedback_joint_count
            if joint_feedback_enabled
            and args.one_shot_selection_policy == "mass_topk"
            else 0.0
        ),
    )

    def make_loss(
        weights: CandidatePruningLossWeights,
        *,
        exact_deletion_supervision: bool,
    ) -> CandidatePruningLoss:
        return CandidatePruningLoss(
            weights,
            knot_position_beta=args.knot_position_beta,
            candidate_match_tolerance=args.candidate_match_tolerance,
            fit_tolerance=args.fit_tolerance,
            positive_keep_weight=args.positive_keep_weight,
            exact_deletion_supervision=exact_deletion_supervision,
            deletion_smoothness_weight=1e-6,
            deletion_control_ridge=0.0,
            teacher_ranking_margin=args.teacher_ranking_margin,
            candidate_coverage_tolerances=tuple(args.candidate_coverage_tolerances),
            position_aware_distribution=True,
            joint_position_supervision=False,
            teacher_relocation_supervision=True,
            teacher_survivor_spacing_weight=(args.teacher_survivor_spacing_weight),
            true_parameter_error_scale=args.proposal_parameter_error_scale,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    required_proposal_pass = _required_proposal_pass_rate(
        deployment_pass_rate_target=args.deployment_pass_rate_target,
        max_teacher_fallback_fraction=args.max_teacher_fallback_fraction,
        teacher_start_mode=args.teacher_start_mode,
    )
    trainer = Trainer(
        model,
        make_loss(pretrain_weights, exact_deletion_supervision=False),
        optimizer,
        device,
        knot_match_tolerance=args.candidate_match_tolerance,
        log_every_batches=args.log_every_batches,
        deployment_pass_rate_target=required_proposal_pass,
        show_progress=args.progress,
    )

    print(
        (
            (
                "v15 objective: deployment-aligned B-spline fit + exact "
                "mass_topk count -> joint parameter/Keep/location refinement; "
                if joint_feedback_enabled
                else "legacy v14 objective: v13 structure plus one-shot "
                "fit-feedback parameter calibration; "
            )
            if args.parameter_feedback_epochs
            else "v13 objective: high-recall local proposals plus set-supervised "
        )
        + "KeepMask-conditioned "
        "survivor relocation "
        "distilled from "
        f"an offline {'delete-then-relax' if args.teacher_survivor_relaxation else 'greedy'} "
        "Hard-RMS teacher at "
        f"epsilon={args.fit_tolerance:g}",
        flush=True,
    )
    print(
        "Data: "
        f"train={args.train_size}, val={args.val_size}, points={args.num_points}, "
        f"source K={args.min_control_points - 4}..{max_source_knots}, "
        f"candidate Kc={args.candidate_knots}",
        flush=True,
    )
    print(
        "Hard-RMS search runs once offline to create cached masks and deletion "
        f"risks from {args.teacher_start_mode}. Deployment uses no labels: one "
        "LearnedKeep mask and one B-spline refit.",
        flush=True,
    )

    histories: dict[str, list[dict[str, float]]] = {}
    if args.candidate_pretrain_epochs:
        # Keep scores stay at their conservative all-open initialization while
        # proposal coverage, parameters and position refinement are pretrained.
        for module_name in (
            "keep_head",
            "adaptive_threshold_head",
            "position_to_keep_feedback",
            "one_shot_selector_attention",
            "one_shot_selector_attention_norm",
            "one_shot_selector_feed_forward",
            "one_shot_selector_output_norm",
            "one_shot_selector_extra_attention",
            "one_shot_selector_extra_attention_norm",
            "one_shot_selector_extra_feed_forward",
            "one_shot_selector_extra_output_norm",
            "joint_preliminary_position_projection",
            "joint_preliminary_position_norm",
            "joint_preliminary_position_head",
            "joint_position_to_keep_feedback",
            "joint_position_to_keep_norm",
            "joint_final_position_projection",
            "joint_final_position_norm",
            "joint_final_position_head",
            "survivor_relative_projection",
            "survivor_relocation_input_norm",
            "survivor_relocation_attention",
            "survivor_relocation_attention_norm",
            "survivor_relocation_feed_forward",
            "survivor_relocation_output_norm",
            "survivor_relocation_head",
        ):
            module = getattr(model.pruning_head, module_name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        proposal_path = args.output.with_name(
            args.output.stem + "_proposal" + args.output.suffix
        )
        proposal_parameter_warmup_epochs = (
            args.proposal_parameter_warmup_epochs
        )
        if proposal_parameter_warmup_epochs:
            original_trainability = {
                name: parameter.requires_grad
                for name, parameter in model.named_parameters()
            }
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            warmup_parameters: list[torch.nn.Parameter] = []
            for module in (model.encoder, model.parameter_head):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
                    warmup_parameters.append(parameter)
            warmup_path = args.output.with_name(
                args.output.stem
                + "_proposal_parameter_warmup"
                + args.output.suffix
            )
            trainer.optimizer = torch.optim.AdamW(
                warmup_parameters,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            trainer.loss_fn = make_loss(
                parameter_warmup_weights,
                exact_deletion_supervision=False,
            ).to(device)
            histories["proposal_parameter_warmup"] = trainer.fit(
                train_loader,
                val_loader,
                proposal_parameter_warmup_epochs,
                gate_warmup_epochs=proposal_parameter_warmup_epochs,
                checkpoint_path=warmup_path,
                stage_name="proposal_parameter_warmup",
            )
            warmup_checkpoint = torch.load(
                warmup_path,
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(
                warmup_checkpoint["model_state_dict"],
                strict=True,
            )
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(original_trainability[name])
            print(
                "Restored best ParameterHead warm-up checkpoint from epoch "
                f"{warmup_checkpoint['epoch']}.",
                flush=True,
            )

        remaining_proposal_epochs = (
            args.candidate_pretrain_epochs
            - proposal_parameter_warmup_epochs
        )
        trainer.optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        trainer.loss_fn = make_loss(
            pretrain_weights,
            exact_deletion_supervision=False,
        ).to(device)
        histories["candidate_pretrain"] = trainer.fit(
            train_loader,
            val_loader,
            remaining_proposal_epochs,
            gate_warmup_epochs=remaining_proposal_epochs,
            checkpoint_path=proposal_path,
            epoch_offset=proposal_parameter_warmup_epochs,
            stage_name="candidate_pretrain",
            deployment_validation=True,
        )
        proposal_checkpoint = torch.load(
            proposal_path, map_location=device, weights_only=True
        )
        model.load_state_dict(proposal_checkpoint["model_state_dict"], strict=True)
        # Trainer checkpoints are intentionally generic.  Persist proposal
        # semantics here so a later --candidate-pretrain-epochs 0 run can
        # reproduce this exact local-attention forward and safely fingerprint
        # its offline teacher cache.
        proposal_checkpoint.update(
            {
                "model_config": dict(model_config),
                "dataset_config": dict(dataset_config),
                "dataset_type": "synthetic_open_cubic_bspline",
                "objective_version": V13_SET_RELOCATION_OBJECTIVE_VERSION,
                "loss_config": {
                    "weights": asdict(pretrain_weights),
                    "candidate_match_tolerance": args.candidate_match_tolerance,
                    "candidate_coverage_tolerances": list(
                        args.candidate_coverage_tolerances
                    ),
                    "fit_tolerance": args.fit_tolerance,
                },
                "training_config": {
                    "stage": "candidate_pretrain",
                    "candidate_pretrain_epochs": args.candidate_pretrain_epochs,
                    "proposal_parameter_warmup_epochs": (
                        proposal_parameter_warmup_epochs
                    ),
                    "candidate_local_attention_bandwidth": (
                        effective_local_attention_bandwidth
                    ),
                },
                "deployment_config": {"role": "fixed_proposal_for_offline_teacher"},
                "selected_stage": "candidate_pretrain",
                "stage_histories": dict(histories),
            }
        )
        torch.save(proposal_checkpoint, proposal_path)
        print(
            f"Restored best proposal checkpoint from epoch "
            f"{proposal_checkpoint['epoch']} before teacher generation.",
            flush=True,
        )
    else:
        if proposal_initializer_checkpoint is None:
            raise RuntimeError("proposal checkpoint was not loaded")
        proposal_checkpoint = proposal_initializer_checkpoint
        state = proposal_checkpoint.get("model_state_dict", proposal_checkpoint)
        _load_model_state(
            model,
            state,
            allow_new_parameter_feedback=args.parameter_feedback_epochs > 0,
            allow_new_joint_feedback=joint_feedback_enabled,
        )
        print(
            f"Loaded fixed proposal checkpoint: {args.proposal_checkpoint}",
            flush=True,
        )

    # The offline teacher must be tied to the proposal/ParameterHead only.
    # Reapply this guard after restoring the selected proposal checkpoint (or
    # loading a historical fixed proposal) in case it contains feedback weights
    # learned by an older training script.
    _reset_and_freeze_late_feedback_heads(model)
    model.set_force_open_gates(False)
    _reset_one_shot_selector_to_neutral(
        model,
        initial_probability=args.initial_keep_probability,
    )
    for parameter in model.pruning_head.parameters():
        parameter.requires_grad_(True)

    proposal_validation_pass = _require_feasible_proposal_checkpoint(
        proposal_checkpoint,
        deployment_pass_rate_target=args.deployment_pass_rate_target,
        max_teacher_fallback_fraction=args.max_teacher_fallback_fraction,
        teacher_start_mode=args.teacher_start_mode,
    )
    if args.teacher_start_mode == "canonical_boehm":
        print(
            "Observed selected proposal exact all-candidate validation pass "
            f"rate: {proposal_validation_pass:.3%}; canonical_boehm teacher "
            "generation is independent of this feasibility rate.",
            flush=True,
        )
    else:
        print(
            "Verified selected proposal exact all-candidate validation pass rate: "
            f"{proposal_validation_pass:.3%} "
            f"(required {required_proposal_pass:.3%})",
            flush=True,
        )

    teacher_feasibility_policy = _teacher_feasibility_policy(
        args.teacher_start_mode
    )
    teacher_fingerprint_material = (
        _model_fingerprint(model)
        + "|teacher_feasibility_policy="
        + teacher_feasibility_policy
    )
    if args.teacher_start_mode == "canonical_boehm":
        teacher_fingerprint_material += (
            "|teacher_start_mode=canonical_boehm|slot_alignment=ordered_rank"
        )
    proposal_fingerprint = hashlib.sha256(
        teacher_fingerprint_material.encode("utf-8")
    ).hexdigest()
    print("Fingerprinting the fixed train/validation samples...", flush=True)
    train_dataset_fingerprint = _dataset_fingerprint(
        train_set,
        split="train",
        seed=args.train_seed,
        dataset_config=dataset_config,
    )
    val_dataset_fingerprint = _dataset_fingerprint(
        val_set,
        split="validation",
        seed=args.val_seed,
        dataset_config=dataset_config,
    )
    teacher_base_config = {
        "error_tolerance": args.fit_tolerance,
        "temperature": args.teacher_risk_temperature,
        "min_internal_knots": 0,
        "degree": 3,
        "smoothness_weight": 1e-6,
        "control_ridge": 0.0,
        "interpolate_endpoints": True,
        "proposal_fingerprint": proposal_fingerprint,
        "relocation_strategy": (
            "delete_then_relax" if args.teacher_survivor_relaxation else "none"
        ),
        "relocation_rounds": args.teacher_relaxation_rounds,
        "relocation_sweeps": args.teacher_relaxation_sweeps,
        "relocation_grid_size": args.teacher_relaxation_grid_size,
        "relocation_restarts": args.teacher_relaxation_restarts,
        "relocation_min_gap": (
            args.min_knot_gap
            if args.teacher_relaxation_min_gap is None
            else args.teacher_relaxation_min_gap
        ),
        "relocation_max_shift": args.one_shot_max_position_shift,
    }
    teacher_config = OneShotTeacherConfig(
        **teacher_base_config,
        dataset_fingerprint=train_dataset_fingerprint,
    )
    val_teacher_config = OneShotTeacherConfig(
        **teacher_base_config,
        dataset_fingerprint=val_dataset_fingerprint,
    )
    teacher_cache_dir = args.teacher_cache_dir or (
        args.output.parent / f"{args.output.stem}_teacher"
    )
    train_teacher = _build_or_load_teacher_cache(
        model=model,
        dataset=train_set,
        cache_path=teacher_cache_dir / "train.pt",
        config=teacher_config,
        batch_size=args.teacher_batch_size,
        device=device,
        reuse=args.reuse_teacher_cache,
        max_fallback_fraction=args.max_teacher_fallback_fraction,
        teacher_start_mode=args.teacher_start_mode,
    )
    val_teacher = _build_or_load_teacher_cache(
        model=model,
        dataset=val_set,
        cache_path=teacher_cache_dir / "val.pt",
        config=val_teacher_config,
        batch_size=args.teacher_batch_size,
        device=device,
        reuse=args.reuse_teacher_cache,
        max_fallback_fraction=args.max_teacher_fallback_fraction,
        teacher_start_mode=args.teacher_start_mode,
    )
    teacher_train_set = TeacherAugmentedDataset(train_set, train_teacher)
    teacher_val_set = TeacherAugmentedDataset(val_set, val_teacher)
    teacher_train_loader = DataLoader(
        teacher_train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )
    teacher_val_loader = DataLoader(
        teacher_val_set,
        batch_size=args.batch_size,
    )

    # Offline labels bind to immutable proposal geometry.  Initial distillation
    # learns only the selection rule; the later low-LR stage activates the
    # separate deployment-position path without changing teacher slot identity.
    selection_parameters = _set_one_shot_trainable(
        model,
        calibrate_positions=False,
    )
    trainer.optimizer = torch.optim.AdamW(
        selection_parameters,
        lr=args.selector_lr,
        weight_decay=args.weight_decay,
    )
    trainer.loss_fn = make_loss(
        one_shot_weights,
        exact_deletion_supervision=False,
    ).to(device)
    # Proposal selection may need a stricter exact-fit target to honor the
    # offline-teacher fallback budget (100% when fallbacks are disabled).
    # LearnedKeep deployment selection retains the user-facing pass target.
    trainer.deployment_pass_rate_target = args.deployment_pass_rate_target
    distill_epochs = (
        args.epochs
        - args.candidate_pretrain_epochs
        - args.keep_position_calibration_epochs
    )
    distill_path = (
        args.output
        if args.keep_position_calibration_epochs == 0
        else args.output.with_name(args.output.stem + "_distill" + args.output.suffix)
    )
    effective_selector_warmup = min(
        args.selector_checkpoint_warmup_epochs,
        max(distill_epochs - 1, 0),
    )
    histories["one_shot_distillation"] = trainer.fit(
        teacher_train_loader,
        teacher_val_loader,
        distill_epochs,
        checkpoint_path=distill_path,
        epoch_offset=args.candidate_pretrain_epochs,
        stage_name="one_shot_selection_distillation",
        deployment_validation=True,
        checkpoint_selection_start_epoch=effective_selector_warmup,
    )
    # Trainer checkpoints are intentionally generic, but a long multi-stage
    # run must remain recoverable if a later calibration/logging stage is
    # interrupted.  Attach the minimum complete construction metadata now.
    distill_checkpoint = torch.load(
        distill_path,
        map_location="cpu",
        weights_only=True,
    )
    distill_checkpoint = _checkpoint_with_metadata(
        distill_checkpoint,
        model_config=dict(model_config),
        dataset_config=dict(dataset_config),
        loss_config={
            "weights": asdict(one_shot_weights),
            "fit_tolerance": args.fit_tolerance,
            "intermediate_role": "recoverable_one_shot_structure",
        },
        training_config={
            "stage": "one_shot_selection_distillation",
            "epochs": distill_epochs,
            "source_output": str(args.output),
        },
        deployment_config={
            "error_tolerance": args.fit_tolerance,
            "role": "recoverable_intermediate_structure",
        },
        histories=dict(histories),
        objective_version=(
            feedback_objective_version
            if args.parameter_feedback_epochs
            else V13_SET_RELOCATION_OBJECTIVE_VERSION
        ),
    )
    torch.save(distill_checkpoint, distill_path)

    candidate_checkpoints = [distill_path]
    if args.keep_position_calibration_epochs:
        model.load_state_dict(distill_checkpoint["model_state_dict"], strict=True)
        calibration_parameters = _set_one_shot_trainable(
            model,
            calibrate_positions=True,
        )
        trainer.optimizer = torch.optim.AdamW(
            calibration_parameters,
            lr=args.selector_lr * args.relocation_lr_scale,
            weight_decay=args.weight_decay,
        )
        trainer.loss_fn = make_loss(
            calibration_weights,
            exact_deletion_supervision=False,
        ).to(device)
        calibration_path = args.output.with_name(
            args.output.stem + "_calibrated" + args.output.suffix
        )
        histories["one_shot_selector_calibration"] = trainer.fit(
            teacher_train_loader,
            teacher_val_loader,
            args.keep_position_calibration_epochs,
            checkpoint_path=calibration_path,
            epoch_offset=args.candidate_pretrain_epochs + distill_epochs,
            stage_name="one_shot_selector_calibration",
            deployment_validation=True,
        )
        calibration_checkpoint = torch.load(
            calibration_path,
            map_location="cpu",
            weights_only=True,
        )
        calibration_checkpoint = _checkpoint_with_metadata(
            calibration_checkpoint,
            model_config=dict(model_config),
            dataset_config=dict(dataset_config),
            loss_config={
                "weights": asdict(calibration_weights),
                "fit_tolerance": args.fit_tolerance,
                "intermediate_role": "recoverable_one_shot_structure",
            },
            training_config={
                "stage": "one_shot_selector_calibration",
                "epochs": args.keep_position_calibration_epochs,
                "source_output": str(args.output),
            },
            deployment_config={
                "error_tolerance": args.fit_tolerance,
                "role": "recoverable_intermediate_structure",
            },
            histories=dict(histories),
            objective_version=(
                feedback_objective_version
                if args.parameter_feedback_epochs
                else V13_SET_RELOCATION_OBJECTIVE_VERSION
            ),
        )
        torch.save(calibration_checkpoint, calibration_path)
        # Calibration is optional refinement, not permission to regress real
        # deployment quality. Keep the distillation/feasibility checkpoint as
        # a valid identity-relocation fallback and rank both using exact
        # standard-B-spline validation metrics.
        candidate_checkpoints.append(calibration_path)

    base_candidates = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in candidate_checkpoints
    ]
    base_best = max(
        base_candidates,
        key=lambda checkpoint: tuple(checkpoint.get("selection_rank", [])),
    )
    if args.parameter_feedback_epochs:
        model.load_state_dict(base_best["model_state_dict"], strict=True)
        with torch.no_grad():
            model.parameter_feedback_head.chord_blend_weight.fill_(
                args.parameter_feedback_initial_chord_blend
            )
        feedback_parameters = _set_parameter_feedback_trainable(model)
        trainer.optimizer = torch.optim.AdamW(
            feedback_parameters,
            lr=args.parameter_feedback_lr,
            weight_decay=args.weight_decay,
        )
        trainer.loss_fn = ParameterFeedbackLoss(
            feedback_weights,
            fit_tolerance=args.fit_tolerance,
            parameter_error_scale=args.feedback_parameter_error_scale,
            knot_position_beta=args.knot_position_beta,
            positive_keep_weight=args.positive_keep_weight,
            degree=model.degree,
            deployment_smoothness_weight=1e-6,
            deployment_control_ridge=0.0,
        ).to(device)
        feedback_path = args.output.with_name(
            args.output.stem + "_parameter_feedback" + args.output.suffix
        )
        histories["parameter_feedback_calibration"] = trainer.fit(
            teacher_train_loader,
            teacher_val_loader,
            args.parameter_feedback_epochs,
            checkpoint_path=feedback_path,
            epoch_offset=args.epochs,
            stage_name="parameter_feedback_calibration",
            deployment_validation=True,
        )
        # The identity (zero-feedback) structure checkpoint remains eligible;
        # the late calibration is selected only if exact B-spline validation
        # improves under the same constrained rank.
        candidate_checkpoints.append(feedback_path)

    best_candidates = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in candidate_checkpoints
    ]
    best = max(
        best_candidates,
        key=lambda checkpoint: tuple(checkpoint.get("selection_rank", [])),
    )
    selected_parameter_feedback = best.get("stage") == (
        "parameter_feedback_calibration"
    )
    selected_structure_checkpoint = base_best if selected_parameter_feedback else best
    selected_position_calibration = selected_structure_checkpoint.get("stage") == (
        "one_shot_selector_calibration"
    )
    selected_loss_weights = (
        calibration_weights if selected_position_calibration else one_shot_weights
    )

    objective_version = (
        feedback_objective_version
        if args.parameter_feedback_epochs
        else V13_SET_RELOCATION_OBJECTIVE_VERSION
    )
    loss_config: dict[str, object] = {
        "weights": asdict(selected_loss_weights),
        "proposal_weights": asdict(pretrain_weights),
        "proposal_parameter_error_scale": args.proposal_parameter_error_scale,
        "knot_position_beta": args.knot_position_beta,
        "candidate_match_tolerance": args.candidate_match_tolerance,
        "fit_tolerance": args.fit_tolerance,
        "positive_keep_weight": args.positive_keep_weight,
        "exact_deletion_supervision": False,
        "deletion_smoothness_weight": 1e-6,
        "deletion_control_ridge": 0.0,
        "one_shot_teacher": True,
        "teacher_start_mode": args.teacher_start_mode,
        "teacher_feasibility_policy": teacher_feasibility_policy,
        "teacher_slot_alignment": "ordered_rank",
        "teacher_position_storage_domain": "fixed_proposal_t0",
        "max_teacher_fallback_fraction": args.max_teacher_fallback_fraction,
        "fixed_proposal_geometry": True,
        "selector_adapter": "final_keepmask_conditioned_survivor_relocation",
        "selector_layers": args.one_shot_selector_layers,
        "straight_through_keep_gate": True,
        "surrogate_selection_role": (
            "optional_auxiliary_only_exact_bspline_validation_selects_checkpoint"
        ),
        "teacher_mask_loss": (
            "teacher_bce_dice_ranking_plus_relocated_set_coverage_and_policy_count"
        ),
        "selection_policy": args.one_shot_selection_policy,
        "selection_safety_sigma": args.one_shot_safety_sigma,
        "selection_coverage_bins": args.one_shot_coverage_bins,
        "teacher_risk_temperature": args.teacher_risk_temperature,
        "teacher_ranking_margin": args.teacher_ranking_margin,
        "candidate_coverage_tolerances": list(args.candidate_coverage_tolerances),
        "position_aware_distribution": True,
        "joint_position_supervision": False,
        "teacher_relocation_supervision": True,
        "teacher_position_assignment": "actual_keepmask_ordered_set_matching",
        "teacher_distribution_target": (
            "relocated_teacher_set_cdf_plus_local_coverage"
        ),
        "stable_pilot_descriptors": True,
        "parameter_feedback_fusion": args.parameter_feedback_epochs > 0,
        "parameter_feedback_fusion_mode": (effective_parameter_feedback_fusion_mode),
        "joint_parameter_structure_feedback": joint_feedback_enabled,
        "joint_feedback_teacher_targets": (
            "final_keep_logits+mass_topk_score+t0_to_t1_warped_survivor_set"
            if joint_feedback_enabled
            else "disabled"
        ),
        "parameter_feedback_loss": {
            **asdict(feedback_weights),
            "parameter_error_scale": args.feedback_parameter_error_scale,
            "initial_chord_blend": (args.parameter_feedback_initial_chord_blend),
            "deployment_refit": (
                "differentiable_standard_bspline_endpoint_constrained"
            ),
            "count_target": (
                "requested_score=K-0.25_in_mass_topk_ceil_interval"
            ),
        },
        "selected_checkpoint_parameter_feedback": selected_parameter_feedback,
        "teacher_survivor_spacing_weight": (args.teacher_survivor_spacing_weight),
        "survivor_position_target": (
            "delete_then_relax_teacher_packed_knots"
            if args.teacher_survivor_relaxation
            else "greedy_teacher_packed_knots"
        ),
        "selected_checkpoint_position_calibrated": selected_position_calibration,
        "candidate_local_attention_bandwidth": (effective_local_attention_bandwidth),
        "joint_position_max_shift": args.one_shot_max_position_shift,
        "survivor_relocation_max_shift": args.one_shot_max_position_shift,
        "joint_calibration_fit_weight": args.lambda_joint_fit,
        "joint_calibration_threshold_weight": (args.lambda_joint_threshold_violation),
        "count_consistency_is_deployment_rule": False,
        "analytic_deletion_cost_is_auxiliary_only": True,
    }
    training_config: dict[str, object] = {
        "structure_mode": "candidate_pruning_one_shot",
        "epochs": args.epochs,
        "candidate_pretrain_epochs": args.candidate_pretrain_epochs,
        "proposal_parameter_warmup_epochs": (
            args.proposal_parameter_warmup_epochs
        ),
        "proposal_checkpoint": (
            str(args.proposal_checkpoint)
            if args.proposal_checkpoint is not None
            else None
        ),
        "one_shot_distillation_epochs": distill_epochs,
        "selector_calibration_epochs": args.keep_position_calibration_epochs,
        "keep_position_calibration_epochs": args.keep_position_calibration_epochs,
        "joint_finetune_epochs": args.keep_position_calibration_epochs,
        "parameter_feedback_epochs": args.parameter_feedback_epochs,
        "total_epochs_including_parameter_feedback": (
            args.epochs + args.parameter_feedback_epochs
        ),
        "train_size": args.train_size,
        "val_size": args.val_size,
        "train_seed": args.train_seed,
        "val_seed": args.val_seed,
        "resample_train_each_epoch": args.resample_train_each_epoch,
        "fit_tolerance": args.fit_tolerance,
        "deployment_pass_rate_target": args.deployment_pass_rate_target,
        "candidate_match_tolerance": args.candidate_match_tolerance,
        "offline_hard_rms_teacher": True,
        "teacher_start_mode": args.teacher_start_mode,
        "teacher_feasibility_policy": teacher_feasibility_policy,
        "teacher_slot_alignment": "ordered_rank",
        "teacher_position_storage_domain": "fixed_proposal_t0",
        "max_teacher_fallback_fraction": args.max_teacher_fallback_fraction,
        "teacher_cache_dir": str(teacher_cache_dir),
        "teacher_config": teacher_config.as_dict(),
        "train_teacher_config": teacher_config.as_dict(),
        "val_teacher_config": val_teacher_config.as_dict(),
        "train_teacher_relocation_summary": _teacher_relocation_metadata(train_teacher),
        "val_teacher_relocation_summary": _teacher_relocation_metadata(val_teacher),
        "proposal_fingerprint": proposal_fingerprint,
        "weight_decay": args.weight_decay,
        "proposal_learning_rate": args.lr,
        "selector_learning_rate": args.selector_lr,
        "relocation_learning_rate": args.selector_lr * args.relocation_lr_scale,
        "relocation_learning_rate_scale": args.relocation_lr_scale,
        "parameter_feedback_learning_rate": args.parameter_feedback_lr,
        "parameter_feedback_fusion_mode": (effective_parameter_feedback_fusion_mode),
        "joint_parameter_structure_feedback": joint_feedback_enabled,
        "joint_feedback_local_bandwidth": args.joint_feedback_local_bandwidth,
        "joint_feedback_max_keep_logit_shift": (
            args.joint_feedback_max_keep_logit_shift
        ),
        "parameter_feedback_max_logit_shift": (args.parameter_feedback_max_logit_shift),
        "parameter_feedback_initial_chord_blend": (
            args.parameter_feedback_initial_chord_blend
        ),
        "feedback_parameter_error_scale": args.feedback_parameter_error_scale,
        "selector_checkpoint_warmup_epochs": effective_selector_warmup,
        "selection_policy": args.one_shot_selection_policy,
        "selection_safety_sigma": args.one_shot_safety_sigma,
        "selection_coverage_bins": args.one_shot_coverage_bins,
        "candidate_local_attention_bandwidth": (effective_local_attention_bandwidth),
        "joint_position_max_shift": args.one_shot_max_position_shift,
        "survivor_relocation_max_shift": args.one_shot_max_position_shift,
        "teacher_survivor_spacing_weight": (args.teacher_survivor_spacing_weight),
        "requested_policy_count_weight": args.lambda_policy_count,
        "effective_policy_count_weight": effective_lambda_policy_count,
        "joint_calibration_fit_weight": args.lambda_joint_fit,
        "joint_calibration_threshold_weight": (args.lambda_joint_threshold_violation),
    }
    deployment_config: dict[str, object] = {
        "selection_rule": (
            (
                (
                    "v14_joint_feedback_mass_topk_then_single_refit"
                    if joint_feedback_enabled
                    else "v14_parameter_feedback_mass_topk_then_single_refit"
                )
                if args.parameter_feedback_epochs
                else "v13_set_supervised_relocation_mass_topk_then_single_refit"
            )
            if args.one_shot_selection_policy == "mass_topk"
            else (
                (
                    "v14_joint_feedback_threshold_then_single_refit"
                    if joint_feedback_enabled
                    else "v14_parameter_feedback_threshold_then_single_refit"
                )
                if args.parameter_feedback_epochs
                else "v13_set_supervised_relocation_threshold_then_single_refit"
            )
        ),
        "error_tolerance": args.fit_tolerance,
        "min_internal_knots": 0,
        "smoothness_weight": 1e-6,
        "control_ridge": 0.0,
        "interpolate_endpoints": True,
        "network_forward_passes": 1,
        "standard_bspline_refits": 1,
        "learned_keep_threshold_is_final": (
            args.one_shot_selection_policy == "threshold"
        ),
        "learned_keep_selection_policy": args.one_shot_selection_policy,
        "uncertainty_safety_sigma": args.one_shot_safety_sigma,
        "parameter_domain_coverage_bins": args.one_shot_coverage_bins,
        "hard_rms_pruning_at_deployment": False,
        "proposal_geometry_frozen_after_teacher": True,
        "deployment_positions_jointly_refined": True,
        "deployment_positions_conditioned_on_final_keep_mask": True,
        "survivor_relocation_max_shift": args.one_shot_max_position_shift,
        "teacher_strategy": teacher_config.relocation_strategy,
        "selected_checkpoint_position_calibrated": selected_position_calibration,
        "parameter_feedback_fusion": args.parameter_feedback_epochs > 0,
        "parameter_feedback_fusion_mode": (effective_parameter_feedback_fusion_mode),
        "joint_parameter_structure_feedback": joint_feedback_enabled,
        "joint_final_mask_reselected_after_parameter_feedback": (
            joint_feedback_enabled
        ),
        "joint_final_relocation_uses_only_final_mask": joint_feedback_enabled,
        "selected_checkpoint_parameter_feedback": selected_parameter_feedback,
        "parameter_feedback_sources": (
            "chord_gaps+pilot_residual+deletion_risk+final_survivor_tokens"
            if args.parameter_feedback_epochs
            else "disabled"
        ),
        "parameter_feedback_additional_spline_solves": 0,
        "threshold_guarantee": "statistical_not_per_sample_exact",
        "checkpoint_selection": (
            "min_retained_knots_subject_to_validation_pass_rate_target"
        ),
        "global_minimum_guaranteed": False,
        "training_forward_internal_proxy_solves": 2,
        "deployment_forward_internal_proxy_solves": 1,
        "deployment_omits_training_only_final_surrogate": True,
    }

    best = _checkpoint_with_metadata(
        best,
        model_config=model_config,
        dataset_config=dataset_config,
        loss_config=loss_config,
        training_config=training_config,
        deployment_config=deployment_config,
        histories=histories,
        objective_version=objective_version,
    )
    torch.save(best, args.output)

    last_path = args.output.with_name(args.output.stem + "_last" + args.output.suffix)
    last = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs + args.parameter_feedback_epochs,
        "stage": "one_shot_training_last",
        "selection_metric": "last_epoch_not_selected",
        "history": histories[
            "parameter_feedback_calibration"
            if args.parameter_feedback_epochs
            else (
                "one_shot_selector_calibration"
                if args.keep_position_calibration_epochs
                else "one_shot_distillation"
            )
        ],
    }
    last_position_calibrated = bool(args.keep_position_calibration_epochs)
    last_loss_config = {
        **loss_config,
        "weights": asdict(
            calibration_weights if last_position_calibrated else one_shot_weights
        ),
        "selected_checkpoint_position_calibrated": last_position_calibrated,
        "selected_checkpoint_parameter_feedback": bool(args.parameter_feedback_epochs),
    }
    last_deployment_config = {
        **deployment_config,
        "selected_checkpoint_position_calibrated": last_position_calibrated,
        "selected_checkpoint_parameter_feedback": bool(args.parameter_feedback_epochs),
    }
    last = _checkpoint_with_metadata(
        last,
        model_config=model_config,
        dataset_config=dataset_config,
        loss_config=last_loss_config,
        training_config=training_config,
        deployment_config=last_deployment_config,
        histories=histories,
        objective_version=objective_version,
    )
    torch.save(last, last_path)
    print(
        f"Selected epoch {best['epoch']} | one-shot deployment pass="
        f"{best.get('best_threshold_satisfied_rate', float('nan')):.3f} | "
        f"K={best.get('best_deployment_retained_knot_count', float('nan')):.2f} | "
        f"teacher mask F1="
        f"{best.get('best_teacher_mask_f1', float('nan')):.3f}",
        flush=True,
    )
    print(f"Saved best checkpoint: {args.output}", flush=True)
    print(f"Saved last checkpoint: {last_path}", flush=True)


if __name__ == "__main__":
    main()
