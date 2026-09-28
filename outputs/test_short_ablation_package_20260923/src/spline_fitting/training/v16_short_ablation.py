"""Small warm-start ablations with a shared, frozen proposal and cached teacher.

These helpers add no model parameters, teacher search, or deployment operations.
``fixed`` and ``legacy`` optimize exactly the same selector objective; only the
decoder used for evaluation differs. ``geometry`` additionally learns to retain
the teacher's geometry and differentiates through two hard-subset spline refits.
The hard Top-K decision itself is not differentiable: refit gradients reach the
decoder and shared selector tokens, not the discrete ranking/count operation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from ..losses.deployment_bspline_loss import differentiable_hard_gated_bspline_fit


MODES = ("fixed", "legacy", "geometry")
SELECTOR_ROOTS = frozenset({
    "coverage_embedding", "selection_blocks", "selection_refinement_blocks",
    "keep_head", "adaptive_threshold_norm", "adaptive_threshold_head",
})
DECODER_ROOTS = frozenset({
    "subset_geometry", "survivor_attention", "parameter_attention",
    "survivor_norm", "parameter_norm", "parameter_update", "relocation_update",
    "relocation_blend_logit", "subset_parameter_trust_head",
    "survivor_refinement_blocks", "coupled_subset_blocks",
})


def configure_ablation(model, mode: str) -> dict:
    """Freeze the proposal and configure one arm without modifying its weights.

    Call on a fresh restoration of the same initialization for each arm, before
    constructing the optimizer. Returned parameter-name lists can be used for
    separate selector/decoder learning rates and are JSON serializable.
    Evaluation mode is intentional during training: autograd remains enabled,
    while dropout cannot change the supposedly frozen proposal features.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode == "fixed":
        # The existing residual scale is shared with coupled proposal rounds.
        # Changing it would invalidate a precomputed teacher for such models.
        if len(model.coupled_proposal_blocks):
            raise ValueError("fixed ablation requires no coupled proposal blocks")
        model.subset_geometry_mode = "anchored"
        model.subset_geometry_residual_scale = 0.0
        model._config["subset_geometry_mode"] = "anchored"
        model._config["subset_geometry_residual_scale"] = 0.0

    selector_names, decoder_names, frozen_names = [], [], []
    for name, parameter in model.named_parameters():
        root = name.split(".", 1)[0]
        is_selector = root in SELECTOR_ROOTS
        is_decoder = root in DECODER_ROOTS and mode == "geometry"
        parameter.requires_grad_(is_selector or is_decoder)
        # Clear stale gradients if a caller inspected/backpropagated before
        # configuring the arm. Frozen weights must not receive optimizer decay.
        parameter.grad = None
        if is_selector:
            selector_names.append(name)
        elif is_decoder:
            decoder_names.append(name)
        else:
            frozen_names.append(name)
    if not selector_names:
        raise ValueError("model has no recognized v16 selector parameters")
    model.eval()
    model._short_ablation_mode = mode
    sizes = {name: parameter.numel() for name, parameter in model.named_parameters()}
    return {
        "mode": mode,
        "selector_parameter_names": selector_names,
        "decoder_parameter_names": decoder_names,
        "frozen_parameter_names": frozen_names,
        "selector_parameter_count": sum(sizes[name] for name in selector_names),
        "decoder_parameter_count": sum(sizes[name] for name in decoder_names),
        "frozen_parameter_count": sum(sizes[name] for name in frozen_names),
        "trainable_parameter_count": sum(
            sizes[name] for name in selector_names + decoder_names
        ),
        "optimization_uses_eval_mode": True,
        "subset_geometry_mode": model.subset_geometry_mode,
        "subset_geometry_residual_scale": model.subset_geometry_residual_scale,
    }


def _batch_tolerance(value, points, name):
    result = torch.as_tensor(value, dtype=points.dtype, device=points.device)
    if result.ndim == 0:
        result = result.expand(points.shape[0])
    if result.shape != points.shape[:1]:
        raise ValueError(f"{name} must be scalar or have shape [B]")
    if not torch.isfinite(result).all() or (result <= 0).any():
        raise ValueError(f"{name} must be finite and positive")
    return result


def _teacher_batch(teacher: Mapping, points, capacity):
    shapes = {
        "mask": (points.shape[0], capacity),
        "params": points.shape[:2],
        "knots": (points.shape[0], capacity),
        "feasible": points.shape[:1],
    }
    result = {}
    for name, shape in shapes.items():
        value = teacher[name]
        if not isinstance(value, torch.Tensor) or value.shape != shape:
            raise ValueError(f"teacher {name} must be a tensor with shape {tuple(shape)}")
        if name in {"mask", "feasible"} and value.dtype != torch.bool:
            raise ValueError(f"teacher {name} must be boolean")
        dtype = torch.bool if name in {"mask", "feasible"} else points.dtype
        result[name] = value.detach().to(device=points.device, dtype=dtype)
    # Infeasible cache rows may intentionally carry sentinel geometry. They are
    # never sent to a loss/refit and must not introduce 0 * NaN contamination.
    valid = result["feasible"]
    for name in ("params", "knots"):
        if not torch.isfinite(result[name][valid]).all():
            raise ValueError(f"feasible teacher {name} must be finite")
    return result


def _valid_context(context, valid):
    return {
        name: value[valid] if isinstance(value, torch.Tensor)
        and value.ndim > 0 and value.shape[0] == valid.shape[0] else value
        for name, value in context.items()
    }


def _fit_objective(output, points, mask, degree, mse_tolerance, peak_tolerance):
    # The same open-clamped basis and fixed endpoint controls as deployment.
    # Float64 and a tiny numerical ridge keep the differentiable solve stable;
    # the runner must still evaluate exact, unregularized deployment separately.
    fit = differentiable_hard_gated_bspline_fit(
        output["params"].double(), points.double(),
        output["internal_knots"].double(), mask, degree=degree,
        smoothness_weight=0.0, control_ridge=0.0, solver_jitter=1e-10,
    )
    mse = fit["per_sample_mse"]
    peak = fit["per_point_squared_error"].amax(-1)
    mse_ratio = mse / mse_tolerance.double()
    peak_ratio = peak / peak_tolerance.double()
    penalty = mse_ratio + F.relu(peak_ratio - 1.0)
    return penalty.mean(), {
        "mse": mse.mean(),
        "max_squared_error": peak.mean(),
        "pass_rate": ((mse_ratio <= 1) & (peak_ratio <= 1)).double().mean(),
    }


def ablation_loss(
    model,
    points: torch.Tensor,
    teacher: Mapping,
    mode: str,
    mse_tolerance,
    max_squared_error_tolerance,
    count_weight: float = 0.5,
    parameter_weight: float = 1.0,
    knot_weight: float = 1.0,
    fit_weight: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Compute feasible-only cached-teacher supervision; never search for masks.

    Teacher fields: boolean ``mask[B,K]`` and ``feasible[B]``, frozen proposal
    ``params[B,M]`` and ``knots[B,K]``. Invalid rows are wholly excluded. Count
    calibration targets the deployed pre-ceil score (including safety reserve)
    at ``K_teacher - .25``; the zero-count target is zero. Geometry distillation
    measures displacement in units of the mean sample/candidate spacing.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if getattr(model, "_short_ablation_mode", None) != mode:
        raise ValueError("call configure_ablation(model, mode) before ablation_loss")
    for name, value in {
        "count_weight": count_weight, "parameter_weight": parameter_weight,
        "knot_weight": knot_weight, "fit_weight": fit_weight,
    }.items():
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    model.eval()
    tolerance = _batch_tolerance(mse_tolerance, points, "mse_tolerance")
    peak_tolerance = _batch_tolerance(
        max_squared_error_tolerance, points, "max_squared_error_tolerance",
    )
    targets = _teacher_batch(teacher, points, model.max_internal_knots)
    context = model.encode_candidates(points, mse_tolerance=tolerance)
    valid = targets["feasible"]
    zero = context["keep_logits"].sum() * 0.0
    metrics = {name: zero.detach() for name in (
        "loss", "selector_loss", "bce_loss", "count_loss", "parameter_loss",
        "knot_loss", "fit_loss", "teacher_fit_loss", "current_fit_loss",
        "teacher_mse", "teacher_max_squared_error", "teacher_pass_rate",
        "current_mse", "current_max_squared_error", "current_pass_rate",
        "teacher_count", "selected_count", "count_mass", "count_score",
        "count_target", "mask_precision", "mask_recall",
    )}
    metrics.update(valid_count=valid.sum().detach(), feasible_fraction=valid.float().mean())
    if not valid.any():
        return zero, metrics

    context = _valid_context(context, valid)
    mask = targets["mask"][valid]
    logits = context["keep_logits"]
    probabilities = context["keep_probabilities"]
    bce = F.binary_cross_entropy_with_logits(logits, mask.to(logits.dtype))
    teacher_count = mask.sum(-1).to(logits.dtype)
    count_target = (teacher_count - 0.25).clamp(0, logits.shape[1])
    count_score = context["one_shot_requested_count_score"]
    count_loss = ((count_score - count_target) / logits.shape[1]).square().mean()
    selector_loss = bce + count_weight * count_loss
    loss = selector_loss
    current_mask = model.select_mask(context)

    if mode == "geometry":
        teacher_output = model.decode_subset(context, mask)
        parameter_delta = (
            teacher_output["params"] - targets["params"][valid]
        ) * (points.shape[1] - 1)
        parameter_loss = parameter_delta.square().mean()
        knot_delta = (
            teacher_output["internal_knots"] - targets["knots"][valid]
        ) * (model.max_internal_knots + 1)
        # Equal curve weighting, including empty teacher sets (zero knot loss).
        knot_loss = (knot_delta.square().masked_fill(~mask, 0).sum(-1)
                     / mask.sum(-1).clamp_min(1)).mean()
        current_output = model.decode_subset(context, current_mask)
        teacher_fit_loss, teacher_fit = _fit_objective(
            teacher_output, points[valid], mask, model.degree,
            tolerance[valid], peak_tolerance[valid],
        )
        current_fit_loss, current_fit = _fit_objective(
            current_output, points[valid], current_mask, model.degree,
            tolerance[valid], peak_tolerance[valid],
        )
        fit_loss = 0.5 * (teacher_fit_loss + current_fit_loss)
        loss = (loss + parameter_weight * parameter_loss + knot_weight * knot_loss
                + fit_weight * fit_loss)
        metrics.update(
            parameter_loss=parameter_loss, knot_loss=knot_loss, fit_loss=fit_loss,
            teacher_fit_loss=teacher_fit_loss, current_fit_loss=current_fit_loss,
            **{f"teacher_{name}": value for name, value in teacher_fit.items()},
            **{f"current_{name}": value for name, value in current_fit.items()},
        )

    true_positive = (current_mask & mask).sum().to(logits.dtype)
    metrics.update(
        loss=loss, selector_loss=selector_loss, bce_loss=bce, count_loss=count_loss,
        teacher_count=teacher_count.mean(), selected_count=current_mask.sum(-1).float().mean(),
        count_mass=probabilities.sum(-1).mean(), count_score=count_score.mean(),
        count_target=count_target.mean(),
        mask_precision=true_positive / current_mask.sum().clamp_min(1),
        mask_recall=true_positive / mask.sum().clamp_min(1),
    )
    return loss, {name: value.detach() for name, value in metrics.items()}
