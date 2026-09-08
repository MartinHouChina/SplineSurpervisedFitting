from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from .bspline_inference import BSplineLeastSquaresFit, refit_bspline_control_points
from .minimal_knot_pruning import (
    MinimalKnotPruningResult,
    prune_knots_to_rms_tolerance,
)
from .knot_diagnostics import warp_internal_knots_to_parameterization


@dataclass(frozen=True)
class VerifiedKnotRepairResult:
    """Exact-refit verification with confidence-ordered proposal add-back.

    The fast learned subset is returned unchanged when its standard B-spline
    refit already meets the requested RMS.  Otherwise the immutable, ordered
    proposal geometry is used: learned proposal identities are retained first,
    the remaining candidates are appended by descending network confidence,
    and a logarithmic prefix search finds an observed feasible state.  Optional
    greedy cleanup starts from that much smaller feasible prefix rather than
    from the complete proposal set.  If even the complete proposal set misses
    the bound, an optional last-resort stage may insert sample-parameter knots
    at the largest current geometric residuals.  Those dynamic knots are kept
    separate from the immutable proposal identities.

    This is an adaptive verified deployment path, not a pure one-shot path and
    not a certificate of the globally smallest feasible knot subset.
    """

    fit_tolerance_rms: float
    learned_fit: BSplineLeastSquaresFit
    final_fit: BSplineLeastSquaresFit
    retained_proposal_indices: torch.Tensor
    retained_proposal_mask: torch.Tensor
    final_source: str
    threshold_satisfied: bool
    learned_threshold_satisfied: bool
    fallback_used: bool
    hard_fallback_used: bool
    cleanup_used: bool
    residual_fallback_used: bool
    inserted_internal_knots: torch.Tensor
    residual_fallback_refit_count: int
    prefix_counts_evaluated: tuple[int, ...]
    direct_refit_count: int
    fit_evaluation_count: int
    parameterization_policy: str
    learned_parameterization: str
    final_parameterization: str
    final_parameters: torch.Tensor
    parameterization_fallback_attempted: bool
    parameterization_fallback_used: bool
    parameterization_fallback_full_fit_mse: float | None
    parameterization_fallback_full_threshold_satisfied: bool | None

    @property
    def final_count(self) -> int:
        return int(self.final_fit.internal_knots.numel())

    @property
    def fit_mse(self) -> torch.Tensor:
        return self.final_fit.fit_mse

    @property
    def proposal_count(self) -> int:
        """Number of immutable network proposal identities."""

        return int(self.retained_proposal_mask.numel())

    @property
    def inserted_count(self) -> int:
        """Number of residual-guided knots retained in the final fit."""

        return int(self.inserted_internal_knots.numel())

    @property
    def deployment_candidate_count(self) -> int:
        """Length of the honest proposal-plus-dynamic deployment slot set."""

        return self.proposal_count + self.inserted_count

    @property
    def deployment_retained_mask(self) -> torch.Tensor:
        """Mask over proposal slots followed by retained dynamic-knot slots.

        The first ``proposal_count`` entries preserve immutable proposal
        identity.  Every trailing entry denotes one dynamically inserted knot;
        their values are stored in ``inserted_internal_knots``.  The spline's
        actual internal knots remain globally sorted independently of this slot
        representation.
        """

        inserted_mask = torch.ones(
            self.inserted_count,
            dtype=torch.bool,
            device=self.retained_proposal_mask.device,
        )
        return torch.cat([self.retained_proposal_mask, inserted_mask])


def _validate_inputs(
    parameters: torch.Tensor,
    points: torch.Tensor,
    proposal_knots: torch.Tensor,
    deployment_knots: torch.Tensor,
    learned_mask: torch.Tensor,
    keep_scores: torch.Tensor,
    *,
    fit_tolerance_rms: float,
    min_internal_knots: int,
) -> torch.Tensor:
    if not math.isfinite(fit_tolerance_rms) or fit_tolerance_rms < 0.0:
        raise ValueError("fit_tolerance_rms must be finite and non-negative")
    if isinstance(min_internal_knots, bool) or not isinstance(min_internal_knots, int):
        raise TypeError("min_internal_knots must be an integer")
    if parameters.ndim != 1:
        raise ValueError("parameters must have shape [M]")
    if points.ndim != 2:
        raise ValueError("points must have shape [M,D]")
    if parameters.shape[0] != points.shape[0]:
        raise ValueError("parameters and points must share a point count")
    for name, value in (
        ("proposal_knots", proposal_knots),
        ("deployment_knots", deployment_knots),
        ("learned_mask", learned_mask),
        ("keep_scores", keep_scores),
    ):
        if value.ndim != 1:
            raise ValueError(f"{name} must have shape [K]")
        if value.shape != proposal_knots.shape:
            raise ValueError("proposal/deployment/mask/scores must share shape [K]")
        if value.device != proposal_knots.device:
            raise ValueError("proposal/deployment/mask/scores must share a device")
    if parameters.device != points.device or parameters.device != proposal_knots.device:
        raise ValueError("parameters, points and candidate tensors must share a device")
    if not parameters.is_floating_point() or not points.is_floating_point():
        raise ValueError("parameters and points must be floating-point")
    if (
        not proposal_knots.is_floating_point()
        or not deployment_knots.is_floating_point()
    ):
        raise ValueError("proposal and deployment knots must be floating-point")
    if not keep_scores.is_floating_point():
        raise ValueError("keep_scores must be floating-point")
    if not (
        parameters.dtype
        == points.dtype
        == proposal_knots.dtype
        == deployment_knots.dtype
        == keep_scores.dtype
    ):
        raise ValueError("all floating-point inputs must share a dtype")
    if not (
        torch.isfinite(proposal_knots).all()
        and torch.isfinite(deployment_knots).all()
        and torch.isfinite(keep_scores).all()
    ):
        raise ValueError("candidate knots and scores must be finite")
    if proposal_knots.numel() > 1 and torch.any(
        proposal_knots[1:] < proposal_knots[:-1]
    ):
        raise ValueError("proposal_knots must be non-decreasing")
    if not 0 <= min_internal_knots <= proposal_knots.numel():
        raise ValueError("min_internal_knots must lie in [0,K]")

    if learned_mask.dtype == torch.bool:
        mask = learned_mask.detach().clone()
    elif learned_mask.is_floating_point():
        if not torch.isfinite(learned_mask).all() or not torch.all(
            (learned_mask == 0.0) | (learned_mask == 1.0)
        ):
            raise ValueError("learned_mask must be Boolean or strictly binary")
        mask = learned_mask.to(torch.bool)
    else:
        raise ValueError("learned_mask must be Boolean or floating-point")

    # The complete v12 deployment tensor is not necessarily ordered because
    # only finally retained slots receive a constrained relocation.  Validate
    # and use only that authoritative selected subset on the learned fast path.
    selected_deployment = deployment_knots[mask]
    if selected_deployment.numel() > 1 and torch.any(
        selected_deployment[1:] < selected_deployment[:-1]
    ):
        raise ValueError("selected deployment knots must be non-decreasing")
    return mask


def _retained_indices_after_pruning(
    initial_indices: torch.Tensor,
    pruning: MinimalKnotPruningResult,
) -> torch.Tensor:
    retained = initial_indices.detach().clone()
    for step in pruning.accepted_steps:
        retained = torch.cat(
            [retained[: step.removed_index], retained[step.removed_index + 1 :]]
        )
    return retained


def _pruning_fit_evaluation_count(result: MinimalKnotPruningResult) -> int:
    # One initial refit, K exact candidate evaluations in every deletion
    # batch, and one materialized winning refit per attempted step.
    return 1 + sum(int(step.all_candidate_rmse.numel()) + 1 for step in result.steps)


def _prepare_alternate_parameters(
    parameters: torch.Tensor,
    alternate_parameters: torch.Tensor,
) -> torch.Tensor:
    """Validate and endpoint-canonicalize a corresponding parameter sequence."""

    if not isinstance(alternate_parameters, torch.Tensor):
        raise TypeError("alternate_parameters must be a torch.Tensor")
    if alternate_parameters.ndim != 1:
        raise ValueError("alternate_parameters must have shape [M]")
    if alternate_parameters.shape != parameters.shape:
        raise ValueError("alternate_parameters and parameters must share shape [M]")
    if not alternate_parameters.is_floating_point():
        raise ValueError("alternate_parameters must be floating-point")
    if (
        alternate_parameters.device != parameters.device
        or alternate_parameters.dtype != parameters.dtype
    ):
        raise ValueError(
            "alternate_parameters and parameters must share dtype and device"
        )
    if not torch.isfinite(alternate_parameters).all():
        raise ValueError("alternate_parameters must contain only finite values")

    result = alternate_parameters.detach().clone()
    tolerance = 32.0 * torch.finfo(result.dtype).eps
    if bool(result[0] < -tolerance) or bool(result[-1] > 1.0 + tolerance):
        raise ValueError("alternate_parameters must lie in [0,1]")
    if not bool(torch.abs(result[0]) <= tolerance) or not bool(
        torch.abs(result[-1] - 1.0) <= tolerance
    ):
        raise ValueError("alternate_parameters must span the complete [0,1] domain")
    result = result.clamp(0.0, 1.0)
    result[0] = 0.0
    result[-1] = 1.0
    if torch.any(result[1:] < result[:-1]):
        raise ValueError("alternate_parameters must be non-decreasing")
    return result


def _warp_candidate_geometry(
    parameters: torch.Tensor,
    alternate_parameters: torch.Tensor,
    proposal_knots: torch.Tensor,
    deployment_knots: torch.Tensor,
    learned_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp immutable proposals and authoritative selected deployment slots."""

    warped_proposal = warp_internal_knots_to_parameterization(
        proposal_knots,
        parameters,
        alternate_parameters,
    )
    # Unselected deployment slots are not authoritative in v12 and can be
    # unordered.  Give them their corresponding warped proposal value; only
    # selected relocated slots are mapped from the deployment geometry.
    warped_deployment = warped_proposal.clone()
    if bool(learned_mask.any()):
        warped_deployment[learned_mask] = warp_internal_knots_to_parameterization(
            deployment_knots[learned_mask],
            parameters,
            alternate_parameters,
        )
    return warped_proposal, warped_deployment


@torch.no_grad()
def _insert_knots_at_largest_residuals(
    parameters: torch.Tensor,
    points: torch.Tensor,
    initial_fit: BSplineLeastSquaresFit,
    *,
    mse_tolerance: float,
    max_insertions: int,
    min_gap: float,
    degree: int,
    smoothness_weight: float,
    control_ridge: float,
    interpolate_endpoints: bool,
    rcond: float | None,
) -> tuple[BSplineLeastSquaresFit, torch.Tensor, int]:
    """Insert legal sample parameters at recomputed largest residuals.

    Each insertion is followed by a complete standard B-spline refit.  A fit
    is accepted only after its measured MSE meets ``mse_tolerance``.  This is a
    deterministic local rescue heuristic, not a globally optimal knot search.
    """

    current_fit = initial_fit
    inserted: list[torch.Tensor] = []
    refit_count = 0
    zero = parameters.new_zeros(())
    one = parameters.new_ones(())

    for _ in range(max_insertions):
        residual_sq = (current_fit.reconstructed_points - points).square().sum(dim=-1)
        residual_order = torch.argsort(residual_sq, descending=True, stable=True)
        existing = torch.cat(
            [zero.reshape(1), current_fit.internal_knots, one.reshape(1)]
        )
        selected_parameter: torch.Tensor | None = None
        for point_index in residual_order.tolist():
            candidate = parameters[point_index]
            if not bool((candidate > zero) & (candidate < one)):
                continue
            distances = torch.abs(existing - candidate)
            # Never insert an exact duplicate, even when min_gap is zero.
            if bool(torch.any(distances == 0.0)):
                continue
            if min_gap > 0.0 and bool(torch.any(distances < min_gap)):
                continue
            selected_parameter = candidate
            break
        if selected_parameter is None:
            break

        next_knots = torch.sort(
            torch.cat([current_fit.internal_knots, selected_parameter.reshape(1)])
        ).values
        current_fit = refit_bspline_control_points(
            parameters,
            points,
            next_knots,
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
            interpolate_endpoints=interpolate_endpoints,
            rcond=rcond,
        )
        inserted.append(selected_parameter.detach().clone())
        refit_count += 1
        if float(current_fit.fit_mse) <= mse_tolerance:
            break

    inserted_tensor = (
        torch.stack(inserted) if inserted else initial_fit.internal_knots.new_empty(0)
    )
    return current_fit, inserted_tensor, refit_count


@torch.no_grad()
def _verified_confidence_repair_network(
    parameters: torch.Tensor,
    points: torch.Tensor,
    proposal_knots: torch.Tensor,
    deployment_knots: torch.Tensor,
    learned_mask: torch.Tensor,
    keep_scores: torch.Tensor,
    *,
    fit_tolerance_rms: float,
    min_internal_knots: int = 0,
    degree: int = 3,
    smoothness_weight: float = 1e-6,
    control_ridge: float = 0.0,
    interpolate_endpoints: bool = True,
    rcond: float | None = None,
    compact: bool = True,
    hard_fallback: bool = True,
    residual_fallback: bool = True,
    max_residual_insertions: int = 8,
    residual_min_gap: float = 1e-3,
) -> VerifiedKnotRepairResult:
    """Verify a learned subset and repair only failed curves.

    The prefix search assumes only that a larger confidence prefix is a useful
    repair direction.  Every returned feasible state is checked by an exact
    standard B-spline refit; monotonic error is never trusted for acceptance.
    With nonzero smoothness regularization the first passing prefix need not be
    the globally smallest passing prefix.
    """

    if not isinstance(residual_fallback, bool):
        raise TypeError("residual_fallback must be Boolean")
    if isinstance(max_residual_insertions, bool) or not isinstance(
        max_residual_insertions, int
    ):
        raise TypeError("max_residual_insertions must be an integer")
    if max_residual_insertions < 0:
        raise ValueError("max_residual_insertions must be non-negative")
    if not math.isfinite(residual_min_gap) or residual_min_gap < 0.0:
        raise ValueError("residual_min_gap must be finite and non-negative")

    mask = _validate_inputs(
        parameters,
        points,
        proposal_knots,
        deployment_knots,
        learned_mask,
        keep_scores,
        fit_tolerance_rms=fit_tolerance_rms,
        min_internal_knots=min_internal_knots,
    )
    mse_tolerance = fit_tolerance_rms * fit_tolerance_rms
    candidate_count = int(proposal_knots.numel())
    learned_indices = torch.nonzero(mask, as_tuple=False).flatten()
    learned_fit = refit_bspline_control_points(
        parameters,
        points,
        deployment_knots[mask],
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        interpolate_endpoints=interpolate_endpoints,
        rcond=rcond,
    )
    direct_refit_count = 1
    learned_satisfied = (
        int(learned_indices.numel()) >= min_internal_knots
        and float(learned_fit.fit_mse) <= mse_tolerance
    )
    if learned_satisfied and not compact:
        return VerifiedKnotRepairResult(
            fit_tolerance_rms=float(fit_tolerance_rms),
            learned_fit=learned_fit,
            final_fit=learned_fit,
            retained_proposal_indices=learned_indices.detach().clone(),
            retained_proposal_mask=mask.detach().clone(),
            final_source="learned_verified",
            threshold_satisfied=True,
            learned_threshold_satisfied=True,
            fallback_used=False,
            hard_fallback_used=False,
            cleanup_used=False,
            residual_fallback_used=False,
            inserted_internal_knots=proposal_knots.new_empty(0),
            residual_fallback_refit_count=0,
            prefix_counts_evaluated=(),
            direct_refit_count=direct_refit_count,
            fit_evaluation_count=direct_refit_count,
            parameterization_policy="network",
            learned_parameterization="network_predicted",
            final_parameterization="network_predicted",
            final_parameters=parameters.detach().clone(),
            parameterization_fallback_attempted=False,
            parameterization_fallback_used=False,
            parameterization_fallback_full_fit_mse=None,
            parameterization_fallback_full_threshold_satisfied=None,
        )
    if learned_satisfied:
        # ``compact=True`` is an explicit request to simplify even a feasible
        # learned fast path.  Start from the authoritative relocated survivor
        # geometry and preserve its immutable proposal identities as deletions
        # are accepted.
        learned_pruning = prune_knots_to_rms_tolerance(
            parameters,
            points,
            learned_fit.internal_knots,
            error_tolerance=fit_tolerance_rms,
            min_internal_knots=min_internal_knots,
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
            interpolate_endpoints=interpolate_endpoints,
            rcond=rcond,
        )
        retained_indices = _retained_indices_after_pruning(
            learned_indices,
            learned_pruning,
        )
        retained_mask = torch.zeros_like(mask)
        if retained_indices.numel():
            retained_mask[retained_indices] = True
        pruning_evaluations = _pruning_fit_evaluation_count(learned_pruning)
        return VerifiedKnotRepairResult(
            fit_tolerance_rms=float(fit_tolerance_rms),
            learned_fit=learned_fit,
            final_fit=learned_pruning.final_fit,
            retained_proposal_indices=retained_indices.detach().clone(),
            retained_proposal_mask=retained_mask,
            final_source="learned_verified_compact",
            threshold_satisfied=learned_pruning.threshold_satisfied,
            learned_threshold_satisfied=True,
            fallback_used=False,
            hard_fallback_used=False,
            cleanup_used=True,
            residual_fallback_used=False,
            inserted_internal_knots=proposal_knots.new_empty(0),
            residual_fallback_refit_count=0,
            prefix_counts_evaluated=(),
            direct_refit_count=direct_refit_count,
            fit_evaluation_count=direct_refit_count + pruning_evaluations,
            parameterization_policy="network",
            learned_parameterization="network_predicted",
            final_parameterization="network_predicted",
            final_parameters=parameters.detach().clone(),
            parameterization_fallback_attempted=False,
            parameterization_fallback_used=False,
            parameterization_fallback_full_fit_mse=None,
            parameterization_fallback_full_threshold_satisfied=None,
        )

    unselected_indices = torch.nonzero(~mask, as_tuple=False).flatten()
    unselected_order = torch.argsort(
        keep_scores[unselected_indices],
        descending=True,
        stable=True,
    )
    priority = torch.cat([learned_indices, unselected_indices[unselected_order]], dim=0)
    base_count = max(int(learned_indices.numel()), min_internal_knots)
    evaluated: dict[int, tuple[BSplineLeastSquaresFit, torch.Tensor]] = {}
    evaluated_order: list[int] = []

    def fit_prefix(count: int) -> tuple[BSplineLeastSquaresFit, torch.Tensor]:
        nonlocal direct_refit_count
        cached = evaluated.get(count)
        if cached is not None:
            return cached
        indices = torch.sort(priority[:count]).values
        fit = refit_bspline_control_points(
            parameters,
            points,
            proposal_knots[indices],
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
            interpolate_endpoints=interpolate_endpoints,
            rcond=rcond,
        )
        direct_refit_count += 1
        evaluated[count] = (fit, indices)
        evaluated_order.append(count)
        return fit, indices

    base_fit, base_indices = fit_prefix(base_count)
    feasible_fit: BSplineLeastSquaresFit | None = None
    feasible_indices: torch.Tensor | None = None
    source = "proposal_same_mask"
    if float(base_fit.fit_mse) <= mse_tolerance:
        feasible_fit, feasible_indices = base_fit, base_indices
    elif base_count < candidate_count:
        full_fit, full_indices = fit_prefix(candidate_count)
        if float(full_fit.fit_mse) <= mse_tolerance:
            low = base_count
            high = candidate_count
            while high - low > 1:
                middle = (low + high) // 2
                middle_fit, _ = fit_prefix(middle)
                if float(middle_fit.fit_mse) <= mse_tolerance:
                    high = middle
                else:
                    low = middle
            feasible_fit, feasible_indices = evaluated[high]
            source = "confidence_add_back"

    pruning: MinimalKnotPruningResult | None = None
    hard_fallback_used = False
    cleanup_used = False
    residual_fallback_used = False
    residual_fallback_refit_count = 0
    final_inserted_knots = proposal_knots.new_empty(0)
    if feasible_fit is not None and feasible_indices is not None and compact:
        pruning = prune_knots_to_rms_tolerance(
            parameters,
            points,
            feasible_fit.internal_knots,
            error_tolerance=fit_tolerance_rms,
            min_internal_knots=min_internal_knots,
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
            interpolate_endpoints=interpolate_endpoints,
            rcond=rcond,
        )
        feasible_fit = pruning.final_fit
        feasible_indices = _retained_indices_after_pruning(feasible_indices, pruning)
        cleanup_used = True
        source += "_compact"

    if (
        (feasible_fit is None or feasible_indices is None)
        and residual_fallback
        and max_residual_insertions > 0
    ):
        # This stage is deliberately gated behind an exact failure of the
        # complete immutable proposal set.  Fast learned and confidence-prefix
        # behavior therefore remain unchanged on all easier samples.
        full_fit = (
            base_fit if base_count == candidate_count else evaluated[candidate_count][0]
        )
        residual_fit, attempted_insertions, residual_fallback_refit_count = (
            _insert_knots_at_largest_residuals(
                parameters,
                points,
                full_fit,
                mse_tolerance=mse_tolerance,
                max_insertions=max_residual_insertions,
                min_gap=residual_min_gap,
                degree=degree,
                smoothness_weight=smoothness_weight,
                control_ridge=control_ridge,
                interpolate_endpoints=interpolate_endpoints,
                rcond=rcond,
            )
        )
        direct_refit_count += residual_fallback_refit_count
        residual_fallback_used = residual_fallback_refit_count > 0
        if float(residual_fit.fit_mse) <= mse_tolerance:
            feasible_fit = residual_fit
            feasible_indices = torch.arange(
                candidate_count,
                device=proposal_knots.device,
            )
            final_inserted_knots = attempted_insertions
            source = "residual_guided_insertion"

    if feasible_fit is None or feasible_indices is None:
        if hard_fallback and candidate_count > 0:
            pruning = prune_knots_to_rms_tolerance(
                parameters,
                points,
                proposal_knots,
                error_tolerance=fit_tolerance_rms,
                min_internal_knots=min_internal_knots,
                degree=degree,
                smoothness_weight=smoothness_weight,
                control_ridge=control_ridge,
                interpolate_endpoints=interpolate_endpoints,
                rcond=rcond,
            )
            feasible_fit = pruning.final_fit
            feasible_indices = _retained_indices_after_pruning(
                torch.arange(candidate_count, device=proposal_knots.device),
                pruning,
            )
            source = "hard_fallback"
            hard_fallback_used = True
        else:
            # Select the measured state with the lowest MSE.  It remains
            # explicitly marked infeasible if no evaluated state met the bound.
            candidates = [(learned_fit, learned_indices)] + list(evaluated.values())
            feasible_fit, feasible_indices = min(
                candidates,
                key=lambda item: float(item[0].fit_mse),
            )
            source = "best_infeasible"

    retained_mask = torch.zeros_like(mask)
    if feasible_indices.numel():
        retained_mask[feasible_indices] = True
    pruning_evaluations = (
        _pruning_fit_evaluation_count(pruning) if pruning is not None else 0
    )
    return VerifiedKnotRepairResult(
        fit_tolerance_rms=float(fit_tolerance_rms),
        learned_fit=learned_fit,
        final_fit=feasible_fit,
        retained_proposal_indices=feasible_indices.detach().clone(),
        retained_proposal_mask=retained_mask,
        final_source=source,
        threshold_satisfied=float(feasible_fit.fit_mse) <= mse_tolerance,
        learned_threshold_satisfied=False,
        fallback_used=True,
        hard_fallback_used=hard_fallback_used,
        cleanup_used=cleanup_used,
        residual_fallback_used=residual_fallback_used,
        inserted_internal_knots=final_inserted_knots.detach().clone(),
        residual_fallback_refit_count=residual_fallback_refit_count,
        prefix_counts_evaluated=tuple(evaluated_order),
        direct_refit_count=direct_refit_count,
        fit_evaluation_count=direct_refit_count + pruning_evaluations,
        parameterization_policy="network",
        learned_parameterization="network_predicted",
        final_parameterization="network_predicted",
        final_parameters=parameters.detach().clone(),
        parameterization_fallback_attempted=False,
        parameterization_fallback_used=False,
        parameterization_fallback_full_fit_mse=None,
        parameterization_fallback_full_threshold_satisfied=None,
    )


@torch.no_grad()
def verified_confidence_repair(
    parameters: torch.Tensor,
    points: torch.Tensor,
    proposal_knots: torch.Tensor,
    deployment_knots: torch.Tensor,
    learned_mask: torch.Tensor,
    keep_scores: torch.Tensor,
    *,
    fit_tolerance_rms: float,
    min_internal_knots: int = 0,
    degree: int = 3,
    smoothness_weight: float = 1e-6,
    control_ridge: float = 0.0,
    interpolate_endpoints: bool = True,
    rcond: float | None = None,
    compact: bool = True,
    hard_fallback: bool = True,
    residual_fallback: bool = True,
    max_residual_insertions: int = 8,
    residual_min_gap: float = 1e-3,
    alternate_parameters: torch.Tensor | None = None,
    parameterization_policy: str = "network",
) -> VerifiedKnotRepairResult:
    """Run exact repair in the network or a corresponding alternate domain.

    ``network`` preserves the predicted parameterization.  ``chord`` maps all
    proposal identities through corresponding samples into the supplied
    chord-length parameterization before any exact refit.  The conservative
    ``chord-fallback`` policy first runs the complete network-domain verifier
    without dynamic insertion; only an observed failure then triggers an exact
    full-proposal chord-domain check followed by chord-domain prefix repair,
    optional compaction, and optional residual insertion.

    Every returned spline stores knots in ``final_parameterization`` and must
    be evaluated using ``final_parameters`` (or new values in that same
    domain).  This avoids silently comparing or evaluating knots across
    incompatible parameterizations.
    """

    allowed_policies = {"network", "chord-fallback", "chord"}
    if parameterization_policy not in allowed_policies:
        raise ValueError(
            "parameterization_policy must be one of network, chord-fallback, chord"
        )
    if parameterization_policy == "network":
        return _verified_confidence_repair_network(
            parameters,
            points,
            proposal_knots,
            deployment_knots,
            learned_mask,
            keep_scores,
            fit_tolerance_rms=fit_tolerance_rms,
            min_internal_knots=min_internal_knots,
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
            interpolate_endpoints=interpolate_endpoints,
            rcond=rcond,
            compact=compact,
            hard_fallback=hard_fallback,
            residual_fallback=residual_fallback,
            max_residual_insertions=max_residual_insertions,
            residual_min_gap=residual_min_gap,
        )

    if alternate_parameters is None:
        raise ValueError(
            f"alternate_parameters are required for {parameterization_policy}"
        )
    # Validate the network-domain inputs and canonicalize the binary mask before
    # using it to identify authoritative relocated deployment slots.
    mask = _validate_inputs(
        parameters,
        points,
        proposal_knots,
        deployment_knots,
        learned_mask,
        keep_scores,
        fit_tolerance_rms=fit_tolerance_rms,
        min_internal_knots=min_internal_knots,
    )
    alternate = _prepare_alternate_parameters(parameters, alternate_parameters)
    alternate_proposal, alternate_deployment = _warp_candidate_geometry(
        parameters,
        alternate,
        proposal_knots,
        deployment_knots,
        mask,
    )

    if parameterization_policy == "chord":
        result = _verified_confidence_repair_network(
            alternate,
            points,
            alternate_proposal,
            alternate_deployment,
            mask,
            keep_scores,
            fit_tolerance_rms=fit_tolerance_rms,
            min_internal_knots=min_internal_knots,
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
            interpolate_endpoints=interpolate_endpoints,
            rcond=rcond,
            compact=compact,
            hard_fallback=hard_fallback,
            residual_fallback=residual_fallback,
            max_residual_insertions=max_residual_insertions,
            residual_min_gap=residual_min_gap,
        )
        return replace(
            result,
            final_source=f"chord_parameterization_{result.final_source}",
            parameterization_policy="chord",
            learned_parameterization="chord_length",
            final_parameterization="chord_length",
            final_parameters=alternate.detach().clone(),
        )

    # Conservative fallback: preserve every feasible network-domain result.
    # Residual insertions and the expensive hard fallback are deliberately
    # deferred until after the deterministic full-proposal chord check.
    network_result = _verified_confidence_repair_network(
        parameters,
        points,
        proposal_knots,
        deployment_knots,
        mask,
        keep_scores,
        fit_tolerance_rms=fit_tolerance_rms,
        min_internal_knots=min_internal_knots,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        interpolate_endpoints=interpolate_endpoints,
        rcond=rcond,
        compact=compact,
        hard_fallback=False,
        residual_fallback=False,
        max_residual_insertions=max_residual_insertions,
        residual_min_gap=residual_min_gap,
    )
    if network_result.threshold_satisfied:
        return replace(
            network_result,
            parameterization_policy="chord-fallback",
        )

    mse_tolerance = fit_tolerance_rms * fit_tolerance_rms
    alternate_full_fit = refit_bspline_control_points(
        alternate,
        points,
        alternate_proposal,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        interpolate_endpoints=interpolate_endpoints,
        rcond=rcond,
    )
    alternate_full_mse = float(alternate_full_fit.fit_mse)
    alternate_full_satisfied = alternate_full_mse <= mse_tolerance
    alternate_result = _verified_confidence_repair_network(
        alternate,
        points,
        alternate_proposal,
        alternate_deployment,
        mask,
        keep_scores,
        fit_tolerance_rms=fit_tolerance_rms,
        min_internal_knots=min_internal_knots,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        interpolate_endpoints=interpolate_endpoints,
        rcond=rcond,
        compact=compact,
        hard_fallback=hard_fallback,
        residual_fallback=residual_fallback,
        max_residual_insertions=max_residual_insertions,
        residual_min_gap=residual_min_gap,
    )

    use_alternate = alternate_result.threshold_satisfied or (
        not network_result.threshold_satisfied
        and float(alternate_result.final_fit.fit_mse)
        <= float(network_result.final_fit.fit_mse)
    )
    selected = alternate_result if use_alternate else network_result
    final_source = (
        f"chord_fallback_{selected.final_source}"
        if use_alternate
        else "chord_fallback_network_best_infeasible"
    )
    return replace(
        selected,
        learned_fit=network_result.learned_fit,
        learned_threshold_satisfied=network_result.learned_threshold_satisfied,
        fallback_used=True,
        final_source=final_source,
        prefix_counts_evaluated=(
            network_result.prefix_counts_evaluated
            + alternate_result.prefix_counts_evaluated
        ),
        direct_refit_count=(
            network_result.direct_refit_count + 1 + alternate_result.direct_refit_count
        ),
        fit_evaluation_count=(
            network_result.fit_evaluation_count
            + 1
            + alternate_result.fit_evaluation_count
        ),
        parameterization_policy="chord-fallback",
        learned_parameterization="network_predicted",
        final_parameterization=(
            "chord_length" if use_alternate else "network_predicted"
        ),
        final_parameters=(
            alternate.detach().clone() if use_alternate else parameters.detach().clone()
        ),
        parameterization_fallback_attempted=True,
        parameterization_fallback_used=use_alternate,
        parameterization_fallback_full_fit_mse=alternate_full_mse,
        parameterization_fallback_full_threshold_satisfied=(alternate_full_satisfied),
    )


__all__ = ["VerifiedKnotRepairResult", "verified_confidence_repair"]
