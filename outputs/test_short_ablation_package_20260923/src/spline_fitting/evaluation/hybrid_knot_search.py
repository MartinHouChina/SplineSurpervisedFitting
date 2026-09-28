from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..spline.bspline_deletion_teacher import single_knot_deletion_mse_batch
from .bspline_inference import BSplineLeastSquaresFit, refit_bspline_control_points


@dataclass(frozen=True)
class KnotPositionRefinementResult:
    """Result of deterministic variable-projection knot-position refinement."""

    initial_fit: BSplineLeastSquaresFit
    final_fit: BSplineLeastSquaresFit
    refit_count: int
    sweeps_completed: int

    @property
    def initial_mse(self) -> float:
        return float(self.initial_fit.fit_mse)

    @property
    def final_mse(self) -> float:
        return float(self.final_fit.fit_mse)


@dataclass(frozen=True)
class HybridKnotSearchResult:
    """Minimum-count feasible state found by the deterministic hybrid search.

    The result is lexicographic among states visited by the search: a feasible
    state with fewer internal knots always wins, and MSE breaks equal-count
    ties.  The exhaustive greedy MSE result is retained as a permanent
    incumbent, so a feasible greedy result cannot be replaced by a structure
    with more knots.
    """

    mse_tolerance: float
    proposal_count: int
    proposal_knots: torch.Tensor
    greedy_fit: BSplineLeastSquaresFit
    greedy_retained_proposal_indices: torch.Tensor
    greedy_retained_proposal_mask: torch.Tensor
    final_fit: BSplineLeastSquaresFit
    retained_proposal_indices: torch.Tensor
    retained_proposal_mask: torch.Tensor
    final_source: str
    threshold_satisfied: bool
    greedy_threshold_satisfied: bool
    visited_state_count: int
    refit_count: int
    levels_explored: tuple[int, ...]
    position_refined_counts: tuple[int, ...]
    start_sources: tuple[str, ...]

    @property
    def final_internal_knots(self) -> torch.Tensor:
        return self.final_fit.internal_knots

    @property
    def final_count(self) -> int:
        return int(self.final_internal_knots.numel())

    @property
    def greedy_count(self) -> int:
        return int(self.greedy_fit.internal_knots.numel())

    @property
    def fit_mse(self) -> torch.Tensor:
        return self.final_fit.fit_mse

    @property
    def retained_proposal_knots(self) -> torch.Tensor:
        """Original proposal positions corresponding to the retained identities."""

        return self.proposal_knots[self.retained_proposal_indices]

    @property
    def final_position_shifts(self) -> torch.Tensor:
        """Signed proposal-to-final displacement for every retained knot."""

        return self.final_internal_knots - self.retained_proposal_knots

    @property
    def mean_absolute_position_shift(self) -> float:
        shifts = self.final_position_shifts
        return float(shifts.abs().mean()) if shifts.numel() else 0.0

    @property
    def max_absolute_position_shift(self) -> float:
        shifts = self.final_position_shifts
        return float(shifts.abs().max()) if shifts.numel() else 0.0


@dataclass(frozen=True)
class _SearchState:
    proposal_indices: torch.Tensor
    fit: BSplineLeastSquaresFit
    source: str

    @property
    def count(self) -> int:
        return int(self.proposal_indices.numel())

    @property
    def mse(self) -> float:
        return float(self.fit.fit_mse)


@dataclass(frozen=True)
class _DeletionCandidate:
    proposal_indices: torch.Tensor
    knots: torch.Tensor
    mse: float
    source: str


@dataclass
class _SearchStatistics:
    refit_count: int = 0
    visited_state_count: int = 0


def _validate_search_options(
    *,
    mse_tolerance: float,
    min_internal_knots: int,
    proposal_count: int,
    beam_width: int,
    branch_factor: int,
    position_sweeps: int,
    position_grid_size: int,
    position_restarts: int,
    position_refine_count_margin: int,
    position_refine_candidate_multiplier: int,
    min_gap: float,
) -> None:
    if not math.isfinite(mse_tolerance) or mse_tolerance < 0.0:
        raise ValueError("mse_tolerance must be finite and non-negative")
    if isinstance(min_internal_knots, bool) or not isinstance(min_internal_knots, int):
        raise TypeError("min_internal_knots must be an integer")
    if not 0 <= min_internal_knots <= proposal_count:
        raise ValueError("min_internal_knots must lie in [0, proposal_count]")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if branch_factor < 0:
        raise ValueError("branch_factor must be non-negative; zero means all")
    if position_sweeps < 0:
        raise ValueError("position_sweeps must be non-negative")
    if position_grid_size < 3 or position_grid_size % 2 == 0:
        raise ValueError("position_grid_size must be an odd integer of at least 3")
    if position_restarts < 1:
        raise ValueError("position_restarts must be positive")
    if position_refine_count_margin < 0:
        raise ValueError("position_refine_count_margin must be non-negative")
    if position_refine_candidate_multiplier < 1:
        raise ValueError("position_refine_candidate_multiplier must be positive")
    if not math.isfinite(min_gap) or min_gap < 0.0:
        raise ValueError("min_gap must be finite and non-negative")
    if proposal_count > 0 and min_gap * (proposal_count + 1) >= 1.0:
        raise ValueError("min_gap is too large for the proposal knot count")


def _validate_optional_start(
    proposal_knots: torch.Tensor,
    deployment_knots: torch.Tensor | None,
    learned_mask: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if (deployment_knots is None) != (learned_mask is None):
        raise ValueError("deployment_knots and learned_mask must be provided together")
    if deployment_knots is None or learned_mask is None:
        return None, None
    if deployment_knots.shape != proposal_knots.shape:
        raise ValueError("deployment_knots must share the proposal shape")
    if deployment_knots.device != proposal_knots.device:
        raise ValueError("deployment_knots must share the proposal device")
    if deployment_knots.dtype != proposal_knots.dtype:
        raise ValueError("deployment_knots must share the proposal dtype")
    if learned_mask.shape != proposal_knots.shape:
        raise ValueError("learned_mask must share the proposal shape")
    if learned_mask.device != proposal_knots.device:
        raise ValueError("learned_mask must share the proposal device")
    if learned_mask.dtype == torch.bool:
        mask = learned_mask.detach().clone()
    elif learned_mask.is_floating_point():
        if not torch.isfinite(learned_mask).all() or not torch.all(
            (learned_mask == 0.0) | (learned_mask == 1.0)
        ):
            raise ValueError("learned_mask must be Boolean or strictly binary")
        mask = learned_mask.to(torch.bool)
    else:
        raise ValueError("learned_mask must be Boolean or floating point")
    return deployment_knots, mask


def _validate_ordered_knots(knots: torch.Tensor, min_gap: float) -> None:
    if knots.ndim != 1:
        raise ValueError("internal knots must have shape [K]")
    if not knots.is_floating_point() or not torch.isfinite(knots).all():
        raise ValueError("internal knots must be finite floating-point values")
    if torch.any((knots <= 0.0) | (knots >= 1.0)):
        raise ValueError("internal knots must lie strictly inside (0, 1)")
    if knots.numel() > 1 and torch.any(knots[1:] < knots[:-1]):
        raise ValueError("internal knots must be non-decreasing")
    if min_gap > 0.0 and knots.numel() > 0:
        boundaries = torch.cat([knots.new_zeros(1), knots, knots.new_ones(1)])
        if torch.any(boundaries[1:] - boundaries[:-1] < min_gap - 1e-7):
            raise ValueError("internal knots do not respect min_gap")


def _mask_from_indices(
    proposal_count: int,
    indices: torch.Tensor,
) -> torch.Tensor:
    mask = torch.zeros(proposal_count, device=indices.device, dtype=torch.bool)
    if indices.numel() > 0:
        mask[indices] = True
    return mask


def _state_key(state: _SearchState) -> tuple[int, ...]:
    return tuple(int(value) for value in state.proposal_indices.tolist())


def _better_feasible_state(
    candidate: _SearchState,
    incumbent: _SearchState,
    mse_tolerance: float,
) -> bool:
    candidate_feasible = candidate.mse <= mse_tolerance
    incumbent_feasible = incumbent.mse <= mse_tolerance
    if candidate_feasible != incumbent_feasible:
        return candidate_feasible
    if candidate_feasible:
        return (candidate.count, candidate.mse) < (incumbent.count, incumbent.mse)
    return (candidate.mse, candidate.count) < (incumbent.mse, incumbent.count)


def _refit(
    parameters: torch.Tensor,
    points: torch.Tensor,
    knots: torch.Tensor,
    *,
    statistics: _SearchStatistics | None,
    degree: int,
    smoothness_weight: float,
    control_ridge: float,
    interpolate_endpoints: bool,
    rcond: float | None,
) -> BSplineLeastSquaresFit:
    fit = refit_bspline_control_points(
        parameters,
        points,
        knots,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        interpolate_endpoints=interpolate_endpoints,
        rcond=rcond,
    )
    if statistics is not None:
        statistics.refit_count += 1
    return fit


def _coordinate_refine(
    parameters: torch.Tensor,
    points: torch.Tensor,
    initial_fit: BSplineLeastSquaresFit,
    *,
    statistics: _SearchStatistics | None,
    degree: int,
    smoothness_weight: float,
    control_ridge: float,
    interpolate_endpoints: bool,
    rcond: float | None,
    min_gap: float,
    sweeps: int,
    grid_size: int,
    restarts: int,
) -> tuple[BSplineLeastSquaresFit, int, int]:
    initial_knots = initial_fit.internal_knots
    knot_count = int(initial_knots.numel())
    if knot_count == 0 or sweeps == 0:
        return initial_fit, 0, 0

    best_fit = initial_fit
    local_refits = 0
    completed_sweeps = 0
    uniform = torch.arange(
        1,
        knot_count + 1,
        device=initial_knots.device,
        dtype=initial_knots.dtype,
    ) / (knot_count + 1)

    for restart in range(restarts):
        if restart == 0:
            restart_fit = initial_fit
        else:
            # Deterministic, ordered multi-starts between the network geometry
            # and a well-spaced uniform geometry.  Convex interpolation keeps
            # the knot order and the configured minimum gap.
            # Include the fully uniform layout as the last restart.  The old
            # 0.35 cap barely moved a clustered proposal and could leave the
            # coordinate search trapped inside that cluster's narrow neighbour
            # intervals.  Interpolation remains ordered and respects min_gap.
            alpha = restart / max(restarts - 1, 1)
            restart_knots = (1.0 - alpha) * initial_knots + alpha * uniform
            restart_fit = _refit(
                parameters,
                points,
                restart_knots,
                statistics=statistics,
                degree=degree,
                smoothness_weight=smoothness_weight,
                control_ridge=control_ridge,
                interpolate_endpoints=interpolate_endpoints,
                rcond=rcond,
            )
            local_refits += 1

        current_fit = restart_fit
        current_knots = restart_fit.internal_knots.detach().clone()
        for sweep in range(sweeps):
            order = (
                range(knot_count) if sweep % 2 == 0 else range(knot_count - 1, -1, -1)
            )
            for knot_index in order:
                machine_margin = 32.0 * torch.finfo(current_knots.dtype).eps
                boundary_gap = max(min_gap, machine_margin)
                lower = (
                    boundary_gap
                    if knot_index == 0
                    else float(current_knots[knot_index - 1]) + min_gap
                )
                upper = (
                    1.0 - boundary_gap
                    if knot_index == knot_count - 1
                    else float(current_knots[knot_index + 1]) - min_gap
                )
                if upper <= lower:
                    continue
                center = float(current_knots[knot_index])
                half_width = 0.5 * (upper - lower) * (0.5**sweep)
                trial_lower = max(lower, center - half_width)
                trial_upper = min(upper, center + half_width)
                trial_values = torch.linspace(
                    trial_lower,
                    trial_upper,
                    grid_size,
                    device=current_knots.device,
                    dtype=current_knots.dtype,
                )
                best_coordinate_fit = current_fit
                best_coordinate_knots = current_knots
                for trial_value in trial_values:
                    trial_knots = current_knots.detach().clone()
                    trial_knots[knot_index] = trial_value
                    trial_fit = _refit(
                        parameters,
                        points,
                        trial_knots,
                        statistics=statistics,
                        degree=degree,
                        smoothness_weight=smoothness_weight,
                        control_ridge=control_ridge,
                        interpolate_endpoints=interpolate_endpoints,
                        rcond=rcond,
                    )
                    local_refits += 1
                    if float(trial_fit.fit_mse) < float(best_coordinate_fit.fit_mse):
                        best_coordinate_fit = trial_fit
                        best_coordinate_knots = trial_knots
                current_fit = best_coordinate_fit
                current_knots = best_coordinate_knots
            completed_sweeps += 1
        if float(current_fit.fit_mse) < float(best_fit.fit_mse):
            best_fit = current_fit

    # Never let a local restart replace the supplied exact fit with a worse
    # state.  This invariant is useful both scientifically and for callers.
    if float(best_fit.fit_mse) > float(initial_fit.fit_mse):
        best_fit = initial_fit
    return best_fit, local_refits, completed_sweeps


@torch.no_grad()
def refine_knot_positions(
    parameters: torch.Tensor,
    points: torch.Tensor,
    internal_knots: torch.Tensor,
    *,
    degree: int = 3,
    smoothness_weight: float = 1e-6,
    control_ridge: float = 0.0,
    interpolate_endpoints: bool = True,
    rcond: float | None = None,
    min_gap: float = 1e-4,
    sweeps: int = 2,
    grid_size: int = 5,
    restarts: int = 1,
) -> KnotPositionRefinementResult:
    """Refine ordered knot positions while exactly eliminating controls.

    Each scalar trial moves one knot inside the legal interval formed by its
    neighbours, then performs a complete standard B-spline control-point
    refit.  The routine is derivative-free and deterministic; it never returns
    a fit with higher MSE than the supplied positions.
    """

    _validate_search_options(
        mse_tolerance=0.0,
        min_internal_knots=0,
        proposal_count=int(internal_knots.numel()),
        beam_width=1,
        branch_factor=0,
        position_sweeps=sweeps,
        position_grid_size=grid_size,
        position_restarts=restarts,
        position_refine_count_margin=0,
        position_refine_candidate_multiplier=1,
        min_gap=min_gap,
    )
    _validate_ordered_knots(internal_knots, min_gap)
    initial_fit = _refit(
        parameters,
        points,
        internal_knots,
        statistics=None,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        interpolate_endpoints=interpolate_endpoints,
        rcond=rcond,
    )
    final_fit, local_refits, completed_sweeps = _coordinate_refine(
        parameters,
        points,
        initial_fit,
        statistics=None,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        interpolate_endpoints=interpolate_endpoints,
        rcond=rcond,
        min_gap=min_gap,
        sweeps=sweeps,
        grid_size=grid_size,
        restarts=restarts,
    )
    return KnotPositionRefinementResult(
        initial_fit=initial_fit,
        final_fit=final_fit,
        refit_count=1 + local_refits,
        sweeps_completed=completed_sweeps,
    )


def _greedy_mse_state(
    parameters: torch.Tensor,
    points: torch.Tensor,
    proposal_knots: torch.Tensor,
    *,
    mse_tolerance: float,
    min_internal_knots: int,
    statistics: _SearchStatistics,
    degree: int,
    smoothness_weight: float,
    control_ridge: float,
    interpolate_endpoints: bool,
    rcond: float | None,
) -> tuple[_SearchState, _SearchState]:
    retained_knots = proposal_knots.detach().clone()
    retained_indices = torch.arange(
        proposal_knots.numel(), device=proposal_knots.device, dtype=torch.long
    )
    initial_fit = _refit(
        parameters,
        points,
        retained_knots,
        statistics=statistics,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        interpolate_endpoints=interpolate_endpoints,
        rcond=rcond,
    )
    initial_state = _SearchState(retained_indices.clone(), initial_fit, "proposal")
    current_fit = initial_fit

    while retained_knots.numel() > min_internal_knots:
        deletion_mse = single_knot_deletion_mse_batch(
            parameters.unsqueeze(0),
            points.unsqueeze(0),
            retained_knots.unsqueeze(0),
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
            interpolate_endpoints=interpolate_endpoints,
            rcond=rcond,
        )[0]
        statistics.refit_count += int(retained_knots.numel())
        statistics.visited_state_count += int(retained_knots.numel())
        best_relative_index = int(torch.argmin(deletion_mse).item())
        child_knots = torch.cat(
            [
                retained_knots[:best_relative_index],
                retained_knots[best_relative_index + 1 :],
            ]
        )
        child_fit = _refit(
            parameters,
            points,
            child_knots,
            statistics=statistics,
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
            interpolate_endpoints=interpolate_endpoints,
            rcond=rcond,
        )
        if float(child_fit.fit_mse) > mse_tolerance:
            break
        retained_knots = child_knots
        retained_indices = torch.cat(
            [
                retained_indices[:best_relative_index],
                retained_indices[best_relative_index + 1 :],
            ]
        )
        current_fit = child_fit

    return initial_state, _SearchState(retained_indices, current_fit, "greedy")


def _deduplicate_best(states: list[_SearchState]) -> list[_SearchState]:
    best_by_subset: dict[tuple[int, ...], _SearchState] = {}
    for state in states:
        key = _state_key(state)
        previous = best_by_subset.get(key)
        if previous is None or state.mse < previous.mse:
            best_by_subset[key] = state
    return sorted(best_by_subset.values(), key=lambda state: (state.mse, state.count))


@torch.no_grad()
def hybrid_minimal_knot_search(
    parameters: torch.Tensor,
    points: torch.Tensor,
    proposal_knots: torch.Tensor,
    *,
    deployment_knots: torch.Tensor | None = None,
    learned_mask: torch.Tensor | None = None,
    mse_tolerance: float,
    min_internal_knots: int = 0,
    degree: int = 3,
    smoothness_weight: float = 1e-6,
    control_ridge: float = 0.0,
    interpolate_endpoints: bool = True,
    rcond: float | None = None,
    beam_width: int = 4,
    branch_factor: int = 4,
    position_sweeps: int = 1,
    position_grid_size: int = 5,
    position_restarts: int = 1,
    position_refine_count_margin: int = 1,
    position_refine_candidate_multiplier: int = 4,
    min_gap: float = 1e-4,
) -> HybridKnotSearchResult:
    """Search for fewer knots under an exact standard-B-spline MSE bound.

    ``branch_factor=0`` exhaustively expands every single deletion from every
    retained beam state.  Positive values retain the most promising exact
    batch-scored deletions before coordinate refinement.  The algorithm is a
    deterministic beam/local search, not a proof of global subset optimality.
    Expensive coordinate refinement is concentrated within
    ``position_refine_count_margin`` levels of the greedy stopping count; high
    count levels use exact refits only, leaving more compute for the actual
    feasibility boundary.
    Nevertheless, its permanent exhaustive-greedy incumbent guarantees that a
    feasible result never contains more knots than that traditional baseline.
    """

    proposal_count = int(proposal_knots.numel())
    _validate_search_options(
        mse_tolerance=mse_tolerance,
        min_internal_knots=min_internal_knots,
        proposal_count=proposal_count,
        beam_width=beam_width,
        branch_factor=branch_factor,
        position_sweeps=position_sweeps,
        position_grid_size=position_grid_size,
        position_restarts=position_restarts,
        position_refine_count_margin=position_refine_count_margin,
        position_refine_candidate_multiplier=position_refine_candidate_multiplier,
        min_gap=min_gap,
    )
    _validate_ordered_knots(proposal_knots, min_gap)
    deployment_knots, learned_mask = _validate_optional_start(
        proposal_knots, deployment_knots, learned_mask
    )
    if deployment_knots is not None and learned_mask is not None:
        _validate_ordered_knots(deployment_knots[learned_mask], min_gap)

    statistics = _SearchStatistics()
    proposal_state, greedy_state = _greedy_mse_state(
        parameters,
        points,
        proposal_knots,
        mse_tolerance=mse_tolerance,
        min_internal_knots=min_internal_knots,
        statistics=statistics,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        interpolate_endpoints=interpolate_endpoints,
        rcond=rcond,
    )
    incumbent = greedy_state
    roots: list[_SearchState] = [proposal_state, greedy_state]
    start_sources = ["proposal", "greedy"]

    if deployment_knots is not None and learned_mask is not None:
        learned_indices = torch.nonzero(learned_mask, as_tuple=False).flatten()
        learned_knots = deployment_knots[learned_mask]
        learned_fit = _refit(
            parameters,
            points,
            learned_knots,
            statistics=statistics,
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
            interpolate_endpoints=interpolate_endpoints,
            rcond=rcond,
        )
        learned_state = _SearchState(learned_indices, learned_fit, "learned")
        start_sources.append("learned")
        if learned_state.count >= min_internal_knots:
            roots.append(learned_state)
            if _better_feasible_state(learned_state, incumbent, mse_tolerance):
                incumbent = learned_state

            # A learned mask is a useful structure proposal, not a reason to
            # freeze its surviving coordinates.  Refine this root even when
            # its initial fit is just above tolerance: relocation can rescue
            # the same low-count structure without re-inserting a knot.
            learned_refined_fit, _, _ = _coordinate_refine(
                parameters,
                points,
                learned_state.fit,
                statistics=statistics,
                degree=degree,
                smoothness_weight=smoothness_weight,
                control_ridge=control_ridge,
                interpolate_endpoints=interpolate_endpoints,
                rcond=rcond,
                min_gap=min_gap,
                sweeps=position_sweeps,
                grid_size=position_grid_size,
                restarts=position_restarts,
            )
            learned_refined = _SearchState(
                learned_indices,
                learned_refined_fit,
                "learned_refined",
            )
            roots.append(learned_refined)
            start_sources.append("learned_refined")
            if _better_feasible_state(learned_refined, incumbent, mse_tolerance):
                incumbent = learned_refined

    # Refine the reliable greedy state before expanding below its count.  The
    # unmodified state remains the permanent incumbent and root.
    greedy_refined_fit, _, _ = _coordinate_refine(
        parameters,
        points,
        greedy_state.fit,
        statistics=statistics,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        interpolate_endpoints=interpolate_endpoints,
        rcond=rcond,
        min_gap=min_gap,
        sweeps=position_sweeps,
        grid_size=position_grid_size,
        restarts=position_restarts,
    )
    greedy_refined = _SearchState(
        greedy_state.proposal_indices,
        greedy_refined_fit,
        "greedy_refined",
    )
    roots.append(greedy_refined)
    start_sources.append("greedy_refined")
    if _better_feasible_state(greedy_refined, incumbent, mse_tolerance):
        incumbent = greedy_refined

    states_by_count: dict[int, list[_SearchState]] = {}
    for root in roots:
        if root.mse <= mse_tolerance or root is proposal_state:
            states_by_count.setdefault(root.count, []).append(root)

    levels_explored: list[int] = []
    position_refined_counts: set[int] = {greedy_state.count}
    if deployment_knots is not None and learned_mask is not None:
        position_refined_counts.add(int(learned_mask.sum().item()))
    for count in range(proposal_count, min_internal_knots, -1):
        parents = _deduplicate_best(states_by_count.get(count, []))[:beam_width]
        # Infeasible learned starts are deliberately not allowed to displace a
        # feasible greedy path.  Proposal is retained only to discover an
        # alternative path when its all-knot fit itself meets the threshold.
        parents = [parent for parent in parents if parent.mse <= mse_tolerance]
        if not parents:
            continue
        levels_explored.append(count)
        raw_candidates: dict[tuple[int, ...], _DeletionCandidate] = {}
        for parent in parents:
            parent_knots = parent.fit.internal_knots
            deletion_mse = single_knot_deletion_mse_batch(
                parameters.unsqueeze(0),
                points.unsqueeze(0),
                parent_knots.unsqueeze(0),
                degree=degree,
                smoothness_weight=smoothness_weight,
                control_ridge=control_ridge,
                interpolate_endpoints=interpolate_endpoints,
                rcond=rcond,
            )[0]
            statistics.refit_count += count
            statistics.visited_state_count += count
            order = torch.argsort(deletion_mse)
            if branch_factor > 0:
                order = order[: min(branch_factor, count)]
            for relative_index_tensor in order:
                relative_index = int(relative_index_tensor.item())
                child_knots = torch.cat(
                    [
                        parent_knots[:relative_index],
                        parent_knots[relative_index + 1 :],
                    ]
                )
                child_indices = torch.cat(
                    [
                        parent.proposal_indices[:relative_index],
                        parent.proposal_indices[relative_index + 1 :],
                    ]
                )
                key = tuple(int(value) for value in child_indices.tolist())
                candidate = _DeletionCandidate(
                    proposal_indices=child_indices,
                    knots=child_knots,
                    mse=float(deletion_mse[relative_index]),
                    source=f"beam:{parent.source}",
                )
                previous = raw_candidates.get(key)
                if previous is None or candidate.mse < previous.mse:
                    raw_candidates[key] = candidate

        # The batched deletion score is already the exact standard-refit MSE.
        # Materialize full fit objects only for the globally best unique
        # subsets, avoiding a second solver call for every discarded branch.
        child_count = count - 1
        refine_positions = (
            child_count <= greedy_state.count + position_refine_count_margin
        )
        # Deletion and relocation are coupled only if candidates are allowed to
        # move before the beam is truncated.  Near the feasibility boundary we
        # therefore refine a wider pre-beam pool.  This retains candidates
        # whose fixed-position deletion score is mediocre but whose remaining
        # knots can move into a much better configuration.
        materialize_width = beam_width
        if refine_positions:
            materialize_width *= position_refine_candidate_multiplier
        screened_candidates = sorted(
            raw_candidates.values(), key=lambda candidate: candidate.mse
        )[:materialize_width]
        screened: list[_SearchState] = []
        for candidate in screened_candidates:
            child_fit = _refit(
                parameters,
                points,
                candidate.knots,
                statistics=statistics,
                degree=degree,
                smoothness_weight=smoothness_weight,
                control_ridge=control_ridge,
                interpolate_endpoints=interpolate_endpoints,
                rcond=rcond,
            )
            screened.append(
                _SearchState(
                    candidate.proposal_indices,
                    child_fit,
                    candidate.source,
                )
            )

        # Refine only the best unique states.  Screening first prevents an
        # O(beam * branch * K * grid) explosion without changing the exact MSE
        # acceptance rule.
        refined_children: list[_SearchState] = []
        for child in screened:
            if refine_positions:
                refined_fit, _, _ = _coordinate_refine(
                    parameters,
                    points,
                    child.fit,
                    statistics=statistics,
                    degree=degree,
                    smoothness_weight=smoothness_weight,
                    control_ridge=control_ridge,
                    interpolate_endpoints=interpolate_endpoints,
                    rcond=rcond,
                    min_gap=min_gap,
                    sweeps=position_sweeps,
                    grid_size=position_grid_size,
                    restarts=position_restarts,
                )
                position_refined_counts.add(child_count)
            else:
                refined_fit = child.fit
            refined = _SearchState(
                child.proposal_indices,
                refined_fit,
                f"refined:{child.source}",
            )
            if refined.mse <= mse_tolerance:
                refined_children.append(refined)
                if _better_feasible_state(refined, incumbent, mse_tolerance):
                    incumbent = refined
        if refined_children:
            states_by_count.setdefault(child_count, []).extend(refined_children)
            states_by_count[child_count] = _deduplicate_best(
                states_by_count[child_count]
            )[:beam_width]

    greedy_mask = _mask_from_indices(proposal_count, greedy_state.proposal_indices)
    final_mask = _mask_from_indices(proposal_count, incumbent.proposal_indices)
    return HybridKnotSearchResult(
        mse_tolerance=float(mse_tolerance),
        proposal_count=proposal_count,
        proposal_knots=proposal_knots.detach().clone(),
        greedy_fit=greedy_state.fit,
        greedy_retained_proposal_indices=(
            greedy_state.proposal_indices.detach().clone()
        ),
        greedy_retained_proposal_mask=greedy_mask,
        final_fit=incumbent.fit,
        retained_proposal_indices=incumbent.proposal_indices.detach().clone(),
        retained_proposal_mask=final_mask,
        final_source=incumbent.source,
        threshold_satisfied=incumbent.mse <= mse_tolerance,
        greedy_threshold_satisfied=greedy_state.mse <= mse_tolerance,
        visited_state_count=statistics.visited_state_count,
        refit_count=statistics.refit_count,
        levels_explored=tuple(levels_explored),
        position_refined_counts=tuple(sorted(position_refined_counts, reverse=True)),
        start_sources=tuple(start_sources),
    )


__all__ = [
    "HybridKnotSearchResult",
    "KnotPositionRefinementResult",
    "hybrid_minimal_knot_search",
    "refine_knot_positions",
]
