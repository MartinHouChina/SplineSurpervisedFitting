"""A small, auditable adaptation of Kang et al. (2015) sparse knot fitting.

The paper solves the scalar constrained problem

``min ||J c^(p)||_1  subject to mean_i |c(s_i) - P_i|^2 <= epsilon``

on a dense, fixed uniform knot vector (their equation (12)).  This module
extends it to vector-valued curves by replacing each scalar absolute jump by
the Euclidean norm of the corresponding vector jump.  The resulting group-L1
problem is convex.  It is solved here by a deliberately simple ADMM routine;
the constraint is enforced approximately by a monotone penalty/bisection
search, rather than by CVX as in the paper.

The second stage is an auditable, Algorithms-4-and-5-inspired approximation.
Nearby active uniform knots are clustered by their initial spacing, every
cluster is merged to one knot, and that knot is positioned with a bounded
midpoint-bisection least-squares search.  It intentionally supports only one
final knot per active cluster; it is not a complete implementation of the
paper's repeated-knot heuristic.
The returned curve is always an exact, unregularized standard B-spline refit.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from ..data.synthetic import bspline_basis_matrix
from .bspline_inference import BSplineLeastSquaresFit, refit_bspline_control_points
from .knot_diagnostics import build_open_knot_vector


_ADAPTATION_LABEL = (
    "Vector-valued group-L1 ADMM adaptation of Kang et al. (2015) equation "
    "(12), followed by Algorithms-4-and-5-inspired active-cluster "
    "midpoint-bisection merging; not the paper's scalar CVX implementation."
)
_RELOCATION_LABEL = (
    "Algorithms 4-5-inspired: classify adjacent active uniform knots by "
    "spacing, merge each cluster to one knot, then use bounded midpoint "
    "bisection with exact least-squares comparisons."
)
_REPAIR_LABEL = (
    "Post-relocation feasibility repair (not in Kang et al.): auditable greedy "
    "addition of active, then dense-initial, candidate knots until the exact "
    "standard B-spline refit satisfies the requested tolerance."
)
_NO_REPAIR_LABEL = (
    "Disabled: report the Algorithms-4-and-5-inspired relocated result without "
    "the repository-specific feasibility add-back."
)


@dataclass(frozen=True)
class SparseKnotPaperResult:
    """Diagnostics and final refit for the sparse-knot paper adaptation."""

    degree: int
    method: str
    relocation_method: str
    data_tolerance: float
    jump_threshold: float
    relative_jump_threshold: float | None
    effective_jump_threshold: float
    initial_internal_knots: torch.Tensor
    initial_spacing: float
    dense_initial_fit_mse: torch.Tensor
    dense_initial_threshold_satisfied: bool
    sparse_control_points: torch.Tensor
    sparse_reconstructed_points: torch.Tensor
    sparse_fit_mse: torch.Tensor
    jump_vectors: torch.Tensor
    jump_norms: torch.Tensor
    active_mask: torch.Tensor
    active_internal_knots: torch.Tensor
    cluster_sizes: tuple[int, ...]
    relocated_internal_knots: torch.Tensor
    final_fit: BSplineLeastSquaresFit
    sparse_iterations: int
    lambda_bisection_iterations: int
    local_refit_count: int
    final_refit_count: int
    repair_method: str
    repair_used: bool
    repair_added_knots: torch.Tensor
    repair_refit_count: int
    sparse_solver_seconds: float
    relocation_seconds: float
    repair_seconds: float
    elapsed_seconds: float

    @property
    def knots(self) -> torch.Tensor:
        """Final sorted internal knots, after local merging and relocation."""
        return self.final_fit.internal_knots

    @property
    def fit(self) -> BSplineLeastSquaresFit:
        """Alias for the final exact standard B-spline refit."""
        return self.final_fit

    @property
    def fit_mse(self) -> torch.Tensor:
        return self.final_fit.fit_mse

    @property
    def active_count(self) -> int:
        return int(self.active_mask.sum().item())

    @property
    def threshold_satisfied(self) -> bool:
        """Whether the sparse stage satisfies equation (12)'s MSE bound."""
        return float(self.sparse_fit_mse) <= self.data_tolerance + 1e-12

    @property
    def final_threshold_satisfied(self) -> bool:
        """Whether the deployed exact refit satisfies the requested MSE bound."""
        return float(self.final_fit.fit_mse) <= self.data_tolerance + 1e-12


@dataclass(frozen=True)
class _SparseState:
    control_points: torch.Tensor
    reconstructed_points: torch.Tensor
    jump_vectors: torch.Tensor
    fit_mse: torch.Tensor
    iterations: int


def _validate_inputs(
    parameters: torch.Tensor,
    points: torch.Tensor,
    degree: int,
    initial_internal_knot_count: int | None,
    data_tolerance: float | None,
    jump_threshold: float,
    relative_jump_threshold: float | None,
    admm_rho: float,
    admm_max_iterations: int,
    admm_tolerance: float,
    lambda_bisection_iterations: int,
    relocation_tolerance: float | None,
    relocation_max_iterations: int,
    cluster_gap_factor: float,
) -> None:
    if parameters.ndim != 1 or points.ndim != 2:
        raise ValueError("parameters must have shape [M] and points [M, D]")
    if parameters.numel() < degree + 1:
        raise ValueError("at least degree + 1 samples are required")
    if parameters.shape[0] != points.shape[0]:
        raise ValueError("parameters and points must share the point count")
    if degree < 1:
        raise ValueError("degree must be positive")
    if not parameters.is_floating_point() or not points.is_floating_point():
        raise ValueError("parameters and points must be floating-point tensors")
    if parameters.device != points.device or parameters.dtype != points.dtype:
        raise ValueError("parameters and points must share device and dtype")
    if not torch.isfinite(parameters).all() or not torch.isfinite(points).all():
        raise ValueError("parameters and points must be finite")
    if torch.any((parameters < 0.0) | (parameters > 1.0)):
        raise ValueError("parameters must lie in [0, 1]")
    if parameters.numel() > 1 and torch.any(parameters[1:] < parameters[:-1]):
        raise ValueError("parameters must be non-decreasing")
    if initial_internal_knot_count is not None and (
        isinstance(initial_internal_knot_count, bool)
        or not isinstance(initial_internal_knot_count, int)
        or initial_internal_knot_count < 0
    ):
        raise ValueError("initial_internal_knot_count must be a non-negative integer")
    if data_tolerance is not None and (
        not math.isfinite(data_tolerance) or data_tolerance < 0.0
    ):
        raise ValueError("data_tolerance must be finite and non-negative")
    if not math.isfinite(jump_threshold) or jump_threshold < 0.0:
        raise ValueError("jump_threshold must be finite and non-negative")
    if relative_jump_threshold is not None and (
        not math.isfinite(relative_jump_threshold)
        or relative_jump_threshold < 0.0
    ):
        raise ValueError("relative_jump_threshold must be finite and non-negative")
    for name, value in (
        ("admm_rho", admm_rho),
        ("admm_tolerance", admm_tolerance),
        ("cluster_gap_factor", cluster_gap_factor),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if relocation_tolerance is not None and (
        not math.isfinite(relocation_tolerance) or relocation_tolerance <= 0.0
    ):
        raise ValueError("relocation_tolerance must be finite and positive")
    for name, value in (
        ("admm_max_iterations", admm_max_iterations),
        ("lambda_bisection_iterations", lambda_bisection_iterations),
        ("relocation_max_iterations", relocation_max_iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def _uniform_internal_knots(
    parameters: torch.Tensor,
    count: int,
) -> torch.Tensor:
    if count == 0:
        return parameters.new_empty(0)
    # Equation (12) assumes a fixed dense initial vector.  The implementation
    # works on [0, 1], the parameter domain enforced by the evaluation API.
    return torch.linspace(
        0.0,
        1.0,
        count + 2,
        device=parameters.device,
        dtype=parameters.dtype,
    )[1:-1]


def _pth_derivative_jump_matrix(
    knot_vector: torch.Tensor,
    degree: int,
    num_control_points: int,
) -> torch.Tensor:
    """Return the linear map from B-spline controls to p-th derivative jumps."""
    transform = torch.eye(
        num_control_points,
        device=knot_vector.device,
        dtype=knot_vector.dtype,
    )
    knots = knot_vector
    current_degree = degree
    for _ in range(degree):
        count = transform.shape[0] - 1
        denominator = (
            knots[current_degree + 1 : current_degree + 1 + count]
            - knots[1 : 1 + count]
        )
        scale = current_degree / denominator.clamp_min(torch.finfo(knots.dtype).eps)
        transform = scale.unsqueeze(-1) * (transform[1:] - transform[:-1])
        knots = knots[1:-1]
        current_degree -= 1
    return transform[1:] - transform[:-1]


def _fit_mse(reconstructed: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return (reconstructed - points).pow(2).sum(dim=-1).mean()


def _least_squares_state(
    basis: torch.Tensor,
    points: torch.Tensor,
    jump_matrix: torch.Tensor,
) -> _SparseState:
    lstsq_kwargs: dict[str, object] = {}
    if basis.device.type == "cpu":
        lstsq_kwargs["driver"] = "gelsd"
    controls = torch.linalg.lstsq(basis, points, **lstsq_kwargs).solution
    reconstructed = basis @ controls
    return _SparseState(
        control_points=controls,
        reconstructed_points=reconstructed,
        jump_vectors=jump_matrix @ controls,
        fit_mse=_fit_mse(reconstructed, points),
        iterations=0,
    )


def _group_shrink(values: torch.Tensor, threshold: float) -> torch.Tensor:
    norms = values.norm(dim=-1, keepdim=True)
    scale = (1.0 - threshold / norms.clamp_min(torch.finfo(values.dtype).eps)).clamp_min(
        0.0
    )
    return scale * values


def _solve_group_l1_admm(
    basis: torch.Tensor,
    points: torch.Tensor,
    jump_matrix: torch.Tensor,
    penalty: float,
    *,
    rho: float,
    max_iterations: int,
    tolerance: float,
) -> _SparseState:
    """Solve a penalized group-L1 jump fit with scaled ADMM."""
    if penalty == 0.0:
        return _least_squares_state(basis, points, jump_matrix)

    # Dense uniform knots make p-th derivative jumps scale as h^{-p}.  Scale
    # the equality constraint globally for ADMM conditioning, while scaling
    # the shrinkage parameter by exactly the same amount.  Thus the objective
    # remains ``penalty * sum ||jump_matrix @ controls||_2``.
    operator_scale = jump_matrix.norm() / math.sqrt(float(jump_matrix.shape[0]))
    scaled_jump_matrix = jump_matrix / operator_scale.clamp_min(1e-12)
    controls_count = basis.shape[1]
    identity = torch.eye(controls_count, device=basis.device, dtype=basis.dtype)
    system = (
        basis.T @ basis
        + rho * (scaled_jump_matrix.T @ scaled_jump_matrix)
        + 1e-12 * identity
    )
    factor = torch.linalg.cholesky(system)
    right = basis.T @ points
    jumps = torch.zeros(
        (jump_matrix.shape[0], points.shape[1]), device=points.device, dtype=points.dtype
    )
    dual = torch.zeros_like(jumps)
    controls = torch.zeros(
        (controls_count, points.shape[1]), device=points.device, dtype=points.dtype
    )
    iterations = max_iterations
    for iteration in range(1, max_iterations + 1):
        controls = torch.cholesky_solve(
            right + rho * (scaled_jump_matrix.T @ (jumps - dual)), factor
        )
        projected = scaled_jump_matrix @ controls + dual
        previous_jumps = jumps
        jumps = _group_shrink(projected, penalty * float(operator_scale) / rho)
        primal = scaled_jump_matrix @ controls - jumps
        dual = dual + primal
        primal_norm = primal.norm()
        dual_norm = (rho * scaled_jump_matrix.T @ (jumps - previous_jumps)).norm()
        scale = max(
            1.0,
            float((scaled_jump_matrix @ controls).norm()),
            float(jumps.norm()),
        )
        if float(primal_norm) <= tolerance * scale and float(dual_norm) <= tolerance * scale:
            iterations = iteration
            break
    reconstructed = basis @ controls
    return _SparseState(
        control_points=controls,
        reconstructed_points=reconstructed,
        jump_vectors=jump_matrix @ controls,
        fit_mse=_fit_mse(reconstructed, points),
        iterations=iterations,
    )


def _select_constrained_sparse_state(
    basis: torch.Tensor,
    points: torch.Tensor,
    jump_matrix: torch.Tensor,
    tolerance: float,
    *,
    rho: float,
    max_iterations: int,
    admm_tolerance: float,
    bisection_iterations: int,
) -> tuple[_SparseState, int]:
    """Approximate the paper's hard MSE constraint by penalty bisection."""
    least_squares = _least_squares_state(basis, points, jump_matrix)
    if float(least_squares.fit_mse) > tolerance + 1e-12:
        raise ValueError(
            "data_tolerance is below the dense initial-knot least-squares MSE"
        )

    # The null space of the p-th derivative jump map is a degree-p polynomial.
    # If it is already feasible, it is the exact sparse optimum (zero jumps).
    _, _, right_singular = torch.linalg.svd(jump_matrix, full_matrices=True)
    null_basis = right_singular[-(basis.shape[1] - jump_matrix.shape[0]) :].T
    polynomial_design = basis @ null_basis
    polynomial_lstsq_kwargs: dict[str, object] = {}
    if polynomial_design.device.type == "cpu":
        polynomial_lstsq_kwargs["driver"] = "gelsd"
    polynomial_controls_reduced = torch.linalg.lstsq(
        polynomial_design,
        points,
        **polynomial_lstsq_kwargs,
    ).solution
    polynomial_controls = null_basis @ polynomial_controls_reduced
    polynomial_reconstructed = basis @ polynomial_controls
    polynomial_mse = _fit_mse(polynomial_reconstructed, points)
    if float(polynomial_mse) <= tolerance + 1e-12:
        return (
            _SparseState(
                control_points=polynomial_controls,
                reconstructed_points=polynomial_reconstructed,
                jump_vectors=jump_matrix @ polynomial_controls,
                fit_mse=polynomial_mse,
                iterations=0,
            ),
            0,
        )

    # Start in units that compensate for the large h^{-p} jump operator on a
    # dense grid, then expand until the MSE constraint becomes infeasible.
    start = float((basis.T @ points).norm() / jump_matrix.norm().clamp_min(1e-12))
    penalty = max(start * 1e-4, 1e-12)
    lower_penalty = 0.0
    best = least_squares
    total_iterations = 0
    upper_penalty: float | None = None
    for _ in range(20):
        state = _solve_group_l1_admm(
            basis,
            points,
            jump_matrix,
            penalty,
            rho=rho,
            max_iterations=max_iterations,
            tolerance=admm_tolerance,
        )
        total_iterations += state.iterations
        if float(state.fit_mse) <= tolerance + 1e-12:
            lower_penalty = penalty
            best = state
            penalty *= 2.0
        else:
            upper_penalty = penalty
            break
    if upper_penalty is None:
        return best, total_iterations

    for _ in range(bisection_iterations):
        middle = 0.5 * (lower_penalty + upper_penalty)
        state = _solve_group_l1_admm(
            basis,
            points,
            jump_matrix,
            middle,
            rho=rho,
            max_iterations=max_iterations,
            tolerance=admm_tolerance,
        )
        total_iterations += state.iterations
        if float(state.fit_mse) <= tolerance + 1e-12:
            lower_penalty = middle
            best = state
        else:
            upper_penalty = middle
    return best, total_iterations


def _cluster_active_knots(
    active_knots: torch.Tensor,
    spacing: float,
    cluster_gap_factor: float,
) -> list[torch.Tensor]:
    """Classify active knots as Algorithm 5 does, using uniform knot spacing."""
    if active_knots.numel() == 0:
        return []
    split = torch.nonzero(
        active_knots[1:] - active_knots[:-1] > spacing * cluster_gap_factor,
        as_tuple=False,
    ).flatten()
    clusters: list[torch.Tensor] = []
    start = 0
    for stop in split.tolist():
        clusters.append(active_knots[start : stop + 1])
        start = stop + 1
    clusters.append(active_knots[start:])
    return clusters


def _refit_mse(
    parameters: torch.Tensor,
    points: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    return refit_bspline_control_points(
        parameters,
        points,
        knots,
        degree=degree,
        smoothness_weight=0.0,
        control_ridge=0.0,
        interpolate_endpoints=False,
    ).fit_mse


def _relocate_clusters(
    parameters: torch.Tensor,
    points: torch.Tensor,
    clusters: list[torch.Tensor],
    degree: int,
    *,
    initial_spacing: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[torch.Tensor, int]:
    """Merge active clusters using a bounded, paper-inspired bisection search."""
    if not clusters:
        return parameters.new_empty(0), 0
    representatives = torch.stack([cluster.mean() for cluster in clusters])
    refits = 0
    for index, cluster in enumerate(clusters):
        if cluster.numel() == 1:
            left = float(cluster[0]) - initial_spacing
            right = float(cluster[0]) + initial_spacing
        else:
            left = float(cluster[0])
            right = float(cluster[-1])
        numerical_gap = max(torch.finfo(parameters.dtype).eps * 32.0, 1e-12)
        left = max(left, numerical_gap)
        right = min(right, 1.0 - numerical_gap)
        if index:
            left = max(left, float(representatives[index - 1]) + numerical_gap)
        if index + 1 < representatives.numel():
            right = min(right, float(representatives[index + 1]) - numerical_gap)
        if right <= left:
            continue
        # This is a transparent interval-halving heuristic inspired by the
        # paper's local error comparisons; it is not an equivalent rewrite of
        # Algorithms 1/4.
        for _ in range(max_iterations):
            if right - left <= tolerance:
                break
            middle = 0.5 * (left + right)
            left_trial = 0.5 * (left + middle)
            right_trial = 0.5 * (middle + right)
            trial_left = representatives.clone()
            trial_right = representatives.clone()
            trial_left[index] = left_trial
            trial_right[index] = right_trial
            left_error = _refit_mse(parameters, points, trial_left, degree)
            right_error = _refit_mse(parameters, points, trial_right, degree)
            refits += 2
            if float(right_error) < float(left_error):
                left = middle
            else:
                right = middle
        representatives[index] = 0.5 * (left + right)
    return torch.sort(representatives).values, refits


def _repair_refit_feasibility(
    parameters: torch.Tensor,
    points: torch.Tensor,
    relocated_knots: torch.Tensor,
    prioritized_candidates: torch.Tensor,
    dense_candidates: torch.Tensor,
    degree: int,
    data_tolerance: float,
) -> tuple[BSplineLeastSquaresFit, bool, torch.Tensor, int]:
    """Restore the MSE bound after merging, without claiming paper provenance.

    The local Algorithms 1/4 merge changes the spline space, so its ordinary
    least-squares refit can exceed the sparse stage's equation-(12) tolerance.
    This repair is intentionally simple and auditable: greedily add the
    candidate whose *exact standard B-spline* refit gives the lowest MSE,
    considering selected active knots before falling back to the complete dense
    initial vector.  Since retaining every initial candidate reproduces the
    dense feasible space, it restores feasibility whenever that initial fit is
    feasible up to solver precision.
    """
    current_knots = torch.sort(relocated_knots).values
    current_fit = refit_bspline_control_points(
        parameters,
        points,
        current_knots,
        degree=degree,
        smoothness_weight=0.0,
        control_ridge=0.0,
        interpolate_endpoints=False,
    )
    refit_count = 1
    if float(current_fit.fit_mse) <= data_tolerance + 1e-12:
        return current_fit, False, current_knots.new_empty(0), refit_count

    added: list[torch.Tensor] = []
    candidate_groups = (prioritized_candidates, dense_candidates)
    for candidates in candidate_groups:
        candidate_pool = torch.unique_consecutive(torch.sort(candidates).values)
        while candidate_pool.numel() > 0:
            # A relocated knot may be close to, but not equal to, a grid
            # candidate. Keep that grid candidate: together they are a valid
            # refinement and retain the dense initial spline space once all
            # are available.
            available_mask = torch.ones(
                candidate_pool.shape[0],
                device=candidate_pool.device,
                dtype=torch.bool,
            )
            if current_knots.numel() > 0:
                exact_duplicate = candidate_pool.unsqueeze(-1).eq(current_knots).any(
                    dim=-1
                )
                available_mask &= ~exact_duplicate
            available = candidate_pool[available_mask]
            if available.numel() == 0:
                break
            best_fit: BSplineLeastSquaresFit | None = None
            best_knot: torch.Tensor | None = None
            for candidate in available:
                trial_knots = torch.sort(
                    torch.cat([current_knots, candidate.view(1)])
                ).values
                trial_fit = refit_bspline_control_points(
                    parameters,
                    points,
                    trial_knots,
                    degree=degree,
                    smoothness_weight=0.0,
                    control_ridge=0.0,
                    interpolate_endpoints=False,
                )
                refit_count += 1
                if best_fit is None or float(trial_fit.fit_mse) < float(best_fit.fit_mse):
                    best_fit = trial_fit
                    best_knot = candidate
            if best_fit is None or best_knot is None:
                break
            current_knots = torch.sort(
                torch.cat([current_knots, best_knot.view(1)])
            ).values
            current_fit = best_fit
            added.append(best_knot.detach().clone())
            if float(current_fit.fit_mse) <= data_tolerance + 1e-12:
                break
        if float(current_fit.fit_mse) <= data_tolerance + 1e-12:
            break
    if added:
        added_knots = torch.stack(added)
    else:
        added_knots = current_knots.new_empty(0)
    return current_fit, bool(added), added_knots, refit_count


@torch.no_grad()
def fit_sparse_knots_paper(
    parameters: torch.Tensor,
    points: torch.Tensor,
    *,
    degree: int = 3,
    initial_internal_knot_count: int | None = None,
    data_tolerance: float | None = None,
    jump_threshold: float = 1e-7,
    relative_jump_threshold: float | None = None,
    admm_rho: float = 1e4,
    admm_max_iterations: int = 1000,
    admm_tolerance: float = 1e-6,
    lambda_bisection_iterations: int = 10,
    relocation_tolerance: float | None = None,
    relocation_max_iterations: int = 12,
    cluster_gap_factor: float = 1.25,
    feasibility_repair: bool = False,
) -> SparseKnotPaperResult:
    """Fit a vector-valued curve by the sparse-knot paper adaptation.

    ``data_tolerance`` is the paper's epsilon: mean squared Euclidean error
    per sampled point, measured in the sparse first stage.  If omitted, it is
    set just above the dense unregularized fit, so callers who care about an
    absolute approximation guarantee should pass it explicitly.  A candidate
    knot is active exactly when its p-th derivative-jump vector has norm
    strictly greater than ``max(jump_threshold, relative_jump_threshold *
    max_jump)``.  Set ``relative_jump_threshold=None`` (the default) to
    retain the backward-compatible absolute-only criterion.  The original
    paper does not greedily add knots after relocation, so
    ``feasibility_repair`` defaults to ``False``.  Set it to ``True`` only for
    the explicitly labelled repository ablation used by older experiments.
    """
    _validate_inputs(
        parameters,
        points,
        degree,
        initial_internal_knot_count,
        data_tolerance,
        jump_threshold,
        relative_jump_threshold,
        admm_rho,
        admm_max_iterations,
        admm_tolerance,
        lambda_bisection_iterations,
        relocation_tolerance,
        relocation_max_iterations,
        cluster_gap_factor,
    )
    started = time.perf_counter()
    count = (
        max(8, parameters.numel() // 2 - 1)
        if initial_internal_knot_count is None
        else initial_internal_knot_count
    )
    initial_knots = _uniform_internal_knots(parameters, count)
    spacing = 1.0 / (count + 1) if count else 1.0
    knot_vector = build_open_knot_vector(initial_knots, degree)
    controls_count = count + degree + 1
    basis = bspline_basis_matrix(parameters, knot_vector, degree, controls_count)
    jump_matrix = _pth_derivative_jump_matrix(knot_vector, degree, controls_count)
    dense = _least_squares_state(basis, points, jump_matrix)
    tolerance = (
        float(dense.fit_mse) * 1.01 + 1e-10
        if data_tolerance is None
        else float(data_tolerance)
    )
    dense_initial_threshold_satisfied = (
        float(dense.fit_mse) <= tolerance + 1e-12
    )
    sparse_started = time.perf_counter()
    if dense_initial_threshold_satisfied:
        sparse, sparse_iterations = _select_constrained_sparse_state(
            basis,
            points,
            jump_matrix,
            tolerance,
            rho=admm_rho,
            max_iterations=admm_max_iterations,
            admm_tolerance=admm_tolerance,
            bisection_iterations=lambda_bisection_iterations,
        )
    else:
        # The constrained problem is infeasible in the supplied dense spline
        # space.  Keep the best dense least-squares state and continue through
        # the disclosed relocation/repair path so batch comparisons can record
        # a failed threshold result instead of aborting the entire experiment.
        sparse = dense
        sparse_iterations = 0
    sparse_solver_seconds = time.perf_counter() - sparse_started
    jump_norms = sparse.jump_vectors.norm(dim=-1)
    relative_floor = (
        0.0
        if relative_jump_threshold is None or jump_norms.numel() == 0
        else relative_jump_threshold * float(jump_norms.max())
    )
    effective_jump_threshold = max(float(jump_threshold), relative_floor)
    active_mask = jump_norms > effective_jump_threshold
    active_knots = initial_knots[active_mask]
    clusters = _cluster_active_knots(active_knots, spacing, cluster_gap_factor)
    relocation_started = time.perf_counter()
    final_knots, local_refit_count = _relocate_clusters(
        parameters,
        points,
        clusters,
        degree,
        initial_spacing=spacing,
        tolerance=(spacing / 32.0 if relocation_tolerance is None else relocation_tolerance),
        max_iterations=relocation_max_iterations,
    )
    relocation_seconds = time.perf_counter() - relocation_started
    relocated_before_repair = final_knots
    repair_started = time.perf_counter()
    if feasibility_repair:
        final_fit, repair_used, repair_added_knots, repair_refit_count = (
            _repair_refit_feasibility(
                parameters,
                points,
                final_knots,
                active_knots,
                initial_knots,
                degree,
                tolerance,
            )
        )
        repair_method = _REPAIR_LABEL
    else:
        final_fit = refit_bspline_control_points(
            parameters,
            points,
            final_knots,
            degree=degree,
            smoothness_weight=0.0,
            control_ridge=0.0,
            interpolate_endpoints=False,
        )
        repair_used = False
        repair_added_knots = final_knots.new_empty(0)
        repair_refit_count = 1
        repair_method = _NO_REPAIR_LABEL
    repair_seconds = time.perf_counter() - repair_started
    final_knots = final_fit.internal_knots
    return SparseKnotPaperResult(
        degree=degree,
        method=_ADAPTATION_LABEL,
        relocation_method=_RELOCATION_LABEL,
        data_tolerance=tolerance,
        jump_threshold=float(jump_threshold),
        relative_jump_threshold=relative_jump_threshold,
        effective_jump_threshold=effective_jump_threshold,
        initial_internal_knots=initial_knots.detach().clone(),
        initial_spacing=spacing,
        dense_initial_fit_mse=dense.fit_mse.detach().clone(),
        dense_initial_threshold_satisfied=dense_initial_threshold_satisfied,
        sparse_control_points=sparse.control_points.detach().clone(),
        sparse_reconstructed_points=sparse.reconstructed_points.detach().clone(),
        sparse_fit_mse=sparse.fit_mse.detach().clone(),
        jump_vectors=sparse.jump_vectors.detach().clone(),
        jump_norms=jump_norms.detach().clone(),
        active_mask=active_mask.detach().clone(),
        active_internal_knots=active_knots.detach().clone(),
        cluster_sizes=tuple(int(cluster.numel()) for cluster in clusters),
        relocated_internal_knots=relocated_before_repair.detach().clone(),
        final_fit=final_fit,
        sparse_iterations=sparse_iterations,
        lambda_bisection_iterations=lambda_bisection_iterations,
        local_refit_count=local_refit_count,
        final_refit_count=repair_refit_count,
        repair_method=repair_method,
        repair_used=repair_used,
        repair_added_knots=repair_added_knots.detach().clone(),
        repair_refit_count=repair_refit_count,
        sparse_solver_seconds=sparse_solver_seconds,
        relocation_seconds=relocation_seconds,
        repair_seconds=repair_seconds,
        elapsed_seconds=time.perf_counter() - started,
    )
