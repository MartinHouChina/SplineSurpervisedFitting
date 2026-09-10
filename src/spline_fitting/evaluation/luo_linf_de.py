r"""Auditable Luo--Kang--Yang (2022) :math:`l_{\infty,1}` + DE adaptation.

The original method has two stages: solve a dense fixed-knot convex model,
select locally maximal derivative jumps, and then globally relocate that fixed
number of candidates with differential evolution.  This module keeps those
three defining operations.  The paper chooses the regularization parameter
case by case and reports maximum point error; the repository wrapper instead
selects the largest feasible regularization parameter for a supplied MSE
budget and always reports a common endpoint-constrained standard B-spline
refit.  Those comparison adaptations are exposed in the returned diagnostics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from ..data.synthetic import bspline_basis_matrix
from .bspline_inference import BSplineLeastSquaresFit, refit_bspline_control_points
from .knot_diagnostics import build_open_knot_vector
from .sparse_knot_paper import _pth_derivative_jump_matrix


@dataclass(frozen=True)
class LuoLinfDEResult:
    """Result of the two-stage numerical adaptation."""

    degree: int
    mse_tolerance: float
    initial_internal_knots: torch.Tensor
    dense_initial_fit_mse: float
    dense_initial_threshold_satisfied: bool
    selected_regularization: float
    sparse_fit_mse: float
    jump_values: torch.Tensor
    candidate_indices: torch.Tensor
    candidate_knots: torch.Tensor
    candidate_refit_mse: float
    optimized_knots: torch.Tensor
    final_fit: BSplineLeastSquaresFit
    admm_iterations: int
    regularization_trials: int
    de_population: int
    de_iterations: int
    de_evaluations: int
    de_initial_max_error: float
    de_final_max_error: float
    elapsed_seconds: float

    @property
    def knots(self) -> torch.Tensor:
        return self.final_fit.internal_knots


@dataclass(frozen=True)
class _LinfState:
    controls: torch.Tensor
    reconstructed: torch.Tensor
    jumps: torch.Tensor
    mse: float
    iterations: int


def _prox_linf_rows(values: torch.Tensor, threshold: float) -> torch.Tensor:
    """Row-wise prox of ``threshold * ||.||_inf`` via Moreau decomposition."""

    if threshold <= 0.0 or values.numel() == 0:
        return values
    absolute = values.abs()
    l1 = absolute.sum(dim=-1, keepdim=True)
    inside = l1 <= threshold
    ordered, _ = torch.sort(absolute, dim=-1, descending=True)
    cumulative = torch.cumsum(ordered, dim=-1)
    ranks = torch.arange(
        1,
        values.shape[-1] + 1,
        device=values.device,
        dtype=values.dtype,
    )
    theta_all = (cumulative - threshold) / ranks
    active = ordered > theta_all
    rho = active.sum(dim=-1, keepdim=True).clamp_min(1)
    theta = torch.gather(theta_all, -1, rho - 1).clamp_min(0.0)
    projection = values.sign() * (absolute - theta).clamp_min(0.0)
    projection = torch.where(inside, values, projection)
    return values - projection


def _least_squares_state(
    basis: torch.Tensor,
    points: torch.Tensor,
    jump_matrix: torch.Tensor,
) -> _LinfState:
    kwargs: dict[str, object] = {"driver": "gelsd"} if basis.device.type == "cpu" else {}
    controls = torch.linalg.lstsq(basis, points, **kwargs).solution
    reconstructed = basis @ controls
    mse = float((reconstructed - points).square().sum(dim=-1).mean())
    return _LinfState(controls, reconstructed, jump_matrix @ controls, mse, 0)


def _solve_linf1_admm(
    basis: torch.Tensor,
    points: torch.Tensor,
    jump_matrix: torch.Tensor,
    regularization: float,
    *,
    rho: float,
    max_iterations: int,
    tolerance: float,
) -> _LinfState:
    """Solve ``||A C-P||_F + lambda sum_i ||(D C)_i||_inf`` by ADMM."""

    if regularization <= 0.0:
        return _least_squares_state(basis, points, jump_matrix)
    controls_count = basis.shape[1]
    identity = torch.eye(controls_count, dtype=basis.dtype, device=basis.device)
    # Scale D without changing the physical jumps.  This substantially improves
    # conditioning for the p-th derivative operator on a dense uniform grid.
    operator_scale = float(
        (jump_matrix.norm() / math.sqrt(max(1, jump_matrix.shape[0]))).clamp_min(1e-12)
    )
    scaled_jump = jump_matrix / operator_scale
    system = basis.T @ basis + scaled_jump.T @ scaled_jump + 1e-12 * identity
    factor = torch.linalg.cholesky(system)

    residual = torch.zeros_like(points)
    jumps = torch.zeros(
        (jump_matrix.shape[0], points.shape[1]),
        dtype=points.dtype,
        device=points.device,
    )
    residual_dual = torch.zeros_like(points)
    jump_dual = torch.zeros_like(jumps)
    controls = torch.zeros(
        (controls_count, points.shape[1]), dtype=points.dtype, device=points.device
    )
    completed = max_iterations
    for iteration in range(1, max_iterations + 1):
        rhs = basis.T @ (points + residual - residual_dual)
        rhs = rhs + scaled_jump.T @ (jumps - jump_dual)
        controls = torch.cholesky_solve(rhs, factor)

        raw_residual = basis @ controls - points + residual_dual
        residual_norm = raw_residual.norm()
        residual_scale = (1.0 - 1.0 / (rho * residual_norm.clamp_min(1e-30))).clamp_min(0.0)
        previous_residual = residual
        residual = residual_scale * raw_residual

        raw_jumps = scaled_jump @ controls + jump_dual
        previous_jumps = jumps
        # lambda ||D C||_{inf,1} = lambda*scale ||D_scaled C||_{inf,1}.
        jumps = _prox_linf_rows(
            raw_jumps, regularization * operator_scale / rho
        )

        residual_primal = basis @ controls - points - residual
        jump_primal = scaled_jump @ controls - jumps
        residual_dual = residual_dual + residual_primal
        jump_dual = jump_dual + jump_primal
        primal = torch.sqrt(residual_primal.square().sum() + jump_primal.square().sum())
        dual = torch.sqrt(
            (basis.T @ (residual - previous_residual)).square().sum()
            + (scaled_jump.T @ (jumps - previous_jumps)).square().sum()
        ) * rho
        scale = max(
            1.0,
            float((basis @ controls - points).norm()),
            float(residual.norm()),
            float((scaled_jump @ controls).norm()),
            float(jumps.norm()),
        )
        if float(primal) <= tolerance * scale and float(dual) <= tolerance * scale:
            completed = iteration
            break

    reconstructed = basis @ controls
    mse = float((reconstructed - points).square().sum(dim=-1).mean())
    return _LinfState(
        controls, reconstructed, jump_matrix @ controls, mse, completed
    )


def _select_regularization(
    basis: torch.Tensor,
    points: torch.Tensor,
    jump_matrix: torch.Tensor,
    mse_tolerance: float,
    *,
    rho: float,
    max_iterations: int,
    admm_tolerance: float,
    bisection_iterations: int,
) -> tuple[_LinfState, float, int, int, float]:
    dense = _least_squares_state(basis, points, jump_matrix)
    if dense.mse > mse_tolerance + 1e-12:
        return dense, 0.0, 1, 0, dense.mse
    scale = float((basis.T @ points).norm() / jump_matrix.norm().clamp_min(1e-12))
    current = max(scale * 1e-4, 1e-12)
    lower = 0.0
    best = dense
    upper: float | None = None
    trials = 1
    iterations = 0
    for _ in range(20):
        state = _solve_linf1_admm(
            basis, points, jump_matrix, current,
            rho=rho, max_iterations=max_iterations, tolerance=admm_tolerance,
        )
        trials += 1
        iterations += state.iterations
        if state.mse <= mse_tolerance + 1e-12:
            lower, best = current, state
            current *= 2.0
        else:
            upper = current
            break
    if upper is None:
        return best, lower, trials, iterations, dense.mse
    for _ in range(bisection_iterations):
        current = 0.5 * (lower + upper)
        state = _solve_linf1_admm(
            basis, points, jump_matrix, current,
            rho=rho, max_iterations=max_iterations, tolerance=admm_tolerance,
        )
        trials += 1
        iterations += state.iterations
        if state.mse <= mse_tolerance + 1e-12:
            lower, best = current, state
        else:
            upper = current
    return best, lower, trials, iterations, dense.mse


def _local_maximum_candidates(
    knots: torch.Tensor,
    jumps: torch.Tensor,
    eta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = jumps.abs().amax(dim=-1) if jumps.numel() else knots.new_empty(0)
    if values.numel() == 0 or float(values.max()) <= 0.0:
        indices = torch.empty(0, dtype=torch.long, device=knots.device)
        return indices, knots.new_empty(0), values
    if values.numel() < 3:
        local = torch.arange(values.numel(), device=knots.device)
    else:
        # Luo--Kang--Yang Algorithm 3.1 loops over i=2,...,n-p-1
        # (one-based notation), so the first and last jump entries are
        # deliberately excluded from the length-three sliding window.
        middle = torch.arange(1, values.numel() - 1, device=knots.device)
        keep = (values[middle] >= values[middle - 1]) & (
            values[middle] >= values[middle + 1]
        )
        local = middle[keep]
    cutoff = eta * float(values.max())
    indices = local[values[local] > cutoff]
    if indices.numel() == 0:
        indices = values.argmax().reshape(1)
    return indices, knots[indices], values


def _ordered_with_gap(values: torch.Tensor, min_gap: float) -> torch.Tensor:
    if values.numel() == 0:
        return values
    ordered = values.clamp(min=min_gap, max=1.0 - min_gap).sort().values
    count = ordered.numel()
    if min_gap * (count + 1) >= 1.0:
        raise ValueError("DE min_gap is too large for the selected knot count")
    # Alternating forward/backward projections make the box and gap constraint
    # deterministic without changing the DE representation dimension.
    for index in range(1, count):
        ordered[index] = torch.maximum(
            ordered[index], ordered[index - 1] + min_gap
        )
    ordered[-1] = torch.minimum(ordered[-1], ordered.new_tensor(1.0 - min_gap))
    for index in range(count - 2, -1, -1):
        ordered[index] = torch.minimum(
            ordered[index], ordered[index + 1] - min_gap
        )
    return ordered


def _fit_and_max_error(
    parameters: torch.Tensor,
    points: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
) -> tuple[BSplineLeastSquaresFit, float]:
    fit = refit_bspline_control_points(
        parameters,
        points,
        knots,
        degree=degree,
        smoothness_weight=0.0,
        control_ridge=0.0,
        interpolate_endpoints=True,
    )
    maximum = float((fit.reconstructed_points - points).norm(dim=-1).max())
    return fit, maximum


def _differential_evolution(
    parameters: torch.Tensor,
    points: torch.Tensor,
    candidates: torch.Tensor,
    *,
    degree: int,
    population_size: int,
    iterations: int,
    differential_weight: float,
    crossover_probability: float,
    min_gap: float,
    seed: int,
) -> tuple[torch.Tensor, float, float, int, float]:
    if candidates.numel() == 0 or iterations == 0:
        fit, error = _fit_and_max_error(parameters, points, candidates, degree)
        return candidates, error, error, 1, float(fit.fit_mse)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = int(candidates.numel())
    population = [candidates.detach().clone()]
    for _ in range(1, population_size):
        raw = torch.rand(count, generator=generator, dtype=candidates.dtype)
        population.append(_ordered_with_gap(raw, min_gap))
    population_tensor = torch.stack(population)
    errors = []
    candidate_refit_mse: float | None = None
    for individual in population_tensor:
        fit, error = _fit_and_max_error(parameters, points, individual, degree)
        if candidate_refit_mse is None:
            candidate_refit_mse = float(fit.fit_mse)
        errors.append(error)
    error_tensor = torch.tensor(errors, dtype=parameters.dtype)
    evaluations = population_size
    initial_error = float(error_tensor[0])

    for _ in range(iterations):
        best = population_tensor[int(error_tensor.argmin())]
        for index in range(population_size):
            pool = [item for item in range(population_size) if item != index]
            permutation = torch.randperm(
                len(pool), generator=generator
            )[:4].tolist()
            r1, r2, r3, r4 = (pool[item] for item in permutation)
            mutant = best + differential_weight * (
                population_tensor[r1] + population_tensor[r2]
                - population_tensor[r3] - population_tensor[r4]
            )
            mutant = _ordered_with_gap(mutant, min_gap)
            crossover = torch.rand(count, generator=generator) <= crossover_probability
            crossover[int(torch.randint(count, (1,), generator=generator))] = True
            trial = torch.where(crossover, mutant, population_tensor[index])
            trial = _ordered_with_gap(trial, min_gap)
            _, trial_error = _fit_and_max_error(parameters, points, trial, degree)
            evaluations += 1
            if trial_error <= float(error_tensor[index]):
                population_tensor[index] = trial
                error_tensor[index] = trial_error
    best_index = int(error_tensor.argmin())
    return (
        population_tensor[best_index].detach().clone(),
        initial_error,
        float(error_tensor[best_index]),
        evaluations,
        float(candidate_refit_mse),
    )


@torch.no_grad()
def fit_luo_linf_de(
    parameters: torch.Tensor,
    points: torch.Tensor,
    *,
    degree: int = 3,
    initial_internal_knot_count: int = 40,
    mse_tolerance: float,
    eta: float = 0.5,
    admm_rho: float = 1.0,
    admm_max_iterations: int = 200,
    admm_tolerance: float = 1e-5,
    lambda_bisection_iterations: int = 6,
    de_population: int = 10,
    de_iterations: int = 30,
    de_weight: float = 0.5,
    de_crossover: float = 0.9,
    min_gap: float = 1e-5,
    seed: int = 2022,
) -> LuoLinfDEResult:
    """Run the Luo et al. two-stage method with disclosed comparison choices."""

    if parameters.ndim != 1 or points.ndim != 2 or parameters.shape[0] != points.shape[0]:
        raise ValueError("parameters/points must have shapes [M] and [M,D]")
    if parameters.dtype != torch.float64 or points.dtype != torch.float64:
        raise ValueError("Luo adaptation requires float64 tensors")
    if parameters.device.type != "cpu" or points.device.type != "cpu":
        raise ValueError("Luo adaptation currently requires CPU tensors")
    if degree < 1 or initial_internal_knot_count < 0:
        raise ValueError("invalid degree or initial knot count")
    if not math.isfinite(mse_tolerance) or mse_tolerance < 0.0:
        raise ValueError("mse_tolerance must be finite and non-negative")
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0,1]")
    if de_population < 5:
        raise ValueError("de_population must be at least 5 for best/2 mutation")
    if de_iterations < 0 or admm_max_iterations < 1 or lambda_bisection_iterations < 1:
        raise ValueError("iteration counts are invalid")
    if not 0.0 <= de_crossover <= 1.0 or not 0.0 < de_weight <= 2.0:
        raise ValueError("invalid DE weight or crossover probability")
    if min_gap <= 0.0:
        raise ValueError("min_gap must be positive")
    if torch.any(parameters[1:] <= parameters[:-1]):
        raise ValueError("parameters must be strictly increasing")

    started = time.perf_counter()
    if initial_internal_knot_count:
        initial_knots = torch.linspace(
            0.0, 1.0, initial_internal_knot_count + 2, dtype=torch.float64
        )[1:-1]
    else:
        initial_knots = torch.empty(0, dtype=torch.float64)
    knot_vector = build_open_knot_vector(initial_knots, degree)
    controls_count = initial_internal_knot_count + degree + 1
    basis = bspline_basis_matrix(parameters, knot_vector, degree, controls_count)
    jump_matrix = _pth_derivative_jump_matrix(
        knot_vector, degree, controls_count
    )
    sparse, regularization, trials, admm_iterations, dense_initial_mse = (
        _select_regularization(
            basis,
            points,
            jump_matrix,
            mse_tolerance,
            rho=admm_rho,
            max_iterations=admm_max_iterations,
            admm_tolerance=admm_tolerance,
            bisection_iterations=lambda_bisection_iterations,
        )
    )
    candidate_indices, candidate_knots, jump_values = _local_maximum_candidates(
        initial_knots, sparse.jumps, eta
    )
    optimized, initial_me, final_me, de_evaluations, candidate_refit_mse = (
        _differential_evolution(
            parameters,
            points,
            candidate_knots,
            degree=degree,
            population_size=de_population,
            iterations=de_iterations,
            differential_weight=de_weight,
            crossover_probability=de_crossover,
            min_gap=min_gap,
            seed=seed,
        )
    )
    final_fit = refit_bspline_control_points(
        parameters,
        points,
        optimized,
        degree=degree,
        smoothness_weight=0.0,
        control_ridge=0.0,
        interpolate_endpoints=True,
    )
    return LuoLinfDEResult(
        degree=degree,
        mse_tolerance=float(mse_tolerance),
        initial_internal_knots=initial_knots,
        dense_initial_fit_mse=dense_initial_mse,
        dense_initial_threshold_satisfied=(
            dense_initial_mse <= mse_tolerance + 1e-12
        ),
        selected_regularization=regularization,
        sparse_fit_mse=sparse.mse,
        jump_values=jump_values.detach().clone(),
        candidate_indices=candidate_indices.detach().clone(),
        candidate_knots=candidate_knots.detach().clone(),
        candidate_refit_mse=candidate_refit_mse,
        optimized_knots=optimized.detach().clone(),
        final_fit=final_fit,
        admm_iterations=admm_iterations,
        regularization_trials=trials,
        de_population=de_population,
        de_iterations=de_iterations,
        de_evaluations=de_evaluations,
        de_initial_max_error=initial_me,
        de_final_max_error=final_me,
        elapsed_seconds=time.perf_counter() - started,
    )
