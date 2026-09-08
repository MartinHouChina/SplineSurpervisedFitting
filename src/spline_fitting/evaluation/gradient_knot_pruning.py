"""A network-free gradient baseline for B-spline knot-count selection.

The baseline deliberately starts from uniformly spaced knots and derives its
parameters solely from the input polyline.  It is therefore useful as an
independent numerical comparison to learned proposal-and-pruning methods.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from ..data.synthetic import bspline_basis_matrix
from .bspline_inference import (
    BSplineLeastSquaresFit,
    refit_bspline_control_points,
    second_difference_matrix,
)
from .knot_diagnostics import build_open_knot_vector


@dataclass(frozen=True)
class GradientKnotDeletionStep:
    """The exact, audited outcome of one greedy deletion round."""

    iteration: int
    knots_before: torch.Tensor
    candidate_fits: tuple[BSplineLeastSquaresFit, ...]
    candidate_updated_knots: tuple[torch.Tensor, ...]
    candidate_location_updates_accepted: tuple[bool, ...]
    selected_index: int | None
    accepted: bool


@dataclass(frozen=True)
class GradientKnotPruningResult:
    """Result from :func:`gradient_knot_pruning_baseline`.

    All stored spline fits and knot values use ``float64``.  ``refit_count``
    counts exact standard-B-spline refits, while ``evaluation_count`` counts
    differentiable variable-projection loss evaluations during location
    optimization.
    """

    mse_tolerance: float
    parameters: torch.Tensor
    initial_fit: BSplineLeastSquaresFit
    initial_relocated_fit: BSplineLeastSquaresFit
    initial_location_update_accepted: bool
    final_fit: BSplineLeastSquaresFit
    steps: tuple[GradientKnotDeletionStep, ...]
    gradient_steps: int
    refit_count: int
    evaluation_count: int
    total_search_time: float

    @property
    def final_internal_knots(self) -> torch.Tensor:
        return self.final_fit.internal_knots

    @property
    def knots(self) -> torch.Tensor:
        """Alias for the final ordered internal knots."""
        return self.final_internal_knots

    @property
    def K(self) -> int:
        """Final number of internal knots."""
        return int(self.final_internal_knots.numel())

    @property
    def deletion_iterations(self) -> int:
        """Number of deletion rounds evaluated (including the final rejection)."""
        return len(self.steps)

    @property
    def accepted_deletions(self) -> int:
        return sum(step.accepted for step in self.steps)

    @property
    def threshold_satisfied(self) -> bool:
        return float(self.final_fit.fit_mse) <= self.mse_tolerance

    @property
    def fit(self) -> BSplineLeastSquaresFit:
        return self.final_fit


def chord_length_parameters(points: torch.Tensor) -> torch.Tensor:
    """Return endpoint-normalized chord-length parameters for ``points``.

    A completely coincident polyline has no meaningful chord direction.  For
    that degenerate but valid input, uniformly spaced parameters are the
    deterministic limiting convention.
    """
    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError("points must have shape [M, D] with M at least 2")
    if not points.is_floating_point() or not torch.isfinite(points).all():
        raise ValueError("points must be finite floating-point values")
    lengths = (points[1:] - points[:-1]).norm(dim=-1)
    total = lengths.sum()
    if float(total) == 0.0:
        return torch.linspace(
            0.0,
            1.0,
            points.shape[0],
            device=points.device,
            dtype=points.dtype,
        )
    parameters = torch.cat([points.new_zeros(1), torch.cumsum(lengths / total, 0)])
    parameters[-1] = 1.0
    return parameters


def _validate_options(
    points: torch.Tensor,
    *,
    max_internal_knots: int,
    mse_tolerance: float,
    min_internal_knots: int,
    min_gap: float,
    optimization_steps: int,
    learning_rate: float,
    optimizer: str,
    degree: int,
    smoothness_weight: float,
    control_ridge: float,
) -> None:
    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError("points must have shape [M, D] with M at least 2")
    if not points.is_floating_point() or not torch.isfinite(points).all():
        raise ValueError("points must be finite floating-point values")
    if isinstance(max_internal_knots, bool) or not isinstance(max_internal_knots, int):
        raise TypeError("max_internal_knots must be an integer")
    if max_internal_knots < 0:
        raise ValueError("max_internal_knots must be non-negative")
    if isinstance(min_internal_knots, bool) or not isinstance(min_internal_knots, int):
        raise TypeError("min_internal_knots must be an integer")
    if not 0 <= min_internal_knots <= max_internal_knots:
        raise ValueError("min_internal_knots must lie in [0, max_internal_knots]")
    if not math.isfinite(mse_tolerance) or mse_tolerance < 0.0:
        raise ValueError("mse_tolerance must be finite and non-negative")
    if not math.isfinite(min_gap) or min_gap < 0.0:
        raise ValueError("min_gap must be finite and non-negative")
    if max_internal_knots and min_gap * (max_internal_knots + 1) >= 1.0:
        raise ValueError("min_gap is too large for max_internal_knots")
    if isinstance(optimization_steps, bool) or not isinstance(optimization_steps, int):
        raise TypeError("optimization_steps must be an integer")
    if optimization_steps < 0:
        raise ValueError("optimization_steps must be non-negative")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if optimizer not in {"adam", "lbfgs"}:
        raise ValueError("optimizer must be 'adam' or 'lbfgs'")
    if degree < 1:
        raise ValueError("degree must be positive")
    if smoothness_weight < 0.0 or control_ridge < 0.0:
        raise ValueError("regularization weights must be non-negative")


def _knots_from_logits(logits: torch.Tensor, min_gap: float) -> torch.Tensor:
    """Map unconstrained logits to ordered knots with boundary gaps."""
    count = logits.numel() - 1
    if count == 0:
        return logits.new_empty(0)
    available = 1.0 - min_gap * (count + 1)
    spans = min_gap + available * torch.softmax(logits, dim=0)
    return torch.cumsum(spans, dim=0)[:-1]


def _logits_for_knots(knots: torch.Tensor, min_gap: float) -> torch.Tensor:
    boundaries = torch.cat([knots.new_zeros(1), knots, knots.new_ones(1)])
    residual_spans = boundaries[1:] - boundaries[:-1] - min_gap
    available = 1.0 - min_gap * (knots.numel() + 1)
    # Inputs already meet the gap invariant.  The tiny floor only protects
    # log/softmax at floating-point boundaries.
    weights = (residual_spans / available).clamp_min(torch.finfo(knots.dtype).tiny)
    return weights.log()


def _variable_projection_mse(
    parameters: torch.Tensor,
    points: torch.Tensor,
    internal_knots: torch.Tensor,
    *,
    degree: int,
    smoothness_weight: float,
    control_ridge: float,
    interpolate_endpoints: bool,
    rcond: float | None,
) -> torch.Tensor:
    """Differentiate geometric MSE through a standard B-spline LS refit."""
    count = int(internal_knots.numel()) + degree + 1
    knot_vector = build_open_knot_vector(internal_knots, degree)
    basis = bspline_basis_matrix(parameters, knot_vector, degree, count)
    difference = second_difference_matrix(
        count, device=points.device, dtype=points.dtype
    )

    if interpolate_endpoints:
        fixed = torch.stack([points[0], points[-1]])
        fixed_columns = torch.stack([basis[:, 0], basis[:, -1]], dim=-1)
        designs = [basis[:, 1:-1]]
        targets = [points - fixed_columns @ fixed]
        if smoothness_weight > 0.0 and difference.shape[0]:
            designs.append(smoothness_weight**0.5 * difference[:, 1:-1])
            targets.append(
                -smoothness_weight**0.5
                * torch.stack([difference[:, 0], difference[:, -1]], dim=-1)
                @ fixed
            )
        if control_ridge > 0.0:
            designs.append(
                control_ridge**0.5
                * torch.eye(count - 2, device=points.device, dtype=points.dtype)
            )
            targets.append(points.new_zeros((count - 2, points.shape[1])))
        solution = torch.linalg.lstsq(
            torch.cat(designs), torch.cat(targets), rcond=rcond
        ).solution
        controls = torch.cat([fixed[:1], solution, fixed[1:]])
    else:
        designs = [basis]
        targets = [points]
        if smoothness_weight > 0.0 and difference.shape[0]:
            designs.append(smoothness_weight**0.5 * difference)
            targets.append(points.new_zeros((difference.shape[0], points.shape[1])))
        if control_ridge > 0.0:
            designs.append(
                control_ridge**0.5
                * torch.eye(count, device=points.device, dtype=points.dtype)
            )
            targets.append(points.new_zeros((count, points.shape[1])))
        controls = torch.linalg.lstsq(
            torch.cat(designs), torch.cat(targets), rcond=rcond
        ).solution
    return (basis @ controls - points).square().sum(dim=-1).mean()


def _optimize_candidate_positions(
    parameters: torch.Tensor,
    points: torch.Tensor,
    starting_knots: torch.Tensor,
    *,
    min_gap: float,
    optimization_steps: int,
    learning_rate: float,
    optimizer: str,
    degree: int,
    smoothness_weight: float,
    control_ridge: float,
    interpolate_endpoints: bool,
    rcond: float | None,
) -> tuple[torch.Tensor, int, int]:
    if starting_knots.numel() == 0 or optimization_steps == 0:
        return starting_knots.detach().clone(), 0, 0
    logits = _logits_for_knots(starting_knots, min_gap).detach().requires_grad_(True)
    evaluations = 0
    steps = 0
    best_logits = logits.detach().clone()
    best_loss = math.inf

    if optimizer == "adam":
        numerical_optimizer: torch.optim.Optimizer = torch.optim.Adam(
            [logits], lr=learning_rate
        )
        for _ in range(optimization_steps):
            numerical_optimizer.zero_grad(set_to_none=True)
            loss = _variable_projection_mse(
                parameters,
                points,
                _knots_from_logits(logits, min_gap),
                degree=degree,
                smoothness_weight=smoothness_weight,
                control_ridge=control_ridge,
                interpolate_endpoints=interpolate_endpoints,
                rcond=rcond,
            )
            evaluations += 1
            if not torch.isfinite(loss):
                break
            if float(loss.detach()) < best_loss:
                best_loss = float(loss.detach())
                best_logits = logits.detach().clone()
            loss.backward()
            numerical_optimizer.step()
            steps += 1
    else:
        numerical_optimizer = torch.optim.LBFGS(
            [logits], lr=learning_rate, max_iter=1, line_search_fn=None
        )
        for _ in range(optimization_steps):
            completed = False

            def closure() -> torch.Tensor:
                nonlocal evaluations, completed, best_logits, best_loss
                numerical_optimizer.zero_grad(set_to_none=True)
                loss = _variable_projection_mse(
                    parameters,
                    points,
                    _knots_from_logits(logits, min_gap),
                    degree=degree,
                    smoothness_weight=smoothness_weight,
                    control_ridge=control_ridge,
                    interpolate_endpoints=interpolate_endpoints,
                    rcond=rcond,
                )
                evaluations += 1
                if torch.isfinite(loss):
                    if float(loss.detach()) < best_loss:
                        best_loss = float(loss.detach())
                        best_logits = logits.detach().clone()
                    loss.backward()
                    completed = True
                    return loss
                # LBFGS requires a scalar with a gradient even in this rare
                # numerical failure path.
                return logits.square().sum() * 0.0

            numerical_optimizer.step(closure)
            if not completed:
                break
            steps += 1

    return _knots_from_logits(best_logits, min_gap), steps, evaluations


def gradient_knot_pruning_baseline(
    points: torch.Tensor,
    *,
    max_internal_knots: int | None = None,
    Kmax: int | None = None,
    mse_tolerance: float | None = None,
    error_tolerance: float | None = None,
    min_internal_knots: int = 0,
    degree: int = 3,
    min_gap: float = 1e-4,
    optimization_steps: int = 40,
    learning_rate: float = 5e-2,
    optimizer: str = "adam",
    smoothness_weight: float = 0.0,
    control_ridge: float = 0.0,
    interpolate_endpoints: bool = True,
    rcond: float | None = None,
) -> GradientKnotPruningResult:
    """Relocate a uniform Kmax set, then greedily prune with relocation.

    No model is accepted or queried.  Parameters are recomputed from the
    observed polyline by chord length.  The full uniform set is optimized once
    before deletion; every deletion candidate then receives its own gradient
    relocation and exact ``float64`` standard B-spline refit.  The greedy path is local, so
    this routine is a reproducible numerical baseline rather than a global
    minimum-knot certificate.
    """
    if max_internal_knots is None:
        max_internal_knots = Kmax
    elif Kmax is not None and Kmax != max_internal_knots:
        raise ValueError("Kmax and max_internal_knots disagree")
    if max_internal_knots is None:
        raise TypeError("max_internal_knots (or Kmax) is required")
    if mse_tolerance is None:
        mse_tolerance = error_tolerance
    elif error_tolerance is not None and error_tolerance != mse_tolerance:
        raise ValueError("error_tolerance and mse_tolerance disagree")
    if mse_tolerance is None:
        raise TypeError("mse_tolerance (or error_tolerance) is required")
    _validate_options(
        points,
        max_internal_knots=max_internal_knots,
        mse_tolerance=mse_tolerance,
        min_internal_knots=min_internal_knots,
        min_gap=min_gap,
        optimization_steps=optimization_steps,
        learning_rate=learning_rate,
        optimizer=optimizer,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
    )
    if rcond is not None and rcond < 0.0:
        raise ValueError("rcond must be non-negative or None")

    started = time.perf_counter()
    # Exact verification is intentionally always float64, independently of
    # the caller's model/training dtype.
    observed = points.detach().to(dtype=torch.float64)
    parameters = chord_length_parameters(observed)
    retained = torch.arange(
        1,
        max_internal_knots + 1,
        device=observed.device,
        dtype=observed.dtype,
    ) / (max_internal_knots + 1)
    initial_fit = refit_bspline_control_points(
        parameters, observed, retained, degree=degree,
        smoothness_weight=smoothness_weight, control_ridge=control_ridge,
        interpolate_endpoints=interpolate_endpoints, rcond=rcond,
    )
    current_fit = initial_fit
    refit_count = 1
    gradient_steps = 0
    evaluation_count = 0
    initial_location_update_accepted = False
    if retained.numel() and optimization_steps:
        optimized_initial_knots, local_steps, local_evaluations = (
            _optimize_candidate_positions(
                parameters,
                observed,
                retained,
                min_gap=min_gap,
                optimization_steps=optimization_steps,
                learning_rate=learning_rate,
                optimizer=optimizer,
                degree=degree,
                smoothness_weight=smoothness_weight,
                control_ridge=control_ridge,
                interpolate_endpoints=interpolate_endpoints,
                rcond=rcond,
            )
        )
        gradient_steps += local_steps
        evaluation_count += local_evaluations
        optimized_initial_fit = refit_bspline_control_points(
            parameters,
            observed,
            optimized_initial_knots,
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
            interpolate_endpoints=interpolate_endpoints,
            rcond=rcond,
        )
        refit_count += 1
        if float(optimized_initial_fit.fit_mse) <= float(initial_fit.fit_mse):
            retained = optimized_initial_knots
            current_fit = optimized_initial_fit
            initial_location_update_accepted = True
    initial_relocated_fit = current_fit
    rounds: list[GradientKnotDeletionStep] = []

    while retained.numel() > min_internal_knots:
        candidate_fits: list[BSplineLeastSquaresFit] = []
        candidate_knots: list[torch.Tensor] = []
        update_accepted: list[bool] = []
        for deleted_index in range(int(retained.numel())):
            start_knots = torch.cat([retained[:deleted_index], retained[deleted_index + 1 :]])
            start_fit = refit_bspline_control_points(
                parameters, observed, start_knots, degree=degree,
                smoothness_weight=smoothness_weight, control_ridge=control_ridge,
                interpolate_endpoints=interpolate_endpoints, rcond=rcond,
            )
            refit_count += 1
            optimized_knots, local_steps, local_evaluations = _optimize_candidate_positions(
                parameters, observed, start_knots, min_gap=min_gap,
                optimization_steps=optimization_steps, learning_rate=learning_rate,
                optimizer=optimizer, degree=degree,
                smoothness_weight=smoothness_weight, control_ridge=control_ridge,
                interpolate_endpoints=interpolate_endpoints, rcond=rcond,
            )
            gradient_steps += local_steps
            evaluation_count += local_evaluations
            optimized_fit = refit_bspline_control_points(
                parameters, observed, optimized_knots, degree=degree,
                smoothness_weight=smoothness_weight, control_ridge=control_ridge,
                interpolate_endpoints=interpolate_endpoints, rcond=rcond,
            )
            refit_count += 1
            # A location update is never allowed to degrade its exact fit.
            keep_update = float(optimized_fit.fit_mse) <= float(start_fit.fit_mse)
            candidate_knots.append(
                (optimized_knots if keep_update else start_knots).detach().clone()
            )
            candidate_fits.append(optimized_fit if keep_update else start_fit)
            update_accepted.append(keep_update)

        feasible = [
            index
            for index, fit in enumerate(candidate_fits)
            if float(fit.fit_mse) <= mse_tolerance
        ]
        selected = (
            min(feasible, key=lambda index: float(candidate_fits[index].fit_mse))
            if feasible
            else None
        )
        accepted = selected is not None
        rounds.append(
            GradientKnotDeletionStep(
                iteration=len(rounds) + 1,
                knots_before=retained.detach().clone(),
                candidate_fits=tuple(candidate_fits),
                candidate_updated_knots=tuple(candidate_knots),
                candidate_location_updates_accepted=tuple(update_accepted),
                selected_index=selected,
                accepted=accepted,
            )
        )
        if not accepted:
            break
        retained = candidate_knots[selected]
        current_fit = candidate_fits[selected]

    return GradientKnotPruningResult(
        mse_tolerance=float(mse_tolerance), parameters=parameters.detach().clone(),
        initial_fit=initial_fit, initial_relocated_fit=initial_relocated_fit,
        initial_location_update_accepted=initial_location_update_accepted,
        final_fit=current_fit, steps=tuple(rounds),
        gradient_steps=gradient_steps, refit_count=refit_count,
        evaluation_count=evaluation_count,
        total_search_time=time.perf_counter() - started,
    )


# Short alias for callers that use the baseline name as a verb.
gradient_prune_knots = gradient_knot_pruning_baseline
fit_gradient_knot_pruning = gradient_knot_pruning_baseline
