from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..spline.bspline_deletion_teacher import single_knot_deletion_rmse_batch
from .bspline_inference import BSplineLeastSquaresFit, refit_bspline_control_points


@dataclass(frozen=True)
class KnotDeletionStep:
    """One greedy single-knot deletion trial.

    ``all_candidate_rmse[j]`` is the true geometric RMS obtained after
    deleting knot ``j`` from ``knots_before`` and refitting every control
    point.  ``spline`` is the best of those refits.  The final step may be
    rejected; retaining it makes the stopping decision auditable.
    """

    iteration: int
    knots_before: torch.Tensor
    removed_index: int
    removed_knot: torch.Tensor
    candidate_knots: torch.Tensor
    all_candidate_rmse: torch.Tensor
    accepted: bool
    spline: BSplineLeastSquaresFit

    @property
    def candidate_rmse(self) -> torch.Tensor:
        """Return the RMS of the best single-knot deletion in this round."""
        return self.spline.fit_rmse


@dataclass(frozen=True)
class MinimalKnotPruningResult:
    """Greedy minimum-complexity fit under a hard geometric RMS tolerance."""

    error_tolerance: float
    min_internal_knots: int
    initial_fit: BSplineLeastSquaresFit
    steps: tuple[KnotDeletionStep, ...]
    final_fit: BSplineLeastSquaresFit

    @property
    def initial_internal_knots(self) -> torch.Tensor:
        return self.initial_fit.internal_knots

    @property
    def final_internal_knots(self) -> torch.Tensor:
        return self.final_fit.internal_knots

    @property
    def initial_count(self) -> int:
        return int(self.initial_internal_knots.numel())

    @property
    def final_count(self) -> int:
        return int(self.final_internal_knots.numel())

    @property
    def fit(self) -> BSplineLeastSquaresFit:
        """Alias for the final standard B-spline fit."""
        return self.final_fit

    @property
    def accepted_steps(self) -> tuple[KnotDeletionStep, ...]:
        return tuple(step for step in self.steps if step.accepted)

    @property
    def removed_knots(self) -> torch.Tensor:
        """Return accepted removals in deletion order."""
        accepted = self.accepted_steps
        if not accepted:
            return self.initial_internal_knots.new_empty(0)
        return torch.stack([step.removed_knot for step in accepted])

    @property
    def rms_trajectory(self) -> torch.Tensor:
        """Return initial and accepted-fit RMS values in order."""
        values = [self.initial_fit.fit_rmse]
        values.extend(step.candidate_rmse for step in self.accepted_steps)
        return torch.stack(values)

    @property
    def threshold_satisfied(self) -> bool:
        """Whether the returned fit actually satisfies the requested RMS."""
        return float(self.final_fit.fit_rmse) <= self.error_tolerance


def _validate_pruning_options(
    error_tolerance: float,
    min_internal_knots: int,
    candidate_count: int,
) -> None:
    if not math.isfinite(error_tolerance) or error_tolerance < 0.0:
        raise ValueError("error_tolerance must be finite and non-negative")
    if isinstance(min_internal_knots, bool) or not isinstance(
        min_internal_knots, int
    ):
        raise TypeError("min_internal_knots must be an integer")
    if min_internal_knots < 0:
        raise ValueError("min_internal_knots must be non-negative")
    if min_internal_knots > candidate_count:
        raise ValueError(
            "min_internal_knots cannot exceed the candidate knot count"
        )


@torch.no_grad()
def prune_knots_to_rms_tolerance(
    parameters: torch.Tensor,
    points: torch.Tensor,
    candidate_internal_knots: torch.Tensor,
    *,
    error_tolerance: float,
    min_internal_knots: int = 0,
    degree: int = 3,
    smoothness_weight: float = 1e-6,
    control_ridge: float = 0.0,
    interpolate_endpoints: bool = True,
    rcond: float | None = None,
) -> MinimalKnotPruningResult:
    """Greedily remove knots while the refitted curve satisfies an RMS bound.

    In every round this function deletes each currently retained knot in turn,
    performs a complete standard B-spline control-point refit, and selects the
    deletion with the lowest *geometric* RMS.  The deletion is committed only
    when that measured RMS is at most ``error_tolerance``.  Consequently a
    learned pruning score can propose or order candidates elsewhere, but it
    cannot override this deployment-time hard check.

    The search is greedy over single-knot deletions.  It therefore returns the
    smallest representation reached along that deterministic deletion path,
    not a certificate of the globally smallest subset among all knot subsets.
    """
    _validate_pruning_options(
        error_tolerance,
        min_internal_knots,
        int(candidate_internal_knots.numel()),
    )

    retained = candidate_internal_knots.detach().clone()
    current_fit = refit_bspline_control_points(
        parameters,
        points,
        retained,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        interpolate_endpoints=interpolate_endpoints,
        rcond=rcond,
    )
    initial_fit = current_fit
    steps: list[KnotDeletionStep] = []

    while retained.numel() > min_internal_knots:
        # Evaluate all current one-knot deletions in one batched Cox--de Boor
        # construction and one batched least-squares call.  We then materialize
        # only the winning fit, preserving the exact exhaustive greedy rule
        # without K separate Python solver launches per round.
        candidate_rmse = single_knot_deletion_rmse_batch(
            parameters.unsqueeze(0),
            points.unsqueeze(0),
            retained.unsqueeze(0),
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
            interpolate_endpoints=interpolate_endpoints,
            rcond=rcond,
        )[0]
        best_index = int(torch.argmin(candidate_rmse).item())
        best_candidate_knots = torch.cat(
            [retained[:best_index], retained[best_index + 1 :]]
        )
        best_fit = refit_bspline_control_points(
            parameters,
            points,
            best_candidate_knots,
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
            interpolate_endpoints=interpolate_endpoints,
            rcond=rcond,
        )
        accepted = float(best_fit.fit_rmse) <= error_tolerance
        step = KnotDeletionStep(
            iteration=len(steps) + 1,
            knots_before=retained.detach().clone(),
            removed_index=best_index,
            removed_knot=retained[best_index].detach().clone(),
            candidate_knots=best_candidate_knots.detach().clone(),
            all_candidate_rmse=candidate_rmse.detach().clone(),
            accepted=accepted,
            spline=best_fit,
        )
        steps.append(step)

        if not accepted:
            break
        retained = best_candidate_knots
        current_fit = best_fit

    return MinimalKnotPruningResult(
        error_tolerance=float(error_tolerance),
        min_internal_knots=min_internal_knots,
        initial_fit=initial_fit,
        steps=tuple(steps),
        final_fit=current_fit,
    )
