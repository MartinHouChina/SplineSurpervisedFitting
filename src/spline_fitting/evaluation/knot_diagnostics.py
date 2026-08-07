from __future__ import annotations

from dataclasses import dataclass

import torch

from ..spline.differentiable_solver import solve_coefficients
from ..spline.truncated_power_basis import build_design_matrix


def _validate_threshold(threshold: float) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("activity threshold must lie in [0, 1]")


def point_fit_statistics(
    reconstructed: torch.Tensor,
    points: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return per-sample point-correspondence fitting statistics.

    ``fit_mse`` follows the training loss convention: mean squared Euclidean
    distance per sampled point. ``coordinate_mse`` is the conventional mean
    over both point and coordinate dimensions.
    """
    if reconstructed.shape != points.shape or points.ndim != 3:
        raise ValueError("reconstructed and points must share shape [B, M, D]")

    squared = (reconstructed - points).pow(2)
    fit_mse = squared.sum(dim=-1).mean(dim=-1)
    coordinate_mse = squared.mean(dim=(-2, -1))
    return {
        "fit_mse": fit_mse,
        "fit_rmse": fit_mse.sqrt(),
        "coordinate_mse": coordinate_mse,
        "coordinate_rmse": coordinate_mse.sqrt(),
    }


def activity_statistics(
    activity: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Summarize soft activity and hard threshold counts per sample."""
    if activity.ndim != 2:
        raise ValueError("activity must have shape [B, K]")
    _validate_threshold(threshold)
    if not torch.isfinite(activity).all():
        raise ValueError("activity must contain only finite values")
    if torch.any((activity < 0.0) | (activity > 1.0)):
        raise ValueError("activity values must lie in [0, 1]")

    batch, candidates = activity.shape
    dtype = activity.dtype
    device = activity.device
    if candidates == 0:
        activity_mean = torch.zeros(batch, device=device, dtype=dtype)
    else:
        activity_mean = activity.mean(dim=-1)

    return {
        "candidate_knot_count": torch.full(
            (batch,), float(candidates), device=device, dtype=dtype
        ),
        "activity_mass": activity.sum(dim=-1),
        "activity_mean": activity_mean,
        "hard_active_count": (activity >= threshold).sum(dim=-1).to(dtype),
    }


def build_open_knot_vector(
    internal_knots: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """Build the standard open-clamped knot vector for reporting/export."""
    if internal_knots.ndim != 1:
        raise ValueError("internal_knots must have shape [K]")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if internal_knots.numel() > 1 and torch.any(
        internal_knots[1:] < internal_knots[:-1]
    ):
        raise ValueError("internal_knots must be sorted")

    boundary_size = degree + 1
    return torch.cat(
        [
            torch.zeros(
                boundary_size,
                device=internal_knots.device,
                dtype=internal_knots.dtype,
            ),
            internal_knots,
            torch.ones(
                boundary_size,
                device=internal_knots.device,
                dtype=internal_knots.dtype,
            ),
        ]
    )


@dataclass(frozen=True)
class PrunedSplineFit:
    """One sample after hard activity thresholding and coefficient refitting."""

    threshold: float
    candidate_count: int
    retained_count: int
    activity_mass: float
    retained_mask: torch.Tensor
    retained_internal_knots: torch.Tensor
    open_knot_vector: torch.Tensor
    coefficients: torch.Tensor
    reconstructed_points: torch.Tensor
    fit_mse: torch.Tensor
    fit_rmse: torch.Tensor
    coordinate_mse: torch.Tensor
    coordinate_rmse: torch.Tensor


@torch.no_grad()
def hard_prune_and_refit(
    params: torch.Tensor,
    internal_knots: torch.Tensor,
    activity: torch.Tensor,
    points: torch.Tensor,
    degree: int,
    *,
    threshold: float = 0.5,
    lambda_poly: float = 1e-6,
    lambda_knot: float = 1e-4,
    gate_eps: float = 0.0,
) -> list[PrunedSplineFit]:
    """Threshold candidate knots, remove inactive columns, and refit coefficients.

    A hard count is meaningful only after the inactive basis columns have been
    removed and the remaining linear coefficients have been solved again. The
    returned fit therefore differs from simply displaying the original soft
    gated forward pass.
    """
    _validate_threshold(threshold)
    if params.ndim != 2 or internal_knots.ndim != 2 or activity.ndim != 2:
        raise ValueError("params, internal_knots and activity must be batched matrices")
    if points.ndim != 3:
        raise ValueError("points must have shape [B, M, D]")
    if internal_knots.shape != activity.shape:
        raise ValueError("internal_knots and activity must have identical shape")
    if params.shape[0] != points.shape[0] or params.shape[0] != activity.shape[0]:
        raise ValueError("all inputs must share the same batch size")
    if params.shape[1] != points.shape[1]:
        raise ValueError("params and points must share the same point count")

    reports: list[PrunedSplineFit] = []
    candidate_count = activity.shape[1]
    for index in range(points.shape[0]):
        retained_mask = activity[index] >= threshold
        retained = internal_knots[index, retained_mask]
        retained_batch = retained.unsqueeze(0)
        hard_activity = torch.ones_like(retained_batch)
        design = build_design_matrix(
            params[index : index + 1],
            retained_batch,
            hard_activity,
            degree,
            gate_eps,
        )["design_matrix"]
        solution = solve_coefficients(
            design,
            points[index : index + 1],
            degree,
            lambda_poly=lambda_poly,
            lambda_knot=lambda_knot,
        )
        reconstructed = design @ solution["coefficients"]
        fit = point_fit_statistics(reconstructed, points[index : index + 1])

        reports.append(
            PrunedSplineFit(
                threshold=threshold,
                candidate_count=candidate_count,
                retained_count=int(retained_mask.sum().item()),
                activity_mass=float(activity[index].sum().item()),
                retained_mask=retained_mask.detach().clone(),
                retained_internal_knots=retained.detach().clone(),
                open_knot_vector=build_open_knot_vector(retained, degree).detach().clone(),
                coefficients=solution["coefficients"][0].detach().clone(),
                reconstructed_points=reconstructed[0].detach().clone(),
                fit_mse=fit["fit_mse"][0].detach().clone(),
                fit_rmse=fit["fit_rmse"][0].detach().clone(),
                coordinate_mse=fit["coordinate_mse"][0].detach().clone(),
                coordinate_rmse=fit["coordinate_rmse"][0].detach().clone(),
            )
        )
    return reports


def knot_contribution_rms(
    weighted_increment_basis: torch.Tensor,
    increment_coefficients: torch.Tensor,
) -> torch.Tensor:
    """Return each column's RMS contribution, shape [B, K].

    Correlated truncated-power columns may cancel each other, so this is a
    descriptive score rather than proof that a knot is individually necessary.
    """
    if weighted_increment_basis.ndim != 3:
        raise ValueError("weighted_increment_basis must have shape [B, M, K]")
    if increment_coefficients.ndim != 3:
        raise ValueError("increment_coefficients must have shape [B, K, D]")
    if (
        weighted_increment_basis.shape[0] != increment_coefficients.shape[0]
        or weighted_increment_basis.shape[2] != increment_coefficients.shape[1]
    ):
        raise ValueError("basis and coefficients must share batch and knot dimensions")

    contribution = (
        weighted_increment_basis.unsqueeze(-1)
        * increment_coefficients.unsqueeze(-3)
    )
    return contribution.pow(2).sum(dim=-1).mean(dim=-2).sqrt()


@dataclass(frozen=True)
class KnotMatchStatistics:
    predicted_count: int
    true_count: int
    matched_count: int
    precision: float
    recall: float
    f1: float
    matched_mae: float


def match_internal_knots(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    tolerance: float = 0.05,
) -> KnotMatchStatistics:
    """Match sorted predicted and true knots under a shared parameterization.

    Knot positions are not directly comparable when the predicted point
    parameterization differs from the ground-truth parameterization.
    """
    if predicted.ndim != 1 or target.ndim != 1:
        raise ValueError("predicted and target knots must be one-dimensional")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    predicted_values = torch.sort(predicted.detach().cpu()).values.tolist()
    target_values = torch.sort(target.detach().cpu()).values.tolist()
    pred_index = 0
    target_index = 0
    errors: list[float] = []
    while pred_index < len(predicted_values) and target_index < len(target_values):
        difference = predicted_values[pred_index] - target_values[target_index]
        if abs(difference) <= tolerance:
            errors.append(abs(difference))
            pred_index += 1
            target_index += 1
        elif difference < 0.0:
            pred_index += 1
        else:
            target_index += 1

    predicted_count = len(predicted_values)
    true_count = len(target_values)
    matched_count = len(errors)
    precision = matched_count / predicted_count if predicted_count else float(true_count == 0)
    recall = matched_count / true_count if true_count else float(predicted_count == 0)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    matched_mae = sum(errors) / matched_count if matched_count else float("nan")
    return KnotMatchStatistics(
        predicted_count=predicted_count,
        true_count=true_count,
        matched_count=matched_count,
        precision=precision,
        recall=recall,
        f1=f1,
        matched_mae=matched_mae,
    )
