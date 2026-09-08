"""Derivative-feature knot placement adapted from Yeh et al. (CAD 2020).

The paper distributes knots so that consecutive knot spans contain equal mass
under a high-order finite-difference feature.  It predicts a knot count with a
dataset-specific error regression.  For the repository's common
minimum-complexity-at-a-fixed-MSE protocol, :func:`fit_feature_cdf_to_tolerance`
uses the published placement rule for each cardinality and scans cardinalities
from small to large.  The scan wrapper is deliberately reported separately
from the paper method.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from .bspline_inference import BSplineLeastSquaresFit, refit_bspline_control_points


@dataclass(frozen=True)
class FeatureCDFKnotPlacementResult:
    """Result and auditable diagnostics for feature-CDF placement."""

    degree: int
    derivative_order: int
    mse_tolerance: float
    max_internal_knots: int
    feature_parameters: torch.Tensor
    feature_values: torch.Tensor
    knots: torch.Tensor
    final_fit: BSplineLeastSquaresFit
    scanned_counts: tuple[int, ...]
    scanned_mse: tuple[float, ...]
    refit_count: int
    threshold_satisfied: bool
    elapsed_seconds: float


def _validate_inputs(
    parameters: torch.Tensor,
    points: torch.Tensor,
    derivative_order: int,
) -> None:
    if parameters.ndim != 1 or points.ndim != 2:
        raise ValueError("parameters/points must have shapes [M] and [M,D]")
    if parameters.shape[0] != points.shape[0]:
        raise ValueError("parameters and points must share the sample count")
    if parameters.device != points.device or parameters.dtype != points.dtype:
        raise ValueError("parameters and points must share device and dtype")
    if not parameters.is_floating_point() or not points.is_floating_point():
        raise ValueError("parameters and points must be floating point")
    if derivative_order < 1 or parameters.numel() <= derivative_order:
        raise ValueError("derivative_order must lie in [1, M-1]")
    if torch.any(parameters[1:] <= parameters[:-1]):
        raise ValueError("feature-CDF placement requires strictly increasing parameters")
    if not torch.isfinite(parameters).all() or not torch.isfinite(points).all():
        raise ValueError("parameters and points must be finite")


def finite_difference_feature(
    parameters: torch.Tensor,
    points: torch.Tensor,
    *,
    derivative_order: int,
    flat_floor_ratio: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Yeh et al. parameter samples and derivative feature values."""

    _validate_inputs(parameters, points, derivative_order)
    if not math.isfinite(flat_floor_ratio) or flat_floor_ratio < 0.0:
        raise ValueError("flat_floor_ratio must be finite and non-negative")
    locations = parameters
    differences = points
    for _ in range(derivative_order):
        widths = locations[1:] - locations[:-1]
        differences = (differences[1:] - differences[:-1]) / widths.unsqueeze(-1)
        locations = 0.5 * (locations[1:] + locations[:-1])
    values = differences.norm(dim=-1).clamp_min(0.0).pow(1.0 / derivative_order)
    peak = float(values.max()) if values.numel() else 0.0
    floor = max(peak * flat_floor_ratio, torch.finfo(points.dtype).eps)
    values = values.clamp_min(floor)
    feature_parameters = torch.cat(
        [parameters[:1], locations, parameters[-1:]], dim=0
    )
    feature_values = torch.cat(
        [values.new_zeros(1), values, values.new_zeros(1)], dim=0
    )
    return feature_parameters, feature_values


def _invert_piecewise_linear_feature_cdf(
    locations: torch.Tensor,
    feature: torch.Tensor,
    targets: torch.Tensor,
    *,
    density_limit_mass: float | None,
) -> torch.Tensor:
    widths = locations[1:] - locations[:-1]
    areas = 0.5 * (feature[:-1] + feature[1:]) * widths
    if density_limit_mass is not None:
        cap = areas.new_tensor(density_limit_mass)
        areas = torch.minimum(areas, cap)
    cumulative = torch.cat([areas.new_zeros(1), torch.cumsum(areas, dim=0)])
    total = cumulative[-1]
    if float(total) <= 0.0:
        return targets
    mass_targets = targets * total
    segment = torch.searchsorted(cumulative[1:], mass_targets, right=False)
    segment = segment.clamp_max(areas.numel() - 1)
    # Yeh et al. define F as the linear interpolant of the cumulative
    # trapezoid samples (their equations 12--14), rather than analytically
    # integrating and inverting the piecewise-linear feature inside a span.
    fraction = (mass_targets - cumulative[segment]) / areas[segment].clamp_min(
        torch.finfo(areas.dtype).eps
    )
    return locations[segment] + fraction.clamp(0.0, 1.0) * widths[segment]


def place_feature_cdf_knots(
    feature_parameters: torch.Tensor,
    feature_values: torch.Tensor,
    internal_knot_count: int,
    *,
    density_limit: bool = True,
) -> torch.Tensor:
    """Place ``K`` internal knots at equal derivative-feature mass."""

    if internal_knot_count < 0:
        raise ValueError("internal_knot_count must be non-negative")
    if internal_knot_count == 0:
        return feature_parameters.new_empty(0)
    fractions = torch.arange(
        1,
        internal_knot_count + 1,
        device=feature_parameters.device,
        dtype=feature_parameters.dtype,
    ) / (internal_knot_count + 1)
    widths = feature_parameters[1:] - feature_parameters[:-1]
    raw_areas = 0.5 * (feature_values[:-1] + feature_values[1:]) * widths
    raw_interval_mass = float(raw_areas.sum()) / (internal_knot_count + 1)
    knots = _invert_piecewise_linear_feature_cdf(
        feature_parameters,
        feature_values,
        fractions,
        density_limit_mass=(raw_interval_mass if density_limit else None),
    )
    eps = max(float(torch.finfo(knots.dtype).eps) * 64.0, 1e-12)
    return torch.sort(knots.clamp(min=eps, max=1.0 - eps)).values


@torch.no_grad()
def fit_feature_cdf_to_tolerance(
    parameters: torch.Tensor,
    points: torch.Tensor,
    *,
    degree: int = 3,
    mse_tolerance: float,
    max_internal_knots: int,
    min_internal_knots: int = 0,
    derivative_order: int | None = None,
    density_limit: bool = True,
    interpolate_endpoints: bool = True,
) -> FeatureCDFKnotPlacementResult:
    """Scan feature-CDF cardinalities and return the first feasible refit."""

    if degree < 1:
        raise ValueError("degree must be positive")
    if not math.isfinite(mse_tolerance) or mse_tolerance < 0.0:
        raise ValueError("mse_tolerance must be finite and non-negative")
    if not 0 <= min_internal_knots <= max_internal_knots:
        raise ValueError("invalid internal-knot count range")
    order = degree + 1 if derivative_order is None else derivative_order
    _validate_inputs(parameters, points, order)
    started = time.perf_counter()
    feature_parameters, feature_values = finite_difference_feature(
        parameters,
        points,
        derivative_order=order,
    )
    counts: list[int] = []
    errors: list[float] = []
    fits: list[BSplineLeastSquaresFit] = []
    for count in range(min_internal_knots, max_internal_knots + 1):
        knots = place_feature_cdf_knots(
            feature_parameters,
            feature_values,
            count,
            density_limit=density_limit,
        )
        fit = refit_bspline_control_points(
            parameters,
            points,
            knots,
            degree=degree,
            smoothness_weight=0.0,
            control_ridge=0.0,
            interpolate_endpoints=interpolate_endpoints,
        )
        counts.append(count)
        errors.append(float(fit.fit_mse))
        fits.append(fit)
        if errors[-1] <= mse_tolerance:
            break
    feasible = [index for index, error in enumerate(errors) if error <= mse_tolerance]
    selected = feasible[0] if feasible else min(
        range(len(fits)), key=lambda index: (errors[index], counts[index])
    )
    final_fit = fits[selected]
    return FeatureCDFKnotPlacementResult(
        degree=degree,
        derivative_order=order,
        mse_tolerance=float(mse_tolerance),
        max_internal_knots=max_internal_knots,
        feature_parameters=feature_parameters.detach().clone(),
        feature_values=feature_values.detach().clone(),
        knots=final_fit.internal_knots.detach().clone(),
        final_fit=final_fit,
        scanned_counts=tuple(counts),
        scanned_mse=tuple(errors),
        refit_count=len(fits),
        threshold_satisfied=float(final_fit.fit_mse) <= mse_tolerance,
        elapsed_seconds=time.perf_counter() - started,
    )
