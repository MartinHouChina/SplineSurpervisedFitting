"""Auditable Liang-inspired feature-integral + IKI B-spline baseline.

Liang et al. describe an error-bounded method that first fits a B-spline with
an intensively uniform knot vector, distributes an initial knot set by equal
increments of a monotone geometric-feature integral, and then applies
iterative knot insertion (IKI).  Their full 2017 article is access restricted,
so its exact feature-combination equation, weight, and IKI constants could not
be verified.  Consequently this module is deliberately named *Liang-inspired*
rather than an exact reproduction.  It makes the comparison choices explicit:

* the feature CDF is ``(1-w) * normalized_arc + w * normalized_abs_curvature``
  with configurable ``w`` (default ``0.5``);
* absolute discrete curvature is the accumulated unsigned turning angle of a
  densely sampled preliminary spline; and
* each IKI step bisects the knot span containing the largest sampled point
  residual.  The midpoint rule is supported by the later implementation
  description in Yeh et al., Computer-Aided Design 2020.

The final and all threshold-scanned fits use the repository's common metric:
mean squared Euclidean point residual from an endpoint-constrained,
unregularized, standard B-spline least-squares refit in ``float64``.

Primary article: F. Liang et al., Measurement Science and Technology 28
(2017) 065015, https://doi.org/10.1088/1361-6501/aa6a05.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from .bspline_inference import BSplineLeastSquaresFit, refit_bspline_control_points
from .gradient_knot_pruning import chord_length_parameters


@dataclass(frozen=True)
class LiangFeatureIKIResult:
    """Result and complete diagnostics for :func:`fit_liang_feature_iki`.

    ``inserted_knots`` records insertion order, whereas ``knots`` and
    ``final_fit.internal_knots`` are sorted.  ``scanned_mse`` contains the
    common MSE after the initial feature-quantile placement and after every IKI
    insertion.  ``refit_count`` also includes the one dense preliminary fit,
    so ``refit_count == 1 + len(scanned_mse)``.
    """

    degree: int
    mse_tolerance: float
    max_internal_knots: int
    curvature_weight: float
    dense_internal_knot_count: int
    feature_sample_count: int
    parameters: torch.Tensor
    dense_knots: torch.Tensor
    preliminary_fit: BSplineLeastSquaresFit
    feature_parameters: torch.Tensor
    feature_curve_points: torch.Tensor
    normalized_arc_length: torch.Tensor
    normalized_absolute_curvature: torch.Tensor
    feature_cdf: torch.Tensor
    initial_knots: torch.Tensor
    inserted_knots: torch.Tensor
    insertion_spans: tuple[tuple[float, float], ...]
    scanned_internal_knot_counts: tuple[int, ...]
    scanned_mse: tuple[float, ...]
    scanned_max_point_error: tuple[float, ...]
    scanned_worst_point_indices: tuple[int, ...]
    final_fit: BSplineLeastSquaresFit
    refit_count: int
    elapsed_seconds: float

    @property
    def knots(self) -> torch.Tensor:
        """Return the final sorted internal knots."""

        return self.final_fit.internal_knots

    @property
    def fit(self) -> BSplineLeastSquaresFit:
        """Alias used by the other evaluation baselines."""

        return self.final_fit

    @property
    def K(self) -> int:
        """Return the final number of internal knots."""

        return int(self.knots.numel())

    @property
    def dense_count(self) -> int:
        """Short audit alias for ``dense_internal_knot_count``."""

        return self.dense_internal_knot_count

    @property
    def initial_internal_knots(self) -> torch.Tensor:
        """Compatibility alias for the initial feature-quantile knot set."""

        return self.initial_knots

    @property
    def iki_refit_count(self) -> int:
        """Number of common-MSE fits, excluding the preliminary dense fit."""

        return len(self.scanned_mse)

    @property
    def scanned_knot_counts(self) -> tuple[int, ...]:
        """Compatibility alias for the scanned internal-knot cardinalities."""

        return self.scanned_internal_knot_counts

    @property
    def threshold_satisfied(self) -> bool:
        """Whether the final common MSE meets the requested tolerance."""

        return float(self.final_fit.fit_mse) <= self.mse_tolerance


def _integer_option(value: int, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validate_curve(points: torch.Tensor, degree: int) -> None:
    if not isinstance(points, torch.Tensor):
        raise TypeError("points must be a torch.Tensor")
    if points.device.type != "cpu":
        raise ValueError("Liang-inspired baseline currently requires CPU points")
    if points.dtype != torch.float64:
        raise ValueError("points must be CPU float64 for comparable refits")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [M, 2]")
    if points.shape[0] < degree + 1:
        raise ValueError("the sample count must be at least degree + 1")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("points must be finite")


def normalized_arc_curvature_feature(
    feature_parameters: torch.Tensor,
    curve_points: torch.Tensor,
    *,
    curvature_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the disclosed monotone feature CDF from sampled planar geometry.

    The returned tensors are the normalized cumulative arc length, normalized
    cumulative absolute discrete curvature, and their weighted feature CDF.
    Absolute discrete curvature is represented by unsigned turning angles and
    split equally over the two segments adjacent to each sampled vertex.  A
    geometrically constant or straight curve uses arc length as the combined
    CDF fallback when the requested weighted mass is zero.
    """

    if feature_parameters.ndim != 1 or curve_points.ndim != 2:
        raise ValueError("feature parameters/points must have shapes [N] and [N,2]")
    if curve_points.shape != (feature_parameters.numel(), 2):
        raise ValueError("feature parameters and planar points must share N")
    if feature_parameters.device != curve_points.device:
        raise ValueError("feature parameters and points must share a device")
    if feature_parameters.dtype != curve_points.dtype:
        raise ValueError("feature parameters and points must share a dtype")
    if not feature_parameters.is_floating_point():
        raise ValueError("feature inputs must be floating point")
    if feature_parameters.numel() < 3:
        raise ValueError("at least three feature samples are required")
    if not (
        bool(torch.isfinite(feature_parameters).all())
        and bool(torch.isfinite(curve_points).all())
    ):
        raise ValueError("feature inputs must be finite")
    if bool(torch.any(feature_parameters[1:] <= feature_parameters[:-1])):
        raise ValueError("feature parameters must be strictly increasing")
    if not math.isfinite(curvature_weight) or not 0.0 <= curvature_weight <= 1.0:
        raise ValueError("curvature_weight must lie in [0, 1]")

    segments = curve_points[1:] - curve_points[:-1]
    arc_increments = segments.norm(dim=-1)
    arc_cumulative = torch.cat(
        [curve_points.new_zeros(1), torch.cumsum(arc_increments, dim=0)]
    )
    arc_total = arc_cumulative[-1]
    if float(arc_total) > 0.0:
        normalized_arc = arc_cumulative / arc_total
    else:
        interval = feature_parameters[-1] - feature_parameters[0]
        normalized_arc = (feature_parameters - feature_parameters[0]) / interval

    incoming = segments[:-1]
    outgoing = segments[1:]
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    dot = (incoming * outgoing).sum(dim=-1)
    scale = incoming.norm(dim=-1) * outgoing.norm(dim=-1)
    turning = torch.atan2(cross.abs(), dot)
    turning = torch.where(scale > 0.0, turning, torch.zeros_like(turning))

    # Allocate a vertex's integrated curvature symmetrically to its adjacent
    # parameter intervals.  The sum remains exactly the total unsigned turn.
    curvature_increments = curve_points.new_zeros(segments.shape[0])
    curvature_increments[:-1] += 0.5 * turning
    curvature_increments[1:] += 0.5 * turning
    curvature_cumulative = torch.cat(
        [curve_points.new_zeros(1), torch.cumsum(curvature_increments, dim=0)]
    )
    curvature_total = curvature_cumulative[-1]
    if float(curvature_total) > 0.0:
        normalized_curvature = curvature_cumulative / curvature_total
    else:
        normalized_curvature = torch.zeros_like(curvature_cumulative)

    feature_cdf = (
        (1.0 - curvature_weight) * normalized_arc
        + curvature_weight * normalized_curvature
    )
    feature_total = feature_cdf[-1]
    if float(feature_total) > 0.0:
        feature_cdf = feature_cdf / feature_total
    else:
        # This occurs for a straight curve with curvature_weight == 1, or for
        # a coincident curve.  Uniform/arc mass is the deterministic limit.
        feature_cdf = normalized_arc.clone()
    feature_cdf = torch.cummax(feature_cdf, dim=0).values
    feature_cdf[0] = 0.0
    feature_cdf[-1] = 1.0
    return normalized_arc, normalized_curvature, feature_cdf


def place_feature_quantile_knots(
    feature_parameters: torch.Tensor,
    feature_cdf: torch.Tensor,
    internal_knot_count: int,
) -> torch.Tensor:
    """Invert a sampled monotone feature CDF at equal-mass quantiles."""

    _integer_option(internal_knot_count, "internal_knot_count", 0)
    if feature_parameters.ndim != 1 or feature_cdf.ndim != 1:
        raise ValueError("feature_parameters and feature_cdf must have shape [N]")
    if feature_parameters.shape != feature_cdf.shape or feature_cdf.numel() < 2:
        raise ValueError("feature_parameters and feature_cdf must share N >= 2")
    if feature_parameters.device != feature_cdf.device:
        raise ValueError("feature_parameters and feature_cdf must share a device")
    if feature_parameters.dtype != feature_cdf.dtype:
        raise ValueError("feature_parameters and feature_cdf must share a dtype")
    if not feature_parameters.is_floating_point():
        raise ValueError("feature arrays must be floating point")
    if not (
        bool(torch.isfinite(feature_parameters).all())
        and bool(torch.isfinite(feature_cdf).all())
    ):
        raise ValueError("feature arrays must be finite")
    if bool(torch.any(feature_parameters[1:] <= feature_parameters[:-1])):
        raise ValueError("feature_parameters must be strictly increasing")
    if bool(torch.any(feature_cdf[1:] < feature_cdf[:-1])):
        raise ValueError("feature_cdf must be non-decreasing")
    tolerance = 64.0 * torch.finfo(feature_cdf.dtype).eps
    if abs(float(feature_cdf[0])) > tolerance or abs(float(feature_cdf[-1]) - 1.0) > tolerance:
        raise ValueError("feature_cdf must start at zero and end at one")
    if internal_knot_count == 0:
        return feature_parameters.new_empty(0)

    targets = torch.arange(
        1,
        internal_knot_count + 1,
        device=feature_cdf.device,
        dtype=feature_cdf.dtype,
    ) / (internal_knot_count + 1)
    upper = torch.searchsorted(feature_cdf, targets, right=False)
    upper = upper.clamp(min=1, max=feature_cdf.numel() - 1)
    lower = upper - 1
    cdf_width = feature_cdf[upper] - feature_cdf[lower]
    fraction = (targets - feature_cdf[lower]) / cdf_width.clamp_min(
        torch.finfo(feature_cdf.dtype).tiny
    )
    knots = feature_parameters[lower] + fraction.clamp(0.0, 1.0) * (
        feature_parameters[upper] - feature_parameters[lower]
    )
    margin = 64.0 * torch.finfo(knots.dtype).eps
    knots = knots.clamp(min=margin, max=1.0 - margin)
    knots = torch.sort(knots).values
    if knots.numel() > 1 and bool(torch.any(knots[1:] <= knots[:-1])):
        raise RuntimeError("feature quantiles did not produce distinct knots")
    return knots


def _uniform_internal_knots(
    count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if count == 0:
        return torch.empty(0, device=device, dtype=dtype)
    return torch.linspace(0.0, 1.0, count + 2, device=device, dtype=dtype)[1:-1]


def _common_refit(
    parameters: torch.Tensor,
    points: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
) -> BSplineLeastSquaresFit:
    return refit_bspline_control_points(
        parameters,
        points,
        knots,
        degree=degree,
        smoothness_weight=0.0,
        control_ridge=0.0,
        interpolate_endpoints=True,
    )


@torch.no_grad()
def fit_liang_feature_iki(
    points: torch.Tensor,
    *,
    mse_tolerance: float,
    max_internal_knots: int,
    degree: int = 3,
    dense_internal_knot_count: int = 40,
    initial_internal_knot_count: int = 4,
    curvature_weight: float = 0.5,
    feature_sample_count: int = 1025,
) -> LiangFeatureIKIResult:
    """Fit an ordered planar curve with the disclosed Liang-inspired method.

    Parameters are computed solely from the observed polyline by chord length.
    The routine then performs one dense-uniform preliminary standard B-spline
    fit, constructs the explicit arc/curvature feature CDF, places a small
    initial knot set at equal feature quantiles, and inserts one span midpoint
    per IKI iteration until the common MSE tolerance is met or
    ``max_internal_knots`` is reached.

    Input points must be CPU ``float64`` so no hidden precision or transfer
    conversion is included.  The elapsed time covers parameterization,
    preliminary fitting, feature construction, IKI, and the final reported
    endpoint-constrained unregularized refit.
    """

    _integer_option(degree, "degree", 1)
    _integer_option(dense_internal_knot_count, "dense_internal_knot_count", 0)
    _integer_option(initial_internal_knot_count, "initial_internal_knot_count", 0)
    _integer_option(max_internal_knots, "max_internal_knots", 0)
    _integer_option(feature_sample_count, "feature_sample_count", 3)
    if initial_internal_knot_count > max_internal_knots:
        raise ValueError(
            "initial_internal_knot_count must not exceed max_internal_knots"
        )
    if not math.isfinite(mse_tolerance) or mse_tolerance < 0.0:
        raise ValueError("mse_tolerance must be finite and non-negative")
    if not math.isfinite(curvature_weight) or not 0.0 <= curvature_weight <= 1.0:
        raise ValueError("curvature_weight must lie in [0, 1]")
    _validate_curve(points, degree)

    started = time.perf_counter()
    observed = points.detach()
    parameters = chord_length_parameters(observed)

    dense_knots = _uniform_internal_knots(
        dense_internal_knot_count,
        device=observed.device,
        dtype=observed.dtype,
    )
    preliminary_fit = _common_refit(parameters, observed, dense_knots, degree)
    feature_parameters = torch.linspace(
        0.0,
        1.0,
        feature_sample_count,
        device=observed.device,
        dtype=observed.dtype,
    )
    feature_curve_points = preliminary_fit.evaluate(feature_parameters)
    normalized_arc, normalized_curvature, feature_cdf = (
        normalized_arc_curvature_feature(
            feature_parameters,
            feature_curve_points,
            curvature_weight=curvature_weight,
        )
    )
    initial_knots = place_feature_quantile_knots(
        feature_parameters,
        feature_cdf,
        initial_internal_knot_count,
    )

    knots = initial_knots.detach().clone()
    inserted: list[torch.Tensor] = []
    insertion_spans: list[tuple[float, float]] = []
    scanned_counts: list[int] = []
    scanned_mse: list[float] = []
    scanned_max_error: list[float] = []
    scanned_worst_indices: list[int] = []

    while True:
        # Every scanned fit already is the required final common refit.  Thus
        # the final loop value can be reported directly without an uncounted,
        # timing-distorting duplicate solve.
        fit = _common_refit(parameters, observed, knots, degree)
        point_error = (fit.reconstructed_points - observed).norm(dim=-1)
        worst_index = int(point_error.argmax())
        mse = float(fit.fit_mse)
        scanned_counts.append(int(knots.numel()))
        scanned_mse.append(mse)
        scanned_max_error.append(float(point_error[worst_index]))
        scanned_worst_indices.append(worst_index)

        if mse <= mse_tolerance or knots.numel() >= max_internal_knots:
            break

        worst_parameter = parameters[worst_index]
        span_index = int(torch.searchsorted(knots, worst_parameter, right=True))
        boundaries = torch.cat(
            [parameters.new_zeros(1), knots, parameters.new_ones(1)]
        )
        left = boundaries[span_index]
        right = boundaries[span_index + 1]
        midpoint = 0.5 * (left + right)
        if not bool((midpoint > left) & (midpoint < right)):
            raise RuntimeError("cannot represent another distinct span midpoint")
        inserted.append(midpoint.detach().clone())
        insertion_spans.append((float(left), float(right)))
        knots = torch.sort(torch.cat([knots, midpoint.reshape(1)])).values

    inserted_knots = (
        torch.stack(inserted) if inserted else observed.new_empty(0)
    )
    return LiangFeatureIKIResult(
        degree=degree,
        mse_tolerance=float(mse_tolerance),
        max_internal_knots=max_internal_knots,
        curvature_weight=float(curvature_weight),
        dense_internal_knot_count=dense_internal_knot_count,
        feature_sample_count=feature_sample_count,
        parameters=parameters.detach().clone(),
        dense_knots=dense_knots.detach().clone(),
        preliminary_fit=preliminary_fit,
        feature_parameters=feature_parameters.detach().clone(),
        feature_curve_points=feature_curve_points.detach().clone(),
        normalized_arc_length=normalized_arc.detach().clone(),
        normalized_absolute_curvature=normalized_curvature.detach().clone(),
        feature_cdf=feature_cdf.detach().clone(),
        initial_knots=initial_knots.detach().clone(),
        inserted_knots=inserted_knots,
        insertion_spans=tuple(insertion_spans),
        scanned_internal_knot_counts=tuple(scanned_counts),
        scanned_mse=tuple(scanned_mse),
        scanned_max_point_error=tuple(scanned_max_error),
        scanned_worst_point_indices=tuple(scanned_worst_indices),
        final_fit=fit,
        refit_count=1 + len(scanned_mse),
        elapsed_seconds=time.perf_counter() - started,
    )


# Descriptive alias for callers that name baselines rather than fit routines.
liang_feature_iki_baseline = fit_liang_feature_iki


__all__ = [
    "LiangFeatureIKIResult",
    "fit_liang_feature_iki",
    "liang_feature_iki_baseline",
    "normalized_arc_curvature_feature",
    "place_feature_quantile_knots",
]
