"""Auditable adaptation of Park and Lee's dominant-point (DOM) method.

The original algorithm selects geometrically significant data points and
places knots by averaging the parameters of consecutive dominant points.  The
implementation below keeps those two defining ingredients and the paper's
adaptive shape-index split.  It deliberately uses the comparison harness's
mean squared Euclidean stopping tolerance and parameter-correspondence
residuals; the paper also reports maximum and orthogonal-distance criteria.
Those deviations are repeated in the returned diagnostics so benchmark output
cannot be mistaken for the authors' original software.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .bspline_inference import BSplineLeastSquaresFit, refit_bspline_control_points
from .gradient_knot_pruning import chord_length_parameters


@dataclass(frozen=True)
class ParkDominantPointResult:
    """Result of :func:`fit_park_dominant_points`.

    ``refit_count`` counts the exact endpoint-constrained standard B-spline
    solves in ``scanned_knot_counts``.  No preliminary curvature fit or hidden
    regularized solve is included in this adaptation.
    """

    mse_tolerance: float
    parameters: torch.Tensor
    knots: torch.Tensor
    dominant_indices: torch.Tensor
    seed_indices: torch.Tensor
    final_fit: BSplineLeastSquaresFit
    refit_count: int
    scanned_knot_counts: tuple[int, ...]
    scanned_mse: tuple[float, ...]
    scanned_max_parameter_residual: tuple[float, ...]
    diagnostics: dict[str, object]

    @property
    def threshold_satisfied(self) -> bool:
        return float(self.final_fit.fit_mse) <= self.mse_tolerance

    @property
    def K(self) -> int:
        return int(self.knots.numel())


def _validate_inputs(
    points: torch.Tensor,
    *,
    degree: int,
    mse_tolerance: float,
    max_internal_knots: int,
    shape_weight: float,
) -> None:
    if not isinstance(points, torch.Tensor):
        raise TypeError("points must be a torch.Tensor")
    if points.device.type != "cpu" or points.dtype != torch.float64:
        raise ValueError("points must be CPU float64 for comparable numerical refits")
    if points.ndim != 2 or points.shape[1] < 1:
        raise ValueError("points must have shape [M, D], with D >= 1")
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be an integer >= 1")
    if points.shape[0] < degree + 1:
        raise ValueError("at least degree + 1 ordered points are required")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("points must be finite")
    if not math.isfinite(mse_tolerance) or mse_tolerance < 0.0:
        raise ValueError("mse_tolerance must be finite and non-negative")
    if (
        isinstance(max_internal_knots, bool)
        or not isinstance(max_internal_knots, int)
        or max_internal_knots < 0
    ):
        raise ValueError("max_internal_knots must be an integer >= 0")
    if not math.isfinite(shape_weight) or not 0.0 <= shape_weight <= 1.0:
        raise ValueError("shape_weight must be finite and lie in [0, 1]")


def _discrete_curvatures(points: torch.Tensor) -> torch.Tensor:
    """Generalized Menger curvature at each interior polyline point.

    Park and Lee allow either local discrete curvature or curvature evaluated
    on a smoothed base curve.  The local formula is used here so the baseline
    has no undisclosed preliminary fit and works in any coordinate dimension.
    """

    curvature = points.new_zeros(points.shape[0])
    if points.shape[0] < 3:
        return curvature
    incoming = points[1:-1] - points[:-2]
    outgoing = points[2:] - points[1:-1]
    diagonal = points[2:] - points[:-2]
    incoming_sq = incoming.square().sum(dim=-1)
    outgoing_sq = outgoing.square().sum(dim=-1)
    gram_area_sq = (
        incoming_sq * outgoing_sq
        - (incoming * outgoing).sum(dim=-1).square()
    ).clamp_min(0.0)
    denominator = incoming.norm(dim=-1) * outgoing.norm(dim=-1) * diagonal.norm(dim=-1)
    valid = denominator > torch.finfo(points.dtype).eps
    values = points.new_zeros(points.shape[0] - 2)
    values[valid] = 2.0 * gram_area_sq[valid].sqrt() / denominator[valid]
    curvature[1:-1] = values
    return curvature


def _local_curvature_maxima(curvatures: torch.Tensor) -> list[int]:
    if curvatures.numel() < 3:
        return []
    lower_bound = float(curvatures.mean()) / 4.0
    maxima = [
        index
        for index in range(1, int(curvatures.numel()) - 1)
        if float(curvatures[index]) > float(curvatures[index - 1])
        and float(curvatures[index]) > float(curvatures[index + 1])
        and float(curvatures[index]) > lower_bound
    ]
    # The paper stores seeds in decreasing order of significance.  A sample
    # index is a deterministic secondary key when two curvatures are equal.
    return sorted(maxima, key=lambda index: (-float(curvatures[index]), index))


def _cumulative_shape_measures(
    points: torch.Tensor,
    parameters: torch.Tensor,
    curvatures: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    curvature_edges = (
        0.5
        * (curvatures[:-1].abs() + curvatures[1:].abs())
        * (parameters[1:] - parameters[:-1])
    )
    length_edges = (points[1:] - points[:-1]).norm(dim=-1)
    return (
        torch.cat([points.new_zeros(1), curvature_edges.cumsum(dim=0)]),
        torch.cat([points.new_zeros(1), length_edges.cumsum(dim=0)]),
    )


def _shape_index(
    start: int,
    end: int,
    *,
    cumulative_curvature: torch.Tensor,
    cumulative_length: torch.Tensor,
    shape_weight: float,
) -> float:
    total_curvature = float(cumulative_curvature[-1])
    total_length = float(cumulative_length[-1])
    local_curvature = float(cumulative_curvature[end] - cumulative_curvature[start])
    local_length = float(cumulative_length[end] - cumulative_length[start])
    curvature_fraction = local_curvature / total_curvature if total_curvature > 0.0 else 0.0
    length_fraction = local_length / total_length if total_length > 0.0 else 0.0
    return shape_weight * curvature_fraction + (1.0 - shape_weight) * length_fraction


def _balanced_shape_split(
    start: int,
    end: int,
    *,
    cumulative_curvature: torch.Tensor,
    cumulative_length: torch.Tensor,
    shape_weight: float,
) -> int | None:
    """Return ``argmin_w |lambda(start,w)-lambda(w,end)|``."""

    if end - start <= 1:
        return None
    candidates: list[tuple[float, int, int]] = []
    for index in range(start + 1, end):
        left = _shape_index(
            start,
            index,
            cumulative_curvature=cumulative_curvature,
            cumulative_length=cumulative_length,
            shape_weight=shape_weight,
        )
        right = _shape_index(
            index,
            end,
            cumulative_curvature=cumulative_curvature,
            cumulative_length=cumulative_length,
            shape_weight=shape_weight,
        )
        # Prefer the sample nearest the index midpoint only for an exact/tiny
        # shape-index tie; this makes a fully coincident curve deterministic.
        candidates.append((abs(left - right), abs(2 * index - start - end), index))
    return min(candidates)[2]


def _widest_shape_segment_split(
    dominant_indices: list[int],
    *,
    cumulative_curvature: torch.Tensor,
    cumulative_length: torch.Tensor,
    shape_weight: float,
) -> int | None:
    """Complete the minimum degree-compatible seed set without a curve fit."""

    candidates: list[tuple[float, int, int, int]] = []
    for start, end in zip(dominant_indices[:-1], dominant_indices[1:]):
        if end - start <= 1:
            continue
        complexity = _shape_index(
            start,
            end,
            cumulative_curvature=cumulative_curvature,
            cumulative_length=cumulative_length,
            shape_weight=shape_weight,
        )
        # Maximize complexity and then available samples.  The final keys keep
        # behavior deterministic when the polyline has zero geometric extent.
        candidates.append((-complexity, -(end - start), start, end))
    if not candidates:
        return None
    _, _, start, end = min(candidates)
    return _balanced_shape_split(
        start,
        end,
        cumulative_curvature=cumulative_curvature,
        cumulative_length=cumulative_length,
        shape_weight=shape_weight,
    )


def _residual_segment_split(
    dominant_indices: list[int],
    residuals: torch.Tensor,
    *,
    cumulative_curvature: torch.Tensor,
    cumulative_length: torch.Tensor,
    shape_weight: float,
) -> int | None:
    """Split the dominant-point segment containing the largest residual."""

    selected = set(dominant_indices)
    residual_order = sorted(
        (index for index in range(int(residuals.numel())) if index not in selected),
        key=lambda index: (-float(residuals[index]), index),
    )
    if not residual_order:
        return None
    worst = residual_order[0]
    for start, end in zip(dominant_indices[:-1], dominant_indices[1:]):
        if start < worst < end:
            return _balanced_shape_split(
                start,
                end,
                cumulative_curvature=cumulative_curvature,
                cumulative_length=cumulative_length,
                shape_weight=shape_weight,
            )
    return None


def _knots_from_dominant_parameters(
    parameters: torch.Tensor,
    dominant_indices: list[int],
    degree: int,
) -> tuple[torch.Tensor, int]:
    """Apply Park--Lee Eq. (4): average ``degree`` dominant parameters."""

    dominant_parameters = parameters[
        torch.tensor(dominant_indices, device=parameters.device, dtype=torch.long)
    ]
    internal_count = len(dominant_indices) - degree - 1
    if internal_count <= 0:
        return parameters.new_empty(0), 0
    knots = torch.stack(
        [dominant_parameters[index : index + degree].mean() for index in range(1, internal_count + 1)]
    )
    # Eq. (4) lies strictly inside the parameter domain for distinct data.
    # Repeated input points can collapse a chord parameter onto an endpoint;
    # clamp only that degenerate boundary case so the common refitter remains
    # defined, and disclose how often it happened.
    boundary_floor = max(16.0 * torch.finfo(parameters.dtype).eps, 1e-12)
    clamped = ((knots <= 0.0) | (knots >= 1.0)).sum().item()
    return knots.clamp(min=boundary_floor, max=1.0 - boundary_floor), int(clamped)


@torch.no_grad()
def fit_park_dominant_points(
    points: torch.Tensor,
    *,
    mse_tolerance: float,
    max_internal_knots: int = 28,
    degree: int = 3,
    shape_weight: float = 0.8,
) -> ParkDominantPointResult:
    """Fit ordered samples with a Park--Lee DOM adaptation.

    The scan starts at zero internal knots and inserts one dominant point at a
    time.  LCM seeds satisfying ``kappa_i > mean(kappa)/4`` have priority in
    decreasing curvature order.  Once those are exhausted, the segment that
    contains the largest parameter-correspondence residual is refined at the
    point balancing the paper's curvature/length shape index (``r=0.8`` by
    default).  The first fit meeting the shared MSE tolerance is returned.
    """

    _validate_inputs(
        points,
        degree=degree,
        mse_tolerance=mse_tolerance,
        max_internal_knots=max_internal_knots,
        shape_weight=shape_weight,
    )
    observed = points.detach()
    parameters = chord_length_parameters(observed)
    curvatures = _discrete_curvatures(observed)
    lcm_indices = _local_curvature_maxima(curvatures)
    cumulative_curvature, cumulative_length = _cumulative_shape_measures(
        observed, parameters, curvatures
    )

    sample_count = int(observed.shape[0])
    effective_max_knots = min(max_internal_knots, sample_count - degree - 1)
    dominant_indices = [0, sample_count - 1]
    pending_seeds = lcm_indices.copy()
    selected_sources: list[str] = ["endpoint", "endpoint"]

    # A degree-p curve needs at least degree+1 control/dominant points.  Consume
    # significant seeds first, then use the shape-index rule to complete this
    # minimum set before the first least-squares solve.
    while len(dominant_indices) < degree + 1:
        if pending_seeds:
            new_index = pending_seeds.pop(0)
            source = "lcm_seed"
        else:
            new_index = _widest_shape_segment_split(
                sorted(dominant_indices),
                cumulative_curvature=cumulative_curvature,
                cumulative_length=cumulative_length,
                shape_weight=shape_weight,
            )
            source = "shape_seed_completion"
        if new_index is None or new_index in dominant_indices:
            raise RuntimeError("could not construct a degree-compatible dominant set")
        dominant_indices.append(new_index)
        selected_sources.append(source)
        dominant_indices.sort()

    scanned_counts: list[int] = []
    scanned_mse: list[float] = []
    scanned_max_residual: list[float] = []
    fit: BSplineLeastSquaresFit | None = None
    total_boundary_clamps = 0

    while True:
        knots, boundary_clamps = _knots_from_dominant_parameters(
            parameters, dominant_indices, degree
        )
        total_boundary_clamps += boundary_clamps
        fit = refit_bspline_control_points(
            parameters,
            observed,
            knots,
            degree=degree,
            smoothness_weight=0.0,
            control_ridge=0.0,
            interpolate_endpoints=True,
        )
        residuals = (fit.reconstructed_points - observed).norm(dim=-1)
        scanned_counts.append(int(knots.numel()))
        scanned_mse.append(float(fit.fit_mse))
        scanned_max_residual.append(float(residuals.max()))
        if float(fit.fit_mse) <= mse_tolerance or knots.numel() >= effective_max_knots:
            break

        if pending_seeds:
            new_index = pending_seeds.pop(0)
            source = "lcm_seed"
        else:
            new_index = _residual_segment_split(
                sorted(dominant_indices),
                residuals,
                cumulative_curvature=cumulative_curvature,
                cumulative_length=cumulative_length,
                shape_weight=shape_weight,
            )
            source = "adaptive_shape_split"
        if new_index is None or new_index in dominant_indices:
            break
        dominant_indices.append(new_index)
        selected_sources.append(source)
        dominant_indices.sort()

    assert fit is not None
    final_dominants = torch.tensor(dominant_indices, dtype=torch.long)
    seed_indices = torch.tensor([0, sample_count - 1, *lcm_indices], dtype=torch.long)
    diagnostics: dict[str, object] = {
        "network_used": False,
        "method": "park_lee_dom_2007_adaptation",
        "reference_doi": "10.1016/j.cad.2006.12.006",
        "parameterization": "chord_length",
        "knot_placement": (
            "Park-Lee Eq. (4): each interior knot averages degree consecutive "
            "dominant-point parameters"
        ),
        "lcm_lower_bound": float(curvatures.mean()) / 4.0,
        "lcm_seed_count": len(lcm_indices),
        "shape_weight_r": float(shape_weight),
        "shape_index": "r*K_s,e/K_0,m + (1-r)*L_s,e/L_0,m",
        "curvature_integral": "trapezoidal in chord parameter",
        "curvature_source": (
            "generalized local Menger curvature; no optional smoothed base curve"
        ),
        "refinement_region": (
            "segment containing largest sampled parameter-correspondence residual"
        ),
        "refinement_point": "argmin_w |lambda_s,w - lambda_w,e|",
        "reported_refit": (
            "unregularized standard B-spline least squares with exact endpoint interpolation"
        ),
        "stopping_metric": "mean squared Euclidean parameter-correspondence error",
        "distance_metric_deviation": (
            "comparison adaptation stops on common MSE; it does not reproduce the "
            "paper's maximum-distance or iterative orthogonal-distance criterion"
        ),
        "seed_completion_deviation": (
            "shape-index splits complete degree+1 dominant points when too few LCM seeds exist"
        ),
        "effective_max_internal_knots": effective_max_knots,
        "requested_max_internal_knots": max_internal_knots,
        "selected_sources": tuple(selected_sources),
        "boundary_knot_clamp_count": total_boundary_clamps,
        "threshold_satisfied": float(fit.fit_mse) <= mse_tolerance,
    }
    return ParkDominantPointResult(
        mse_tolerance=float(mse_tolerance),
        parameters=parameters.detach().clone(),
        knots=fit.internal_knots.detach().clone(),
        dominant_indices=final_dominants,
        seed_indices=seed_indices,
        final_fit=fit,
        refit_count=len(scanned_counts),
        scanned_knot_counts=tuple(scanned_counts),
        scanned_mse=tuple(scanned_mse),
        scanned_max_parameter_residual=tuple(scanned_max_residual),
        diagnostics=diagnostics,
    )


__all__ = ["ParkDominantPointResult", "fit_park_dominant_points"]
