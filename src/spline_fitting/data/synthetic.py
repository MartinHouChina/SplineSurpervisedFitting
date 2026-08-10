from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CubicBSplineSample:
    """One synthetic open cubic B-spline and its sampled point cloud.

    Ground-truth parameters supervise ParameterHead; internal knots supervise
    query positions and existence. Control points and the full knot vector are
    retained for evaluation and visualization.
    """

    points: torch.Tensor
    parameters: torch.Tensor
    control_points: torch.Tensor
    knot_vector: torch.Tensor
    degree: int


def build_open_clamped_knot_vector(
    num_control_points: int,
    degree: int = 3,
    *,
    nonuniformity: float = 0.65,
    min_span: float = 0.02,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a legal open clamped knot vector on ``[0, 1]``.

    ``nonuniformity=0`` produces uniform knot spans. Larger values mix in
    random positive spans while preserving a minimum span length.
    """
    if degree < 1:
        raise ValueError("degree must be positive")
    if num_control_points < degree + 1:
        raise ValueError(
            f"num_control_points must be at least degree + 1 ({degree + 1})"
        )
    if not 0.0 <= nonuniformity <= 1.0:
        raise ValueError("nonuniformity must lie in [0, 1]")

    num_internal = num_control_points - degree - 1
    num_spans = num_internal + 1
    if min_span * num_spans >= 1.0:
        raise ValueError("min_span is too large for the requested knot count")

    uniform = torch.full((num_spans,), 1.0 / num_spans, dtype=dtype)
    random_weights = torch.rand(num_spans, generator=generator, dtype=dtype).clamp_min(1e-6)
    random_weights = random_weights / random_weights.sum()
    mixed = (1.0 - nonuniformity) * uniform + nonuniformity * random_weights

    # Reserve a fixed minimum width for every span, then distribute the rest.
    remaining = 1.0 - min_span * num_spans
    spans = min_span + remaining * mixed
    spans = spans / spans.sum()
    internal = torch.cumsum(spans, dim=0)[:-1]

    return torch.cat(
        [
            torch.zeros(degree + 1, dtype=dtype),
            internal,
            torch.ones(degree + 1, dtype=dtype),
        ]
    )


def bspline_basis_matrix(
    parameters: torch.Tensor,
    knot_vector: torch.Tensor,
    degree: int,
    num_control_points: int,
) -> torch.Tensor:
    """Evaluate all B-spline basis functions with Cox--de Boor recursion."""
    if parameters.ndim != 1:
        raise ValueError("parameters must have shape [M]")
    if knot_vector.ndim != 1:
        raise ValueError("knot_vector must have shape [L]")
    expected_length = num_control_points + degree + 1
    if knot_vector.numel() != expected_length:
        raise ValueError(
            f"Expected knot vector length {expected_length}, got {knot_vector.numel()}"
        )

    t = parameters.unsqueeze(-1)
    left = knot_vector[:-1]
    right = knot_vector[1:]
    basis = ((t >= left) & (t < right)).to(parameters.dtype)

    # Recursion decreases the number of available basis functions by one.
    for order in range(1, degree + 1):
        count = knot_vector.numel() - order - 1
        left_den = knot_vector[order : order + count] - knot_vector[:count]
        right_den = (
            knot_vector[order + 1 : order + 1 + count]
            - knot_vector[1 : 1 + count]
        )

        left_num = t - knot_vector[:count]
        right_num = knot_vector[order + 1 : order + 1 + count] - t

        left_term = torch.where(
            left_den.abs() > 1e-12,
            left_num / left_den.clamp_min(1e-12) * basis[:, :count],
            torch.zeros_like(basis[:, :count]),
        )
        right_term = torch.where(
            right_den.abs() > 1e-12,
            right_num / right_den.clamp_min(1e-12) * basis[:, 1 : count + 1],
            torch.zeros_like(basis[:, :count]),
        )
        basis = left_term + right_term

    basis = basis[:, :num_control_points]

    # The half-open interval definition excludes t=1. Enforce the standard
    # endpoint convention for an open clamped B-spline.
    endpoint_mask = parameters >= 1.0 - 1e-7
    if endpoint_mask.any():
        basis = basis.clone()
        basis[endpoint_mask] = 0.0
        basis[endpoint_mask, -1] = 1.0
    return basis


def evaluate_bspline_curve(
    parameters: torch.Tensor,
    control_points: torch.Tensor,
    knot_vector: torch.Tensor,
    degree: int = 3,
) -> torch.Tensor:
    """Evaluate an open B-spline curve at ordered parameter values."""
    basis = bspline_basis_matrix(
        parameters,
        knot_vector,
        degree,
        num_control_points=control_points.shape[0],
    )
    return basis @ control_points


def fit_control_points_for_internal_knots(
    parameters: torch.Tensor,
    points: torch.Tensor,
    internal_knots: torch.Tensor,
    *,
    degree: int = 3,
    ridge: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refit control points and return ``(control_points, RMS distance)``."""
    knot_vector = torch.cat(
        [
            torch.zeros(
                degree + 1, device=points.device, dtype=points.dtype
            ),
            internal_knots.to(device=points.device, dtype=points.dtype),
            torch.ones(
                degree + 1, device=points.device, dtype=points.dtype
            ),
        ]
    )
    num_control_points = int(internal_knots.numel()) + degree + 1
    basis = bspline_basis_matrix(
        parameters,
        knot_vector,
        degree,
        num_control_points,
    )
    normal = basis.transpose(0, 1) @ basis
    normal = normal + ridge * torch.eye(
        num_control_points, device=points.device, dtype=points.dtype
    )
    right_hand_side = basis.transpose(0, 1) @ points
    control_points = torch.linalg.solve(normal, right_hand_side)
    reconstructed = basis @ control_points
    rms_distance = (
        (reconstructed - points).pow(2).sum(dim=-1).mean().sqrt()
    )
    return control_points, rms_distance


def canonicalize_internal_knots(
    parameters: torch.Tensor,
    points: torch.Tensor,
    internal_knots: torch.Tensor,
    *,
    degree: int = 3,
    error_tolerance: float = 5e-3,
    ridge: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedily remove redundant knots under a geometric RMS tolerance.

    The returned representation is a deterministic, parsimonious target for
    supervised learning.  It describes the smallest representation reached by
    greedy single-knot removal, rather than the arbitrary representation used
    by the random curve generator.
    """
    if error_tolerance < 0.0:
        raise ValueError("error_tolerance must be non-negative")
    retained = torch.sort(internal_knots.detach().clone()).values
    control_points, rms_distance = fit_control_points_for_internal_knots(
        parameters,
        points,
        retained,
        degree=degree,
        ridge=ridge,
    )

    while retained.numel() > 0:
        best_index = -1
        best_rms: torch.Tensor | None = None
        best_control: torch.Tensor | None = None
        for index in range(retained.numel()):
            candidate = torch.cat([retained[:index], retained[index + 1 :]])
            candidate_control, candidate_rms = fit_control_points_for_internal_knots(
                parameters,
                points,
                candidate,
                degree=degree,
                ridge=ridge,
            )
            if best_rms is None or bool(candidate_rms < best_rms):
                best_index = index
                best_rms = candidate_rms
                best_control = candidate_control

        if best_rms is None or float(best_rms) > error_tolerance:
            break
        retained = torch.cat([retained[:best_index], retained[best_index + 1 :]])
        control_points = best_control
        rms_distance = best_rms

    if control_points is None:
        raise RuntimeError("canonical knot refit did not produce control points")
    return retained, control_points, rms_distance


def _random_unit_vector(
    dimension: int,
    generator: torch.Generator | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    vector = torch.randn(dimension, generator=generator, dtype=dtype)
    return vector / vector.norm().clamp_min(1e-8)


def generate_control_polygon(
    num_control_points: int,
    point_dim: int,
    *,
    turn_strength: float = 0.45,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate an ordered, smooth random control polygon.

    Successive edge directions are correlated. This produces free-form open
    curves that are more representative of industrial spline geometry than a
    fixed analytic sine function, while avoiding a purely chaotic random walk.
    """
    if point_dim not in (2, 3):
        raise ValueError("point_dim must be 2 or 3")
    if num_control_points < 4:
        raise ValueError("A cubic B-spline needs at least four control points")

    direction = _random_unit_vector(point_dim, generator, dtype)
    point = torch.zeros(point_dim, dtype=dtype)
    control_points = [point]

    for _ in range(num_control_points - 1):
        random_direction = _random_unit_vector(point_dim, generator, dtype)
        direction = (1.0 - turn_strength) * direction + turn_strength * random_direction
        direction = direction / direction.norm().clamp_min(1e-8)
        step = 0.75 + 0.55 * torch.rand((), generator=generator, dtype=dtype)
        point = point + step * direction
        control_points.append(point)

    control = torch.stack(control_points, dim=0)

    # Add a mild anisotropic affine deformation to enlarge shape diversity.
    scales = 0.65 + 0.8 * torch.rand(point_dim, generator=generator, dtype=dtype)
    control = control * scales
    control = control - control.mean(dim=0, keepdim=True)
    scale = control.norm(dim=-1).amax().clamp_min(1e-8)
    return control / scale


def generate_sampling_parameters(
    num_points: int,
    *,
    nonuniformity: float = 0.45,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate ordered parameters with optional non-uniform sampling density."""
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if not 0.0 <= nonuniformity <= 1.0:
        raise ValueError("nonuniformity must lie in [0, 1]")

    num_gaps = num_points - 1
    uniform = torch.full((num_gaps,), 1.0 / num_gaps, dtype=dtype)
    random_gaps = torch.rand(num_gaps, generator=generator, dtype=dtype).clamp_min(1e-5)
    random_gaps = random_gaps / random_gaps.sum()
    gaps = (1.0 - nonuniformity) * uniform + nonuniformity * random_gaps
    parameters = torch.cat(
        [torch.zeros(1, dtype=dtype), torch.cumsum(gaps, dim=0)]
    )
    parameters[-1] = 1.0
    return parameters


def generate_cubic_bspline_sample(
    num_points: int = 64,
    point_dim: int = 2,
    min_control_points: int = 5,
    max_control_points: int = 10,
    noise_std: float = 0.001,
    knot_nonuniformity: float = 0.65,
    sampling_nonuniformity: float = 0.45,
    turn_strength: float = 0.45,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> CubicBSplineSample:
    """Generate sampled points from a random open cubic B-spline curve."""
    degree = 3
    if min_control_points < degree + 1:
        raise ValueError("min_control_points must be at least 4 for cubic splines")
    if max_control_points < min_control_points:
        raise ValueError("max_control_points must be >= min_control_points")

    num_control_points = int(
        torch.randint(
            min_control_points,
            max_control_points + 1,
            (1,),
            generator=generator,
        ).item()
    )
    control_points = generate_control_polygon(
        num_control_points,
        point_dim,
        turn_strength=turn_strength,
        generator=generator,
        dtype=dtype,
    )
    knot_vector = build_open_clamped_knot_vector(
        num_control_points,
        degree,
        nonuniformity=knot_nonuniformity,
        generator=generator,
        dtype=dtype,
    )
    parameters = generate_sampling_parameters(
        num_points,
        nonuniformity=sampling_nonuniformity,
        generator=generator,
        dtype=dtype,
    )
    points = evaluate_bspline_curve(parameters, control_points, knot_vector, degree)

    if noise_std > 0.0:
        noise = torch.randn(points.shape, generator=generator, dtype=dtype)
        points = points + noise_std * noise

    return CubicBSplineSample(
        points=points,
        parameters=parameters,
        control_points=control_points,
        knot_vector=knot_vector,
        degree=degree,
    )


def generate_synthetic_curve(
    num_points: int = 64,
    point_dim: int = 2,
    noise_std: float = 0.001,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Backward-compatible wrapper returning cubic B-spline sample points."""
    return generate_cubic_bspline_sample(
        num_points=num_points,
        point_dim=point_dim,
        noise_std=noise_std,
        generator=generator,
    ).points


class SyntheticCubicBSplineDataset(Dataset):
    """Deterministic-on-index dataset of sampled open cubic B-splines.

    Ground-truth parameters, knot vectors and control points are padded and
    returned with every sample. Training uses ``true_params`` and true internal
    knots; control points and the full knot vector support diagnostics.
    """

    def __init__(
        self,
        size: int = 1000,
        num_points: int = 64,
        point_dim: int = 2,
        min_control_points: int = 5,
        max_control_points: int = 10,
        noise_std: float = 0.001,
        knot_nonuniformity: float = 0.65,
        sampling_nonuniformity: float = 0.45,
        turn_strength: float = 0.45,
        seed: int = 42,
        normalize: bool = True,
        return_ground_truth: bool = True,
        canonical_knot_tolerance: float = 5e-3,
        cache_samples: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if max_control_points < min_control_points:
            raise ValueError("max_control_points must be >= min_control_points")
        self.size = size
        self.num_points = num_points
        self.point_dim = point_dim
        self.min_control_points = min_control_points
        self.max_control_points = max_control_points
        self.noise_std = noise_std
        self.knot_nonuniformity = knot_nonuniformity
        self.sampling_nonuniformity = sampling_nonuniformity
        self.turn_strength = turn_strength
        self.seed = seed
        self.normalize = normalize
        self.return_ground_truth = return_ground_truth
        if canonical_knot_tolerance < 0.0:
            raise ValueError("canonical_knot_tolerance must be non-negative")
        self.canonical_knot_tolerance = canonical_knot_tolerance
        self.cache_samples = bool(cache_samples)
        self._sample_cache: dict[int, dict[str, torch.Tensor | int]] = {}
        self.dtype = dtype
        self.degree = 3
        self.max_knot_vector_length = max_control_points + self.degree + 1
        self.max_internal_knots = max_control_points - self.degree - 1

    def __len__(self) -> int:
        return self.size

    @staticmethod
    def _chord_length_parameters(points: torch.Tensor) -> torch.Tensor:
        segment_lengths = (points[1:] - points[:-1]).norm(dim=-1)
        total = segment_lengths.sum().clamp_min(1e-8)
        return torch.cat(
            [
                torch.zeros(1, dtype=points.dtype),
                torch.cumsum(segment_lengths / total, dim=0),
            ]
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        if index < 0 or index >= self.size:
            raise IndexError(index)
        if self.cache_samples and index in self._sample_cache:
            return self._sample_cache[index]
        generator = torch.Generator().manual_seed(self.seed + index)
        sample = generate_cubic_bspline_sample(
            num_points=self.num_points,
            point_dim=self.point_dim,
            min_control_points=self.min_control_points,
            max_control_points=self.max_control_points,
            noise_std=self.noise_std,
            knot_nonuniformity=self.knot_nonuniformity,
            sampling_nonuniformity=self.sampling_nonuniformity,
            turn_strength=self.turn_strength,
            generator=generator,
            dtype=self.dtype,
        )

        center = sample.points.mean(dim=0)
        centered = sample.points - center
        scale = centered.norm(dim=-1).amax().clamp_min(1e-8)
        points = centered / scale if self.normalize else sample.points

        result: dict[str, torch.Tensor | int] = {
            "points": points,
            "chord_params": self._chord_length_parameters(points),
            "center": center,
            "scale": scale,
            "sample_id": index,
            "curve_degree": sample.degree,
        }

        if not self.return_ground_truth:
            return result

        source_internal = sample.knot_vector[
            sample.degree + 1 : -(sample.degree + 1)
        ]
        internal, control_points, canonical_fit_rms = canonicalize_internal_knots(
            sample.parameters,
            points,
            source_internal,
            degree=sample.degree,
            error_tolerance=self.canonical_knot_tolerance,
        )
        num_control_points = control_points.shape[0]
        padded_control = torch.zeros(
            self.max_control_points,
            self.point_dim,
            dtype=self.dtype,
        )
        control_mask = torch.zeros(self.max_control_points, dtype=torch.bool)
        padded_control[:num_control_points] = control_points
        control_mask[:num_control_points] = True

        padded_knots = torch.ones(self.max_knot_vector_length, dtype=self.dtype)
        knot_mask = torch.zeros(self.max_knot_vector_length, dtype=torch.bool)
        canonical_knot_vector = torch.cat(
            [
                torch.zeros(sample.degree + 1, dtype=self.dtype),
                internal,
                torch.ones(sample.degree + 1, dtype=self.dtype),
            ]
        )
        knot_length = canonical_knot_vector.numel()
        padded_knots[:knot_length] = canonical_knot_vector
        knot_mask[:knot_length] = True

        padded_internal = torch.zeros(self.max_internal_knots, dtype=self.dtype)
        internal_mask = torch.zeros(self.max_internal_knots, dtype=torch.bool)
        padded_internal[: internal.numel()] = internal
        internal_mask[: internal.numel()] = True

        result.update(
            {
                "true_params": sample.parameters,
                "true_control_points": padded_control,
                "true_control_mask": control_mask,
                "true_knot_vector": padded_knots,
                "true_knot_mask": knot_mask,
                "true_internal_knots": padded_internal,
                "true_internal_knot_mask": internal_mask,
                "num_control_points": num_control_points,
                "source_num_control_points": sample.control_points.shape[0],
                "source_internal_knot_count": source_internal.numel(),
                "canonical_fit_rms": canonical_fit_rms,
            }
        )
        if self.cache_samples:
            self._sample_cache[index] = result
        return result


# Backward-compatible name used by the original training scripts.
SyntheticCurveDataset = SyntheticCubicBSplineDataset
