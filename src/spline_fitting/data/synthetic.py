from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from ..spline.bspline_deletion_teacher import single_knot_deletion_rmse_batch


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


@dataclass(frozen=True)
class SourceKnotMinimalityCertificate:
    """Numerical certificate for a clean source knot set.

    The certificate is deliberately scoped to subsets of ``internal_knots``.
    With unregularized least squares, every proper subset is contained in at
    least one of the one-knot-deletion spline spaces.  Consequently, if the
    full fit satisfies the threshold and every one-knot deletion violates a
    guarded threshold, no proper subset of the source knots can satisfy it.

    This does *not* certify the global optimum over arbitrary relocated knot
    positions.  That harder continuous problem remains outside the dataset
    generator's claim.
    """

    certified: bool
    full_fit_rms: torch.Tensor
    single_deletion_rms: torch.Tensor
    error_tolerance: float
    margin: float

    @property
    def required_single_deletion_rms(self) -> float:
        return self.error_tolerance * (1.0 + self.margin)

    @property
    def minimum_single_deletion_rms(self) -> torch.Tensor:
        if self.single_deletion_rms.numel() == 0:
            return self.full_fit_rms.new_tensor(float("inf"))
        return self.single_deletion_rms.min()


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
    random_weights = torch.rand(num_spans, generator=generator, dtype=dtype).clamp_min(
        1e-6
    )
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
            knot_vector[order + 1 : order + 1 + count] - knot_vector[1 : 1 + count]
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
    smoothness_weight: float = 1e-6,
    ridge: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Endpoint-constrained B-spline refit used by canonical label pruning.

    This intentionally uses a least-squares factorization instead of normal
    equations.  The same endpoint convention is used by v7 deployment, so a
    training deletion and a deployed deletion are judged on the same curve
    domain rather than on two subtly different solvers.
    """
    if smoothness_weight < 0.0 or ridge < 0.0:
        raise ValueError("smoothness_weight and ridge must be non-negative")
    knot_vector = torch.cat(
        [
            torch.zeros(degree + 1, device=points.device, dtype=points.dtype),
            internal_knots.to(device=points.device, dtype=points.dtype),
            torch.ones(degree + 1, device=points.device, dtype=points.dtype),
        ]
    )
    num_control_points = int(internal_knots.numel()) + degree + 1
    basis = bspline_basis_matrix(
        parameters,
        knot_vector,
        degree,
        num_control_points,
    )
    control_points = points.new_zeros(num_control_points, points.shape[-1])
    control_points[0] = points[0]
    control_points[-1] = points[-1]
    if num_control_points > 2:
        interior_basis = basis[:, 1:-1]
        right_hand_side = (
            points
            - basis[:, :1] * control_points[:1]
            - basis[:, -1:] * control_points[-1:]
        )
        if smoothness_weight > 0.0:
            difference = points.new_zeros(
                num_control_points - 2,
                num_control_points,
            )
            row = torch.arange(num_control_points - 2, device=points.device)
            difference[row, row] = 1.0
            difference[row, row + 1] = -2.0
            difference[row, row + 2] = 1.0
            fixed_control = torch.stack([control_points[0], control_points[-1]], dim=0)
            fixed_difference = torch.stack(
                [difference[:, 0], difference[:, -1]], dim=-1
            )
            smooth_scale = smoothness_weight**0.5
            interior_basis = torch.cat(
                [interior_basis, smooth_scale * difference[:, 1:-1]],
                dim=0,
            )
            right_hand_side = torch.cat(
                [
                    right_hand_side,
                    -smooth_scale * (fixed_difference @ fixed_control),
                ],
                dim=0,
            )
        if ridge > 0.0:
            ridge_rows = ridge**0.5 * torch.eye(
                num_control_points - 2,
                device=points.device,
                dtype=points.dtype,
            )
            interior_basis = torch.cat([interior_basis, ridge_rows], dim=0)
            right_hand_side = torch.cat(
                [
                    right_hand_side,
                    points.new_zeros(num_control_points - 2, points.shape[-1]),
                ],
                dim=0,
            )
        control_points[1:-1] = torch.linalg.lstsq(
            interior_basis,
            right_hand_side,
        ).solution
    reconstructed = basis @ control_points
    rms_distance = (reconstructed - points).pow(2).sum(dim=-1).mean().sqrt()
    return control_points, rms_distance


@torch.no_grad()
def certify_source_knot_minimality(
    parameters: torch.Tensor,
    clean_points: torch.Tensor,
    internal_knots: torch.Tensor,
    *,
    degree: int = 3,
    error_tolerance: float = 5e-3,
    margin: float = 0.2,
) -> SourceKnotMinimalityCertificate:
    """Certify threshold minimality within subsets of a source knot set.

    Labels are checked against a *clean* curve and unregularized endpoint-
    constrained least squares.  A positive margin rejects samples that sit on
    the decision boundary and would otherwise flip labels under small amounts
    of observation noise or floating-point variation.

    The result is a cardinality certificate only in the finite family formed
    by subsets of ``internal_knots``.  It must not be reported as a proof over
    arbitrary continuous knot relocation.
    """
    if parameters.ndim != 1:
        raise ValueError("parameters must have shape [M]")
    if clean_points.ndim != 2 or clean_points.shape[0] != parameters.shape[0]:
        raise ValueError("clean_points must have shape [M, D]")
    if internal_knots.ndim != 1:
        raise ValueError("internal_knots must have shape [K]")
    if error_tolerance <= 0.0:
        raise ValueError("error_tolerance must be positive")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")

    # Certificates are generated offline, so use float64 even when the
    # training tensors are float32.  This avoids accepting a boundary case due
    # to a low-precision least-squares solve.
    certificate_parameters = parameters.detach().to(dtype=torch.float64)
    certificate_points = clean_points.detach().to(dtype=torch.float64)
    certificate_knots = torch.sort(
        internal_knots.detach().to(dtype=torch.float64)
    ).values
    _, full_fit_rms = fit_control_points_for_internal_knots(
        certificate_parameters,
        certificate_points,
        certificate_knots,
        degree=degree,
        smoothness_weight=0.0,
        ridge=0.0,
    )
    if certificate_knots.numel() == 0:
        deletion_rms = certificate_knots.new_empty(0)
    else:
        deletion_rms = single_knot_deletion_rmse_batch(
            certificate_parameters.unsqueeze(0),
            certificate_points.unsqueeze(0),
            certificate_knots.unsqueeze(0),
            degree=degree,
            smoothness_weight=0.0,
            control_ridge=0.0,
            interpolate_endpoints=True,
        )[0]

    required_deletion_rms = error_tolerance * (1.0 + margin)
    full_fit_is_feasible = float(full_fit_rms) <= error_tolerance
    every_deletion_is_infeasible = deletion_rms.numel() == 0 or bool(
        torch.all(deletion_rms > required_deletion_rms)
    )
    return SourceKnotMinimalityCertificate(
        certified=full_fit_is_feasible and every_deletion_is_infeasible,
        full_fit_rms=full_fit_rms,
        single_deletion_rms=deletion_rms,
        error_tolerance=float(error_tolerance),
        margin=float(margin),
    )


def canonicalize_internal_knots(
    parameters: torch.Tensor,
    points: torch.Tensor,
    internal_knots: torch.Tensor,
    *,
    degree: int = 3,
    error_tolerance: float = 5e-3,
    smoothness_weight: float = 1e-6,
    ridge: float = 0.0,
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
        smoothness_weight=smoothness_weight,
        ridge=ridge,
    )
    # A zero tolerance is the explicit "preserve source representation" mode.
    # Avoid evaluating every possible single-knot deletion when the caller has
    # requested no canonical reduction. This matters for Kmax=20 datasets.
    if error_tolerance == 0.0:
        return retained, control_points, rms_distance

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
                smoothness_weight=smoothness_weight,
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


def generate_complexity_aligned_control_polygon(
    num_control_points: int,
    point_dim: int,
    *,
    oscillation_amplitude: float = 0.3,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a curve whose requested knot count carries geometric signal.

    The legacy correlated random walk becomes smoother as its control count
    grows after normalization.  It can therefore assign a large source knot
    count to a curve that is accurately represented by far fewer knots.  This
    construction combines monotone progress with alternating transverse
    detail, then applies a random orthogonal transform.  Each added local span
    contributes visible geometry, while the final certificate remains the
    authority that decides whether the sample is accepted.
    """
    if point_dim not in (2, 3):
        raise ValueError("point_dim must be 2 or 3")
    if num_control_points < 4:
        raise ValueError("A cubic B-spline needs at least four control points")
    if oscillation_amplitude <= 0.0:
        raise ValueError("oscillation_amplitude must be positive")

    coordinates = torch.zeros(num_control_points, point_dim, dtype=dtype)
    coordinates[:, 0] = torch.linspace(-1.0, 1.0, num_control_points, dtype=dtype)
    phase = int(torch.randint(0, 2, (1,), generator=generator).item())
    alternating = torch.where(
        (torch.arange(num_control_points) + phase) % 2 == 0,
        torch.ones(num_control_points, dtype=dtype),
        -torch.ones(num_control_points, dtype=dtype),
    )
    amplitude_jitter = 0.75 + 0.5 * torch.rand(
        num_control_points,
        generator=generator,
        dtype=dtype,
    )
    coordinates[:, 1] = oscillation_amplitude * alternating * amplitude_jitter
    if point_dim == 3:
        # A phase-shifted, lower-amplitude component avoids making every 3-D
        # sample planar without destroying the strong local detail certificate.
        shifted = torch.roll(alternating, shifts=1)
        depth_jitter = 0.5 + torch.rand(
            num_control_points,
            generator=generator,
            dtype=dtype,
        )
        coordinates[:, 2] = 0.45 * oscillation_amplitude * shifted * depth_jitter

    random_frame = torch.randn(
        point_dim,
        point_dim,
        generator=generator,
        dtype=dtype,
    )
    orthogonal, _ = torch.linalg.qr(random_frame)
    control = coordinates @ orthogonal.transpose(0, 1)
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
    parameters = torch.cat([torch.zeros(1, dtype=dtype), torch.cumsum(gaps, dim=0)])
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
    control_polygon_mode: str = "smooth_random_walk",
    oscillation_amplitude: float = 0.3,
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
    if control_polygon_mode == "smooth_random_walk":
        control_points = generate_control_polygon(
            num_control_points,
            point_dim,
            turn_strength=turn_strength,
            generator=generator,
            dtype=dtype,
        )
    elif control_polygon_mode == "complexity_aligned":
        control_points = generate_complexity_aligned_control_polygon(
            num_control_points,
            point_dim,
            oscillation_amplitude=oscillation_amplitude,
            generator=generator,
            dtype=dtype,
        )
    else:
        raise ValueError(
            "control_polygon_mode must be 'smooth_random_walk' or 'complexity_aligned'"
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
        certified_minimal_source: bool = False,
        minimality_margin: float = 0.2,
        minimality_max_attempts: int = 16,
        minimality_audit_points: int = 0,
        oscillation_amplitude: float = 0.3,
        seed: int = 42,
        normalize: bool = True,
        return_ground_truth: bool = True,
        canonical_knot_tolerance: float = 5e-3,
        cache_samples: bool = True,
        resample_each_epoch: bool = False,
        epoch_seed_stride: int = 1_000_003,
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
        self.certified_minimal_source = bool(certified_minimal_source)
        if minimality_margin < 0.0:
            raise ValueError("minimality_margin must be non-negative")
        self.minimality_margin = float(minimality_margin)
        if minimality_max_attempts < 1:
            raise ValueError("minimality_max_attempts must be positive")
        self.minimality_max_attempts = int(minimality_max_attempts)
        if minimality_audit_points not in (0,) and minimality_audit_points < 2:
            raise ValueError("minimality_audit_points must be 0 or at least 2")
        self.minimality_audit_points = int(minimality_audit_points)
        if oscillation_amplitude <= 0.0:
            raise ValueError("oscillation_amplitude must be positive")
        self.oscillation_amplitude = float(oscillation_amplitude)
        self.seed = seed
        self.normalize = normalize
        self.return_ground_truth = return_ground_truth
        if canonical_knot_tolerance < 0.0:
            raise ValueError("canonical_knot_tolerance must be non-negative")
        if self.certified_minimal_source and canonical_knot_tolerance <= 0.0:
            raise ValueError(
                "certified_minimal_source requires a positive canonical_knot_tolerance"
            )
        self.canonical_knot_tolerance = canonical_knot_tolerance
        self.cache_samples = bool(cache_samples)
        self.resample_each_epoch = bool(resample_each_epoch)
        if epoch_seed_stride <= 0:
            raise ValueError("epoch_seed_stride must be positive")
        self.epoch_seed_stride = int(epoch_seed_stride)
        self.epoch = 0
        self._sample_cache: dict[int, dict[str, torch.Tensor | int]] = {}
        self.dtype = dtype
        self.degree = 3
        self.max_knot_vector_length = max_control_points + self.degree + 1
        self.max_internal_knots = max_control_points - self.degree - 1

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic fresh training population for an epoch."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        next_epoch = int(epoch) if self.resample_each_epoch else 0
        if next_epoch != self.epoch:
            self.epoch = next_epoch
            self._sample_cache.clear()

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
        sample_seed = self.seed + self.epoch * self.epoch_seed_stride + index
        generator = torch.Generator().manual_seed(sample_seed)
        minimality_certificate: SourceKnotMinimalityCertificate | None = None
        generation_attempts = 1
        if self.certified_minimal_source:
            # Draw K once so rejection cannot bias the requested count
            # distribution toward easier, smaller curves.
            source_control_count = int(
                torch.randint(
                    self.min_control_points,
                    self.max_control_points + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            for generation_attempts in range(1, self.minimality_max_attempts + 1):
                sample = generate_cubic_bspline_sample(
                    num_points=self.num_points,
                    point_dim=self.point_dim,
                    min_control_points=source_control_count,
                    max_control_points=source_control_count,
                    noise_std=0.0,
                    knot_nonuniformity=self.knot_nonuniformity,
                    sampling_nonuniformity=self.sampling_nonuniformity,
                    turn_strength=self.turn_strength,
                    control_polygon_mode="complexity_aligned",
                    oscillation_amplitude=self.oscillation_amplitude,
                    generator=generator,
                    dtype=self.dtype,
                )
                clean_center = sample.points.mean(dim=0)
                clean_centered = sample.points - clean_center
                clean_scale = clean_centered.norm(dim=-1).amax().clamp_min(1e-8)
                source_internal_for_certificate = sample.knot_vector[
                    sample.degree + 1 : -(sample.degree + 1)
                ]
                if self.minimality_audit_points:
                    audit_parameters = torch.linspace(
                        0.0,
                        1.0,
                        self.minimality_audit_points,
                        dtype=self.dtype,
                    )
                    audit_points = evaluate_bspline_curve(
                        audit_parameters,
                        sample.control_points,
                        sample.knot_vector,
                        sample.degree,
                    )
                    if self.normalize:
                        audit_points = (audit_points - clean_center) / clean_scale
                else:
                    audit_parameters = sample.parameters
                    audit_points = (
                        clean_centered / clean_scale
                        if self.normalize
                        else sample.points
                    )
                minimality_certificate = certify_source_knot_minimality(
                    audit_parameters,
                    audit_points,
                    source_internal_for_certificate,
                    degree=sample.degree,
                    error_tolerance=self.canonical_knot_tolerance,
                    margin=self.minimality_margin,
                )
                if minimality_certificate.certified:
                    break
            else:
                raise RuntimeError(
                    "failed to generate a certified-minimal source curve after "
                    f"{self.minimality_max_attempts} attempts for sample {index}; "
                    "increase oscillation_amplitude or minimality_max_attempts"
                )

            center = clean_center
            scale = clean_scale
            clean_points = clean_centered / scale if self.normalize else sample.points
            if self.noise_std > 0.0:
                observation_noise = torch.randn(
                    sample.points.shape,
                    generator=generator,
                    dtype=self.dtype,
                )
                observed_points = sample.points + self.noise_std * observation_noise
            else:
                observed_points = sample.points.detach().clone()
            points = (
                (observed_points - center) / scale
                if self.normalize
                else observed_points
            )
        else:
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
            clean_points = points

        result: dict[str, torch.Tensor | int] = {
            "points": points,
            "chord_params": self._chord_length_parameters(points),
            "center": center,
            "scale": scale,
            "sample_id": index,
            "sample_epoch": self.epoch,
            "curve_degree": sample.degree,
        }
        if self.certified_minimal_source:
            if minimality_certificate is None:
                raise RuntimeError("minimality certificate was not generated")
            result.update(
                {
                    "clean_points": clean_points,
                    "source_minimality_certified": minimality_certificate.certified,
                    "source_full_fit_rms": minimality_certificate.full_fit_rms.to(
                        dtype=self.dtype
                    ),
                    "source_min_single_deletion_rms": (
                        minimality_certificate.minimum_single_deletion_rms.to(
                            dtype=self.dtype
                        )
                    ),
                    "source_minimality_required_rms": torch.tensor(
                        minimality_certificate.required_single_deletion_rms,
                        dtype=self.dtype,
                    ),
                    "source_generation_attempts": generation_attempts,
                }
            )

        if not self.return_ground_truth:
            return result

        source_internal = sample.knot_vector[sample.degree + 1 : -(sample.degree + 1)]
        source_control_points = (
            (sample.control_points - center) / scale
            if self.normalize
            else sample.control_points.detach().clone()
        )
        source_control_count = source_control_points.shape[0]
        padded_source_control = torch.zeros(
            self.max_control_points,
            self.point_dim,
            dtype=self.dtype,
        )
        source_control_mask = torch.zeros(
            self.max_control_points,
            dtype=torch.bool,
        )
        padded_source_control[:source_control_count] = source_control_points
        source_control_mask[:source_control_count] = True
        padded_source_knots = torch.ones(
            self.max_knot_vector_length,
            dtype=self.dtype,
        )
        source_knot_mask = torch.zeros(
            self.max_knot_vector_length,
            dtype=torch.bool,
        )
        source_knot_length = sample.knot_vector.numel()
        padded_source_knots[:source_knot_length] = sample.knot_vector
        source_knot_mask[:source_knot_length] = True
        if self.certified_minimal_source:
            if minimality_certificate is None:
                raise RuntimeError("minimality certificate was not generated")
            # The source itself is the clean, threshold-minimal label in the
            # certified subset domain.  Observation noise belongs only to the
            # network input and is never allowed to change K or the knot set.
            internal = source_internal.detach().clone()
            control_points = source_control_points.detach().clone()
            canonical_fit_rms = minimality_certificate.full_fit_rms.to(dtype=self.dtype)
            _, observation_fit_rms = fit_control_points_for_internal_knots(
                sample.parameters,
                points,
                internal,
                degree=sample.degree,
                smoothness_weight=1e-6,
                ridge=0.0,
            )
            result["source_observation_fit_rms"] = observation_fit_rms
        elif self.canonical_knot_tolerance == 0.0:
            internal = source_internal.detach().clone()
            control_points = source_control_points.detach().clone()
            source_reconstruction = evaluate_bspline_curve(
                sample.parameters,
                control_points,
                sample.knot_vector,
                sample.degree,
            )
            canonical_fit_rms = (
                (source_reconstruction - points).pow(2).sum(dim=-1).mean().sqrt()
            )
        else:
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
                "source_control_points": padded_source_control,
                "source_control_mask": source_control_mask,
                "source_knot_vector": padded_source_knots,
                "source_knot_mask": source_knot_mask,
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
