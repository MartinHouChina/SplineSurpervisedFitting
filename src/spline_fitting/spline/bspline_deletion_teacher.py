from __future__ import annotations

import math

import torch


def _validate_inputs(
    parameters: torch.Tensor,
    points: torch.Tensor,
    internal_knots: torch.Tensor,
    degree: int,
    smoothness_weight: float,
    control_ridge: float,
    rcond: float | None,
) -> None:
    if parameters.ndim != 2:
        raise ValueError("parameters must have shape [B, M]")
    if points.ndim != 3:
        raise ValueError("points must have shape [B, M, D]")
    if internal_knots.ndim != 2:
        raise ValueError("internal_knots must have shape [B, K]")
    if parameters.shape[0] != points.shape[0] or parameters.shape[0] != internal_knots.shape[0]:
        raise ValueError("parameters, points and internal_knots must share a batch size")
    if parameters.shape[1] != points.shape[1]:
        raise ValueError("parameters and points must share a point count")
    if internal_knots.shape[1] < 1:
        raise ValueError("internal_knots must contain at least one candidate (K >= 1)")
    if parameters.shape[0] < 1:
        raise ValueError("batch size must be positive")
    if parameters.shape[1] < 1 or points.shape[2] < 1:
        raise ValueError("point count and point dimension must be positive")
    if degree < 1:
        raise ValueError("degree must be positive")
    if smoothness_weight < 0.0:
        raise ValueError("smoothness_weight must be non-negative")
    if control_ridge < 0.0:
        raise ValueError("control_ridge must be non-negative")
    if rcond is not None and rcond < 0.0:
        raise ValueError("rcond must be non-negative or None")
    if not parameters.is_floating_point() or not points.is_floating_point():
        raise ValueError("parameters and points must be floating-point tensors")
    if not internal_knots.is_floating_point():
        raise ValueError("internal_knots must be a floating-point tensor")
    if parameters.device != points.device or parameters.device != internal_knots.device:
        raise ValueError("all inputs must share a device")
    if parameters.dtype != points.dtype or parameters.dtype != internal_knots.dtype:
        raise ValueError("all inputs must share a dtype")
    if not (
        torch.isfinite(parameters).all()
        and torch.isfinite(points).all()
        and torch.isfinite(internal_knots).all()
    ):
        raise ValueError("all inputs must contain only finite values")
    if torch.any((parameters < 0.0) | (parameters > 1.0)):
        raise ValueError("parameters must lie in [0, 1]")
    if parameters.shape[1] > 1 and torch.any(parameters[:, 1:] < parameters[:, :-1]):
        raise ValueError("parameters must be non-decreasing")
    if torch.any((internal_knots <= 0.0) | (internal_knots >= 1.0)):
        raise ValueError("internal knots must lie strictly inside (0, 1)")
    if internal_knots.shape[1] > 1 and torch.any(
        internal_knots[:, 1:] < internal_knots[:, :-1]
    ):
        raise ValueError("internal knots must be non-decreasing")


def _all_single_deletions(internal_knots: torch.Tensor) -> torch.Tensor:
    """Return all K one-knot deletions as ``[B, K, K - 1]``."""
    batch_size, candidate_count = internal_knots.shape
    slot = torch.arange(candidate_count, device=internal_knots.device)
    keep = slot.unsqueeze(0) != slot.unsqueeze(1)
    retained_indices = slot.expand(candidate_count, -1)[keep].reshape(
        candidate_count, candidate_count - 1
    )
    return torch.gather(
        internal_knots[:, None, :].expand(-1, candidate_count, -1),
        dim=-1,
        index=retained_indices[None, :, :].expand(batch_size, -1, -1),
    )


def _batched_open_bspline_basis(
    parameters: torch.Tensor,
    retained_knots: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """Cox--de Boor basis for B*K fixed-size deletion states.

    ``parameters`` has shape ``[B, M]`` and ``retained_knots`` has shape
    ``[B, K, K - 1]``.  The result is ``[B, K, M, K + degree]``.
    """
    batch_size, candidate_count, _ = retained_knots.shape
    boundary = retained_knots.new_zeros(batch_size, candidate_count, degree + 1)
    knot_vector = torch.cat(
        [boundary, retained_knots, boundary + 1.0],
        dim=-1,
    )
    num_control_points = candidate_count + degree
    parameter_column = parameters[:, None, :, None]
    left = knot_vector[..., :-1]
    right = knot_vector[..., 1:]
    basis = (
        (parameter_column >= left[:, :, None, :])
        & (parameter_column < right[:, :, None, :])
    ).to(parameters.dtype)

    for order in range(1, degree + 1):
        count = knot_vector.shape[-1] - order - 1
        left_denominator = (
            knot_vector[..., order : order + count] - knot_vector[..., :count]
        )
        right_denominator = (
            knot_vector[..., order + 1 : order + 1 + count]
            - knot_vector[..., 1 : 1 + count]
        )
        left_numerator = parameter_column - knot_vector[..., None, :count]
        right_numerator = (
            knot_vector[..., None, order + 1 : order + 1 + count]
            - parameter_column
        )
        left_term = torch.where(
            left_denominator[..., None, :].abs() > 1e-12,
            left_numerator
            / left_denominator[..., None, :].clamp_min(1e-12)
            * basis[..., :count],
            torch.zeros_like(basis[..., :count]),
        )
        right_term = torch.where(
            right_denominator[..., None, :].abs() > 1e-12,
            right_numerator
            / right_denominator[..., None, :].clamp_min(1e-12)
            * basis[..., 1 : count + 1],
            torch.zeros_like(basis[..., :count]),
        )
        basis = left_term + right_term

    basis = basis[..., :num_control_points]
    endpoint = parameters[:, None, :, None] >= 1.0 - 1e-7
    endpoint_basis = torch.zeros_like(basis)
    endpoint_basis[..., -1] = 1.0
    return torch.where(endpoint, endpoint_basis, basis)


def _second_difference_matrix(
    num_control_points: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    difference = reference.new_zeros(num_control_points - 2, num_control_points)
    row = torch.arange(num_control_points - 2, device=reference.device)
    difference[row, row] = 1.0
    difference[row, row + 1] = -2.0
    difference[row, row + 2] = 1.0
    return difference


@torch.no_grad()
def single_knot_deletion_rmse_batch(
    parameters: torch.Tensor,
    points: torch.Tensor,
    internal_knots: torch.Tensor,
    *,
    degree: int = 3,
    smoothness_weight: float = 1e-6,
    control_ridge: float = 0.0,
    interpolate_endpoints: bool = True,
    rcond: float | None = None,
) -> torch.Tensor:
    """Compute the real refit RMS after deleting each candidate knot.

    The returned tensor has shape ``[B, K]``. Entry ``[b, j]`` is the mean
    Euclidean RMS obtained by physically deleting knot ``j`` and refitting a
    standard open-clamped B-spline to curve ``b``.  All ``B*K`` deletion
    states are evaluated together: Cox--de Boor basis construction and the
    augmented least-squares solve are both batched.

    The objective and numerical conventions intentionally match
    :func:`spline_fitting.evaluation.refit_bspline_control_points`, including
    its default endpoint interpolation and control-polygon smoothness weight.
    The function is a non-differentiable training-label teacher; gradients
    should be learned through a surrogate pruning head, not through this
    discrete deletion operation.
    """
    _validate_inputs(
        parameters,
        points,
        internal_knots,
        degree,
        smoothness_weight,
        control_ridge,
        rcond,
    )
    batch_size, candidate_count = internal_knots.shape
    point_count, point_dimension = points.shape[1:]
    num_control_points = candidate_count + degree

    retained_knots = _all_single_deletions(internal_knots)
    basis = _batched_open_bspline_basis(parameters, retained_knots, degree)
    target = points[:, None, :, :].expand(-1, candidate_count, -1, -1)
    difference = _second_difference_matrix(num_control_points, points)

    if interpolate_endpoints:
        fixed_control_points = torch.stack([points[:, 0], points[:, -1]], dim=1)
        fixed_control_points = fixed_control_points[:, None, :, :].expand(
            -1, candidate_count, -1, -1
        )
        fixed_columns = torch.stack([basis[..., 0], basis[..., -1]], dim=-1)
        design_blocks = [basis[..., 1:-1]]
        target_blocks = [target - fixed_columns @ fixed_control_points]

        if smoothness_weight > 0.0:
            square_root_weight = math.sqrt(smoothness_weight)
            fixed_difference = torch.stack(
                [difference[:, 0], difference[:, -1]], dim=-1
            )
            smooth_design = (
                square_root_weight
                * difference[:, 1:-1][None, None, :, :].expand(
                    batch_size, candidate_count, -1, -1
                )
            )
            smooth_target = -square_root_weight * (
                fixed_difference[None, None, :, :] @ fixed_control_points
            )
            design_blocks.append(smooth_design)
            target_blocks.append(smooth_target)
        if control_ridge > 0.0:
            interior_count = num_control_points - 2
            ridge_design = (
                math.sqrt(control_ridge)
                * torch.eye(
                    interior_count,
                    device=points.device,
                    dtype=points.dtype,
                )[None, None, :, :].expand(batch_size, candidate_count, -1, -1)
            )
            design_blocks.append(ridge_design)
            target_blocks.append(
                points.new_zeros(
                    batch_size,
                    candidate_count,
                    interior_count,
                    point_dimension,
                )
            )
    else:
        fixed_control_points = None
        design_blocks = [basis]
        target_blocks = [target]
        if smoothness_weight > 0.0:
            smooth_design = (
                math.sqrt(smoothness_weight)
                * difference[None, None, :, :].expand(
                    batch_size, candidate_count, -1, -1
                )
            )
            design_blocks.append(smooth_design)
            target_blocks.append(
                points.new_zeros(
                    batch_size,
                    candidate_count,
                    difference.shape[0],
                    point_dimension,
                )
            )
        if control_ridge > 0.0:
            ridge_design = (
                math.sqrt(control_ridge)
                * torch.eye(
                    num_control_points,
                    device=points.device,
                    dtype=points.dtype,
                )[None, None, :, :].expand(batch_size, candidate_count, -1, -1)
            )
            design_blocks.append(ridge_design)
            target_blocks.append(
                points.new_zeros(
                    batch_size,
                    candidate_count,
                    num_control_points,
                    point_dimension,
                )
            )

    augmented_design = torch.cat(design_blocks, dim=-2)
    augmented_target = torch.cat(target_blocks, dim=-2)
    lstsq_kwargs: dict[str, object] = {"rcond": rcond}
    if augmented_design.device.type == "cpu":
        lstsq_kwargs["driver"] = "gelsy"
    solution = torch.linalg.lstsq(
        augmented_design,
        augmented_target,
        **lstsq_kwargs,
    ).solution

    if fixed_control_points is None:
        control_points = solution
    else:
        control_points = torch.cat(
            [fixed_control_points[..., :1, :], solution, fixed_control_points[..., 1:, :]],
            dim=-2,
        )
    reconstructed = basis @ control_points
    return (
        (reconstructed - target)
        .pow(2)
        .sum(dim=-1)
        .mean(dim=-1)
        .sqrt()
    )


__all__ = ["single_knot_deletion_rmse_batch"]
