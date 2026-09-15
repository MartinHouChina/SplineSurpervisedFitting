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
    if (
        parameters.shape[0] != points.shape[0]
        or parameters.shape[0] != internal_knots.shape[0]
    ):
        raise ValueError(
            "parameters, points and internal_knots must share a batch size"
        )
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
            knot_vector[..., None, order + 1 : order + 1 + count] - parameter_column
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


def _is_cuda_rank_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "torch.linalg.lstsq" in message and (
        "full rank" in message or "rank deficient" in message
    )


def _cpu_svd_lstsq(
    design: torch.Tensor,
    target: torch.Tensor,
    *,
    rcond: float | None,
    output_device: torch.device,
) -> torch.Tensor:
    """Use the same rank-aware solver as the standard B-spline refit."""
    return torch.linalg.lstsq(
        design.to(device="cpu"),
        target.to(device="cpu"),
        rcond=rcond,
        driver="gelsd",
    ).solution.to(device=output_device)


def _replace_nonfinite_cuda_solutions(
    design: torch.Tensor,
    target: torch.Tensor,
    solution: torch.Tensor,
    *,
    rcond: float | None,
) -> torch.Tensor:
    """Retry only the states whose CUDA QR solution is not finite."""
    bad = ~torch.isfinite(solution).all(dim=(-2, -1))
    if not bool(bad.any()):
        return solution
    repaired = solution.clone()
    repaired[bad] = _cpu_svd_lstsq(
        design[bad], target[bad], rcond=rcond, output_device=design.device
    )
    return repaired


def _rank_aware_batched_lstsq(
    design: torch.Tensor,
    target: torch.Tensor,
    *,
    rcond: float | None,
) -> torch.Tensor:
    """Keep CUDA QR fast for full-rank states; use CPU SVD on singular states.

    CUDA ``lstsq`` supports only the full-rank QR driver. Dense knot proposals
    can leave B-spline columns entirely unobserved, so a single singular
    deletion state otherwise aborts the whole batch. The fallback works one
    source curve at a time to avoid moving unrelated states to the CPU.
    """
    if design.device.type == "cpu":
        return _cpu_svd_lstsq(
            design, target, rcond=rcond, output_device=design.device
        )

    try:
        solution = torch.linalg.lstsq(design, target, rcond=rcond).solution
    except RuntimeError as error:
        if not _is_cuda_rank_error(error):
            raise
    else:
        return _replace_nonfinite_cuda_solutions(
            design, target, solution, rcond=rcond
        )

    # A failed [B, K] call does not identify all singular states. Retry K
    # deletion states per source curve; only curves whose QR call fails move to
    # CPU SVD. In the common all-full-rank case there is just one GPU solve.
    curve_solutions = []
    for curve_design, curve_target in zip(design.unbind(0), target.unbind(0)):
        try:
            curve_solution = torch.linalg.lstsq(
                curve_design, curve_target, rcond=rcond
            ).solution
        except RuntimeError as error:
            if not _is_cuda_rank_error(error):
                raise
            curve_solution = _cpu_svd_lstsq(
                curve_design,
                curve_target,
                rcond=rcond,
                output_device=design.device,
            )
        else:
            curve_solution = _replace_nonfinite_cuda_solutions(
                curve_design, curve_target, curve_solution, rcond=rcond
            )
        curve_solutions.append(curve_solution)
    return torch.stack(curve_solutions, dim=0)


@torch.no_grad()
def single_knot_deletion_mse_batch(
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
    """Compute mean squared Euclidean refit error for every one-knot deletion.

    The returned tensor has shape ``[B, K]``. Entry ``[b, j]`` is the mean
    squared Euclidean error obtained by physically deleting knot ``j`` and
    refitting a standard open-clamped B-spline to curve ``b``.  All ``B*K`` deletion
    states are evaluated together: Cox--de Boor basis construction and the
    augmented least-squares solve are batched when the design has full rank.
    Singular CUDA states are retried with a rank-aware CPU SVD solver.

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
            smooth_design = square_root_weight * difference[:, 1:-1][
                None, None, :, :
            ].expand(batch_size, candidate_count, -1, -1)
            smooth_target = -square_root_weight * (
                fixed_difference[None, None, :, :] @ fixed_control_points
            )
            design_blocks.append(smooth_design)
            target_blocks.append(smooth_target)
        if control_ridge > 0.0:
            interior_count = num_control_points - 2
            ridge_design = math.sqrt(control_ridge) * torch.eye(
                interior_count,
                device=points.device,
                dtype=points.dtype,
            )[None, None, :, :].expand(batch_size, candidate_count, -1, -1)
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
            smooth_design = math.sqrt(smoothness_weight) * difference[
                None, None, :, :
            ].expand(batch_size, candidate_count, -1, -1)
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
            ridge_design = math.sqrt(control_ridge) * torch.eye(
                num_control_points,
                device=points.device,
                dtype=points.dtype,
            )[None, None, :, :].expand(batch_size, candidate_count, -1, -1)
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
    solution = _rank_aware_batched_lstsq(
        augmented_design, augmented_target, rcond=rcond
    )

    if fixed_control_points is None:
        control_points = solution
    else:
        control_points = torch.cat(
            [
                fixed_control_points[..., :1, :],
                solution,
                fixed_control_points[..., 1:, :],
            ],
            dim=-2,
        )
    reconstructed = basis @ control_points
    mse = (reconstructed - target).pow(2).sum(dim=-1).mean(dim=-1)
    if augmented_design.device.type == "cuda" and not bool(torch.isfinite(mse).all()):
        # A CUDA QR call can also return finite but explosive coefficients,
        # producing an invalid reconstructed error without raising. Retry the
        # affected deletion states with the rank-aware deployment solver.
        bad = ~torch.isfinite(mse)
        robust_solution = _cpu_svd_lstsq(
            augmented_design[bad],
            augmented_target[bad],
            rcond=rcond,
            output_device=augmented_design.device,
        )
        if fixed_control_points is None:
            robust_controls = robust_solution
        else:
            robust_controls = torch.cat(
                [
                    fixed_control_points[bad, :1, :],
                    robust_solution,
                    fixed_control_points[bad, 1:, :],
                ],
                dim=-2,
            )
        robust_reconstruction = basis[bad] @ robust_controls
        mse = mse.clone()
        mse[bad] = (
            (robust_reconstruction - target[bad]).pow(2).sum(dim=-1).mean(dim=-1)
        )
    if not bool(torch.isfinite(mse).all()):
        raise RuntimeError("one-knot deletion refit produced non-finite MSE")
    return mse


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
    """Backward-compatible RMS view of one-knot deletion refit errors."""

    return single_knot_deletion_mse_batch(
        parameters,
        points,
        internal_knots,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        interpolate_endpoints=interpolate_endpoints,
        rcond=rcond,
    ).sqrt()


__all__ = [
    "single_knot_deletion_mse_batch",
    "single_knot_deletion_rmse_batch",
]
