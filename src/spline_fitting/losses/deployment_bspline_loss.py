from __future__ import annotations

import torch

from ..data.synthetic import bspline_basis_matrix
from ..evaluation.bspline_inference import second_difference_matrix
from ..evaluation.knot_diagnostics import build_open_knot_vector


def differentiable_hard_gated_bspline_fit(
    parameters: torch.Tensor,
    points: torch.Tensor,
    candidate_knots: torch.Tensor,
    hard_keep_mask: torch.Tensor,
    *,
    degree: int = 3,
    smoothness_weight: float = 1e-6,
    control_ridge: float = 0.0,
    solver_jitter: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Differentiate through the B-spline refit used after hard selection.

    The production refitter intentionally runs under ``torch.no_grad`` and uses
    an SVD-capable least-squares backend.  Training needs the same B-spline
    basis, endpoint constraints and regularizers *with* gradients, so this
    helper solves an augmented, full-column-rank least-squares problem.  A tiny
    ridge row keeps the backward solve well-defined without changing the hard
    mask or the one-shot deployment path.

    Candidate counts are ragged after hard selection; the small per-curve loop
    is confined to the final calibration stage and is never used by
    ``forward_deployment``.
    """

    if parameters.ndim != 2:
        raise ValueError("parameters must have shape [B,M]")
    if points.ndim != 3:
        raise ValueError("points must have shape [B,M,D]")
    if candidate_knots.ndim != 2 or hard_keep_mask.ndim != 2:
        raise ValueError("candidate knots and keep mask must have shape [B,K]")
    if parameters.shape != points.shape[:2]:
        raise ValueError("parameters and points must share [B,M]")
    if candidate_knots.shape != hard_keep_mask.shape:
        raise ValueError("candidate knots and keep mask must share shape")
    if candidate_knots.shape[0] != points.shape[0]:
        raise ValueError("all inputs must share the batch dimension")
    if degree < 1:
        raise ValueError("degree must be positive")
    if smoothness_weight < 0.0 or control_ridge < 0.0 or solver_jitter < 0.0:
        raise ValueError("refit regularization values must be non-negative")

    mask = hard_keep_mask.to(torch.bool)
    per_sample_mse: list[torch.Tensor] = []
    per_sample_coordinate_mse: list[torch.Tensor] = []
    retained_counts: list[torch.Tensor] = []

    for batch_index in range(points.shape[0]):
        sample_points = points[batch_index]
        sample_parameters = parameters[batch_index]
        internal_knots = torch.sort(
            candidate_knots[batch_index, mask[batch_index]]
        ).values
        knot_vector = build_open_knot_vector(internal_knots, degree)
        control_count = int(internal_knots.numel()) + degree + 1
        basis = bspline_basis_matrix(
            sample_parameters,
            knot_vector,
            degree,
            control_count,
        )

        # Open-clamped endpoint interpolation matches deployment: P0=Q0 and
        # Plast=Qlast.  Only the interior control points are solved.
        fixed_controls = torch.stack(
            [sample_points[0], sample_points[-1]], dim=0
        )
        fixed_columns = torch.stack([basis[:, 0], basis[:, -1]], dim=-1)
        interior_basis = basis[:, 1:-1]
        design_blocks = [interior_basis]
        target_blocks = [sample_points - fixed_columns @ fixed_controls]

        difference = second_difference_matrix(
            control_count,
            device=points.device,
            dtype=points.dtype,
        )
        if smoothness_weight > 0.0 and difference.shape[0] > 0:
            fixed_difference = torch.stack(
                [difference[:, 0], difference[:, -1]], dim=-1
            )
            scale = smoothness_weight**0.5
            design_blocks.append(scale * difference[:, 1:-1])
            target_blocks.append(-scale * (fixed_difference @ fixed_controls))

        interior_count = control_count - 2
        if control_ridge > 0.0:
            scale = control_ridge**0.5
            design_blocks.append(
                scale
                * torch.eye(
                    interior_count,
                    device=points.device,
                    dtype=points.dtype,
                )
            )
            target_blocks.append(
                torch.zeros(
                    (interior_count, points.shape[-1]),
                    device=points.device,
                    dtype=points.dtype,
                )
            )
        if solver_jitter > 0.0:
            # This row is solely a numerical guard for autograd.  It is much
            # smaller than the deployment smoothness term by default.
            scale = solver_jitter**0.5
            design_blocks.append(
                scale
                * torch.eye(
                    interior_count,
                    device=points.device,
                    dtype=points.dtype,
                )
            )
            target_blocks.append(
                torch.zeros(
                    (interior_count, points.shape[-1]),
                    device=points.device,
                    dtype=points.dtype,
                )
            )

        augmented_design = torch.cat(design_blocks, dim=0)
        augmented_target = torch.cat(target_blocks, dim=0)
        # The regularization rows make the matrix full-column-rank.  ``gels``
        # is differentiable on both CPU and CUDA and avoids normal equations.
        interior_controls = torch.linalg.lstsq(
            augmented_design,
            augmented_target,
            driver="gels",
        ).solution
        controls = torch.cat(
            [fixed_controls[:1], interior_controls, fixed_controls[1:]], dim=0
        )
        reconstructed = basis @ controls
        squared = (reconstructed - sample_points).square()
        per_sample_mse.append(squared.sum(dim=-1).mean())
        per_sample_coordinate_mse.append(squared.mean())
        retained_counts.append(
            points.new_tensor(float(internal_knots.numel()))
        )

    mse = torch.stack(per_sample_mse)
    coordinate_mse = torch.stack(per_sample_coordinate_mse)
    return {
        "per_sample_mse": mse,
        "fit_mse": mse.mean(),
        "per_sample_rms": mse.clamp_min(0.0).sqrt(),
        "coordinate_mse": coordinate_mse.mean(),
        "retained_count": torch.stack(retained_counts),
    }
