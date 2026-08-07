from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ..data.synthetic import bspline_basis_matrix
from .knot_diagnostics import build_open_knot_vector, point_fit_statistics


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _validate_single_curve_inputs(
    parameters: torch.Tensor,
    points: torch.Tensor,
    internal_knots: torch.Tensor,
    degree: int,
) -> None:
    if parameters.ndim != 1:
        raise ValueError("parameters must have shape [M]")
    if points.ndim != 2:
        raise ValueError("points must have shape [M, D]")
    if internal_knots.ndim != 1:
        raise ValueError("internal_knots must have shape [K]")
    if parameters.shape[0] != points.shape[0]:
        raise ValueError("parameters and points must share the same point count")
    if degree < 1:
        raise ValueError("degree must be positive")
    if not parameters.is_floating_point() or not points.is_floating_point():
        raise ValueError("parameters and points must be floating-point tensors")
    if parameters.device != points.device or internal_knots.device != points.device:
        raise ValueError("parameters, points and internal_knots must share a device")
    if parameters.dtype != points.dtype or internal_knots.dtype != points.dtype:
        raise ValueError("parameters, points and internal_knots must share a dtype")
    if not (
        torch.isfinite(parameters).all()
        and torch.isfinite(points).all()
        and torch.isfinite(internal_knots).all()
    ):
        raise ValueError("parameters, points and internal_knots must be finite")
    if torch.any((parameters < 0.0) | (parameters > 1.0)):
        raise ValueError("parameters must lie in [0, 1]")
    if parameters.numel() > 1 and torch.any(parameters[1:] < parameters[:-1]):
        raise ValueError("parameters must be non-decreasing")
    if torch.any((internal_knots <= 0.0) | (internal_knots >= 1.0)):
        raise ValueError("internal knots must lie strictly inside (0, 1)")
    if internal_knots.numel() > 1 and torch.any(
        internal_knots[1:] < internal_knots[:-1]
    ):
        raise ValueError("internal knots must be non-decreasing")


def second_difference_matrix(
    num_control_points: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the control-polygon second-difference operator ``D2``."""
    if num_control_points < 1:
        raise ValueError("num_control_points must be positive")
    if num_control_points < 3:
        return torch.empty(
            (0, num_control_points),
            device=device,
            dtype=dtype,
        )

    result = torch.zeros(
        (num_control_points - 2, num_control_points),
        device=device,
        dtype=dtype,
    )
    row = torch.arange(num_control_points - 2, device=device)
    result[row, row] = 1.0
    result[row, row + 1] = -2.0
    result[row, row + 2] = 1.0
    return result


@dataclass(frozen=True)
class BSplineLeastSquaresFit:
    """A standard open-clamped B-spline after control-point refitting."""

    degree: int
    internal_knots: torch.Tensor
    knot_vector: torch.Tensor
    basis_matrix: torch.Tensor
    control_points: torch.Tensor
    reconstructed_points: torch.Tensor
    fit_mse: torch.Tensor
    fit_rmse: torch.Tensor
    coordinate_mse: torch.Tensor
    coordinate_rmse: torch.Tensor
    data_squared_error: torch.Tensor
    smoothness_squared: torch.Tensor
    control_squared: torch.Tensor
    augmented_objective: torch.Tensor
    solver_rank: int | None

    def evaluate(self, parameters: torch.Tensor) -> torch.Tensor:
        """Evaluate the refitted curve at new parameter values."""
        basis = bspline_basis_matrix(
            parameters,
            self.knot_vector,
            self.degree,
            num_control_points=self.control_points.shape[0],
        )
        return basis @ self.control_points


@torch.no_grad()
def refit_bspline_control_points(
    parameters: torch.Tensor,
    points: torch.Tensor,
    internal_knots: torch.Tensor,
    *,
    degree: int = 3,
    smoothness_weight: float = 1e-6,
    control_ridge: float = 0.0,
    rcond: float | None = None,
) -> BSplineLeastSquaresFit:
    """Refit a standard B-spline without forming normal equations.

    The control points ``P`` solve the augmented least-squares problem

    ``min ||B P - Q||_F^2 + lambda_s ||D2 P||_F^2
         + lambda_r ||P||_F^2``.

    On CPU, PyTorch's pivoted-QR least-squares driver (``gelsy``) is used.
    Other devices use their supported ``torch.linalg.lstsq`` backend.  This is
    deliberately more stable than solving ``(B.T @ B) P = B.T @ Q``.
    """
    _validate_non_negative("smoothness_weight", smoothness_weight)
    _validate_non_negative("control_ridge", control_ridge)
    if rcond is not None and rcond < 0.0:
        raise ValueError("rcond must be non-negative or None")
    _validate_single_curve_inputs(parameters, points, internal_knots, degree)

    num_control_points = int(internal_knots.numel()) + degree + 1
    knot_vector = build_open_knot_vector(internal_knots, degree)
    basis = bspline_basis_matrix(
        parameters,
        knot_vector,
        degree,
        num_control_points,
    )
    difference = second_difference_matrix(
        num_control_points,
        device=points.device,
        dtype=points.dtype,
    )

    design_blocks = [basis]
    target_blocks = [points]
    if smoothness_weight > 0.0 and difference.shape[0] > 0:
        design_blocks.append(smoothness_weight**0.5 * difference)
        target_blocks.append(
            torch.zeros(
                (difference.shape[0], points.shape[1]),
                device=points.device,
                dtype=points.dtype,
            )
        )
    if control_ridge > 0.0:
        identity = torch.eye(
            num_control_points,
            device=points.device,
            dtype=points.dtype,
        )
        design_blocks.append(control_ridge**0.5 * identity)
        target_blocks.append(
            torch.zeros(
                (num_control_points, points.shape[1]),
                device=points.device,
                dtype=points.dtype,
            )
        )

    augmented_design = torch.cat(design_blocks, dim=0)
    augmented_target = torch.cat(target_blocks, dim=0)
    lstsq_kwargs: dict[str, object] = {"rcond": rcond}
    if augmented_design.device.type == "cpu":
        # gelsy uses a rank-revealing complete orthogonal factorization with
        # column pivoting and is robust to nearly coincident retained knots.
        lstsq_kwargs["driver"] = "gelsy"
    solution = torch.linalg.lstsq(
        augmented_design,
        augmented_target,
        **lstsq_kwargs,
    )
    control_points = solution.solution
    reconstructed = basis @ control_points
    statistics = point_fit_statistics(
        reconstructed.unsqueeze(0),
        points.unsqueeze(0),
    )

    data_squared_error = (reconstructed - points).pow(2).sum()
    smoothness_squared = (difference @ control_points).pow(2).sum()
    control_squared = control_points.pow(2).sum()
    augmented_objective = (
        data_squared_error
        + smoothness_weight * smoothness_squared
        + control_ridge * control_squared
    )
    solver_rank = (
        int(solution.rank.item()) if solution.rank.numel() == 1 else None
    )

    return BSplineLeastSquaresFit(
        degree=degree,
        internal_knots=internal_knots.detach().clone(),
        knot_vector=knot_vector.detach().clone(),
        basis_matrix=basis.detach().clone(),
        control_points=control_points.detach().clone(),
        reconstructed_points=reconstructed.detach().clone(),
        fit_mse=statistics["fit_mse"][0].detach().clone(),
        fit_rmse=statistics["fit_rmse"][0].detach().clone(),
        coordinate_mse=statistics["coordinate_mse"][0].detach().clone(),
        coordinate_rmse=statistics["coordinate_rmse"][0].detach().clone(),
        data_squared_error=data_squared_error.detach().clone(),
        smoothness_squared=smoothness_squared.detach().clone(),
        control_squared=control_squared.detach().clone(),
        augmented_objective=augmented_objective.detach().clone(),
        solver_rank=solver_rank,
    )


def hard_gate_mask(
    hard_gate: torch.Tensor,
    *,
    binary_tolerance: float = 1e-6,
) -> torch.Tensor:
    """Convert a deterministic Hard-Concrete gate to a Boolean keep mask.

    The deployment API intentionally accepts only binary gates.  Passing the
    stochastic/soft training gate, or ``P(z > 0)``, is an error; callers must
    run the model in evaluation mode first.  This prevents an inference-time
    threshold sweep from silently changing the selected spline structure.
    """
    _validate_non_negative("binary_tolerance", binary_tolerance)
    if hard_gate.ndim != 1:
        raise ValueError("hard_gate must have shape [K]")
    if hard_gate.dtype == torch.bool:
        return hard_gate.detach().clone()
    if not hard_gate.is_floating_point():
        raise ValueError("hard_gate must be Boolean or floating point")
    if not torch.isfinite(hard_gate).all():
        raise ValueError("hard_gate must contain only finite values")
    close_to_zero = torch.isclose(
        hard_gate,
        torch.zeros((), device=hard_gate.device, dtype=hard_gate.dtype),
        atol=binary_tolerance,
        rtol=0.0,
    )
    close_to_one = torch.isclose(
        hard_gate,
        torch.ones((), device=hard_gate.device, dtype=hard_gate.dtype),
        atol=binary_tolerance,
        rtol=0.0,
    )
    if not torch.all(close_to_zero | close_to_one):
        raise ValueError(
            "hard_gate must be binary; call model.eval() and pass the "
            "deterministic activity_gate, not l0_probability"
        )
    return close_to_one


@dataclass(frozen=True)
class HardGatedBSplineFit:
    """Per-sample deployment result after physical candidate-knot deletion."""

    sample_index: int
    candidate_count: int
    retained_count: int
    retained_mask: torch.Tensor
    hard_gate: torch.Tensor
    spline: BSplineLeastSquaresFit

    @property
    def retained_internal_knots(self) -> torch.Tensor:
        return self.spline.internal_knots

    @property
    def knot_vector(self) -> torch.Tensor:
        return self.spline.knot_vector

    @property
    def control_points(self) -> torch.Tensor:
        return self.spline.control_points

    @property
    def reconstructed_points(self) -> torch.Tensor:
        return self.spline.reconstructed_points

    @property
    def fit_mse(self) -> torch.Tensor:
        return self.spline.fit_mse

    @property
    def fit_rmse(self) -> torch.Tensor:
        return self.spline.fit_rmse


@torch.no_grad()
def refit_hard_gated_bspline_batch(
    parameters: torch.Tensor,
    candidate_knots: torch.Tensor,
    hard_gates: torch.Tensor,
    points: torch.Tensor,
    *,
    degree: int = 3,
    smoothness_weight: float = 1e-6,
    control_ridge: float = 0.0,
    rcond: float | None = None,
    binary_tolerance: float = 1e-6,
) -> list[HardGatedBSplineFit]:
    """Delete hard-gated knots and refit variable-size standard B-splines.

    Returning a list is intentional: a batch can retain different numbers of
    knots, hence its knot vectors and control-point matrices have different
    lengths.  ``K=0`` is legal and produces an open cubic Bezier representation
    with four control points when ``degree=3``.
    """
    if parameters.ndim != 2:
        raise ValueError("parameters must have shape [B, M]")
    if candidate_knots.ndim != 2 or hard_gates.ndim != 2:
        raise ValueError("candidate_knots and hard_gates must have shape [B, K]")
    if points.ndim != 3:
        raise ValueError("points must have shape [B, M, D]")
    if candidate_knots.shape != hard_gates.shape:
        raise ValueError("candidate_knots and hard_gates must share shape [B, K]")
    if parameters.shape[0] != points.shape[0] or parameters.shape[0] != hard_gates.shape[0]:
        raise ValueError("all inputs must share the same batch size")
    if parameters.shape[1] != points.shape[1]:
        raise ValueError("parameters and points must share the same point count")
    if hard_gates.device != candidate_knots.device:
        raise ValueError("candidate_knots and hard_gates must share a device")
    if parameters.shape[0] == 0:
        return []

    results: list[HardGatedBSplineFit] = []
    candidate_count = int(candidate_knots.shape[1])
    for index in range(parameters.shape[0]):
        retained_mask = hard_gate_mask(
            hard_gates[index],
            binary_tolerance=binary_tolerance,
        )
        retained_knots = candidate_knots[index, retained_mask]
        spline = refit_bspline_control_points(
            parameters[index],
            points[index],
            retained_knots,
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
            rcond=rcond,
        )
        results.append(
            HardGatedBSplineFit(
                sample_index=index,
                candidate_count=candidate_count,
                retained_count=int(retained_mask.sum().item()),
                retained_mask=retained_mask.detach().clone(),
                hard_gate=hard_gates[index].detach().clone(),
                spline=spline,
            )
        )
    return results


@torch.no_grad()
def refit_model_output_as_bsplines(
    output: Mapping[str, torch.Tensor],
    points: torch.Tensor,
    *,
    degree: int = 3,
    smoothness_weight: float = 1e-6,
    control_ridge: float = 0.0,
    rcond: float | None = None,
    binary_tolerance: float = 1e-6,
) -> list[HardGatedBSplineFit]:
    """Deployment adapter for ``SplineFittingNetwork.eval()`` output."""
    required = ("params", "internal_knots", "activity_gate")
    missing = [name for name in required if name not in output]
    if missing:
        raise KeyError("model output is missing: " + ", ".join(missing))
    return refit_hard_gated_bspline_batch(
        parameters=output["params"],
        candidate_knots=output["internal_knots"],
        hard_gates=output["activity_gate"],
        points=points,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        rcond=rcond,
        binary_tolerance=binary_tolerance,
    )
