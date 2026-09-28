"""Offline labels shared by the small fixed-Proposal V16 ablations.

The teacher never moves parameters/knots, inserts knots, or calls the subset
decoder.  It is deliberately separate from inference.  Only ``feasible`` rows
are training targets; a failed row's all-false mask is a sentinel, NOT a
zero-knot solution.  Keep failed rows in evaluation denominators.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ..evaluation.bspline_inference import refit_bspline_control_points
from ..spline.bspline_deletion_teacher import single_knot_deletion_mse_batch


def _validate_shapes(points: torch.Tensor, params: torch.Tensor,
                     knots: torch.Tensor, anchors: torch.Tensor) -> None:
    if points.ndim != 3 or params.ndim != 2 or knots.ndim != 2:
        raise ValueError("expected points [B,M,D], params [B,M], knots [B,K]")
    if params.shape != points.shape[:2] or knots.shape[0] != points.shape[0]:
        raise ValueError("geometry batch sizes and point counts must agree")
    if points.shape[1] < 2 or points.shape[2] < 1:
        raise ValueError("at least two points with positive dimension are required")
    if knots.shape[1] > 32:
        raise ValueError("the bounded ablation supports at most 32 candidate knots")
    if anchors.shape != knots.shape or anchors.dtype != torch.bool:
        raise ValueError("anchors must be boolean with shape [B,K]")
    if any(not tensor.is_floating_point() for tensor in (points, params, knots)):
        raise ValueError("points, params, and knots must be floating-point tensors")


def _valid_geometry(points: torch.Tensor, params: torch.Tensor,
                    knots: torch.Tensor, degree: int) -> bool:
    if not all(bool(torch.isfinite(value).all()) for value in (points, params, knots)):
        return False
    if float(params[0]) != 0.0 or float(params[-1]) != 1.0:
        return False
    if bool(((params < 0) | (params > 1)).any()) or bool((params.diff() < 0).any()):
        return False
    if bool(((knots <= 0) | (knots >= 1)).any()) or bool((knots.diff() < 0).any()):
        return False
    if knots.numel() and int(torch.unique_consecutive(knots, return_counts=True)[1].max()) > degree + 1:
        return False
    return True


def _refit_errors(points: torch.Tensor, params: torch.Tensor,
                  knots: torch.Tensor, degree: int) -> tuple[float, float]:
    fit = refit_bspline_control_points(
        params, points, knots, degree=degree, smoothness_weight=0.0,
        control_ridge=0.0, interpolate_endpoints=True,
    )
    squared = (fit.reconstructed_points - points).square().sum(dim=-1)
    if not bool(torch.isfinite(squared).all()):
        raise RuntimeError("production refit returned non-finite errors")
    return float(squared.mean()), float(squared.max())


def _deletion_order(points: torch.Tensor, params: torch.Tensor,
                    knots: torch.Tensor, degree: int,
                    mse_tolerance: float) -> tuple[list[int], bool]:
    """Use the batched teacher for ordering, never as a feasibility oracle.

    Its CPU gelsy solve can underestimate numerical rank; production gelsd
    verification must therefore also consider apparently infeasible entries.
    Round negligible ordering differences so exact-fit ties use slot order.
    """
    try:
        errors = single_knot_deletion_mse_batch(
            params.unsqueeze(0), points.unsqueeze(0), knots.unsqueeze(0),
            degree=degree, smoothness_weight=0.0, control_ridge=0.0,
            interpolate_endpoints=True,
        )[0].tolist()
    except (RuntimeError, ValueError):
        return list(range(knots.numel())), True
    tie_resolution = max(1e-14, mse_tolerance * 1e-9)
    def key(index: int) -> tuple[float, int]:
        error = errors[index]
        rank = round(error / tie_resolution) if math.isfinite(error) else math.inf
        return rank, index
    return sorted(range(len(errors)), key=key), False


@torch.no_grad()
def build_teacher_from_geometry(
    points: torch.Tensor,
    params: torch.Tensor,
    knots: torch.Tensor,
    *,
    anchors: torch.Tensor | None = None,
    min_selected_knots: int = 0,
    degree: int = 3,
    mse_tolerance: float = 5e-5,
    max_squared_error_tolerance: float = 5e-4,
    max_deletions: int = 32,
) -> list[dict[str, Any]]:
    """Greedily delete fixed candidates using two exact observed-error bounds.

    Inputs are batched; returned tensors are independent, detached CPU clones
    in their ORIGINAL dtype.  ``knots`` in each record retains all K original
    slots, and the Boolean ``mask`` identifies survivors.  Computation alone
    uses CPU float64.  ``steps`` counts accepted deletions, ``refits`` counts
    exact production-refit attempts (including dense), and ``ranking_calls``
    counts batched deletion solves.  No files or caches are written here.

    Structural misuse raises ValueError. Invalid/numerically failed curves
    instead produce explicit failure records.  Dense-infeasible curves are
    not assigned any positive mask label and are not searched further.
    """
    for name, tolerance in (("mse_tolerance", mse_tolerance),
                            ("max_squared_error_tolerance", max_squared_error_tolerance)):
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name, value, minimum in (("degree", degree, 1),
                                 ("min_selected_knots", min_selected_knots, 0),
                                 ("max_deletions", max_deletions, 0)):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if anchors is None:
        anchors = torch.zeros_like(knots, dtype=torch.bool)
    _validate_shapes(points, params, knots, anchors)
    if min_selected_knots > knots.shape[1]:
        raise ValueError("min_selected_knots cannot exceed candidate count")
    original_params = params.detach().cpu().clone()
    original_knots = knots.detach().cpu().clone()
    cpu_points = points.detach().to(device="cpu", dtype=torch.float64)
    cpu_params = original_params.to(torch.float64)
    cpu_knots = original_knots.to(torch.float64)
    cpu_anchors = anchors.detach().cpu().clone()
    records: list[dict[str, Any]] = []

    def passes(mse: float, maxse: float) -> bool:
        return mse <= mse_tolerance and maxse <= max_squared_error_tolerance

    for index in range(points.shape[0]):
        curve, parameter, candidate = cpu_points[index], cpu_params[index], cpu_knots[index]
        record: dict[str, Any] = {
            "mask": torch.zeros_like(candidate, dtype=torch.bool),
            "params": original_params[index].clone(),
            "knots": original_knots[index].clone(),
            "anchors": cpu_anchors[index].clone(),
            "feasible": False, "mse": math.inf, "maxse": math.inf, "count": 0,
            "dense_mse": math.inf, "dense_maxse": math.inf, "dense_pass": False,
            "steps": 0, "refits": 0, "ranking_calls": 0,
            "ranking_failures": 0, "refit_failures": 0, "termination": "invalid_geometry",
        }
        records.append(record)
        if not _valid_geometry(curve, parameter, candidate, degree):
            continue
        record["refits"] += 1
        try:
            mse, maxse = _refit_errors(curve, parameter, candidate, degree)
        except (RuntimeError, ValueError) as error:
            record.update(termination="dense_refit_failed", refit_failures=1, error=str(error))
            continue
        record.update(mse=mse, maxse=maxse, dense_mse=mse, dense_maxse=maxse,
                      dense_pass=passes(mse, maxse), termination="dense_infeasible")
        if not record["dense_pass"]:
            continue
        mask = torch.ones_like(candidate, dtype=torch.bool)
        record.update(mask=mask, count=int(mask.sum()), feasible=True)
        while True:
            if record["count"] <= min_selected_knots:
                record["termination"] = "minimum_count"
                break
            permitted = mask & ~cpu_anchors[index]
            if not bool(permitted.any()):
                record["termination"] = "anchors_only"
                break
            if record["steps"] >= max_deletions:
                record["termination"] = "deletion_budget"
                break
            retained_indices = mask.nonzero(as_tuple=False).flatten()
            record["ranking_calls"] += 1
            ordering, ranking_failed = _deletion_order(
                curve, parameter, candidate[mask], degree, mse_tolerance,
            )
            record["ranking_failures"] += int(ranking_failed)
            accepted = False
            # MaxSE need not be monotone across candidates: a failed first
            # deletion is not grounds to stop before testing the remaining ones.
            for local_index in ordering:
                deleted_index = int(retained_indices[local_index])
                if not bool(permitted[deleted_index]):
                    continue
                trial = mask.clone()
                trial[deleted_index] = False
                record["refits"] += 1
                try:
                    mse, maxse = _refit_errors(curve, parameter, candidate[trial], degree)
                except (RuntimeError, ValueError):
                    record["refit_failures"] += 1
                    continue
                if passes(mse, maxse):
                    mask = trial
                    record.update(mask=mask, count=int(mask.sum()), mse=mse, maxse=maxse)
                    record["steps"] += 1
                    accepted = True
                    break
            if not accepted:
                record["termination"] = "no_feasible_deletion"
                break
    return records


@torch.no_grad()
def build_fixed_teacher(
    model: torch.nn.Module,
    points: torch.Tensor,
    *,
    mse_tolerance: float = 5e-5,
    max_squared_error_tolerance: float = 5e-4,
    max_deletions: int = 32,
) -> list[dict[str, Any]]:
    """Encode once with the INITIAL model, then freeze geometry and anchors.

    The caller caches these records once for all arms; do not regenerate them
    from fine-tuned models. Only synthetic training/validation observations
    belong here, never real-test reference labels. Existing model train/eval
    flags are restored, including deliberately frozen submodules.
    """
    if points.ndim != 3 or points.shape[1] < 2 or points.shape[2] < 1:
        raise ValueError("points must have shape [B,M,D] with M >= 2, D >= 1")
    if not points.is_floating_point():
        raise ValueError("points must be floating point")
    if points.shape[0] == 0:
        return []
    candidate_count = int(model.max_internal_knots)
    if candidate_count > 32:
        raise ValueError("the bounded ablation supports at most 32 candidate knots")
    weight = next(model.parameters())
    model_points = points.detach().to(device=weight.device, dtype=weight.dtype)
    params = torch.full(points.shape[:2], math.nan, dtype=weight.dtype)
    knots = torch.full((points.shape[0], candidate_count), math.nan, dtype=weight.dtype)
    anchors = torch.zeros_like(knots, dtype=torch.bool)
    valid_indices = torch.isfinite(model_points).all(dim=-1).all(dim=-1).nonzero().flatten()
    mode_flags = [(module, module.training) for module in model.modules()]

    def encode(indices: torch.Tensor) -> None:
        context = model.encode_candidates(model_points[indices], mse_tolerance=mse_tolerance)
        cpu_indices = indices.cpu()
        params[cpu_indices] = context["proposal_params"].detach().cpu()
        knots[cpu_indices] = context["proposal_internal_knots"].detach().cpu()
        anchors[cpu_indices] = model._coverage_anchors(context).detach().cpu()

    proposal_errors: dict[int, str] = {}
    try:
        model.eval()
        if valid_indices.numel():
            try:
                encode(valid_indices)
            except RuntimeError:
                # A bad numerical row must not discard otherwise valid curves.
                for index in valid_indices:
                    try:
                        encode(index.reshape(1))
                    except RuntimeError as error:
                        proposal_errors[int(index)] = str(error)
    finally:
        for module, training in mode_flags:
            module.training = training
    records = build_teacher_from_geometry(
        points, params, knots, anchors=anchors,
        min_selected_knots=int(model.min_selected_knots), degree=int(model.degree),
        mse_tolerance=mse_tolerance,
        max_squared_error_tolerance=max_squared_error_tolerance,
        max_deletions=max_deletions,
    )
    for index, error in proposal_errors.items():
        records[index].update(termination="proposal_failed", error=error)
    return records


__all__ = ["build_fixed_teacher", "build_teacher_from_geometry"]
