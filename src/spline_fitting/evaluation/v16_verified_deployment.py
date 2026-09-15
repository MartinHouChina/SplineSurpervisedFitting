"""Optional, explicitly numerical MSE verification of a v16 one-shot subset.

The learned mask and its standard B-spline refit are always retained as the
unmodified comparison result.  Verification may add candidate knots and may
switch back to the proposal parameterization; neither operation is part of the
one-shot network.  No globally minimum knot count is claimed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping

import torch

from .bspline_inference import BSplineLeastSquaresFit, refit_bspline_control_points
from .knot_diagnostics import warp_internal_knots_to_parameterization
from .verified_knot_repair import VerifiedKnotRepairResult, verified_confidence_repair


@dataclass(frozen=True)
class V16VerifiedDeploymentComparison:
    """Separate raw one-shot and optional MSE-verified deployment measurements."""

    mse_tolerance: float
    raw_fit: BSplineLeastSquaresFit
    verified_fit: BSplineLeastSquaresFit
    raw_parameters: torch.Tensor
    verified_parameters: torch.Tensor
    raw_keep_mask: torch.Tensor
    verified_proposal_mask: torch.Tensor
    raw_pass: bool
    verified_pass: bool
    status: str
    verified_source: str
    verified_parameterization: str
    network_only_ms: float | None
    numerical_verification_ms: float
    direct_standard_refit_count: int
    # This counts extra direct CPU refit calls beyond the raw mask refit.
    # Batched pruning solves, when compact=True, live in fit_evaluation_count.
    additional_standard_refit_count: int
    fit_evaluation_count: int
    decoded_full_candidate_mse: float | None
    proposal_full_candidate_mse: float | None
    proposal_domain_fallback_attempted: bool

    @property
    def raw_k(self) -> int:
        return int(self.raw_fit.internal_knots.numel())

    @property
    def verified_k(self) -> int:
        return int(self.verified_fit.internal_knots.numel())

    @property
    def raw_mse(self) -> float:
        return float(self.raw_fit.fit_mse)

    @property
    def verified_mse(self) -> float:
        return float(self.verified_fit.fit_mse)

    @property
    def method_labels(self) -> tuple[str, str]:
        return ("ours_one_shot", "ours_mse_verified_numerical_repair")

    def diagnostics(self) -> dict:
        """Plain JSON-ready diagnostics for benchmark/visualization adapters."""

        return {
            "method_labels": list(self.method_labels),
            "mse_tolerance": self.mse_tolerance,
            "raw_one_shot_mse": self.raw_mse,
            "raw_one_shot_k": self.raw_k,
            "raw_one_shot_pass": self.raw_pass,
            "verified_mse": self.verified_mse,
            "verified_k": self.verified_k,
            "verified_pass": self.verified_pass,
            "verified_status": self.status,
            "verified_source": self.verified_source,
            "verified_parameterization": self.verified_parameterization,
            "network_only_ms": self.network_only_ms,
            "numerical_verification_ms": self.numerical_verification_ms,
            "direct_standard_refit_count": self.direct_standard_refit_count,
            "additional_standard_refit_count": self.additional_standard_refit_count,
            "fit_evaluation_count": self.fit_evaluation_count,
            "decoded_full_candidate_mse": self.decoded_full_candidate_mse,
            "proposal_full_candidate_mse": self.proposal_full_candidate_mse,
            "proposal_domain_fallback_attempted": self.proposal_domain_fallback_attempted,
            "globally_minimum_k_certified": False,
        }


def _one_curve_tensor(
    output: Mapping[str, torch.Tensor], key: str, sample_index: int
) -> torch.Tensor:
    value = output.get(key)
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        raise ValueError(f"v16 output {key!r} must be a batched tensor")
    if not 0 <= sample_index < value.shape[0]:
        raise ValueError("sample_index is outside the v16 output batch")
    return value[sample_index].detach().to(device="cpu")


def _check_numeric(name: str, value: float, *, strictly_positive: bool) -> float:
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if strictly_positive else result < 0):
        adjective = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must be finite and {adjective}")
    return result


@torch.no_grad()
def verify_v16_one_shot_subset(
    raw_output: Mapping[str, torch.Tensor],
    points: torch.Tensor,
    *,
    mse_tolerance: float,
    sample_index: int = 0,
    network_only_ms: float | None = None,
    proposal_domain_fallback: bool = True,
    compact: bool = False,
    degree: int = 3,
    smoothness_weight: float = 0.0,
    min_internal_knots: int = 0,
) -> V16VerifiedDeploymentComparison:
    """Verify a v16 raw subset against the *actual* unrooted MSE threshold.

    ``raw_output`` is the result of exactly one ``forward_deployment`` call.
    The verifier measures its own numerical work; callers may supply a
    separately measured ``network_only_ms``.  On failure it tries confidence
    add-back in the decoded parameter domain, followed optionally by the
    frozen Proposal domain.  If both complete candidate sets miss the budget,
    the result is marked ``candidate_budget_infeasible`` without inventing a
    passing curve or silently inserting new knot identities.
    """

    tolerance = _check_numeric("mse_tolerance", mse_tolerance, strictly_positive=True)
    _check_numeric("smoothness_weight", smoothness_weight, strictly_positive=False)
    if network_only_ms is not None:
        network_only_ms = _check_numeric(
            "network_only_ms", network_only_ms, strictly_positive=False
        )
    if not isinstance(proposal_domain_fallback, bool) or not isinstance(compact, bool):
        raise TypeError("proposal_domain_fallback and compact must be Boolean")
    if points.ndim != 2 or not points.is_floating_point():
        raise ValueError("points must be a floating-point [M,D] curve")
    if not torch.isfinite(points).all():
        raise ValueError("points must be finite")

    observed = points.detach().to(device="cpu", dtype=torch.float64)
    decoded_params = _one_curve_tensor(raw_output, "params", sample_index).double()
    proposal_params = _one_curve_tensor(raw_output, "proposal_params", sample_index).double()
    proposal_knots = _one_curve_tensor(
        raw_output, "proposal_internal_knots", sample_index
    ).double()
    deployed_knots = _one_curve_tensor(raw_output, "internal_knots", sample_index).double()
    keep_scores = _one_curve_tensor(
        raw_output, "keep_probabilities", sample_index
    ).double()
    keep_mask = _one_curve_tensor(raw_output, "learned_keep_mask", sample_index).bool()
    if "warped_proposal_internal_knots" in raw_output:
        decoded_proposal_knots = _one_curve_tensor(
            raw_output, "warped_proposal_internal_knots", sample_index
        ).double()
    else:
        decoded_proposal_knots = warp_internal_knots_to_parameterization(
            proposal_knots, proposal_params, decoded_params
        )

    started = time.perf_counter()
    fit_tolerance_rms = math.sqrt(tolerance)
    decoded: VerifiedKnotRepairResult = verified_confidence_repair(
        decoded_params,
        observed,
        decoded_proposal_knots,
        deployed_knots,
        keep_mask,
        keep_scores,
        fit_tolerance_rms=fit_tolerance_rms,
        min_internal_knots=min_internal_knots,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=0.0,
        interpolate_endpoints=True,
        compact=compact,
        hard_fallback=False,
        residual_fallback=False,
        parameterization_policy="network",
    )
    raw_fit = decoded.learned_fit
    selected = decoded
    selected_parameters = decoded_params
    selected_domain = "decoded_network"
    source = decoded.final_source
    refit_count = decoded.direct_refit_count
    evaluation_count = decoded.fit_evaluation_count
    fallback_attempted = False
    decoded_full_mse: float | None = None
    proposal_full_mse: float | None = None

    if not decoded.threshold_satisfied:
        # The complete decoded candidate set is explicitly measured rather
        # than assuming that a numerical regularized fit is monotone in K.
        decoded_full_fit = refit_bspline_control_points(
            decoded_params, observed, decoded_proposal_knots,
            degree=degree, smoothness_weight=smoothness_weight,
            control_ridge=0.0, interpolate_endpoints=True,
        )
        decoded_full_mse = float(decoded_full_fit.fit_mse)
        refit_count += 1
        evaluation_count += 1

    if not decoded.threshold_satisfied and proposal_domain_fallback:
        fallback_attempted = True
        proposal: VerifiedKnotRepairResult = verified_confidence_repair(
            proposal_params,
            observed,
            proposal_knots,
            proposal_knots,
            keep_mask,
            keep_scores,
            fit_tolerance_rms=fit_tolerance_rms,
            min_internal_knots=min_internal_knots,
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=0.0,
            interpolate_endpoints=True,
            compact=compact,
            hard_fallback=False,
            residual_fallback=False,
            parameterization_policy="network",
        )
        refit_count += proposal.direct_refit_count
        evaluation_count += proposal.fit_evaluation_count
        if proposal.threshold_satisfied or (
            float(proposal.final_fit.fit_mse) < float(selected.final_fit.fit_mse)
        ):
            selected = proposal
            selected_parameters = proposal_params
            selected_domain = "frozen_proposal"
            source = f"proposal_domain_{proposal.final_source}"
        if not proposal.threshold_satisfied:
            proposal_full_fit = refit_bspline_control_points(
                proposal_params, observed, proposal_knots,
                degree=degree, smoothness_weight=smoothness_weight,
                control_ridge=0.0, interpolate_endpoints=True,
            )
            proposal_full_mse = float(proposal_full_fit.fit_mse)
            refit_count += 1
            evaluation_count += 1

    raw_pass = float(raw_fit.fit_mse) <= tolerance
    verified_pass = float(selected.final_fit.fit_mse) <= tolerance
    if decoded.learned_threshold_satisfied and selected.final_count < int(raw_fit.internal_knots.numel()):
        status = "numerically_compacted"
    elif decoded.learned_threshold_satisfied:
        status = "raw_feasible"
    elif verified_pass:
        status = (
            "repaired_proposal_domain"
            if selected_domain == "frozen_proposal"
            else "repaired_decoded_domain"
        )
    else:
        status = "candidate_budget_infeasible"
    verification_ms = (time.perf_counter() - started) * 1000.0
    return V16VerifiedDeploymentComparison(
        mse_tolerance=tolerance,
        raw_fit=raw_fit,
        verified_fit=selected.final_fit,
        raw_parameters=decoded_params.detach().clone(),
        verified_parameters=selected_parameters.detach().clone(),
        raw_keep_mask=keep_mask.detach().clone(),
        verified_proposal_mask=selected.retained_proposal_mask.detach().clone(),
        raw_pass=raw_pass,
        verified_pass=verified_pass,
        status=status,
        verified_source=source,
        verified_parameterization=selected_domain,
        network_only_ms=network_only_ms,
        numerical_verification_ms=verification_ms,
        direct_standard_refit_count=refit_count,
        additional_standard_refit_count=max(0, refit_count - 1),
        fit_evaluation_count=evaluation_count,
        decoded_full_candidate_mse=decoded_full_mse,
        proposal_full_candidate_mse=proposal_full_mse,
        proposal_domain_fallback_attempted=fallback_attempted,
    )


__all__ = ["V16VerifiedDeploymentComparison", "verify_v16_one_shot_subset"]
