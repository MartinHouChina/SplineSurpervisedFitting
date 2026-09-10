"""Shared, network-free baselines for synthetic and real-curve comparisons.

All reported errors come from unregularized, endpoint-constrained standard
B-spline least squares.  The elapsed time covers the entire numerical method,
including chord parameterization and the final reported fit.  Published
methods are disclosed repository adaptations, not the authors' original code.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from .bspline_inference import BSplineLeastSquaresFit, refit_bspline_control_points
from .dung_direct_knot import fit_dung_direct_knots
from .feature_cdf_knot_placement import fit_feature_cdf_to_tolerance
from .gradient_knot_pruning import (
    chord_length_parameters,
    gradient_knot_pruning_baseline,
)
from .liang_feature_iki import fit_liang_feature_iki
from .luo_linf_de import fit_luo_linf_de
from .park_dominant_point import fit_park_dominant_points
from .sparse_knot_paper import fit_sparse_knots_paper


PUBLISHED_ADAPTATION_METHODS = (
    "park_dominant_point_2007_adaptation",
    "liang_feature_iki_2017_adaptation",
    "dung_direct_knot_2017_adaptation",
    "kang_sparse_2015_adaptation",
    "luo_linf_de_2022_adaptation",
    "yeh_feature_cdf_2020",
)
NUMERICAL_BASELINE_METHODS = (
    "uniform_gradient_pruning",
)
COMPARISON_BASELINE_METHODS = (
    *PUBLISHED_ADAPTATION_METHODS,
    *NUMERICAL_BASELINE_METHODS,
)
# Backward-compatible public name.  It now has the semantically correct scope:
# disclosed adaptations of published methods only.  Benchmark dispatch accepts
# COMPARISON_BASELINE_METHODS so repository-native numerical controls remain
# available without being presented as literature reproductions.
PUBLISHED_BASELINE_METHODS = PUBLISHED_ADAPTATION_METHODS
_COMMON_REFIT = (
    "unregularized standard B-spline least squares with exact endpoint interpolation"
)
_TIMING_SCOPE = (
    "complete numerical method: chord parameterization, knot selection/relocation, "
    "and final endpoint-constrained refit"
)


@dataclass(frozen=True)
class PublishedBaselineResult:
    method: str
    fit: BSplineLeastSquaresFit
    parameters: torch.Tensor
    elapsed_ms: float
    diagnostics: dict[str, object]


def _integer_bound(value: int, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def run_published_baseline(
    method: str,
    points: torch.Tensor,
    *,
    mse_tolerance: float,
    max_internal_knots: int = 28,
    degree: int = 3,
    gradient_steps: int = 12,
    paper_initial_knots: int = 40,
    paper_admm_iterations: int = 400,
    paper_lambda_bisections: int = 8,
    paper_relocation_iterations: int = 8,
    park_shape_weight: float = 0.8,
    liang_dense_knots: int | None = None,
    liang_initial_knots: int = 4,
    liang_curvature_weight: float = 0.5,
    liang_feature_samples: int = 1025,
    dung_max_error: float | None = None,
    dung_scan_intervals: int = 10,
    dung_optimization_iterations: int = 10,
    luo_eta: float = 0.5,
    luo_de_population: int = 10,
    luo_de_iterations: int = 50,
    luo_seed: int = 2022,
) -> PublishedBaselineResult:
    """Run a baseline on a single already-normalized ordered CPU float64 curve.

    ``mse_tolerance`` is mean squared Euclidean distance, with no square root
    and no division by coordinate dimension.  No network, true parameters or
    true knots are used.  ``max_internal_knots`` bounds the Yeh scan and the
    uniform deletion baseline; Kang uses its separately disclosed dense count
    ``paper_initial_knots``.  Liang's preliminary dense fit and Luo's dense
    sparse stage use ``liang_dense_knots`` and ``paper_initial_knots``
    respectively.  Input normalization/resampling is a shared dataset operation
    and must be performed before this call.
    """
    if method not in COMPARISON_BASELINE_METHODS:
        raise ValueError(f"unknown baseline method: {method!r}")
    _integer_bound(degree, "degree", 1)
    _integer_bound(max_internal_knots, "max_internal_knots", 0)
    _integer_bound(gradient_steps, "gradient_steps", 0)
    _integer_bound(paper_initial_knots, "paper_initial_knots", 0)
    _integer_bound(paper_admm_iterations, "paper_admm_iterations", 1)
    _integer_bound(paper_lambda_bisections, "paper_lambda_bisections", 1)
    _integer_bound(paper_relocation_iterations, "paper_relocation_iterations", 1)
    if liang_dense_knots is None:
        liang_dense_knots = paper_initial_knots
    _integer_bound(liang_dense_knots, "liang_dense_knots", 0)
    _integer_bound(liang_initial_knots, "liang_initial_knots", 0)
    _integer_bound(liang_feature_samples, "liang_feature_samples", 3)
    _integer_bound(dung_scan_intervals, "dung_scan_intervals", 1)
    _integer_bound(
        dung_optimization_iterations, "dung_optimization_iterations", 0
    )
    _integer_bound(luo_de_population, "luo_de_population", 5)
    _integer_bound(luo_de_iterations, "luo_de_iterations", 0)
    _integer_bound(luo_seed, "luo_seed", 0)
    if not math.isfinite(mse_tolerance) or mse_tolerance < 0.0:
        raise ValueError("mse_tolerance must be finite and non-negative")
    for value, name in (
        (park_shape_weight, "park_shape_weight"),
        (liang_curvature_weight, "liang_curvature_weight"),
        (luo_eta, "luo_eta"),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if dung_max_error is not None and (
        not math.isfinite(dung_max_error) or dung_max_error < 0.0
    ):
        raise ValueError("dung_max_error must be finite and non-negative")
    if (
        method == "liang_feature_iki_2017_adaptation"
        and liang_initial_knots > max_internal_knots
    ):
        raise ValueError("liang_initial_knots must not exceed max_internal_knots")
    if not isinstance(points, torch.Tensor):
        raise TypeError("points must be a CPU float64 torch.Tensor")
    if points.device.type != "cpu" or points.dtype != torch.float64:
        raise ValueError("points must be CPU float64 for comparable numerical refits")
    minimum_samples = degree + (2 if method == "yeh_feature_cdf_2020" else 1)
    if points.ndim != 2 or points.shape[0] < minimum_samples or points.shape[1] < 1:
        raise ValueError(
            f"points must have shape [M, D], with M >= {minimum_samples} and D >= 1"
        )
    if not bool(torch.isfinite(points).all()):
        raise ValueError("points must be finite")

    started = time.perf_counter()
    observed = points.detach()
    diagnostics: dict[str, object] = {
        "network_used": False,
        "parameterization": "chord_length",
        "reported_refit": _COMMON_REFIT,
        "timing_scope": _TIMING_SCOPE,
        "mse_tolerance": float(mse_tolerance),
        "smoothness_weight": 0.0,
        "control_ridge": 0.0,
        "interpolate_endpoints": True,
    }

    if method == "uniform_gradient_pruning":
        # The baseline computes chord parameters internally.  Enable gradients
        # locally even when an evaluation caller uses no_grad/inference_mode.
        with torch.inference_mode(False), torch.enable_grad():
            result = gradient_knot_pruning_baseline(
                observed.clone(),
                max_internal_knots=max_internal_knots,
                mse_tolerance=mse_tolerance,
                min_internal_knots=0,
                degree=degree,
                min_gap=1e-4,
                optimization_steps=gradient_steps,
                learning_rate=0.05,
                optimizer="adam",
                smoothness_weight=0.0,
                control_ridge=0.0,
                interpolate_endpoints=True,
            )
        parameters = result.parameters
        # This fit already uses exactly the shared settings; an extra solve
        # would only duplicate work and distort the recorded method latency.
        fit = result.final_fit
        diagnostics.update({
            "initial_internal_knot_count": max_internal_knots,
            "initial_location_update_accepted": result.initial_location_update_accepted,
            "uniform_initial_fit_mse": float(result.initial_fit.fit_mse),
            "relocated_initial_fit_mse": float(result.initial_relocated_fit.fit_mse),
            "accepted_deletions": result.accepted_deletions,
            "gradient_steps": result.gradient_steps,
            "optimization_steps_per_candidate": gradient_steps,
            "variable_projection_evaluations": result.evaluation_count,
            "exact_refit_count": result.refit_count,
            "native_final_fit_mse": float(result.final_fit.fit_mse),
        })
    elif method == "park_dominant_point_2007_adaptation":
        result = fit_park_dominant_points(
            observed,
            mse_tolerance=mse_tolerance,
            max_internal_knots=max_internal_knots,
            degree=degree,
            shape_weight=park_shape_weight,
        )
        parameters = result.parameters
        fit = result.final_fit
        diagnostics.update(result.diagnostics)
        diagnostics.update({
            "scanned_counts": result.scanned_knot_counts,
            "scanned_mse": result.scanned_mse,
            "scanned_max_parameter_residual": (
                result.scanned_max_parameter_residual
            ),
            "exact_refit_count": result.refit_count,
            "method_fidelity": "core DOM equations with a common-MSE stopping wrapper",
        })
    elif method == "liang_feature_iki_2017_adaptation":
        result = fit_liang_feature_iki(
            observed,
            mse_tolerance=mse_tolerance,
            max_internal_knots=max_internal_knots,
            degree=degree,
            dense_internal_knot_count=liang_dense_knots,
            initial_internal_knot_count=liang_initial_knots,
            curvature_weight=liang_curvature_weight,
            feature_sample_count=liang_feature_samples,
        )
        parameters = result.parameters
        fit = result.final_fit
        diagnostics.update({
            "reference_doi": "10.1088/1361-6501/aa6a05",
            "variant": "feature_integral_plus_iki_disclosed_adaptation",
            "dense_internal_knot_count": result.dense_internal_knot_count,
            "initial_internal_knot_count": int(result.initial_knots.numel()),
            "inserted_internal_knot_count": int(result.inserted_knots.numel()),
            "curvature_weight": result.curvature_weight,
            "feature_sample_count": result.feature_sample_count,
            "scanned_counts": result.scanned_internal_knot_counts,
            "scanned_mse": result.scanned_mse,
            "scanned_max_parameter_residual": result.scanned_max_point_error,
            "exact_refit_count": result.refit_count,
            "paper_native_objective": (
                "error-bounded feature-integral initialization plus iterative knot insertion"
            ),
            "comparison_wrapper": (
                "explicit 0.5 arc/curvature feature blend by default, worst-residual "
                "span midpoint insertion, and common-MSE stopping"
            ),
            "method_fidelity": (
                "Liang-inspired adaptation; the access-controlled article's exact "
                "feature equation and constants are not claimed"
            ),
        })
    elif method == "dung_direct_knot_2017_adaptation":
        result = fit_dung_direct_knots(
            observed,
            degree=degree,
            mse_tolerance=mse_tolerance,
            max_internal_knots=max_internal_knots,
            max_error=dung_max_error,
            scan_intervals=dung_scan_intervals,
            optimization_iterations=dung_optimization_iterations,
        )
        parameters = result.parameters
        fit = result.final_fit
        diagnostics.update(result.diagnostics)
        diagnostics.update({
            "method_fidelity": (
                "serial simple-knot adaptation; parallel operations and knot "
                "multiplicity classification are not reproduced"
            ),
            "threshold_satisfied_native_max_error": (
                result.final_max_error <= result.native_max_error_tolerance
            ),
        })
    else:
        parameters = chord_length_parameters(observed)
        if method == "kang_sparse_2015_adaptation":
            result = fit_sparse_knots_paper(
                parameters,
                observed,
                degree=degree,
                initial_internal_knot_count=paper_initial_knots,
                data_tolerance=mse_tolerance,
                jump_threshold=1e-7,
                relative_jump_threshold=1e-3,
                admm_rho=1e4,
                admm_max_iterations=paper_admm_iterations,
                admm_tolerance=1e-6,
                lambda_bisection_iterations=paper_lambda_bisections,
                relocation_max_iterations=paper_relocation_iterations,
                feasibility_repair=False,
            )
            # Preserve the native unconstrained result for auditing.  The
            # endpoint constraint is applied uniformly to the reported metric.
            fit = refit_bspline_control_points(
                parameters,
                observed,
                result.knots,
                degree=degree,
                smoothness_weight=0.0,
                control_ridge=0.0,
                interpolate_endpoints=True,
            )
            diagnostics.update({
                "initial_internal_knot_count": int(result.initial_internal_knots.numel()),
                "dense_initial_fit_mse": float(result.dense_initial_fit_mse),
                "dense_initial_threshold_satisfied": result.dense_initial_threshold_satisfied,
                "active_internal_knot_count": result.active_count,
                "cluster_sizes": result.cluster_sizes,
                "active_cluster_count": len(result.cluster_sizes),
                "relocated_internal_knot_count": int(
                    result.relocated_internal_knots.numel()
                ),
                "relocated_unique_knot_count": int(
                    torch.unique_consecutive(
                        result.relocated_internal_knots
                    ).numel()
                ),
                "relocated_knot_multiplicities": tuple(
                    int(value)
                    for value in torch.unique_consecutive(
                        result.relocated_internal_knots,
                        return_counts=True,
                    )[1].tolist()
                ),
                "active_to_relocated_compression_ratio": (
                    float(result.relocated_internal_knots.numel())
                    / max(1, result.active_count)
                ),
                "sparse_stage_mse": float(result.sparse_fit_mse),
                "sparse_stage_threshold_satisfied": result.threshold_satisfied,
                # The sparse-stage value belongs to Kang's native ADMM fit,
                # whereas ``fit`` is the shared endpoint-constrained standard
                # refit. Record the transition without attributing the whole
                # difference to relocation alone.
                "native_sparse_feasible_final_common_refit_failed": (
                    result.threshold_satisfied
                    and float(fit.fit_mse) > mse_tolerance + 1e-12
                ),
                "sparse_and_final_mse_are_directly_comparable": False,
                "sparse_iterations": result.sparse_iterations,
                "sparse_solver_seconds": result.sparse_solver_seconds,
                "relocation_seconds": result.relocation_seconds,
                "non_paper_feasibility_repair_enabled": False,
                "paper_feasibility_repair_used": result.repair_used,
                "paper_feasibility_repair_added_count": int(result.repair_added_knots.numel()),
                "local_refit_count": result.local_refit_count,
                "method_note": result.method,
                "absolute_jump_threshold": result.jump_threshold,
                "relative_jump_threshold": result.relative_jump_threshold,
                "effective_jump_threshold": result.effective_jump_threshold,
                "native_final_fit_mse_without_endpoint_constraint": float(result.final_fit.fit_mse),
                "native_endpoint_constrained_mse": float(fit.fit_mse),
                "endpoint_constraint_fallback": "disabled",
            })
        elif method == "luo_linf_de_2022_adaptation":
            result = fit_luo_linf_de(
                parameters,
                observed,
                degree=degree,
                initial_internal_knot_count=paper_initial_knots,
                mse_tolerance=mse_tolerance,
                eta=luo_eta,
                admm_max_iterations=paper_admm_iterations,
                lambda_bisection_iterations=paper_lambda_bisections,
                de_population=luo_de_population,
                de_iterations=luo_de_iterations,
                seed=luo_seed,
            )
            fit = result.final_fit
            diagnostics.update({
                "reference_doi": "10.4208/jcm.2012-m2020-0203",
                "variant": "linf1_jump_sparsity_plus_differential_evolution",
                "initial_internal_knot_count": int(
                    result.initial_internal_knots.numel()
                ),
                "dense_initial_fit_mse": result.dense_initial_fit_mse,
                "dense_initial_threshold_satisfied": (
                    result.dense_initial_threshold_satisfied
                ),
                "selected_regularization": result.selected_regularization,
                "sparse_stage_mse": result.sparse_fit_mse,
                "sparse_stage_threshold_satisfied": (
                    result.sparse_fit_mse <= mse_tolerance + 1e-12
                ),
                "jump_local_maximum_eta": luo_eta,
                "candidate_internal_knot_count": int(
                    result.candidate_knots.numel()
                ),
                "initial_to_candidate_compression_ratio": (
                    float(result.candidate_knots.numel())
                    / max(1, result.initial_internal_knots.numel())
                ),
                "candidate_refit_mse_before_de": result.candidate_refit_mse,
                "candidate_refit_threshold_satisfied_before_de": (
                    result.candidate_refit_mse <= mse_tolerance + 1e-12
                ),
                "candidate_selection_lost_sparse_feasibility": (
                    result.sparse_fit_mse <= mse_tolerance + 1e-12
                    and result.candidate_refit_mse > mse_tolerance + 1e-12
                ),
                "de_population": result.de_population,
                "de_iterations": result.de_iterations,
                "de_evaluations": result.de_evaluations,
                "de_initial_max_error": result.de_initial_max_error,
                "de_final_max_error": result.de_final_max_error,
                "de_restored_common_mse_feasibility": (
                    result.candidate_refit_mse > mse_tolerance + 1e-12
                    and float(result.final_fit.fit_mse)
                    <= mse_tolerance + 1e-12
                ),
                "paper_native_objective": (
                    "nonsquared Frobenius data term plus l-infinity,1 derivative-jump "
                    "penalty; DE minimizes maximum Euclidean error"
                ),
                "comparison_wrapper": (
                    "regularization is selected against the common MSE budget and "
                    "the final fit uses the common endpoint-constrained refit"
                ),
                "method_fidelity": (
                    "defining mixed-norm, local-maximum selection, and DE stages "
                    "are reproduced; hyperparameter selection is adapted"
                ),
            })
        else:
            result = fit_feature_cdf_to_tolerance(
                parameters,
                observed,
                degree=degree,
                mse_tolerance=mse_tolerance,
                max_internal_knots=max_internal_knots,
                min_internal_knots=0,
                density_limit=True,
                interpolate_endpoints=True,
            )
            fit = result.final_fit
            diagnostics.update({
                "paper_placement_rule": "equal high-order derivative-feature mass",
                "comparison_wrapper": (
                    "ascending cardinality scan replaces the paper's dataset-specific "
                    "target-error regression"
                ),
                "derivative_order": result.derivative_order,
                "density_limit": True,
                "scanned_counts": result.scanned_counts,
                "scanned_mse": result.scanned_mse,
                "exact_refit_count": result.refit_count,
                "max_internal_knots": max_internal_knots,
            })

    diagnostics["threshold_satisfied"] = float(fit.fit_mse) <= mse_tolerance
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return PublishedBaselineResult(
        method=method,
        fit=fit,
        parameters=parameters,
        elapsed_ms=elapsed_ms,
        diagnostics=diagnostics,
    )
