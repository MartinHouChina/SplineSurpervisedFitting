"""Evaluation helpers for spline fitting and knot reduction."""

from .bspline_inference import (
    BSplineLeastSquaresFit,
    HardGatedBSplineFit,
    hard_gate_mask,
    refit_bspline_control_points,
    refit_hard_gated_bspline_batch,
    refit_model_output_as_bsplines,
    second_difference_matrix,
)
from .knot_diagnostics import (
    KnotMatchStatistics,
    PrunedSplineFit,
    activity_statistics,
    build_open_knot_vector,
    hard_prune_and_refit,
    knot_contribution_rms,
    match_internal_knots,
    point_fit_statistics,
)

__all__ = [
    "BSplineLeastSquaresFit",
    "HardGatedBSplineFit",
    "KnotMatchStatistics",
    "PrunedSplineFit",
    "activity_statistics",
    "build_open_knot_vector",
    "hard_gate_mask",
    "hard_prune_and_refit",
    "knot_contribution_rms",
    "match_internal_knots",
    "point_fit_statistics",
    "refit_bspline_control_points",
    "refit_hard_gated_bspline_batch",
    "refit_model_output_as_bsplines",
    "second_difference_matrix",
]
