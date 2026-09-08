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
from .certified_real_world import (
    PolylineSimplificationResult,
    REFERENCE_CERTIFICATE_SCOPE,
    ReferenceCertifiedFitResult,
    certify_reference_mse,
    exact_polyline_bspline_fit,
    simplify_polyline_to_mse,
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
    warp_internal_knots_to_parameterization,
)
from .hybrid_knot_search import (
    HybridKnotSearchResult,
    KnotPositionRefinementResult,
    hybrid_minimal_knot_search,
    refine_knot_positions,
)
from .gradient_knot_pruning import (
    GradientKnotDeletionStep,
    GradientKnotPruningResult,
    chord_length_parameters,
    fit_gradient_knot_pruning,
    gradient_knot_pruning_baseline,
    gradient_prune_knots,
)
from .feature_cdf_knot_placement import (
    FeatureCDFKnotPlacementResult,
    finite_difference_feature,
    fit_feature_cdf_to_tolerance,
    place_feature_cdf_knots,
)
from .minimal_knot_pruning import (
    KnotDeletionStep,
    MinimalKnotPruningResult,
    prune_knots_to_rms_tolerance,
)
from .sparse_knot_paper import (
    SparseKnotPaperResult,
    fit_sparse_knots_paper,
)
from .verified_knot_repair import (
    VerifiedKnotRepairResult,
    verified_confidence_repair,
)

__all__ = [
    "BSplineLeastSquaresFit",
    "HardGatedBSplineFit",
    "GradientKnotDeletionStep",
    "GradientKnotPruningResult",
    "FeatureCDFKnotPlacementResult",
    "HybridKnotSearchResult",
    "KnotMatchStatistics",
    "KnotDeletionStep",
    "MinimalKnotPruningResult",
    "KnotPositionRefinementResult",
    "PrunedSplineFit",
    "PolylineSimplificationResult",
    "REFERENCE_CERTIFICATE_SCOPE",
    "ReferenceCertifiedFitResult",
    "SparseKnotPaperResult",
    "VerifiedKnotRepairResult",
    "activity_statistics",
    "build_open_knot_vector",
    "certify_reference_mse",
    "exact_polyline_bspline_fit",
    "chord_length_parameters",
    "fit_gradient_knot_pruning",
    "finite_difference_feature",
    "fit_feature_cdf_to_tolerance",
    "gradient_knot_pruning_baseline",
    "gradient_prune_knots",
    "hard_gate_mask",
    "hard_prune_and_refit",
    "hybrid_minimal_knot_search",
    "knot_contribution_rms",
    "match_internal_knots",
    "point_fit_statistics",
    "place_feature_cdf_knots",
    "prune_knots_to_rms_tolerance",
    "fit_sparse_knots_paper",
    "refit_bspline_control_points",
    "refit_hard_gated_bspline_batch",
    "refit_model_output_as_bsplines",
    "refine_knot_positions",
    "second_difference_matrix",
    "simplify_polyline_to_mse",
    "verified_confidence_repair",
    "warp_internal_knots_to_parameterization",
]
