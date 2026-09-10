from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

from torch import nn

from .models.spline_network import SplineFittingNetwork


CROSS_ATTENTION_OBJECTIVE_VERSION = "cross_attention_true_params_hard_concrete_v1"
INDEPENDENT_QUERY_V1_OBJECTIVE_VERSION = "independent_query_supervised_hard_concrete_v1"
INDEPENDENT_QUERY_V2_OBJECTIVE_VERSION = "independent_query_supervised_hard_concrete_v2"
PREVIOUS_OBJECTIVE_VERSION = "independent_query_two_stage_hard_concrete_v3"
COUNT_CONDITIONED_V4_OBJECTIVE_VERSION = "count_conditioned_structured_knots_v4"
COUNT_CONDITIONED_V5_OBJECTIVE_VERSION = "canonical_ordinal_count_conditioned_v5"
# Public compatibility name used by historical scripts/tests.  Do not retarget
# it: changing its meaning would silently reinterpret existing v6 checkpoints.
CURRENT_OBJECTIVE_VERSION = "interactive_structure_dynamic_knots_v6"
CANDIDATE_PRUNING_OBJECTIVE_VERSION = "candidate_pruning_minimal_rms_v7"
ONE_SHOT_PRUNING_OBJECTIVE_VERSION = "candidate_pruning_one_shot_teacher_v8"
V9_ONE_SHOT_PRUNING_OBJECTIVE_VERSION = "candidate_pruning_fixed_proposal_teacher_v9"
V10_FEASIBLE_ONE_SHOT_PRUNING_OBJECTIVE_VERSION = (
    "candidate_pruning_structured_feasible_teacher_v10"
)
V11_JOINT_ONE_SHOT_PRUNING_OBJECTIVE_VERSION = (
    "candidate_pruning_joint_refinement_teacher_v11"
)
V12_COUPLED_RELOCATION_OBJECTIVE_VERSION = (
    "candidate_pruning_coupled_relocation_teacher_v12"
)
V13_SET_RELOCATION_OBJECTIVE_VERSION = (
    "candidate_pruning_one_shot_set_relocation_teacher_v13"
)
V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION = (
    "candidate_pruning_one_shot_parameter_feedback_v14"
)
V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION = (
    "candidate_pruning_joint_parameter_structure_feedback_v14"
)
V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION = (
    "candidate_pruning_deployment_aligned_feedback_v15"
)
V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION = (
    "candidate_selection_counterfactual_bspline_v16"
)
V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION = (
    "candidate_selection_supervised_bspline_v16"
)
V16_OBJECTIVE_VERSIONS = (
    V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
    V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION,
)
V16_ADAPTIVE_SELECTION_REVISION = "v16_supervised_ordered_assignment_mass_topk"
V16_CERTIFIED_SYNTHETIC_CONTRACT = (
    "source_subset_threshold_minimal_k4_56_span001_v2"
)
V16_SIMPLIFICATION_CONTRACT = (
    "synthetic_ground_truth_ordered_keep_and_relocation_v4"
)
V16_CHECKPOINT_SELECTION_CONTRACT = "mean_per_curve_subset_cost_v1"
V16_JOINT_CHECKPOINT_QUALITY = "supervised_fit_count_selected"
V16_MINIMALITY_MARGIN = 0.2
V16_MINIMALITY_AUDIT_POINTS = 512
# Historical reporting reference only.  It is deliberately distinct from the
# per-curve MSE tolerance and does not control checkpoint eligibility.
V16_FORMAL_PASS_RATE = 0.90
V16_FORMAL_KNOT_MATCH_TOLERANCE = 0.01
V16_FORMAL_SYNTHETIC_COUNT_MAE_MAX = 2.0
V16_FORMAL_SYNTHETIC_KNOT_F1_MIN = 0.60
V16_FORMAL_SYNTHETIC_MATCHED_MAE_MAX = 0.005
V16_FORMAL_SYNTHETIC_MIN_INTERNAL_KNOTS = 4
V16_FORMAL_SYNTHETIC_MAX_INTERNAL_KNOTS = 56
V16_FORMAL_SYNTHETIC_MIN_CONTROL_POINTS = 8
V16_FORMAL_SYNTHETIC_MAX_CONTROL_POINTS = 60
V16_FORMAL_SYNTHETIC_KNOT_MIN_SPAN = 0.01
V16_FORMAL_SYNTHETIC_BOUNDARY_SAMPLES_MIN = 32
V16_FORMAL_PROPOSAL_HIGH_K_MIN_KNOTS = 40
V16_FORMAL_PROPOSAL_HIGH_K_FRACTION = 0.5
V16_FORMAL_FINAL_SAFETY_SIGMA_MAX = 0.05
# Historical training scripts import this alias. Keep their v15 defaults and
# checkpoint labels unchanged; v16 has a dedicated trainer and explicit label.
LATEST_OBJECTIVE_VERSION = V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION


def _finite_checkpoint_float(value: Any) -> float | None:
    """Return a finite scalar checkpoint value without accepting booleans."""
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _checkpoint_int(value: Any) -> int | None:
    """Return an exact checkpoint integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def assess_v16_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    required_pass_rate: float = V16_FORMAL_PASS_RATE,
    required_mse_tolerance: float | None = None,
) -> dict[str, Any]:
    """Audit structural eligibility while reporting empirical diagnostics.

    Pass rates and knot/count accuracy never decide eligibility.  The hard
    checks cover protocol integrity, certified data, a completed/mature joint
    curriculum, final deployment safety and the requested MSE contract.
    """
    required = _finite_checkpoint_float(required_pass_rate)
    if required is None or not 0.0 <= required <= 1.0:
        raise ValueError("required_pass_rate must be finite and lie in [0,1]")
    required_tolerance = None
    if required_mse_tolerance is not None:
        required_tolerance = _finite_checkpoint_float(required_mse_tolerance)
        if required_tolerance is None or required_tolerance <= 0.0:
            raise ValueError("required_mse_tolerance must be finite and positive")

    training = checkpoint.get("training_config")
    validation = checkpoint.get("validation_metrics")
    deployment = checkpoint.get("deployment_config")
    model_config = checkpoint.get("model_config")
    dataset_config = checkpoint.get("dataset_config")
    loss_config = checkpoint.get("loss_config")
    simplification_curriculum = checkpoint.get("simplification_curriculum")
    training = training if isinstance(training, Mapping) else {}
    validation = validation if isinstance(validation, Mapping) else {}
    deployment = deployment if isinstance(deployment, Mapping) else {}
    model_config = model_config if isinstance(model_config, Mapping) else {}
    dataset_config = dataset_config if isinstance(dataset_config, Mapping) else {}
    loss_config = loss_config if isinstance(loss_config, Mapping) else {}
    simplification_curriculum = (
        simplification_curriculum
        if isinstance(simplification_curriculum, Mapping)
        else {}
    )
    loss_weights = loss_config.get("weights")
    loss_weights = loss_weights if isinstance(loss_weights, Mapping) else {}
    configured_proposal = _finite_checkpoint_float(
        training.get("proposal_pass_target")
    )
    configured_deployment = _finite_checkpoint_float(
        training.get("deployment_pass_target")
    )
    proposal_high_k_fraction = _finite_checkpoint_float(
        training.get("proposal_high_k_fraction")
    )
    proposal_high_k_min_knots = _checkpoint_int(
        training.get("proposal_high_k_min_knots")
    )
    proposal_knot_assignment_weight = _finite_checkpoint_float(
        loss_weights.get("proposal_knot_assignment_weight")
    )
    training_real_fraction = _finite_checkpoint_float(training.get("real_fraction"))
    joint_supervision = loss_config.get("joint_supervision")
    synthetic_count_role = loss_config.get("synthetic_count_role")
    observed_dense = _finite_checkpoint_float(
        validation.get("worst_dense_pass_rate")
    )
    observed_deployment = _finite_checkpoint_float(
        validation.get("worst_deployment_pass_rate")
    )
    recorded_tolerance = _finite_checkpoint_float(
        deployment.get("mse_tolerance", training.get("mse_tolerance"))
    )
    recorded_match_tolerance = _finite_checkpoint_float(
        deployment.get(
            "knot_match_tolerance", training.get("knot_match_tolerance")
        )
    )
    certificate_rms_tolerance = _finite_checkpoint_float(
        dataset_config.get("canonical_knot_tolerance")
    )
    minimality_margin = _finite_checkpoint_float(
        dataset_config.get("minimality_margin")
    )
    minimality_audit_points = dataset_config.get("minimality_audit_points")
    candidate_capacity = _finite_checkpoint_float(
        model_config.get("max_internal_knots")
    )
    synthetic_min_control_points = _checkpoint_int(
        dataset_config.get("min_control_points")
    )
    synthetic_max_control_points = _checkpoint_int(
        dataset_config.get("max_control_points")
    )
    synthetic_knot_min_span = _finite_checkpoint_float(
        dataset_config.get("knot_min_span")
    )
    synthetic_boundary_knot_count = _checkpoint_int(
        validation.get("synthetic_boundary_knot_count")
    )
    synthetic_boundary_sample_count = _checkpoint_int(
        validation.get("synthetic_boundary_sample_count")
    )
    synthetic_boundary_dense_pass_rate = _finite_checkpoint_float(
        validation.get("synthetic_boundary_dense_pass_rate")
    )
    synthetic_boundary_deployment_pass_rate = _finite_checkpoint_float(
        validation.get("synthetic_boundary_deployment_pass_rate")
    )
    observed_qualification_dense = _finite_checkpoint_float(
        validation.get("qualification_dense_pass_rate")
    )
    observed_qualification_deployment = _finite_checkpoint_float(
        validation.get("qualification_deployment_pass_rate")
    )
    observed_keep_count = _finite_checkpoint_float(
        validation.get("keep_count")
    )
    final_safety_sigma = _finite_checkpoint_float(
        training.get("final_safety_sigma")
    )
    applied_safety_sigma = _finite_checkpoint_float(
        model_config.get(
            "one_shot_safety_sigma", deployment.get("one_shot_safety_sigma")
        )
    )
    final_safety_knots = training.get("final_safety_knots")
    applied_safety_knots = model_config.get(
        "one_shot_safety_knots", deployment.get("one_shot_safety_knots")
    )
    synthetic_count_mae = _finite_checkpoint_float(
        validation.get("synthetic_count_mae")
    )
    synthetic_knot_match_f1 = _finite_checkpoint_float(
        validation.get("synthetic_knot_match_f1")
    )
    synthetic_knot_matched_mae = _finite_checkpoint_float(
        validation.get("synthetic_knot_matched_mae")
    )
    configured_target_met = (
        configured_deployment is not None
        and observed_qualification_deployment is not None
        and observed_qualification_deployment >= configured_deployment
    )
    configured_checkpoint_accepted = (
        checkpoint.get("stage") == "joint" and configured_target_met
    )
    reasons: list[str] = []
    if checkpoint.get("objective_version") != V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION:
        reasons.append("objective_version is not supervised v16")
    architecture_revision = checkpoint.get("architecture_revision")
    selection_policy = model_config.get(
        "one_shot_selection_policy",
        deployment.get("one_shot_selection_policy"),
    )
    adaptive_threshold = model_config.get(
        "one_shot_adaptive_threshold",
        deployment.get("adaptive_keep_threshold"),
    )
    if architecture_revision != V16_ADAPTIVE_SELECTION_REVISION:
        reasons.append(
            "checkpoint is not the current v16 adaptive-beta mass-TopK revision"
        )
    if selection_policy != "mass_topk":
        reasons.append("formal v16 selection policy is not mass_topk")
    if adaptive_threshold is not True:
        reasons.append("formal v16 adaptive keep threshold is not enabled")
    simplification_contract = checkpoint.get("simplification_contract")
    if simplification_contract != V16_SIMPLIFICATION_CONTRACT:
        reasons.append(
            "checkpoint is not the synthetic-ground-truth ordered KeepMask "
            "and relocation revision"
        )
    if training_real_fraction is None or not math.isclose(
        training_real_fraction, 0.0, rel_tol=0.0, abs_tol=0.0
    ):
        reasons.append(
            "formal v16 training must use labelled synthetic curves only "
            "(real_fraction=0)"
        )
    if (
        proposal_high_k_fraction is None
        or not math.isclose(
            proposal_high_k_fraction,
            V16_FORMAL_PROPOSAL_HIGH_K_FRACTION,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        reasons.append(
            "formal proposal training must allocate exactly "
            f"{V16_FORMAL_PROPOSAL_HIGH_K_FRACTION:.0%} of synthetic draws "
            "to the high-K stratum"
        )
    if proposal_high_k_min_knots != V16_FORMAL_PROPOSAL_HIGH_K_MIN_KNOTS:
        reasons.append(
            "formal proposal high-K stratum must start at K="
            f"{V16_FORMAL_PROPOSAL_HIGH_K_MIN_KNOTS}"
        )
    if checkpoint.get("simplification_ready") is not True:
        reasons.append("joint simplification curriculum was not mature when saved")
    if simplification_curriculum.get("aggregate_pass_feedback") is not False:
        reasons.append(
            "simplification curriculum must not use aggregate pass-rate feedback"
        )
    if (
        final_safety_sigma is None
        or applied_safety_sigma is None
        or not math.isclose(
            final_safety_sigma, applied_safety_sigma, rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        reasons.append("checkpoint was not evaluated at the final safety sigma")
    elif final_safety_sigma > V16_FORMAL_FINAL_SAFETY_SIGMA_MAX:
        reasons.append(
            "final safety sigma exceeds the formal simplification allowance"
        )
    if (
        isinstance(final_safety_knots, bool)
        or not isinstance(final_safety_knots, int)
        or isinstance(applied_safety_knots, bool)
        or not isinstance(applied_safety_knots, int)
        or final_safety_knots != applied_safety_knots
    ):
        reasons.append("checkpoint was not evaluated at the final safety-knot reserve")
    elif final_safety_knots != 0:
        reasons.append("formal simplification requires zero fixed safety knots")
    synthetic_contract = checkpoint.get("synthetic_data_contract")
    certified_synthetic = dataset_config.get("certified_minimal_source")
    if synthetic_contract != V16_CERTIFIED_SYNTHETIC_CONTRACT:
        reasons.append(
            "synthetic data contract is not the certified K=4..56 source-subset "
            "minimality revision"
        )
    if certified_synthetic is not True:
        reasons.append("certified-minimal synthetic source generation is not enabled")
    if synthetic_min_control_points != V16_FORMAL_SYNTHETIC_MIN_CONTROL_POINTS:
        reasons.append(
            "formal synthetic source minimum must be 8 control points "
            "(4 internal knots)"
        )
    if synthetic_max_control_points != V16_FORMAL_SYNTHETIC_MAX_CONTROL_POINTS:
        reasons.append(
            "formal synthetic source maximum must be 60 control points "
            "(56 internal knots)"
        )
    if (
        synthetic_knot_min_span is None
        or not math.isclose(
            synthetic_knot_min_span,
            V16_FORMAL_SYNTHETIC_KNOT_MIN_SPAN,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        reasons.append(
            "formal synthetic knot_min_span must equal 0.01 for the K=4..56 protocol"
        )
    if certificate_rms_tolerance is None or certificate_rms_tolerance <= 0:
        reasons.append("synthetic minimality RMS tolerance is missing or invalid")
    elif (
        recorded_tolerance is not None
        and not math.isclose(
            certificate_rms_tolerance * certificate_rms_tolerance,
            recorded_tolerance,
            rel_tol=1e-12,
            abs_tol=0.0,
        )
    ):
        reasons.append("synthetic certificate RMS tolerance does not match sqrt(MSE tolerance)")
    if minimality_margin is None or minimality_margin < V16_MINIMALITY_MARGIN:
        reasons.append(
            f"synthetic minimality margin is below {V16_MINIMALITY_MARGIN:g}"
        )
    if (
        isinstance(minimality_audit_points, bool)
        or not isinstance(minimality_audit_points, int)
        or minimality_audit_points < V16_MINIMALITY_AUDIT_POINTS
    ):
        reasons.append(
            f"synthetic minimality audit grid has fewer than "
            f"{V16_MINIMALITY_AUDIT_POINTS} points"
        )
    if candidate_capacity is None or not math.isclose(
        candidate_capacity,
        V16_FORMAL_SYNTHETIC_MAX_INTERNAL_KNOTS,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        reasons.append(
            "formal v16 candidate-knot capacity must equal 56 internal knots"
        )
    # Retained K, all-keep behavior and knot/count accuracy are empirical
    # benchmark results.  Keep them in the report below without gating access
    # to the benchmark itself.
    if (
        synthetic_boundary_knot_count
        != V16_FORMAL_SYNTHETIC_MAX_INTERNAL_KNOTS
    ):
        reasons.append(
            "formal validation boundary must be the K=56 synthetic stratum"
        )
    if (
        synthetic_boundary_sample_count is None
        or synthetic_boundary_sample_count
        < V16_FORMAL_SYNTHETIC_BOUNDARY_SAMPLES_MIN
    ):
        reasons.append(
            "formal validation K=56 boundary audit must contain at least "
            f"{V16_FORMAL_SYNTHETIC_BOUNDARY_SAMPLES_MIN} synthetic samples"
        )
    expected_qualification_dense = None
    if (
        observed_dense is not None
        and synthetic_boundary_dense_pass_rate is not None
    ):
        expected_qualification_dense = min(
            observed_dense, synthetic_boundary_dense_pass_rate
        )
    expected_qualification_deployment = None
    if (
        observed_deployment is not None
        and synthetic_boundary_deployment_pass_rate is not None
    ):
        expected_qualification_deployment = min(
            observed_deployment, synthetic_boundary_deployment_pass_rate
        )
    # Aggregate and boundary pass rates are descriptive.  Their recomputed
    # minima are exposed to flag reporting inconsistencies without rejecting
    # a structurally valid checkpoint.
    if (
        recorded_match_tolerance is None
        or not math.isclose(
            recorded_match_tolerance,
            V16_FORMAL_KNOT_MATCH_TOLERANCE,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        reasons.append(
            "formal knot diagnostics must use match@"
            f"{V16_FORMAL_KNOT_MATCH_TOLERANCE:g}"
        )
    required_positive_weights = (
        "fit_weight",
        "distillation_weight",
        "count_weight",
        "supervised_count_weight",
        "supervised_over_count_weight",
        "true_parameter_weight",
        "proposal_knot_coverage_weight",
        "proposal_knot_assignment_weight",
        "selected_knot_position_weight",
    )
    inactive_weights = [
        name
        for name in required_positive_weights
        if (
            (weight := _finite_checkpoint_float(loss_weights.get(name))) is None
            or weight <= 0
        )
    ]
    if inactive_weights:
        reasons.append(
            "formal simplification supervision is inactive: "
            + ", ".join(inactive_weights)
        )
    if joint_supervision != "synthetic_ground_truth":
        reasons.append("Joint supervision is not direct synthetic ground truth")
    if loss_config.get("ranked_prefix_teacher") is not False:
        reasons.append("online ranked-prefix teacher must be disabled")
    if loss_config.get("online_teacher") is not False:
        reasons.append("online self-teacher must be disabled")
    if synthetic_count_role != "exact":
        reasons.append("certified synthetic knot count must be an exact label")
    if checkpoint.get("stage") != "joint":
        reasons.append("checkpoint is not from the joint stage")
    if checkpoint.get("checkpoint_selection") != V16_CHECKPOINT_SELECTION_CONTRACT:
        reasons.append("checkpoint was not selected by mean per-curve subset cost")
    if checkpoint.get("checkpoint_quality") != V16_JOINT_CHECKPOINT_QUALITY:
        reasons.append("joint checkpoint quality is not supervised fit/count selected")
    if required_tolerance is not None:
        if recorded_tolerance is None:
            reasons.append("checkpoint MSE tolerance is missing or invalid")
        elif not math.isclose(
            recorded_tolerance, required_tolerance, rel_tol=1e-12, abs_tol=0.0
        ):
            reasons.append(
                f"checkpoint MSE tolerance {recorded_tolerance:.6g} differs from "
                f"the requested formal tolerance {required_tolerance:.6g}"
            )

    return {
        "schema_version": 6,
        "qualification_contract": (
            "v16_supervised_synthetic_only_pass_rates_report_only_v4"
        ),
        "required_reporting_pass_rate": required,
        "pass_rate_reference_only": True,
        "pass_rate_reference_met": bool(
            observed_qualification_dense is not None
            and observed_qualification_deployment is not None
            and observed_qualification_dense >= required
            and observed_qualification_deployment >= required
        ),
        "required_mse_tolerance": required_tolerance,
        "configured_proposal_pass_target": configured_proposal,
        "configured_deployment_pass_target": configured_deployment,
        "observed_worst_dense_pass_rate": observed_dense,
        "observed_worst_deployment_pass_rate": observed_deployment,
        "observed_qualification_dense_pass_rate": observed_qualification_dense,
        "observed_qualification_deployment_pass_rate": (
            observed_qualification_deployment
        ),
        "qualification_dense_consistent": bool(
            expected_qualification_dense is not None
            and observed_qualification_dense is not None
            and math.isclose(
                observed_qualification_dense,
                expected_qualification_dense,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ),
        "qualification_deployment_consistent": bool(
            expected_qualification_deployment is not None
            and observed_qualification_deployment is not None
            and math.isclose(
                observed_qualification_deployment,
                expected_qualification_deployment,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ),
        "recorded_mse_tolerance": recorded_tolerance,
        "recorded_knot_match_tolerance": recorded_match_tolerance,
        "candidate_knot_capacity": candidate_capacity,
        "formal_proposal_high_k_fraction": (
            V16_FORMAL_PROPOSAL_HIGH_K_FRACTION
        ),
        "formal_proposal_high_k_min_knots": (
            V16_FORMAL_PROPOSAL_HIGH_K_MIN_KNOTS
        ),
        "configured_proposal_high_k_fraction": proposal_high_k_fraction,
        "configured_proposal_high_k_min_knots": proposal_high_k_min_knots,
        "proposal_knot_assignment_weight": proposal_knot_assignment_weight,
        "training_real_fraction": training_real_fraction,
        "joint_supervision": joint_supervision,
        "online_teacher": loss_config.get("online_teacher"),
        "synthetic_count_role": synthetic_count_role,
        "formal_synthetic_min_internal_knots": (
            V16_FORMAL_SYNTHETIC_MIN_INTERNAL_KNOTS
        ),
        "formal_synthetic_max_internal_knots": (
            V16_FORMAL_SYNTHETIC_MAX_INTERNAL_KNOTS
        ),
        "formal_synthetic_min_control_points": (
            V16_FORMAL_SYNTHETIC_MIN_CONTROL_POINTS
        ),
        "formal_synthetic_max_control_points": (
            V16_FORMAL_SYNTHETIC_MAX_CONTROL_POINTS
        ),
        "formal_synthetic_knot_min_span": V16_FORMAL_SYNTHETIC_KNOT_MIN_SPAN,
        "formal_synthetic_boundary_samples_min": (
            V16_FORMAL_SYNTHETIC_BOUNDARY_SAMPLES_MIN
        ),
        "synthetic_min_control_points": synthetic_min_control_points,
        "synthetic_max_control_points": synthetic_max_control_points,
        "synthetic_knot_min_span": synthetic_knot_min_span,
        "synthetic_boundary_knot_count": synthetic_boundary_knot_count,
        "synthetic_boundary_sample_count": synthetic_boundary_sample_count,
        "synthetic_boundary_dense_pass_rate": (
            synthetic_boundary_dense_pass_rate
        ),
        "synthetic_boundary_deployment_pass_rate": (
            synthetic_boundary_deployment_pass_rate
        ),
        "observed_mean_retained_knots": observed_keep_count,
        "configured_target_met": configured_target_met,
        "configured_checkpoint_accepted": configured_checkpoint_accepted,
        "checkpoint_selection": checkpoint.get("checkpoint_selection"),
        "checkpoint_quality": checkpoint.get("checkpoint_quality"),
        "architecture_revision": architecture_revision,
        "selection_policy": selection_policy,
        "adaptive_keep_threshold": adaptive_threshold,
        "simplification_contract": simplification_contract,
        "simplification_ready": checkpoint.get("simplification_ready"),
        "aggregate_pass_feedback": simplification_curriculum.get(
            "aggregate_pass_feedback"
        ),
        "final_safety_sigma": final_safety_sigma,
        "applied_safety_sigma": applied_safety_sigma,
        "final_safety_knots": final_safety_knots,
        "applied_safety_knots": applied_safety_knots,
        "synthetic_count_mae": synthetic_count_mae,
        "synthetic_knot_match_f1": synthetic_knot_match_f1,
        "synthetic_knot_matched_mae": synthetic_knot_matched_mae,
        "formal_synthetic_count_mae_max": V16_FORMAL_SYNTHETIC_COUNT_MAE_MAX,
        "formal_synthetic_knot_f1_min": V16_FORMAL_SYNTHETIC_KNOT_F1_MIN,
        "formal_synthetic_matched_mae_max": (
            V16_FORMAL_SYNTHETIC_MATCHED_MAE_MAX
        ),
        "synthetic_data_contract": synthetic_contract,
        "certified_minimal_source": certified_synthetic,
        "synthetic_certificate_rms_tolerance": certificate_rms_tolerance,
        "synthetic_minimality_margin": minimality_margin,
        "synthetic_minimality_audit_points": minimality_audit_points,
        "formal_reporting_eligible": not reasons,
        "reasons": reasons,
    }


LEGACY_LOSS_CONFIG: dict[str, Any] = {
    "weights": {
        "fit": 1.0,
        "l0": 0.0,
        "activity": 2e-3,
        "binary": 2e-4,
        "orthogonal": 5e-2,
        "gap": 1e-2,
        "parameter_prior": 1e-2,
        "true_parameter": 0.0,
        "existence": 0.0,
        "knot_position": 0.0,
        "count": 0.0,
        "over_count": 0.0,
    },
    "min_knot_gap": 1e-3,
}

A_SCHEME_WITH_ORTHOGONAL_LOSS_CONFIG: dict[str, Any] = {
    "weights": {
        "fit": 1.0,
        "l0": 2e-5,
        "activity": 0.0,
        "binary": 0.0,
        "orthogonal": 5e-2,
        "gap": 1e-2,
        "parameter_prior": 1e-2,
        "true_parameter": 0.0,
        "existence": 0.0,
        "knot_position": 0.0,
        "count": 0.0,
        "over_count": 0.0,
    },
    "min_knot_gap": 1e-3,
}

CROSS_ATTENTION_LOSS_CONFIG: dict[str, Any] = {
    "weights": {
        "fit": 1.0,
        "l0": 2e-5,
        "activity": 0.0,
        "binary": 0.0,
        "orthogonal": 0.0,
        "gap": 1e-2,
        "parameter_prior": 0.0,
        "true_parameter": 1e-2,
        "existence": 0.0,
        "knot_position": 0.0,
        "count": 0.0,
        "over_count": 0.0,
    },
    "min_knot_gap": 1e-3,
}

INDEPENDENT_QUERY_V1_LOSS_CONFIG: dict[str, Any] = {
    "weights": {
        "fit": 1.0,
        "l0": 0.0,
        "activity": 0.0,
        "binary": 0.0,
        "orthogonal": 0.0,
        "gap": 0.0,
        "parameter_prior": 0.0,
        "true_parameter": 1e-2,
        "existence": 1e-3,
        "knot_position": 1e-2,
        "count": 1e-3,
        "over_count": 0.0,
    },
    "min_knot_gap": 1e-3,
    "knot_position_beta": 0.02,
}

A_SCHEME_LOSS_CONFIG: dict[str, Any] = {
    "weights": {
        "fit": 1.0,
        "l0": 0.0,
        "activity": 0.0,
        "binary": 0.0,
        # Compatibility field only; it is absent from the current objective.
        "orthogonal": 0.0,
        "gap": 0.0,
        "parameter_prior": 0.0,
        "true_parameter": 1e-2,
        "existence": 0.0,
        "knot_position": 1e-2,
        "count": 1e-2,
        "over_count": 0.0,
    },
    "min_knot_gap": 1e-3,
}

PREVIOUS_OBJECTIVE_LOSS_CONFIG: dict[str, Any] = {
    **deepcopy(A_SCHEME_LOSS_CONFIG),
    "weights": {
        **A_SCHEME_LOSS_CONFIG["weights"],
        "existence": 5e-3,
    },
}

INDEPENDENT_QUERY_V2_LOSS_CONFIG: dict[str, Any] = {
    **deepcopy(PREVIOUS_OBJECTIVE_LOSS_CONFIG),
    "weights": {
        **PREVIOUS_OBJECTIVE_LOSS_CONFIG["weights"],
        "count": 2e-3,
    },
}

CURRENT_OBJECTIVE_LOSS_CONFIG: dict[str, Any] = {
    **deepcopy(A_SCHEME_LOSS_CONFIG),
    "weights": {
        **A_SCHEME_LOSS_CONFIG["weights"],
        "true_parameter": 5e-2,
        "knot_position": 5e-2,
        "count": 5e-3,
        "over_count": 2e-3,
    },
    "count_loss": "structured_continuation_binary_cross_entropy",
}

COUNT_CONDITIONED_V5_LOSS_CONFIG: dict[str, Any] = {
    **deepcopy(CURRENT_OBJECTIVE_LOSS_CONFIG),
    "count_loss": "ordinal_binary_cross_entropy",
}

CANDIDATE_PRUNING_LOSS_CONFIG: dict[str, Any] = {
    "weights": {
        "fit": 0.25,
        "threshold_violation": 5.0,
        "true_parameter": 5e-2,
        "candidate_coverage": 5.0,
        "candidate_repulsion": 5e-2,
        "keep": 2.5e-1,
        "remove_action": 1.0,
        "knot_position": 2.0,
        "count_consistency": 0.0,
        "deletion_cost": 5e-2,
    },
    "knot_position_beta": 0.01,
    "candidate_match_tolerance": 0.02,
    "fit_tolerance": 5e-3,
    "positive_keep_weight": 2.0,
    "exact_deletion_supervision": True,
    "deletion_smoothness_weight": 1e-6,
    "deletion_control_ridge": 0.0,
}

ONE_SHOT_PRUNING_LOSS_CONFIG: dict[str, Any] = {
    "weights": {
        **CANDIDATE_PRUNING_LOSS_CONFIG["weights"],
        "fit": 0.0,
        "threshold_violation": 0.0,
        "true_parameter": 0.0,
        "candidate_coverage": 0.0,
        "candidate_repulsion": 0.0,
        "keep": 1.0,
        "remove_action": 0.0,
        "deletion_cost": 0.0,
        "teacher_risk": 0.5,
        "teacher_ranking": 1.0,
        "teacher_count": 2.0,
        "complexity": 0.25,
        "knot_position": 4.0,
    },
    **{
        key: value
        for key, value in CANDIDATE_PRUNING_LOSS_CONFIG.items()
        if key != "weights"
    },
    "positive_keep_weight": 1.0,
    "candidate_match_tolerance": 0.01,
    "teacher_ranking_margin": 1.0,
    "one_shot_teacher": True,
    "straight_through_keep_gate": True,
    "surrogate_selection_role": "diagnostic_only",
    "teacher_mask_loss": "unweighted_bce_plus_soft_dice",
}

V9_ONE_SHOT_PRUNING_LOSS_CONFIG: dict[str, Any] = {
    **deepcopy(ONE_SHOT_PRUNING_LOSS_CONFIG),
    "weights": {
        **ONE_SHOT_PRUNING_LOSS_CONFIG["weights"],
        # Candidate localization is completed before teacher generation.
        # Distillation must not move slots away from cached Hard-RMS labels.
        "knot_position": 0.0,
    },
    "fixed_proposal_geometry": True,
    "selector_adapter": "position_aware_self_attention",
}

V10_FEASIBLE_ONE_SHOT_PRUNING_LOSS_CONFIG: dict[str, Any] = {
    **deepcopy(V9_ONE_SHOT_PRUNING_LOSS_CONFIG),
    "weights": {
        **V9_ONE_SHOT_PRUNING_LOSS_CONFIG["weights"],
        "teacher_distribution": 2.0,
        "teacher_critical_recall": 1.0,
        "teacher_count": 4.0,
        "complexity": 0.1,
    },
    "positive_keep_weight": 2.0,
    "teacher_mask_loss": ("weighted_bce_plus_soft_dice_plus_cdf_plus_critical_recall"),
    "selection_policy": "mass_topk",
}

V11_JOINT_ONE_SHOT_PRUNING_LOSS_CONFIG: dict[str, Any] = {
    **deepcopy(V10_FEASIBLE_ONE_SHOT_PRUNING_LOSS_CONFIG),
    "weights": {
        **V10_FEASIBLE_ONE_SHOT_PRUNING_LOSS_CONFIG["weights"],
        "fit": 0.05,
        "threshold_violation": 0.5,
        "knot_position": 1.0,
        "teacher_false_positive": 1.0,
        "policy_count": 4.0,
        "canonical_selection": 0.5,
    },
    "positive_keep_weight": 1.5,
    "candidate_coverage_tolerances": [0.005, 0.01, 0.02],
    "position_aware_distribution": True,
    "joint_position_supervision": True,
    "teacher_mask_loss": (
        "teacher_bce_dice_ranking_plus_canonical_set_and_policy_count"
    ),
    "selection_policy": "mass_topk",
}

# v12 supervises the deployed survivor locations with the packed positions
# produced by the delete-then-relax teacher.  These flags matter when a compact
# checkpoint omits ``loss_config`` (for example, an exported inference model):
# falling back to v11's canonical matching would silently reconstruct the
# wrong training objective.
V12_COUPLED_RELOCATION_LOSS_CONFIG: dict[str, Any] = {
    **deepcopy(V11_JOINT_ONE_SHOT_PRUNING_LOSS_CONFIG),
    "joint_position_supervision": False,
    "teacher_relocation_supervision": True,
    "teacher_survivor_spacing_weight": 0.25,
    "selector_adapter": "final_keepmask_conditioned_survivor_relocation",
}

# v13 keeps v12's deployed architecture but fixes the training semantics:
# surrogate truncated-power terms are opt-in diagnostics, survivor positions
# are matched as ordered sets against the relocated teacher, and the existing
# distribution weight also provides slot-invariant soft teacher-set coverage.
V13_SET_RELOCATION_LOSS_CONFIG: dict[str, Any] = {
    **deepcopy(V12_COUPLED_RELOCATION_LOSS_CONFIG),
    "weights": {
        **V12_COUPLED_RELOCATION_LOSS_CONFIG["weights"],
        "fit": 0.0,
        "threshold_violation": 0.0,
    },
    "teacher_position_assignment": "actual_keepmask_ordered_set_matching",
    "teacher_distribution_target": ("relocated_teacher_set_cdf_plus_local_coverage"),
    "stable_pilot_descriptors": True,
    "surrogate_selection_role": "diagnostic_only",
}

# v14 leaves proposal, selection and survivor relocation in the v13 parameter
# domain, then calibrates a small late head from chord/pilot/Keep feedback.  The
# stage-specific loss is stored separately by the training script; these
# defaults keep compact inference checkpoints interpretable.
V14_PARAMETER_FEEDBACK_LOSS_CONFIG: dict[str, Any] = {
    **deepcopy(V13_SET_RELOCATION_LOSS_CONFIG),
    "parameter_feedback_fusion": True,
    "parameter_feedback_loss": {
        "fit": 0.10,
        "threshold_violation": 0.50,
        "true_parameter": 1.00,
        "chord_prior": 0.05,
        "identity": 0.01,
        "parameter_error_scale": 0.02,
        "initial_chord_blend": 0.60,
    },
}

V14_JOINT_PARAMETER_STRUCTURE_LOSS_CONFIG: dict[str, Any] = {
    **deepcopy(V14_PARAMETER_FEEDBACK_LOSS_CONFIG),
    "joint_parameter_structure_feedback": True,
    "parameter_feedback_loss": {
        **V14_PARAMETER_FEEDBACK_LOSS_CONFIG["parameter_feedback_loss"],
        "joint_keep": 1.0,
        "joint_position": 2.0,
        "joint_count": 2.0,
    },
}

# v15 keeps the v14 one-shot network intact, but calibrates it against the
# actual hard-mask standard-B-spline refit.  Its count target follows the exact
# mass_topk rounding score (including the uncertainty reserve), and its set
# losses supervise every deployed survivor instead of only matched pairs.
V15_DEPLOYMENT_ALIGNED_LOSS_CONFIG: dict[str, Any] = {
    **deepcopy(V14_JOINT_PARAMETER_STRUCTURE_LOSS_CONFIG),
    "parameter_feedback_loss": {
        **V14_JOINT_PARAMETER_STRUCTURE_LOSS_CONFIG["parameter_feedback_loss"],
        "fit": 0.0,
        "threshold_violation": 0.0,
        "deployment_fit": 0.25,
        "deployment_threshold_violation": 2.0,
        "joint_critical_recall": 0.5,
        "joint_set_position": 1.0,
        "joint_spacing": 0.5,
        "joint_count_target": "mass_topk_requested_score_inside_ceil_interval",
        "deployment_refit": "differentiable_standard_bspline_endpoint_constrained",
    },
    "deployment_aligned_feedback": True,
}


def migrate_model_config(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Return an explicit model config and whether the checkpoint is legacy.

    Pre-A checkpoints have no ``gate_mode``.  Treating those sigmoid-head
    weights as Hard-Concrete logits would silently change their meaning, so
    absence of the field is deliberately migrated to ``legacy_soft``.
    """
    if "model_config" not in checkpoint:
        raise KeyError("checkpoint is missing model_config")
    config = dict(checkpoint["model_config"])
    objective_version = checkpoint.get("objective_version")
    if objective_version in V16_OBJECTIVE_VERSIONS:
        # A separate architecture, not another set of defaults for the legacy
        # SplineFittingNetwork. Never inject old head/gate configuration fields.
        config.setdefault("structure_mode", "candidate_pruning_one_shot")
        if config["structure_mode"] != "candidate_pruning_one_shot":
            raise ValueError("v16 requires candidate_pruning_one_shot structure_mode")
        return config, False
    if "structure_mode" not in config:
        if objective_version in {
            ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
            V9_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
            V10_FEASIBLE_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
            V11_JOINT_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
            V12_COUPLED_RELOCATION_OBJECTIVE_VERSION,
            V13_SET_RELOCATION_OBJECTIVE_VERSION,
            V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
            V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION,
            V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
        }:
            config["structure_mode"] = "candidate_pruning_one_shot"
        elif objective_version == CANDIDATE_PRUNING_OBJECTIVE_VERSION:
            config["structure_mode"] = "candidate_pruning"
        elif objective_version == CURRENT_OBJECTIVE_VERSION:
            config["structure_mode"] = "interactive_dynamic"
        elif objective_version in {
            COUNT_CONDITIONED_V5_OBJECTIVE_VERSION,
            COUNT_CONDITIONED_V4_OBJECTIVE_VERSION,
        }:
            config["structure_mode"] = "count_conditioned"
        else:
            config["structure_mode"] = "hard_concrete"
    # These are compatibility defaults, not new-training defaults.  Explicit
    # values recorded by a checkpoint always win; missing historical metadata
    # must retain the old ParameterHead/CandidateKnotHead forward semantics.
    config.setdefault("parameter_gap_reference", "learned")
    config.setdefault("parameter_residual_logit_limit", 0.5)
    config.setdefault("count_attention_heads", 4)
    is_v5 = (
        checkpoint.get("objective_version") == COUNT_CONDITIONED_V5_OBJECTIVE_VERSION
    )
    config.setdefault(
        "count_head_mode",
        "ordinal_local_attention" if is_v5 else "categorical_global",
    )
    config.setdefault("count_query_count", 4)
    config.setdefault("structure_attention_heads", 4)
    if config["structure_mode"] in {
        "candidate_pruning",
        "candidate_pruning_one_shot",
    }:
        config.setdefault("pruning_residual_bandwidth", 0.05)
        config.setdefault("pruning_initial_keep_probability", 0.9)
        config.setdefault(
            "one_shot_fixed_proposal_geometry",
            objective_version
            in {
                V9_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
                V10_FEASIBLE_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
                V11_JOINT_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
                V12_COUPLED_RELOCATION_OBJECTIVE_VERSION,
                V13_SET_RELOCATION_OBJECTIVE_VERSION,
                V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
                V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION,
                V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
            },
        )
        config.setdefault(
            "one_shot_selection_policy",
            "mass_topk"
            if objective_version
            in {
                V10_FEASIBLE_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
                V11_JOINT_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
                V12_COUPLED_RELOCATION_OBJECTIVE_VERSION,
                V13_SET_RELOCATION_OBJECTIVE_VERSION,
                V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
                V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION,
                V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
            }
            else "threshold",
        )
        config.setdefault("one_shot_safety_sigma", 0.0)
        config.setdefault(
            "one_shot_selector_layers",
            2
            if objective_version
            in {
                V10_FEASIBLE_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
                V11_JOINT_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
                V12_COUPLED_RELOCATION_OBJECTIVE_VERSION,
                V13_SET_RELOCATION_OBJECTIVE_VERSION,
                V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
                V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION,
                V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
            }
            else 1,
        )
        config.setdefault(
            "one_shot_coverage_bins",
            4
            if objective_version == V10_FEASIBLE_ONE_SHOT_PRUNING_OBJECTIVE_VERSION
            else 0,
        )
        config.setdefault("candidate_local_attention_bandwidth", 0.0)
        config.setdefault("candidate_interval_logit_limit", 0.0)
        config.setdefault(
            "candidate_position_parameterization",
            "interval_softmax",
        )
        config.setdefault(
            "one_shot_joint_position_refinement",
            objective_version
            in {
                V11_JOINT_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
                V12_COUPLED_RELOCATION_OBJECTIVE_VERSION,
                V13_SET_RELOCATION_OBJECTIVE_VERSION,
                V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
                V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION,
                V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
            },
        )
        config.setdefault(
            "one_shot_max_position_shift",
            0.15
            if objective_version
            in {
                V12_COUPLED_RELOCATION_OBJECTIVE_VERSION,
                V13_SET_RELOCATION_OBJECTIVE_VERSION,
                V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
                V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION,
                V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
            }
            else 0.05,
        )
        config.setdefault(
            "one_shot_survivor_relocation",
            objective_version
            in {
                V12_COUPLED_RELOCATION_OBJECTIVE_VERSION,
                V13_SET_RELOCATION_OBJECTIVE_VERSION,
                V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
                V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION,
                V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
            },
        )
        config.setdefault(
            "stable_pilot_descriptors",
            objective_version
            in {
                V13_SET_RELOCATION_OBJECTIVE_VERSION,
                V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
                V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION,
                V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
            },
        )
        config.setdefault(
            "parameter_feedback_fusion",
            objective_version
            in {
                V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
                V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION,
                V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
            },
        )
        config.setdefault("parameter_feedback_attention_heads", None)
        config.setdefault("parameter_feedback_max_logit_shift", 0.5)
        config.setdefault(
            "parameter_feedback_fusion_mode",
            (
                "cross_attention"
                if objective_version
                in {
                    V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION,
                    V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
                }
                else "fast_global"
            ),
        )
        config.setdefault(
            "joint_parameter_structure_feedback",
            objective_version
            in {
                V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION,
                V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
            },
        )
        config.setdefault("joint_parameter_structure_local_bandwidth", 0.08)
        config.setdefault(
            "joint_parameter_structure_max_keep_logit_shift",
            2.0,
        )
        config.setdefault("enforce_ordered_joint_candidates", False)
    if config["structure_mode"] == "interactive_dynamic":
        # v6 checkpoints written before categorical count prediction contain
        # stop_head/stop_bias tensors. Missing metadata must therefore rebuild
        # the historical hazard layout for strict state-dict loading.
        config.setdefault("structure_count_mode", "hazard")
        if "min_internal_knots" not in config:
            dataset_config = checkpoint.get("dataset_config", {})
            canonical_tolerance = float(
                dataset_config.get("canonical_knot_tolerance", 5e-3)
            )
            if canonical_tolerance == 0.0:
                degree = int(config.get("degree", 3))
                source_minimum = int(
                    dataset_config.get("min_control_points", degree + 1)
                ) - (degree + 1)
                config["min_internal_knots"] = max(
                    0,
                    min(
                        source_minimum,
                        int(config.get("max_internal_knots", source_minimum)),
                    ),
                )
            else:
                config["min_internal_knots"] = 0
    config.setdefault(
        "count_decoder_mode",
        "shared_count_embedding" if is_v5 else "independent_branches",
    )
    config.setdefault(
        "geometry_feature_mode",
        (
            "chord_derivatives"
            if checkpoint.get("objective_version")
            in {
                CANDIDATE_PRUNING_OBJECTIVE_VERSION,
                ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
                V9_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
                V10_FEASIBLE_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
                V11_JOINT_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
                V12_COUPLED_RELOCATION_OBJECTIVE_VERSION,
                V13_SET_RELOCATION_OBJECTIVE_VERSION,
                V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
                V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION,
                V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
                CURRENT_OBJECTIVE_VERSION,
                COUNT_CONDITIONED_V5_OBJECTIVE_VERSION,
            }
            else "raw_differences"
        ),
    )
    legacy = config["structure_mode"] == "hard_concrete" and "gate_mode" not in config
    if legacy:
        config["gate_mode"] = "legacy_soft"
        config["activity_use_local_context"] = False
        config["gap_parameterization"] = "legacy"
        config["activity_use_pilot_importance"] = False
    elif "activity_use_pilot_importance" not in config:
        # Preserve checkpoints trained before the pilot-importance feature was
        # introduced instead of silently changing their activity logits.
        config["activity_use_pilot_importance"] = False
    if "knot_use_local_cross_attention" not in config:
        # Every checkpoint created before the cross-attention KnotHead used the
        # global-only MLP. Reconstruct that exact module layout for strict load.
        config["knot_use_local_cross_attention"] = False
    if "knot_parameterization" not in config:
        # v0.4 and earlier shared K+1 interval logits. Recreate that exact
        # parameter layout; every new checkpoint records this field explicitly.
        config["knot_parameterization"] = "interval"
    if "activity_use_query_features" not in config:
        config["activity_use_query_features"] = False
    if "activity_use_candidate_self_attention" not in config:
        # v2 and earlier activity logits were independent across candidates.
        # A partially specified current config follows the current model
        # default, while historical checkpoints retain their exact layout.
        config["activity_use_candidate_self_attention"] = (
            checkpoint.get("objective_version") == PREVIOUS_OBJECTIVE_VERSION
        )
    if "activity_candidate_attention_heads" not in config:
        config["activity_candidate_attention_heads"] = 4
    if "detach_activity_gate_for_fit" not in config:
        # v1 and all earlier checkpoints allowed fit gradients through the
        # gate. Incomplete v2/v3 configs receive stop-gradient behavior;
        # historical resumed training keeps its exact optimization semantics.
        config["detach_activity_gate_for_fit"] = checkpoint.get(
            "objective_version"
        ) in {
            PREVIOUS_OBJECTIVE_VERSION,
            INDEPENDENT_QUERY_V2_OBJECTIVE_VERSION,
        }
    if "compute_first_derivative" not in config:
        saved_weights = checkpoint.get("loss_config", {}).get("weights", {})
        if "orthogonal" in saved_weights:
            needs_derivative = float(saved_weights["orthogonal"]) != 0.0
        else:
            # Every checkpoint predating CURRENT_OBJECTIVE_VERSION used the
            # projection-orthogonality objective (or its historical default).
            needs_derivative = checkpoint.get("objective_version") not in {
                CURRENT_OBJECTIVE_VERSION,
                CANDIDATE_PRUNING_OBJECTIVE_VERSION,
                ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
                V9_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
                V10_FEASIBLE_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
                V11_JOINT_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
                V12_COUPLED_RELOCATION_OBJECTIVE_VERSION,
                V13_SET_RELOCATION_OBJECTIVE_VERSION,
                V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
                V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION,
                V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
                COUNT_CONDITIONED_V5_OBJECTIVE_VERSION,
                COUNT_CONDITIONED_V4_OBJECTIVE_VERSION,
                PREVIOUS_OBJECTIVE_VERSION,
                INDEPENDENT_QUERY_V2_OBJECTIVE_VERSION,
                INDEPENDENT_QUERY_V1_OBJECTIVE_VERSION,
                CROSS_ATTENTION_OBJECTIVE_VERSION,
            }
        config["compute_first_derivative"] = needs_derivative
    return config, legacy


def migrate_loss_config(
    checkpoint: Mapping[str, Any],
    *,
    legacy: bool,
) -> tuple[dict[str, Any], bool]:
    """Load loss metadata without inventing an L0 term for old checkpoints."""
    assumed = "loss_config" not in checkpoint
    objective_version = checkpoint.get("objective_version")
    current_objective = objective_version == CURRENT_OBJECTIVE_VERSION
    no_orthogonal_objective = objective_version in {
        CURRENT_OBJECTIVE_VERSION,
        CANDIDATE_PRUNING_OBJECTIVE_VERSION,
        ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
        V9_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
        V10_FEASIBLE_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
        V11_JOINT_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
        V12_COUPLED_RELOCATION_OBJECTIVE_VERSION,
        V13_SET_RELOCATION_OBJECTIVE_VERSION,
        V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
        V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION,
        V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
        COUNT_CONDITIONED_V5_OBJECTIVE_VERSION,
        COUNT_CONDITIONED_V4_OBJECTIVE_VERSION,
        PREVIOUS_OBJECTIVE_VERSION,
        INDEPENDENT_QUERY_V2_OBJECTIVE_VERSION,
        INDEPENDENT_QUERY_V1_OBJECTIVE_VERSION,
        CROSS_ATTENTION_OBJECTIVE_VERSION,
    }
    if assumed:
        if legacy:
            default_config = LEGACY_LOSS_CONFIG
        elif objective_version == V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION:
            default_config = V15_DEPLOYMENT_ALIGNED_LOSS_CONFIG
        elif objective_version == V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION:
            default_config = V14_JOINT_PARAMETER_STRUCTURE_LOSS_CONFIG
        elif objective_version == V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION:
            default_config = V14_PARAMETER_FEEDBACK_LOSS_CONFIG
        elif objective_version == V13_SET_RELOCATION_OBJECTIVE_VERSION:
            default_config = V13_SET_RELOCATION_LOSS_CONFIG
        elif objective_version == V12_COUPLED_RELOCATION_OBJECTIVE_VERSION:
            default_config = V12_COUPLED_RELOCATION_LOSS_CONFIG
        elif objective_version == V11_JOINT_ONE_SHOT_PRUNING_OBJECTIVE_VERSION:
            default_config = V11_JOINT_ONE_SHOT_PRUNING_LOSS_CONFIG
        elif objective_version == V10_FEASIBLE_ONE_SHOT_PRUNING_OBJECTIVE_VERSION:
            default_config = V10_FEASIBLE_ONE_SHOT_PRUNING_LOSS_CONFIG
        elif objective_version == V9_ONE_SHOT_PRUNING_OBJECTIVE_VERSION:
            default_config = V9_ONE_SHOT_PRUNING_LOSS_CONFIG
        elif objective_version == ONE_SHOT_PRUNING_OBJECTIVE_VERSION:
            default_config = ONE_SHOT_PRUNING_LOSS_CONFIG
        elif objective_version == CANDIDATE_PRUNING_OBJECTIVE_VERSION:
            default_config = CANDIDATE_PRUNING_LOSS_CONFIG
        elif current_objective:
            default_config = CURRENT_OBJECTIVE_LOSS_CONFIG
        elif objective_version == COUNT_CONDITIONED_V5_OBJECTIVE_VERSION:
            default_config = COUNT_CONDITIONED_V5_LOSS_CONFIG
        elif objective_version == COUNT_CONDITIONED_V4_OBJECTIVE_VERSION:
            default_config = A_SCHEME_LOSS_CONFIG
        elif objective_version == PREVIOUS_OBJECTIVE_VERSION:
            default_config = PREVIOUS_OBJECTIVE_LOSS_CONFIG
        elif objective_version == INDEPENDENT_QUERY_V2_OBJECTIVE_VERSION:
            default_config = INDEPENDENT_QUERY_V2_LOSS_CONFIG
        elif objective_version == INDEPENDENT_QUERY_V1_OBJECTIVE_VERSION:
            default_config = INDEPENDENT_QUERY_V1_LOSS_CONFIG
        elif objective_version == CROSS_ATTENTION_OBJECTIVE_VERSION:
            default_config = CROSS_ATTENTION_LOSS_CONFIG
        else:
            default_config = A_SCHEME_WITH_ORTHOGONAL_LOSS_CONFIG
        config = deepcopy(default_config)
    else:
        config = deepcopy(checkpoint["loss_config"])
        if objective_version in {
            CANDIDATE_PRUNING_OBJECTIVE_VERSION,
            ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
            V9_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
            V10_FEASIBLE_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
            V11_JOINT_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
            V12_COUPLED_RELOCATION_OBJECTIVE_VERSION,
            V13_SET_RELOCATION_OBJECTIVE_VERSION,
            V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
            V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION,
            V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
        }:
            defaults = (
                V15_DEPLOYMENT_ALIGNED_LOSS_CONFIG
                if objective_version == V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION
                else (
                    V14_JOINT_PARAMETER_STRUCTURE_LOSS_CONFIG
                    if objective_version
                    == V14_JOINT_PARAMETER_STRUCTURE_OBJECTIVE_VERSION
                    else (
                        V14_PARAMETER_FEEDBACK_LOSS_CONFIG
                        if objective_version == V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION
                        else (
                            V13_SET_RELOCATION_LOSS_CONFIG
                            if objective_version
                            == V13_SET_RELOCATION_OBJECTIVE_VERSION
                            else (
                                V12_COUPLED_RELOCATION_LOSS_CONFIG
                                if objective_version
                                == V12_COUPLED_RELOCATION_OBJECTIVE_VERSION
                                else (
                                    V11_JOINT_ONE_SHOT_PRUNING_LOSS_CONFIG
                                    if objective_version
                                    == V11_JOINT_ONE_SHOT_PRUNING_OBJECTIVE_VERSION
                                    else (
                                        V10_FEASIBLE_ONE_SHOT_PRUNING_LOSS_CONFIG
                                        if objective_version
                                        == V10_FEASIBLE_ONE_SHOT_PRUNING_OBJECTIVE_VERSION
                                        else (
                                            V9_ONE_SHOT_PRUNING_LOSS_CONFIG
                                            if objective_version
                                            == V9_ONE_SHOT_PRUNING_OBJECTIVE_VERSION
                                            else (
                                                ONE_SHOT_PRUNING_LOSS_CONFIG
                                                if objective_version
                                                == ONE_SHOT_PRUNING_OBJECTIVE_VERSION
                                                else CANDIDATE_PRUNING_LOSS_CONFIG
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            weights = config.setdefault("weights", {})
            for name, value in defaults["weights"].items():
                weights.setdefault(name, value)
            for name, value in defaults.items():
                if name != "weights":
                    config.setdefault(name, value)
            return config, assumed
        weights = config.setdefault("weights", {})
        weights.setdefault(
            "l0",
            0.0 if legacy else A_SCHEME_LOSS_CONFIG["weights"]["l0"],
        )
        weights.setdefault("activity", 0.0)
        weights.setdefault("binary", 0.0)
        weights.setdefault("true_parameter", 0.0)
        weights.setdefault("existence", 0.0)
        weights.setdefault("knot_position", 0.0)
        weights.setdefault("count", 0.0)
        weights.setdefault("over_count", 0.0)
        weights.setdefault(
            "orthogonal",
            (
                0.0
                if no_orthogonal_objective
                else LEGACY_LOSS_CONFIG["weights"]["orthogonal"]
            ),
        )
    config.setdefault("knot_position_beta", 0.02)
    return config, assumed


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> tuple[nn.Module, dict[str, Any], bool]:
    """Strictly restore v16 or the unchanged historical network architecture."""
    if checkpoint.get("objective_version") in V16_OBJECTIVE_VERSIONS:
        from .models.v16_network import V16CandidateSelectionNetwork

        config, legacy = migrate_model_config(checkpoint)
        model = V16CandidateSelectionNetwork(**config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model, model.get_config(), legacy
    config, legacy = migrate_model_config(checkpoint)
    model = SplineFittingNetwork(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    temperature = checkpoint.get("gate_temperature")
    if temperature is not None and hasattr(model, "set_gate_temperature"):
        model.set_gate_temperature(float(temperature))
    if hasattr(model, "set_force_open_gates"):
        model.set_force_open_gates(False)
    return model, config, legacy
