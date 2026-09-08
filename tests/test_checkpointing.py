from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import (  # noqa: E402
    COUNT_CONDITIONED_V4_OBJECTIVE_VERSION,
    COUNT_CONDITIONED_V5_OBJECTIVE_VERSION,
    CURRENT_OBJECTIVE_VERSION,
    PREVIOUS_OBJECTIVE_VERSION,
    V11_JOINT_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
    V12_COUPLED_RELOCATION_OBJECTIVE_VERSION,
    V13_SET_RELOCATION_OBJECTIVE_VERSION,
    V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
    V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
    build_model_from_checkpoint,
    migrate_loss_config,
    migrate_model_config,
)
from spline_fitting.models.spline_network import SplineFittingNetwork  # noqa: E402


class CheckpointMigrationTests(unittest.TestCase):
    def test_v15_compact_checkpoint_restores_deployment_aligned_objective(self) -> None:
        checkpoint = {
            "objective_version": V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
            "model_config": {"point_dim": 2},
        }

        model_config, legacy = migrate_model_config(checkpoint)
        loss_config, assumed = migrate_loss_config(checkpoint, legacy=legacy)

        self.assertFalse(legacy)
        self.assertTrue(assumed)
        self.assertEqual(model_config["structure_mode"], "candidate_pruning_one_shot")
        self.assertTrue(model_config["joint_parameter_structure_feedback"])
        self.assertEqual(model_config["parameter_feedback_fusion_mode"], "cross_attention")
        feedback = loss_config["parameter_feedback_loss"]
        self.assertEqual(feedback["deployment_fit"], 0.25)
        self.assertEqual(feedback["deployment_threshold_violation"], 2.0)
        self.assertEqual(
            feedback["joint_count_target"],
            "mass_topk_requested_score_inside_ceil_interval",
        )

    def test_explicit_uniform_parameter_reference_restores_strictly(self) -> None:
        config = {
            "point_dim": 2,
            "hidden_dim": 16,
            "encoder_layers": 1,
            "max_internal_knots": 3,
            "structure_mode": "count_conditioned",
            "count_attention_heads": 4,
            "parameter_gap_reference": "uniform_residual",
            "parameter_residual_logit_limit": 2.5,
        }
        reference = SplineFittingNetwork(**config).eval()
        checkpoint = {
            "objective_version": COUNT_CONDITIONED_V5_OBJECTIVE_VERSION,
            "model_config": config,
            "model_state_dict": reference.state_dict(),
        }

        restored, migrated, legacy = build_model_from_checkpoint(checkpoint)

        self.assertFalse(legacy)
        self.assertEqual(migrated["parameter_gap_reference"], "uniform_residual")
        self.assertEqual(migrated["parameter_residual_logit_limit"], 2.5)
        points = torch.randn(2, 12, 2)
        with torch.no_grad():
            expected = reference(points)
            actual = restored.eval()(points)
        torch.testing.assert_close(actual["params"], expected["params"])
        torch.testing.assert_close(actual["internal_knots"], expected["internal_knots"])

    def test_pre_a_checkpoint_restores_legacy_forward_semantics(self) -> None:
        torch.manual_seed(4)
        base_config = {
            "point_dim": 2,
            "degree": 3,
            "hidden_dim": 16,
            "encoder_layers": 1,
            "max_internal_knots": 3,
            "min_parameter_gap": 1e-4,
            "min_knot_gap": 1e-3,
            "lambda_poly": 1e-6,
            "lambda_knot": 1e-4,
            "gate_eps": 1e-6,
        }
        reference_model = SplineFittingNetwork(
            **base_config,
            gate_mode="legacy_soft",
            gap_parameterization="legacy",
            activity_use_local_context=False,
            compute_first_derivative=True,
        ).eval()
        legacy_state = {
            key: value
            for key, value in reference_model.state_dict().items()
            if not key.endswith("_temperature_buffer")
        }
        checkpoint = {
            "model_config": base_config,
            "model_state_dict": legacy_state,
            "loss_config": {
                "weights": {
                    "fit": 1.0,
                    "activity": 0.002,
                    "binary": 0.0002,
                    "orthogonal": 0.05,
                    "gap": 0.01,
                    "parameter_prior": 0.01,
                }
            },
        }
        restored, config, legacy = build_model_from_checkpoint(checkpoint)
        restored.eval()
        points = torch.randn(2, 16, 2)

        with torch.no_grad():
            expected = reference_model(points)
            actual = restored(points)

        self.assertTrue(legacy)
        self.assertEqual(config["gate_mode"], "legacy_soft")
        self.assertEqual(config["gap_parameterization"], "legacy")
        self.assertFalse(config["activity_use_local_context"])
        self.assertTrue(config["compute_first_derivative"])
        for key in ("params", "internal_knots", "activity", "reconstructed_points"):
            torch.testing.assert_close(actual[key], expected[key])

        loss_config, assumed = migrate_loss_config(checkpoint, legacy=True)
        self.assertFalse(assumed)
        self.assertEqual(loss_config["weights"]["l0"], 0.0)

    def test_explicit_a_config_is_not_marked_legacy(self) -> None:
        checkpoint = {
            "model_config": {
                "point_dim": 2,
                "gate_mode": "hard_concrete",
                "gap_parameterization": "strict",
            }
        }
        config, legacy = migrate_model_config(checkpoint)
        self.assertFalse(legacy)
        self.assertEqual(config["gate_mode"], "hard_concrete")
        self.assertTrue(config["compute_first_derivative"])

    def test_current_objective_disables_derivative_and_orthogonal_loss(self) -> None:
        checkpoint = {
            "objective_version": CURRENT_OBJECTIVE_VERSION,
            "model_config": {
                "point_dim": 2,
                "gate_mode": "hard_concrete",
                "gap_parameterization": "strict",
            },
        }

        config, legacy = migrate_model_config(checkpoint)
        loss_config, assumed = migrate_loss_config(checkpoint, legacy=legacy)

        self.assertFalse(legacy)
        self.assertFalse(config["compute_first_derivative"])
        self.assertEqual(config["structure_mode"], "interactive_dynamic")
        self.assertEqual(config["structure_count_mode"], "hazard")
        self.assertEqual(config["min_internal_knots"], 0)
        self.assertTrue(assumed)
        self.assertEqual(loss_config["weights"]["orthogonal"], 0.0)
        self.assertEqual(loss_config["weights"]["true_parameter"], 5e-2)
        self.assertEqual(loss_config["weights"]["over_count"], 2e-3)
        self.assertEqual(loss_config["weights"]["existence"], 0.0)
        self.assertEqual(loss_config["weights"]["count"], 5e-3)
        self.assertFalse(config["activity_use_candidate_self_attention"])

    def test_cross_attention_checkpoint_restores_strictly(self) -> None:
        config = {
            "point_dim": 2,
            "hidden_dim": 16,
            "encoder_layers": 1,
            "max_internal_knots": 3,
            "gate_mode": "hard_concrete",
            "activity_use_pilot_importance": False,
            "knot_use_local_cross_attention": True,
            "knot_attention_heads": 4,
            "knot_parameterization": "independent_queries",
            "activity_use_query_features": True,
            "detach_activity_gate_for_fit": True,
        }
        reference = SplineFittingNetwork(**config).eval()
        checkpoint = {
            "objective_version": PREVIOUS_OBJECTIVE_VERSION,
            "model_config": config,
            "model_state_dict": reference.state_dict(),
        }

        restored, migrated, legacy = build_model_from_checkpoint(checkpoint)

        self.assertFalse(legacy)
        self.assertEqual(migrated["structure_mode"], "hard_concrete")
        self.assertTrue(migrated["knot_use_local_cross_attention"])
        points = torch.randn(2, 12, 2)
        with torch.no_grad():
            expected = reference(points)
            actual = restored.eval()(points)
        torch.testing.assert_close(actual["internal_knots"], expected["internal_knots"])

    def test_current_interactive_checkpoint_restores_strictly(self) -> None:
        config = {
            "point_dim": 2,
            "hidden_dim": 16,
            "encoder_layers": 1,
            "max_internal_knots": 3,
            "structure_mode": "interactive_dynamic",
            "structure_attention_heads": 4,
        }
        reference = SplineFittingNetwork(
            **config,
            structure_count_mode="hazard",
            min_internal_knots=0,
        ).eval()
        checkpoint = {
            "objective_version": CURRENT_OBJECTIVE_VERSION,
            "model_config": config,
            "model_state_dict": reference.state_dict(),
        }

        restored, migrated, legacy = build_model_from_checkpoint(checkpoint)

        self.assertFalse(legacy)
        self.assertEqual(migrated["structure_mode"], "interactive_dynamic")
        self.assertEqual(migrated["structure_count_mode"], "hazard")
        self.assertEqual(migrated["min_internal_knots"], 0)
        points = torch.randn(2, 12, 2)
        with torch.no_grad():
            expected = reference(points)
            actual = restored.eval()(points)
        torch.testing.assert_close(actual["count_logits"], expected["count_logits"])
        torch.testing.assert_close(actual["internal_knots"], expected["internal_knots"])

    def test_old_high_k_interactive_checkpoint_infers_legal_minimum(self) -> None:
        saved_config = {
            "point_dim": 2,
            "hidden_dim": 16,
            "encoder_layers": 1,
            "max_internal_knots": 20,
            "structure_mode": "interactive_dynamic",
            "structure_attention_heads": 4,
        }
        reference = SplineFittingNetwork(
            **saved_config,
            structure_count_mode="hazard",
            min_internal_knots=4,
        ).eval()
        checkpoint = {
            "objective_version": CURRENT_OBJECTIVE_VERSION,
            "model_config": saved_config,
            "dataset_config": {
                "min_control_points": 8,
                "max_control_points": 24,
                "canonical_knot_tolerance": 0.0,
            },
            "model_state_dict": reference.state_dict(),
        }

        restored, migrated, _ = build_model_from_checkpoint(checkpoint)

        self.assertEqual(migrated["structure_count_mode"], "hazard")
        self.assertEqual(migrated["min_internal_knots"], 4)
        with torch.no_grad():
            output = restored.eval()(torch.randn(2, 24, 2))
        torch.testing.assert_close(
            output["count_probabilities"][:, :4], torch.zeros(2, 4)
        )
        self.assertTrue(torch.all(output["predicted_knot_count"] >= 4))

    def test_categorical_interactive_checkpoint_restores_strictly(self) -> None:
        config = {
            "point_dim": 2,
            "hidden_dim": 16,
            "encoder_layers": 1,
            "max_internal_knots": 5,
            "structure_mode": "interactive_dynamic",
            "structure_attention_heads": 4,
            "structure_count_mode": "categorical",
            "min_internal_knots": 2,
        }
        reference = SplineFittingNetwork(**config).eval()
        checkpoint = {
            "objective_version": CURRENT_OBJECTIVE_VERSION,
            "model_config": config,
            "model_state_dict": reference.state_dict(),
        }

        restored, migrated, legacy = build_model_from_checkpoint(checkpoint)

        self.assertFalse(legacy)
        self.assertEqual(migrated["structure_count_mode"], "categorical")
        self.assertEqual(migrated["min_internal_knots"], 2)
        points = torch.randn(2, 12, 2)
        with torch.no_grad():
            expected = reference(points)
            actual = restored.eval()(points)
        torch.testing.assert_close(actual["count_logits"], expected["count_logits"])
        torch.testing.assert_close(actual["internal_knots"], expected["internal_knots"])

    def test_v5_count_conditioned_checkpoint_restores_strictly(self) -> None:
        config = {
            "point_dim": 2,
            "hidden_dim": 16,
            "encoder_layers": 1,
            "max_internal_knots": 3,
            "structure_mode": "count_conditioned",
            "count_attention_heads": 4,
            "count_head_mode": "ordinal_local_attention",
            "count_decoder_mode": "shared_count_embedding",
            "geometry_feature_mode": "chord_derivatives",
        }
        reference = SplineFittingNetwork(**config).eval()
        checkpoint = {
            "objective_version": COUNT_CONDITIONED_V5_OBJECTIVE_VERSION,
            "model_config": config,
            "model_state_dict": reference.state_dict(),
        }
        restored, migrated, legacy = build_model_from_checkpoint(checkpoint)
        self.assertFalse(legacy)
        self.assertEqual(migrated["structure_mode"], "count_conditioned")
        points = torch.randn(2, 12, 2)
        with torch.no_grad():
            expected = reference(points)
            actual = restored.eval()(points)
        torch.testing.assert_close(actual["count_logits"], expected["count_logits"])
        torch.testing.assert_close(actual["internal_knots"], expected["internal_knots"])

    def test_v4_count_checkpoint_restores_independent_branches(self) -> None:
        config = {
            "point_dim": 2,
            "hidden_dim": 16,
            "encoder_layers": 1,
            "max_internal_knots": 3,
            "structure_mode": "count_conditioned",
            "count_attention_heads": 4,
            "count_head_mode": "categorical_global",
            "count_decoder_mode": "independent_branches",
            "geometry_feature_mode": "raw_differences",
        }
        reference = SplineFittingNetwork(**config).eval()
        saved_config = {
            key: value
            for key, value in config.items()
            if key
            not in {
                "count_head_mode",
                "count_decoder_mode",
                "geometry_feature_mode",
            }
        }
        checkpoint = {
            "objective_version": COUNT_CONDITIONED_V4_OBJECTIVE_VERSION,
            "model_config": saved_config,
            "model_state_dict": reference.state_dict(),
        }
        restored, migrated, _ = build_model_from_checkpoint(checkpoint)
        self.assertEqual(migrated["count_head_mode"], "categorical_global")
        self.assertEqual(migrated["count_decoder_mode"], "independent_branches")
        points = torch.randn(2, 12, 2)
        with torch.no_grad():
            expected = reference(points)
            actual = restored.eval()(points)
        torch.testing.assert_close(actual["count_logits"], expected["count_logits"])
        torch.testing.assert_close(actual["internal_knots"], expected["internal_knots"])

    def test_v12_checkpoint_infers_coupled_relocation_and_restores_strictly(
        self,
    ) -> None:
        saved_config = {
            "point_dim": 2,
            "hidden_dim": 16,
            "encoder_layers": 1,
            "max_internal_knots": 3,
        }
        reference = SplineFittingNetwork(
            **saved_config,
            structure_mode="candidate_pruning_one_shot",
            geometry_feature_mode="chord_derivatives",
            one_shot_fixed_proposal_geometry=True,
            one_shot_selection_policy="mass_topk",
            one_shot_selector_layers=2,
            one_shot_joint_position_refinement=True,
            one_shot_survivor_relocation=True,
            one_shot_max_position_shift=0.15,
        ).eval()
        checkpoint = {
            "objective_version": V12_COUPLED_RELOCATION_OBJECTIVE_VERSION,
            "model_config": saved_config,
            "model_state_dict": reference.state_dict(),
        }

        restored, migrated, legacy = build_model_from_checkpoint(checkpoint)
        loss_config, assumed = migrate_loss_config(checkpoint, legacy=legacy)

        self.assertFalse(legacy)
        self.assertEqual(migrated["structure_mode"], "candidate_pruning_one_shot")
        self.assertTrue(migrated["one_shot_fixed_proposal_geometry"])
        self.assertEqual(migrated["one_shot_selection_policy"], "mass_topk")
        self.assertTrue(migrated["one_shot_joint_position_refinement"])
        self.assertTrue(migrated["one_shot_survivor_relocation"])
        self.assertEqual(migrated["one_shot_max_position_shift"], 0.15)
        self.assertFalse(migrated["stable_pilot_descriptors"])
        self.assertFalse(migrated["compute_first_derivative"])
        self.assertTrue(assumed)
        self.assertEqual(loss_config["weights"]["policy_count"], 4.0)
        self.assertEqual(loss_config["weights"]["knot_position"], 1.0)
        self.assertFalse(loss_config["joint_position_supervision"])
        self.assertTrue(loss_config["teacher_relocation_supervision"])
        self.assertEqual(loss_config["teacher_survivor_spacing_weight"], 0.25)

        points = torch.randn(2, 12, 2)
        with torch.no_grad():
            expected = reference(points)
            actual = restored.eval()(points)
        torch.testing.assert_close(actual["internal_knots"], expected["internal_knots"])
        torch.testing.assert_close(
            actual["keep_probability"], expected["keep_probability"]
        )

    def test_v11_migration_keeps_survivor_relocation_disabled(self) -> None:
        checkpoint = {
            "objective_version": V11_JOINT_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
            "model_config": {"point_dim": 2},
        }

        config, legacy = migrate_model_config(checkpoint)

        self.assertFalse(legacy)
        self.assertTrue(config["one_shot_joint_position_refinement"])
        self.assertFalse(config["one_shot_survivor_relocation"])
        self.assertEqual(config["one_shot_max_position_shift"], 0.05)

    def test_v13_compact_checkpoint_restores_set_supervised_defaults(self) -> None:
        checkpoint = {
            "objective_version": V13_SET_RELOCATION_OBJECTIVE_VERSION,
            "model_config": {"point_dim": 2},
        }

        model_config, legacy = migrate_model_config(checkpoint)
        loss_config, assumed = migrate_loss_config(checkpoint, legacy=legacy)

        self.assertFalse(legacy)
        self.assertTrue(assumed)
        self.assertEqual(
            model_config["structure_mode"],
            "candidate_pruning_one_shot",
        )
        self.assertTrue(model_config["one_shot_survivor_relocation"])
        self.assertEqual(model_config["one_shot_max_position_shift"], 0.15)
        self.assertTrue(model_config["stable_pilot_descriptors"])
        self.assertEqual(loss_config["weights"]["fit"], 0.0)
        self.assertEqual(loss_config["weights"]["threshold_violation"], 0.0)
        self.assertEqual(
            loss_config["teacher_position_assignment"],
            "actual_keepmask_ordered_set_matching",
        )
        self.assertEqual(
            loss_config["teacher_distribution_target"],
            "relocated_teacher_set_cdf_plus_local_coverage",
        )
        self.assertTrue(loss_config["stable_pilot_descriptors"])

    def test_v14_compact_checkpoint_enables_late_parameter_feedback(self) -> None:
        checkpoint = {
            "objective_version": V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
            "model_config": {"point_dim": 2},
        }

        model_config, legacy = migrate_model_config(checkpoint)
        loss_config, assumed = migrate_loss_config(checkpoint, legacy=legacy)

        self.assertFalse(legacy)
        self.assertTrue(assumed)
        self.assertEqual(
            model_config["structure_mode"],
            "candidate_pruning_one_shot",
        )
        self.assertTrue(model_config["one_shot_survivor_relocation"])
        self.assertTrue(model_config["stable_pilot_descriptors"])
        self.assertTrue(model_config["parameter_feedback_fusion"])
        self.assertEqual(model_config["parameter_feedback_max_logit_shift"], 0.5)
        self.assertEqual(model_config["parameter_feedback_fusion_mode"], "fast_global")
        self.assertEqual(model_config["parameter_gap_reference"], "learned")
        self.assertEqual(model_config["parameter_residual_logit_limit"], 0.5)
        self.assertEqual(model_config["candidate_interval_logit_limit"], 0.0)
        self.assertEqual(
            model_config["candidate_position_parameterization"],
            "interval_softmax",
        )
        self.assertFalse(model_config["enforce_ordered_joint_candidates"])
        self.assertFalse(model_config["compute_first_derivative"])
        self.assertTrue(loss_config["parameter_feedback_fusion"])
        self.assertEqual(
            loss_config["parameter_feedback_loss"]["parameter_error_scale"],
            0.02,
        )
        self.assertEqual(
            loss_config["parameter_feedback_loss"]["initial_chord_blend"],
            0.6,
        )

    def test_v14_explicit_historical_logit_limits_are_preserved(self) -> None:
        checkpoint = {
            "objective_version": V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
            "model_config": {
                "point_dim": 2,
                "parameter_residual_logit_limit": 0.5,
                "candidate_interval_logit_limit": 0.5,
            },
        }

        model_config, legacy = migrate_model_config(checkpoint)

        self.assertFalse(legacy)
        self.assertEqual(model_config["parameter_residual_logit_limit"], 0.5)
        self.assertEqual(model_config["candidate_interval_logit_limit"], 0.5)

    def test_v14_checkpoint_restores_feedback_head_strictly(self) -> None:
        config = {
            "point_dim": 2,
            "hidden_dim": 16,
            "encoder_layers": 1,
            "max_internal_knots": 3,
            "structure_mode": "candidate_pruning_one_shot",
            "structure_attention_heads": 4,
            "geometry_feature_mode": "chord_derivatives",
            "one_shot_fixed_proposal_geometry": True,
            "one_shot_selection_policy": "mass_topk",
            "one_shot_selector_layers": 2,
            "one_shot_joint_position_refinement": True,
            "one_shot_survivor_relocation": True,
            "one_shot_max_position_shift": 0.15,
            "stable_pilot_descriptors": True,
            "parameter_feedback_fusion": True,
            "parameter_feedback_attention_heads": 4,
            "parameter_feedback_max_logit_shift": 0.5,
            "compute_first_derivative": False,
        }
        reference = SplineFittingNetwork(**config).eval()
        checkpoint = {
            "objective_version": V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
            "model_config": config,
            "model_state_dict": reference.state_dict(),
        }

        restored, migrated, legacy = build_model_from_checkpoint(checkpoint)

        self.assertFalse(legacy)
        self.assertTrue(migrated["parameter_feedback_fusion"])
        points = torch.randn(2, 12, 2)
        with torch.no_grad():
            expected = reference(points)
            actual = restored.eval()(points)
        torch.testing.assert_close(actual["params"], expected["params"])
        torch.testing.assert_close(
            actual["parameter_feedback_gap_logit_delta"],
            expected["parameter_feedback_gap_logit_delta"],
        )
        torch.testing.assert_close(
            actual["reconstructed_points"], expected["reconstructed_points"]
        )

    def test_v1_checkpoint_preserves_coupled_fit_gate_semantics(self) -> None:
        checkpoint = {
            "objective_version": "independent_query_supervised_hard_concrete_v1",
            "model_config": {
                "point_dim": 2,
                "gate_mode": "hard_concrete",
                "knot_parameterization": "independent_queries",
                "knot_use_local_cross_attention": True,
                "activity_use_query_features": True,
            },
        }

        config, legacy = migrate_model_config(checkpoint)
        loss_config, assumed = migrate_loss_config(checkpoint, legacy=legacy)

        self.assertFalse(legacy)
        self.assertFalse(config["detach_activity_gate_for_fit"])
        self.assertTrue(assumed)
        self.assertEqual(loss_config["weights"]["existence"], 1e-3)
        self.assertEqual(loss_config["weights"]["count"], 1e-3)

    def test_v2_checkpoint_preserves_independent_activity_layout(self) -> None:
        checkpoint = {
            "objective_version": "independent_query_supervised_hard_concrete_v2",
            "model_config": {
                "point_dim": 2,
                "gate_mode": "hard_concrete",
                "knot_parameterization": "independent_queries",
                "knot_use_local_cross_attention": True,
                "activity_use_query_features": True,
            },
        }

        config, legacy = migrate_model_config(checkpoint)
        loss_config, assumed = migrate_loss_config(checkpoint, legacy=legacy)

        self.assertFalse(legacy)
        self.assertTrue(config["detach_activity_gate_for_fit"])
        self.assertFalse(config["activity_use_candidate_self_attention"])
        self.assertTrue(assumed)
        self.assertEqual(loss_config["weights"]["count"], 2e-3)


if __name__ == "__main__":
    unittest.main()
