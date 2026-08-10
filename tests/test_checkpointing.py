from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import (
    COUNT_CONDITIONED_V4_OBJECTIVE_VERSION,
    CURRENT_OBJECTIVE_VERSION,
    PREVIOUS_OBJECTIVE_VERSION,
    build_model_from_checkpoint,
    migrate_loss_config,
    migrate_model_config,
)
from spline_fitting.models.spline_network import SplineFittingNetwork


class CheckpointMigrationTests(unittest.TestCase):
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
        self.assertEqual(config["structure_mode"], "count_conditioned")
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

    def test_current_count_conditioned_checkpoint_restores_strictly(self) -> None:
        config = {
            "point_dim": 2,
            "hidden_dim": 16,
            "encoder_layers": 1,
            "max_internal_knots": 3,
            "structure_mode": "count_conditioned",
            "count_attention_heads": 4,
        }
        reference = SplineFittingNetwork(**config).eval()
        checkpoint = {
            "objective_version": CURRENT_OBJECTIVE_VERSION,
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
