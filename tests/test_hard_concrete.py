from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.models.activity_head import ActivityHead
from spline_fitting.models.hard_concrete import HardConcreteGate
from spline_fitting.models.spline_network import SplineFittingNetwork
from spline_fitting.spline.differentiable_solver import (
    coefficient_drop_objective_delta,
)
from spline_fitting.spline.truncated_power_basis import build_design_matrix


class HardConcreteGateTests(unittest.TestCase):
    def test_l0_probability_and_eval_gate_follow_closed_form(self) -> None:
        module = HardConcreteGate(
            temperature=0.5,
            stretch_low=-0.1,
            stretch_high=1.1,
            threshold=0.5,
        )
        module.eval()
        logits = torch.tensor([[-5.0, 0.0, 5.0]])

        output = module(logits)
        expected = torch.sigmoid(
            logits - 0.5 * math.log(-module.stretch_low / module.stretch_high)
        )

        torch.testing.assert_close(output["l0_probability"], expected)
        torch.testing.assert_close(output["activity"], expected)
        torch.testing.assert_close(output["expected_l0"], expected.sum(dim=-1))
        torch.testing.assert_close(
            output["activity_gate"], (expected >= 0.5).to(logits.dtype)
        )
        self.assertTrue(
            bool(((output["activity_gate"] == 0) | (output["activity_gate"] == 1)).all())
        )

    def test_training_sample_is_bounded_stochastic_and_differentiable(self) -> None:
        torch.manual_seed(7)
        module = HardConcreteGate(temperature=2.0 / 3.0)
        module.train()
        logits = torch.zeros(4, 128, requires_grad=True)

        first = module(logits)["activity_gate"]
        second = module(logits)["activity_gate"]
        first.mean().backward()

        self.assertTrue(bool((first >= 0.0).all() and (first <= 1.0).all()))
        self.assertFalse(torch.equal(first, second))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_temperature_is_checkpointed_and_old_state_is_accepted(self) -> None:
        source = HardConcreteGate(temperature=0.37)
        restored = HardConcreteGate(temperature=1.5)
        restored.load_state_dict(source.state_dict(), strict=True)
        self.assertAlmostEqual(restored.temperature, 0.37, places=6)

        old_checkpoint: dict[str, torch.Tensor] = {}
        fallback = HardConcreteGate(temperature=0.8)
        fallback.load_state_dict(old_checkpoint, strict=True)
        self.assertAlmostEqual(fallback.temperature, 0.8, places=6)

    def test_force_open_only_overrides_actual_gate(self) -> None:
        module = HardConcreteGate(temperature=0.5)
        module.eval()
        logits = torch.full((2, 3), -20.0)
        module.set_force_open_gates(True)

        output = module(logits)

        torch.testing.assert_close(output["activity_gate"], torch.ones_like(logits))
        self.assertTrue(bool((output["activity"] < 0.5).all()))

    def test_deployment_threshold_can_be_frozen_or_overridden(self) -> None:
        module = HardConcreteGate(temperature=0.5, threshold=0.8).eval()
        logits = torch.zeros(1, 1)
        closed = module(logits)["activity_gate"]
        module.set_threshold(0.7)
        opened = module(logits)["activity_gate"]

        torch.testing.assert_close(closed, torch.zeros_like(closed))
        torch.testing.assert_close(opened, torch.ones_like(opened))

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HardConcreteGate(temperature=0.0)
        with self.assertRaises(ValueError):
            HardConcreteGate(stretch_low=0.0)
        with self.assertRaises(ValueError):
            HardConcreteGate(stretch_high=1.0)
        with self.assertRaises(ValueError):
            HardConcreteGate(threshold=1.1)


class ActivityHeadTests(unittest.TestCase):
    def test_pilot_importance_adds_node_specific_logit_bias(self) -> None:
        head = ActivityHead(
            hidden_dim=8,
            initial_bias=0.0,
            use_pilot_importance=True,
            pilot_importance_gain=2.0,
        ).eval()
        with torch.no_grad():
            for parameter in head.mlp.parameters():
                parameter.zero_()

        output = head(
            torch.zeros(1, 8),
            torch.tensor([[0.25, 0.75]]),
            normalized_knot_importance=torch.tensor([[-1.0, 1.0]]),
        )

        torch.testing.assert_close(
            output["activity_logits"], torch.tensor([[-2.0, 2.0]])
        )

    def test_local_context_is_knot_specific_without_changing_mlp_width(self) -> None:
        hidden_dim = 8
        head = ActivityHead(
            hidden_dim=hidden_dim,
            use_local_context=True,
            context_bandwidth=0.01,
        )
        captured: list[torch.Tensor] = []

        def capture_input(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            captured.append(args[0].detach())

        handle = head.mlp[0].register_forward_pre_hook(capture_input)
        try:
            global_features = torch.zeros(1, hidden_dim)
            internal_knots = torch.tensor([[0.2, 0.8]])
            params = torch.tensor([[0.0, 0.2, 0.8, 1.0]])
            local_features = torch.zeros(1, 4, hidden_dim)
            local_features[0, 1, 0] = 1.0
            local_features[0, 2, 0] = 5.0
            output = head(
                global_features,
                internal_knots,
                local_features=local_features,
                params=params,
            )
        finally:
            handle.remove()

        self.assertEqual(head.mlp[0].in_features, hidden_dim + 1)
        self.assertEqual(output["activity"].shape, (1, 2))
        self.assertEqual(output["activity_gate"].shape, (1, 2))
        self.assertAlmostEqual(float(captured[0][0, 0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(captured[0][0, 1, 0]), 5.0, places=5)

    def test_local_context_requires_pointwise_inputs(self) -> None:
        head = ActivityHead(hidden_dim=8, use_local_context=True)
        with self.assertRaises(ValueError):
            head(torch.zeros(1, 8), torch.tensor([[0.5]]))


class HardConcreteSplineNetworkTests(unittest.TestCase):
    def test_drop_objective_delta_matches_explicit_constrained_optimum(self) -> None:
        torch.manual_seed(3)
        batch, columns, dimension = 2, 5, 2
        matrix = torch.randn(batch, columns, columns, dtype=torch.float64)
        normal = matrix.transpose(-1, -2) @ matrix
        normal = normal + 0.3 * torch.eye(columns, dtype=torch.float64)
        rhs = torch.randn(batch, columns, dimension, dtype=torch.float64)
        coefficients = torch.linalg.solve(normal, rhs)
        predicted = coefficient_drop_objective_delta(
            coefficients,
            normal,
            first_column=2,
        )

        expected_columns = []
        full_minimum = -(rhs * coefficients).sum(dim=(-2, -1))
        for dropped in range(2, columns):
            keep = torch.tensor(
                [index for index in range(columns) if index != dropped]
            )
            reduced_normal = normal[:, keep][:, :, keep]
            reduced_rhs = rhs[:, keep]
            reduced_coefficients = torch.linalg.solve(
                reduced_normal,
                reduced_rhs,
            )
            reduced_minimum = -(
                reduced_rhs * reduced_coefficients
            ).sum(dim=(-2, -1))
            expected_columns.append(reduced_minimum - full_minimum)
        expected = torch.stack(expected_columns, dim=-1)

        torch.testing.assert_close(predicted, expected)

    def test_direct_gate_has_finite_gradient_at_zero(self) -> None:
        params = torch.linspace(0.0, 1.0, 8).unsqueeze(0)
        knots = torch.tensor([[0.3, 0.7]])
        gate = torch.tensor([[0.0, 0.5]], requires_grad=True)

        output = build_design_matrix(
            params,
            knots,
            gate,
            degree=3,
            eps=0.0,
            gate_transform="direct",
        )
        output["design_matrix"].sum().backward()

        self.assertTrue(bool(torch.isfinite(gate.grad).all()))
        torch.testing.assert_close(output["activity_gate"], gate.detach())
        self.assertTrue(bool((output["weighted_increment_basis"][..., 0] == 0).all()))

    def test_network_eval_uses_binary_gate_and_warmup_can_force_all_open(self) -> None:
        model = SplineFittingNetwork(
            hidden_dim=16,
            encoder_layers=1,
            max_internal_knots=3,
            lambda_knot=1e-3,
            gate_mode="hard_concrete",
            gate_temperature=0.5,
            hard_concrete_gamma=-0.1,
            hard_concrete_zeta=1.1,
            activity_use_local_context=True,
        )
        with torch.no_grad():
            model.activity_head.mlp[-1].weight.zero_()
            model.activity_head.mlp[-1].bias.fill_(-20.0)
        model.eval()
        points = torch.randn(2, 16, 2)

        closed = model(points)
        self.assertTrue(bool((closed["activity_gate"] == 0).all()))
        self.assertTrue(bool((closed["weighted_increment_basis"] == 0).all()))

        model.set_force_open_gates(True)
        opened = model(points)
        self.assertTrue(bool((opened["activity_gate"] == 1).all()))
        torch.testing.assert_close(
            opened["weighted_increment_basis"], opened["increment_basis"]
        )

    def test_legacy_state_without_temperature_loads_strictly(self) -> None:
        legacy = SplineFittingNetwork(
            hidden_dim=16,
            encoder_layers=1,
            max_internal_knots=2,
            gate_mode="legacy_soft",
            activity_use_local_context=False,
        )
        old_state = {
            key: value
            for key, value in legacy.state_dict().items()
            if not key.endswith("_temperature_buffer")
        }
        restored = SplineFittingNetwork(
            hidden_dim=16,
            encoder_layers=1,
            max_internal_knots=2,
            gate_mode="legacy_soft",
            activity_use_local_context=False,
        )

        incompatibility = restored.load_state_dict(old_state, strict=True)

        self.assertEqual(incompatibility.missing_keys, [])
        self.assertEqual(incompatibility.unexpected_keys, [])


if __name__ == "__main__":
    unittest.main()
