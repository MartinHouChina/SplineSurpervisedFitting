from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.losses.total_loss import LossWeights, SplineFittingLoss
from spline_fitting.models.knot_head import KnotHead
from spline_fitting.models.parameter_head import ParameterHead


class ASchemeLossAndGapTests(unittest.TestCase):
    def test_l0_loss_is_expected_count_per_curve(self) -> None:
        points = torch.zeros(2, 4, 2)
        output = {
            "reconstructed_points": points.clone(),
            "activity": torch.tensor([[0.2, 0.8, 0.5], [0.1, 0.3, 0.6]]),
            "l0_probability": torch.tensor([[0.2, 0.8, 0.5], [0.1, 0.3, 0.6]]),
            "internal_knots": torch.tensor([[0.2, 0.5, 0.8], [0.2, 0.5, 0.8]]),
            "params": torch.linspace(0.0, 1.0, 4).expand(2, -1),
        }
        loss_fn = SplineFittingLoss(LossWeights(l0=2.0, gap=0.0))
        losses = loss_fn(output, points)

        # Per-curve expected counts are 1.5 and 1.0, so the batch mean is 1.25.
        self.assertAlmostEqual(float(losses["l0_loss"]), 1.25)
        self.assertAlmostEqual(float(losses["loss"]), 2.5)
        self.assertNotIn("orthogonal_loss", losses)

    def test_historical_orthogonal_loss_requires_the_derivative(self) -> None:
        points = torch.zeros(1, 4, 2)
        output = {
            "reconstructed_points": points.clone(),
            "activity": torch.empty(1, 0),
            "internal_knots": torch.empty(1, 0),
            "params": torch.linspace(0.0, 1.0, 4).unsqueeze(0),
        }
        loss_fn = SplineFittingLoss(LossWeights(orthogonal=0.05, gap=0.0))

        with self.assertRaisesRegex(KeyError, "first_derivative"):
            loss_fn(output, points)

        output["first_derivative"] = torch.ones_like(points)
        losses = loss_fn(output, points)
        self.assertIn("orthogonal_loss", losses)

    def test_gap_loss_accepts_no_internal_knots(self) -> None:
        loss_fn = SplineFittingLoss()
        value = loss_fn._gap_loss(torch.empty(3, 0))
        self.assertEqual(value.ndim, 0)
        self.assertEqual(float(value), 0.0)

    def test_true_parameter_supervision_contributes_to_total_loss(self) -> None:
        points = torch.zeros(1, 3, 2)
        output = {
            "reconstructed_points": points.clone(),
            "activity": torch.empty(1, 0),
            "internal_knots": torch.empty(1, 0),
            "params": torch.tensor([[0.0, 0.5, 1.0]]),
        }
        target = torch.tensor([[0.0, 0.25, 1.0]])
        loss_fn = SplineFittingLoss(LossWeights(true_parameter=2.0, gap=0.0))

        losses = loss_fn(output, points, true_params=target)

        expected = torch.tensor((0.25**2) / 3.0)
        torch.testing.assert_close(losses["true_parameter_loss"], expected)
        torch.testing.assert_close(losses["loss"], 2.0 * expected)

    def test_local_cross_attention_knot_head_uses_positioned_sequence(self) -> None:
        torch.manual_seed(8)
        head = KnotHead(
            hidden_dim=16,
            max_internal_knots=4,
            min_gap=0.01,
            use_local_cross_attention=True,
            attention_heads=4,
        )
        global_features = torch.randn(2, 16)
        local_features = torch.randn(2, 12, 16, requires_grad=True)
        positions = torch.linspace(0.0, 1.0, 12).expand(2, -1)

        output = head(
            global_features,
            local_features=local_features,
            positions=positions,
        )

        self.assertEqual(output["internal_knots"].shape, (2, 4))
        self.assertEqual(output["knot_query_features"].shape, (2, 4, 16))
        self.assertEqual(output["knot_attention_weights"].shape, (2, 4, 12))
        torch.testing.assert_close(
            output["knot_attention_weights"].sum(dim=-1),
            torch.ones(2, 4),
        )
        self.assertTrue(
            torch.all(
                output["internal_knots"][:, 1:] >= output["internal_knots"][:, :-1]
            )
        )
        self.assertTrue(torch.all(output["internal_knots"] >= 0.01 - 1e-7))
        self.assertTrue(torch.all(output["internal_knots"] <= 0.99 + 1e-7))
        output["internal_knots"].sum().backward()
        self.assertGreater(float(local_features.grad.abs().sum()), 0.0)

    def test_knot_head_enforces_configured_gap_after_normalization(self) -> None:
        head = KnotHead(
            hidden_dim=16,
            max_internal_knots=4,
            min_gap=0.05,
            knot_parameterization="interval",
            use_local_cross_attention=False,
        )
        output = head(torch.randn(3, 16))
        gaps = output["knot_intervals"]
        self.assertTrue(torch.all(gaps >= 0.05 - 1e-7))
        torch.testing.assert_close(gaps.sum(dim=-1), torch.ones(3))

    def test_supervised_matching_marks_exists_and_trains_matched_positions(
        self,
    ) -> None:
        predicted = torch.tensor([[0.10, 0.40, 0.80]], requires_grad=True)
        logits = torch.zeros(1, 3, requires_grad=True)
        true_knots = torch.tensor([[0.12, 0.78, 0.0]])
        true_mask = torch.tensor([[True, True, False]])
        targets, position_targets, matched = SplineFittingLoss._ordered_knot_assignment(
            predicted, true_knots, true_mask
        )
        torch.testing.assert_close(targets, torch.tensor([[1.0, 0.0, 1.0]]))
        torch.testing.assert_close(
            position_targets[matched], torch.tensor([0.12, 0.78])
        )

        points = torch.zeros(1, 4, 2)
        output = {
            "reconstructed_points": points.clone(),
            "activity": torch.sigmoid(logits),
            "activity_probability_logits": logits,
            "l0_probability": torch.sigmoid(logits),
            "internal_knots": predicted,
            "params": torch.linspace(0.0, 1.0, 4).unsqueeze(0),
        }
        loss_fn = SplineFittingLoss(
            LossWeights(existence=1.0, knot_position=1.0, count=1.0, gap=0.0)
        )
        losses = loss_fn(
            output,
            points,
            true_internal_knots=true_knots,
            true_internal_knot_mask=true_mask,
        )
        losses["loss"].backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        self.assertGreater(float(predicted.grad.abs().sum()), 0.0)

    def test_parameter_head_enforces_configured_gap_after_normalization(self) -> None:
        head = ParameterHead(hidden_dim=16, min_gap=0.02)
        local = torch.randn(2, 12, 16)
        global_features = torch.randn(2, 16)
        output = head(local, global_features)
        gaps = output["parameter_gaps"]
        self.assertTrue(torch.all(gaps >= 0.02 - 1e-7))
        torch.testing.assert_close(gaps.sum(dim=-1), torch.ones(2))


if __name__ == "__main__":
    unittest.main()
