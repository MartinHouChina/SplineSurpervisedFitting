from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.evaluation.bspline_inference import (
    refit_model_output_as_bsplines,
    select_count_conditioned_output_by_bic,
)
from spline_fitting.losses.total_loss import LossWeights, SplineFittingLoss
from spline_fitting.models.count_conditioned_knot_head import (
    CountConditionedKnotHead,
)
from spline_fitting.models.count_head import CountHead
from spline_fitting.models.spline_network import SplineFittingNetwork


class CountConditionedTests(unittest.TestCase):
    def test_count_head_returns_normalized_ordinal_distribution(self) -> None:
        head = CountHead(hidden_dim=16, max_internal_knots=4)
        features = torch.randn(3, 16, requires_grad=True)
        local_features = torch.randn(3, 12, 16, requires_grad=True)
        positions = torch.linspace(0.0, 1.0, 12).expand(3, -1)
        output = head(features, local_features, positions)

        self.assertEqual(output["count_logits"].shape, (3, 5))
        self.assertEqual(output["count_ordinal_logits"].shape, (3, 4))
        self.assertEqual(output["predicted_knot_count"].shape, (3,))
        survival = torch.sigmoid(output["count_ordinal_logits"])
        self.assertTrue(torch.all(survival[:, :-1] >= survival[:, 1:]))
        torch.testing.assert_close(
            output["count_probabilities"].sum(dim=-1), torch.ones(3)
        )
        output["expected_knot_count"].sum().backward()
        self.assertGreater(float(features.grad.abs().sum()), 0.0)

    def test_decoder_returns_exact_strictly_ordered_count(self) -> None:
        head = CountConditionedKnotHead(
            hidden_dim=16,
            max_internal_knots=4,
            min_gap=0.01,
            attention_heads=4,
        )
        selected_count = torch.tensor([0, 2, 4])
        output = head(
            torch.randn(3, 16),
            torch.randn(3, 12, 16),
            torch.linspace(0.0, 1.0, 12).expand(3, -1),
            selected_count,
        )

        torch.testing.assert_close(
            output["knot_mask"].sum(dim=-1), selected_count
        )
        self.assertEqual(output["branch_internal_knots"].shape, (3, 5, 4))
        for index, count in enumerate(selected_count.tolist()):
            knots = output["internal_knots"][index, :count]
            if count:
                self.assertTrue(torch.all(knots > 0.0))
                self.assertTrue(torch.all(knots < 1.0))
            if count > 1:
                self.assertTrue(torch.all(knots[1:] - knots[:-1] >= 0.01))

    def test_network_uses_teacher_count_for_training_branch(self) -> None:
        model = SplineFittingNetwork(
            hidden_dim=16,
            encoder_layers=1,
            max_internal_knots=4,
            structure_mode="count_conditioned",
            count_attention_heads=4,
        ).train()
        points = torch.randn(2, 16, 2)
        true_count = torch.tensor([1, 3])
        output = model(points, true_internal_knot_count=true_count)

        torch.testing.assert_close(output["count_used_for_knots"], true_count)
        torch.testing.assert_close(output["knot_mask"].sum(dim=-1), true_count)
        self.assertNotIn("activity", output)
        self.assertEqual(output["reconstructed_points"].shape, points.shape)

    def test_count_loss_and_selected_branch_receive_gradients(self) -> None:
        model = SplineFittingNetwork(
            hidden_dim=16,
            encoder_layers=1,
            max_internal_knots=3,
            structure_mode="count_conditioned",
            count_attention_heads=4,
        ).train()
        points = torch.randn(2, 16, 2)
        true_count = torch.tensor([1, 2])
        output = model(points, true_internal_knot_count=true_count)
        true_knots = torch.tensor([[0.35, 0.0], [0.3, 0.7]])
        true_mask = torch.tensor([[True, False], [True, True]])
        loss_fn = SplineFittingLoss(
            LossWeights(fit=1.0, count=0.01, knot_position=0.01)
        )
        losses = loss_fn(
            output,
            points,
            true_internal_knots=true_knots,
            true_internal_knot_mask=true_mask,
        )
        losses["loss"].backward()

        self.assertGreater(
            float(model.count_head.evidence_head[-1].weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(model.knot_head.interval_queries.grad.abs().sum()), 0.0
        )
        self.assertTrue(torch.isfinite(losses["count_loss"]))

    def test_deployment_uses_count_mask_without_threshold(self) -> None:
        model = SplineFittingNetwork(
            hidden_dim=16,
            encoder_layers=1,
            max_internal_knots=3,
            structure_mode="count_conditioned",
            count_attention_heads=4,
        ).eval()
        points = torch.randn(2, 16, 2)
        with torch.no_grad():
            output = model(points)
            fitted = refit_model_output_as_bsplines(output, points)

        for index, item in enumerate(fitted):
            self.assertEqual(
                item.retained_count,
                int(output["predicted_knot_count"][index]),
            )

    def test_bic_selection_compares_complete_count_branches(self) -> None:
        model = SplineFittingNetwork(
            hidden_dim=16,
            encoder_layers=1,
            max_internal_knots=3,
            structure_mode="count_conditioned",
            count_attention_heads=4,
        ).eval()
        points = torch.randn(2, 16, 2)
        with torch.no_grad():
            output = model(points)
            selected, scores = select_count_conditioned_output_by_bic(output, points)
        self.assertEqual(scores.shape, (2, 4))
        torch.testing.assert_close(
            selected["knot_mask"].sum(dim=-1),
            selected["deployment_selected_knot_count"],
        )


if __name__ == "__main__":
    unittest.main()
