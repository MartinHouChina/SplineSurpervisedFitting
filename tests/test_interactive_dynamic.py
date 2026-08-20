from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.evaluation.bspline_inference import refit_model_output_as_bsplines
from spline_fitting.losses.total_loss import LossWeights, SplineFittingLoss
from spline_fitting.models.dynamic_knot_decoder import DynamicKnotDecoder
from spline_fitting.models.interactive_structure_head import InteractiveStructureHead
from spline_fitting.models.spline_network import SplineFittingNetwork


class InteractiveDynamicTests(unittest.TestCase):
    def test_structure_head_returns_monotone_normalized_count_distribution(self) -> None:
        head = InteractiveStructureHead(
            hidden_dim=16, max_internal_knots=5, attention_heads=4
        )
        global_features = torch.randn(3, 16, requires_grad=True)
        local_features = torch.randn(3, 12, 16, requires_grad=True)
        positions = torch.linspace(0.0, 1.0, 12).expand(3, -1)
        output = head(global_features, local_features, positions)

        self.assertEqual(output["structure_query_features"].shape, (3, 5, 16))
        self.assertEqual(output["count_probabilities"].shape, (3, 6))
        torch.testing.assert_close(
            output["count_probabilities"].sum(dim=-1), torch.ones(3)
        )
        survival = output["structure_survival_probabilities"]
        self.assertTrue(torch.all(survival[:, :-1] >= survival[:, 1:]))
        output["expected_knot_count"].sum().backward()
        self.assertGreater(float(local_features.grad.abs().sum()), 0.0)

    def test_dynamic_decoder_only_reports_selected_query_counts(self) -> None:
        decoder = DynamicKnotDecoder(
            hidden_dim=16,
            max_internal_knots=20,
            min_gap=1e-3,
            attention_heads=4,
        )
        selected = torch.tensor([0, 2, 5, 2])
        output = decoder(
            torch.randn(4, 16),
            torch.randn(4, 12, 16),
            torch.linspace(0.0, 1.0, 12).expand(4, -1),
            torch.randn(4, 20, 16),
            selected,
        )
        self.assertNotIn("branch_internal_knots", output)
        torch.testing.assert_close(output["knot_mask"].sum(dim=-1), selected)
        torch.testing.assert_close(
            output["decoded_interval_query_count"], torch.tensor([0, 3, 6, 3])
        )
        for index, count in enumerate(selected.tolist()):
            knots = output["internal_knots"][index, :count]
            if count > 1:
                self.assertTrue(torch.all(knots[1:] > knots[:-1]))

    def test_network_uses_teacher_count_without_standalone_count_head(self) -> None:
        model = SplineFittingNetwork(
            hidden_dim=16,
            encoder_layers=1,
            max_internal_knots=4,
            structure_mode="interactive_dynamic",
            structure_attention_heads=4,
        ).train()
        points = torch.randn(3, 16, 2)
        true_count = torch.tensor([1, 3, 2])
        output = model(points, true_internal_knot_count=true_count)
        self.assertFalse(hasattr(model, "count_head"))
        self.assertNotIn("branch_internal_knots", output)
        torch.testing.assert_close(output["count_used_for_knots"], true_count)
        torch.testing.assert_close(output["knot_mask"].sum(dim=-1), true_count)

    def test_teacher_forcing_ratio_controls_dynamic_decoder_count_source(self) -> None:
        model = SplineFittingNetwork(
            hidden_dim=16,
            encoder_layers=1,
            max_internal_knots=4,
            structure_mode="interactive_dynamic",
            structure_attention_heads=4,
        ).train()
        points = torch.randn(2, 16, 2)
        true_count = torch.tensor([1, 2])
        teacher_output = model(
            points,
            true_internal_knot_count=true_count,
            teacher_forcing_ratio=1.0,
        )
        predicted_output = model(
            points,
            true_internal_knot_count=true_count,
            teacher_forcing_ratio=0.0,
        )
        torch.testing.assert_close(
            teacher_output["count_used_for_knots"], true_count
        )
        torch.testing.assert_close(
            predicted_output["count_used_for_knots"],
            predicted_output["predicted_knot_count"],
        )

    def test_structure_and_dynamic_decoder_receive_supervised_gradients(self) -> None:
        model = SplineFittingNetwork(
            hidden_dim=16,
            encoder_layers=1,
            max_internal_knots=3,
            structure_mode="interactive_dynamic",
            structure_attention_heads=4,
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
            float(model.structure_head.stop_head.weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(model.knot_head.interval_queries.grad.abs().sum()), 0.0
        )

    def test_standard_deployment_refits_only_selected_nodes(self) -> None:
        model = SplineFittingNetwork(
            hidden_dim=16,
            encoder_layers=1,
            max_internal_knots=3,
            structure_mode="interactive_dynamic",
            structure_attention_heads=4,
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


if __name__ == "__main__":
    unittest.main()
