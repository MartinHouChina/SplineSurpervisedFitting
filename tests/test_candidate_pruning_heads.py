from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.models.candidate_knot_head import CandidateKnotHead  # noqa: E402
from spline_fitting.models.interactive_pruning_head import (  # noqa: E402
    InteractivePruningHead,
)


class CandidateKnotHeadTests(unittest.TestCase):
    def test_candidates_are_strictly_ordered_and_cover_the_domain(self) -> None:
        head = CandidateKnotHead(
            hidden_dim=16,
            num_candidates=8,
            min_gap=0.01,
            attention_heads=4,
        )
        global_features = torch.randn(3, 16, requires_grad=True)
        local_features = torch.randn(3, 24, 16, requires_grad=True)
        positions = torch.linspace(0.0, 1.0, 24).expand(3, -1)
        output = head(global_features, local_features, positions)

        self.assertEqual(output["candidate_positions"].shape, (3, 8))
        self.assertEqual(output["candidate_tokens"].shape, (3, 8, 16))
        self.assertEqual(output["candidate_intervals"].shape, (3, 9))
        self.assertEqual(output["candidate_attention_weights"].shape, (3, 9, 24))
        torch.testing.assert_close(
            output["candidate_intervals"].sum(dim=-1),
            torch.ones(3),
        )
        self.assertTrue(torch.all(output["candidate_intervals"] >= 0.01 - 1e-7))
        self.assertTrue(torch.all(output["candidate_positions"] > 0.0))
        self.assertTrue(torch.all(output["candidate_positions"] < 1.0))
        self.assertTrue(
            torch.all(
                output["candidate_positions"][:, 1:]
                - output["candidate_positions"][:, :-1]
                >= 0.01 - 1e-7
            )
        )

        loss = (
            output["candidate_positions"].square().sum()
            + output["candidate_tokens"].square().mean()
        )
        loss.backward()
        self.assertGreater(float(global_features.grad.abs().sum()), 0.0)
        self.assertGreater(float(local_features.grad.abs().sum()), 0.0)
        self.assertGreater(float(head.interval_queries.grad.abs().sum()), 0.0)

    def test_near_uniform_initialization_spreads_candidates(self) -> None:
        torch.manual_seed(7)
        head = CandidateKnotHead(
            hidden_dim=16,
            num_candidates=20,
            min_gap=1e-3,
            attention_heads=4,
        ).eval()
        with torch.no_grad():
            output = head(
                torch.zeros(1, 16),
                torch.zeros(1, 32, 16),
                torch.linspace(0.0, 1.0, 32).unsqueeze(0),
            )
        # A high-recall head should not initialize all candidates in one local
        # region; both outer quarters of the domain receive candidates.
        candidates = output["candidate_positions"][0]
        self.assertGreater(int((candidates < 0.25).sum()), 0)
        self.assertGreater(int((candidates > 0.75).sum()), 0)


class InteractivePruningHeadTests(unittest.TestCase):
    def _inputs(self) -> tuple[torch.Tensor, ...]:
        tokens = torch.randn(2, 6, 16, requires_grad=True)
        positions = torch.tensor(
            [[0.08, 0.22, 0.40, 0.58, 0.76, 0.92]]
        ).expand(2, -1).clone().requires_grad_()
        coefficient_energy = torch.rand(2, 6, requires_grad=True)
        deletion_delta = torch.rand(2, 6, requires_grad=True)
        residual_features = torch.randn(2, 6, 2, requires_grad=True)
        return (
            tokens,
            positions,
            coefficient_energy,
            deletion_delta,
            residual_features,
        )

    def test_outputs_have_expected_shapes_and_receive_gradients(self) -> None:
        head = InteractivePruningHead(
            hidden_dim=16,
            residual_feature_dim=2,
            attention_heads=4,
            min_gap=0.01,
        )
        tokens, positions, energy, delta, residual = self._inputs()
        output = head(
            tokens,
            positions,
            coefficient_energy=energy,
            deletion_delta=delta,
            residual_features=residual,
        )

        self.assertEqual(output["keep_logits"].shape, (2, 6))
        self.assertEqual(output["keep_probabilities"].shape, (2, 6))
        self.assertEqual(output["predicted_deletion_cost"].shape, (2, 6))
        self.assertEqual(output["refined_candidate_positions"].shape, (2, 6))
        self.assertEqual(output["pruning_tokens"].shape, (2, 6, 16))
        self.assertTrue(torch.all(output["keep_probabilities"] >= 0.0))
        self.assertTrue(torch.all(output["keep_probabilities"] <= 1.0))
        self.assertTrue(torch.all(output["predicted_deletion_cost"] >= 0.0))

        loss = (
            output["keep_logits"].mean()
            + output["predicted_deletion_cost"].mean()
            + output["refined_candidate_positions"].square().mean()
        )
        loss.backward()
        self.assertGreater(float(tokens.grad.abs().sum()), 0.0)
        self.assertGreater(float(energy.grad.abs().sum()), 0.0)
        self.assertGreater(float(delta.grad.abs().sum()), 0.0)
        self.assertGreater(float(residual.grad.abs().sum()), 0.0)
        self.assertGreater(float(head.keep_head.weight.grad.abs().sum()), 0.0)

    def test_position_refinement_cannot_cross_neighbors(self) -> None:
        head = InteractivePruningHead(
            hidden_dim=16,
            residual_feature_dim=1,
            attention_heads=4,
            min_gap=0.01,
            max_position_fraction=0.45,
        ).eval()
        # Force alternating maximally positive/negative signals after the
        # shared token processing.  Even this adversarial update cannot cross.
        with torch.no_grad():
            head.position_residual_head.weight.fill_(1000.0)
            head.position_residual_head.bias.zero_()
        positions = torch.tensor([[0.10, 0.25, 0.50, 0.72, 0.90]])
        output = head(
            torch.randn(1, 5, 16),
            positions,
            coefficient_energy=torch.rand(1, 5),
            deletion_delta=torch.rand(1, 5),
            residual_features=torch.rand(1, 5),
        )
        refined = output["refined_candidate_positions"]
        self.assertTrue(torch.all(refined > 0.0))
        self.assertTrue(torch.all(refined < 1.0))
        self.assertTrue(torch.all(refined[:, 1:] - refined[:, :-1] >= 0.01 - 1e-6))

    def test_extreme_analytic_features_remain_finite_and_mask_is_respected(self) -> None:
        head = InteractivePruningHead(
            hidden_dim=16,
            residual_feature_dim=1,
            attention_heads=4,
            min_gap=0.01,
        ).eval()
        mask = torch.tensor([[True, True, False, True]])
        output = head(
            torch.randn(1, 4, 16),
            torch.tensor([[0.1, 0.35, 0.60, 0.9]]),
            coefficient_energy=torch.tensor([[float("inf"), 1e30, 0.0, 1.0]]),
            deletion_delta=torch.tensor([[float("nan"), -1e30, 1e30, 0.0]]),
            residual_features=torch.tensor([[float("inf"), float("-inf"), 0.0, 1e30]]),
            candidate_mask=mask,
        )
        for name in (
            "keep_probabilities",
            "predicted_deletion_cost",
            "refined_candidate_positions",
            "analytic_contribution_features",
        ):
            self.assertTrue(torch.all(torch.isfinite(output[name])), name)
        self.assertEqual(float(output["keep_probabilities"][0, 2].detach()), 0.0)
        self.assertEqual(
            float(output["predicted_deletion_cost"][0, 2].detach()), 0.0
        )
        self.assertEqual(float(output["position_residual"][0, 2].detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
