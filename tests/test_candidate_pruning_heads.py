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

    def test_local_cross_attention_is_anchor_centred(self) -> None:
        torch.manual_seed(13)
        head = CandidateKnotHead(
            hidden_dim=16,
            num_candidates=7,
            min_gap=0.01,
            attention_heads=4,
            local_attention_bandwidth=0.05,
        ).eval()
        positions = torch.linspace(0.0, 1.0, 101).unsqueeze(0)
        with torch.no_grad():
            output = head(
                torch.zeros(1, 16),
                torch.randn(1, 101, 16),
                positions,
            )
        attention = output["candidate_attention_weights"][0]
        anchors = head.interval_query_anchors
        mean_distance = (
            attention * (positions[0].unsqueeze(0) - anchors.unsqueeze(-1)).abs()
        ).sum(dim=-1)
        self.assertTrue(torch.all(mean_distance < 0.10))


class InteractivePruningHeadTests(unittest.TestCase):
    def _inputs(self) -> tuple[torch.Tensor, ...]:
        tokens = torch.randn(2, 6, 16, requires_grad=True)
        positions = (
            torch.tensor([[0.08, 0.22, 0.40, 0.58, 0.76, 0.92]])
            .expand(2, -1)
            .clone()
            .requires_grad_()
        )
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

    def test_one_shot_threshold_is_curve_adaptive_and_receives_gradients(self) -> None:
        torch.manual_seed(19)
        head = InteractivePruningHead(
            hidden_dim=16,
            residual_feature_dim=1,
            attention_heads=4,
            min_gap=0.01,
            one_shot_adaptive=True,
        )
        # Keep all knot-local evidence identical between samples.  Only the
        # optional curve-level descriptor can then change beta.
        tokens = torch.randn(1, 5, 16).expand(2, -1, -1).clone().requires_grad_()
        positions = torch.tensor([[0.1, 0.28, 0.47, 0.69, 0.9]]).expand(2, -1)
        energy = torch.rand(1, 5).expand(2, -1)
        delta = torch.rand(1, 5).expand(2, -1)
        residual = torch.rand(1, 5).expand(2, -1)
        global_features = torch.zeros(2, 16, requires_grad=True)
        with torch.no_grad():
            global_features[1, 0] = 5.0
            first = head.adaptive_threshold_head[0]
            last = head.adaptive_threshold_head[-1]
            first.weight.zero_()
            first.weight.copy_(torch.eye(16))
            first.bias.zero_()
            last.weight.zero_()
            last.weight[0, 0] = 1.0
            last.bias.zero_()

        output = head(
            tokens,
            positions,
            coefficient_energy=energy,
            deletion_delta=delta,
            residual_features=residual,
            global_features=global_features,
        )
        torch.testing.assert_close(
            output["preliminary_raw_importance"][0],
            output["preliminary_raw_importance"][1],
        )
        self.assertFalse(
            torch.allclose(
                output["adaptive_keep_threshold"][0],
                output["adaptive_keep_threshold"][1],
            )
        )
        self.assertFalse(
            torch.allclose(
                output["keep_probability"][0],
                output["keep_probability"][1],
            )
        )
        torch.testing.assert_close(
            output["keep_probability"],
            torch.sigmoid(
                output["raw_importance"]
                - output["adaptive_keep_threshold"].unsqueeze(-1)
            ),
        )

        loss = (
            output["keep_probability"].mean()
            + output["refined_candidate_positions"].square().mean()
        )
        loss.backward()
        self.assertGreater(
            float(head.adaptive_threshold_head[-1].weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(float(global_features.grad.abs().sum()), 0.0)

    def test_provisional_position_feeds_back_into_final_keep_decision(self) -> None:
        torch.manual_seed(41)
        hidden_dim = 16
        head = InteractivePruningHead(
            hidden_dim=hidden_dim,
            residual_feature_dim=1,
            attention_heads=4,
            min_gap=0.01,
            one_shot_adaptive=True,
        ).eval()
        with torch.no_grad():
            # Make the named feedback path deterministic and strongly
            # sensitive to its provisional-position state block.
            feedback_in = head.position_to_keep_feedback[0]
            feedback_out = head.position_to_keep_feedback[-1]
            feedback_in.weight.zero_()
            feedback_in.bias.zero_()
            feedback_in.weight[:, hidden_dim : 2 * hidden_dim].copy_(
                torch.eye(hidden_dim)
            )
            feedback_out.weight.copy_(torch.eye(hidden_dim))
            feedback_out.bias.zero_()
            head.keep_head.weight.zero_()
            head.keep_head.weight[0, 0] = 1.0
            head.keep_head.bias.zero_()
            head.adaptive_threshold_head[-1].weight.zero_()
            head.adaptive_threshold_head[-1].bias.zero_()
            head.position_residual_head.weight.zero_()

        tokens = torch.randn(1, 6, hidden_dim)
        positions = torch.tensor([[0.08, 0.22, 0.40, 0.58, 0.76, 0.92]])
        keyword_inputs = {
            "coefficient_energy": torch.rand(1, 6),
            "deletion_delta": torch.rand(1, 6),
            "residual_features": torch.rand(1, 6),
        }
        with torch.no_grad():
            head.position_residual_head.bias.fill_(-4.0)
            shifted_left = head(tokens, positions, **keyword_inputs)
            head.position_residual_head.bias.fill_(4.0)
            shifted_right = head(tokens, positions, **keyword_inputs)

        # The preliminary decision precedes position decoding and is exactly
        # unchanged; the second keep decision must see the proposed motion.
        torch.testing.assert_close(
            shifted_left["preliminary_keep_probability"],
            shifted_right["preliminary_keep_probability"],
        )
        self.assertGreater(
            float(
                (
                    shifted_left["provisional_candidate_positions"]
                    - shifted_right["provisional_candidate_positions"]
                )
                .abs()
                .max()
            ),
            1e-3,
        )
        self.assertGreater(
            float(
                (
                    shifted_left["final_keep_probability"]
                    - shifted_right["final_keep_probability"]
                )
                .abs()
                .max()
            ),
            1e-4,
        )

        head.zero_grad(set_to_none=True)
        head.position_residual_head.bias.data.fill_(0.5)
        differentiable = head(tokens, positions, **keyword_inputs)
        differentiable["final_keep_probability"].mean().backward()
        self.assertGreater(
            float(head.position_residual_head.bias.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(head.position_to_keep_feedback[-1].weight.grad.abs().sum()),
            0.0,
        )

    def test_v9_keep_changes_cannot_move_fixed_proposal_geometry(self) -> None:
        torch.manual_seed(89)
        head = InteractivePruningHead(
            hidden_dim=16,
            residual_feature_dim=1,
            attention_heads=4,
            min_gap=0.01,
            one_shot_adaptive=True,
            one_shot_fixed_proposal_geometry=True,
        ).eval()
        positions = torch.tensor([[0.08, 0.22, 0.40, 0.58, 0.76, 0.92]])
        inputs = {
            "coefficient_energy": torch.rand(1, 6),
            "deletion_delta": torch.rand(1, 6),
            "residual_features": torch.rand(1, 6),
        }
        tokens = torch.randn(1, 6, 16)
        with torch.no_grad():
            head.keep_head.weight.zero_()
            head.keep_head.bias.fill_(-8.0)
            removed = head(tokens, positions, **inputs)
            head.keep_head.bias.fill_(8.0)
            retained = head(tokens, positions, **inputs)

        self.assertFalse(torch.any(removed["final_hard_keep_mask"]))
        self.assertTrue(torch.all(retained["final_hard_keep_mask"]))
        torch.testing.assert_close(
            removed["refined_candidate_positions"],
            retained["refined_candidate_positions"],
        )

    def test_v11_keep_mask_conditions_parallel_position_update(self) -> None:
        torch.manual_seed(97)
        head = InteractivePruningHead(
            hidden_dim=16,
            residual_feature_dim=1,
            attention_heads=4,
            min_gap=0.01,
            one_shot_adaptive=True,
            one_shot_fixed_proposal_geometry=True,
            one_shot_selection_policy="mass_topk",
            one_shot_joint_position_refinement=True,
        ).eval()
        positions = torch.tensor([[0.08, 0.22, 0.40, 0.58, 0.76, 0.92]])
        inputs = {
            "coefficient_energy": torch.rand(1, 6),
            "deletion_delta": torch.rand(1, 6),
            "residual_features": torch.rand(1, 6),
        }
        tokens = torch.randn(1, 6, 16)
        with torch.no_grad():
            head.keep_head.weight.zero_()
            head.adaptive_threshold_head[-1].weight.zero_()
            head.adaptive_threshold_head[-1].bias.zero_()
            head.joint_final_position_head.weight.zero_()
            head.joint_final_position_head.bias.fill_(3.0)
            head.keep_head.bias.fill_(-0.4)
            smaller = head(tokens, positions, **inputs)
            head.keep_head.bias.fill_(1.4)
            larger = head(tokens, positions, **inputs)

        self.assertLess(
            int(smaller["final_hard_keep_mask"].sum()),
            int(larger["final_hard_keep_mask"].sum()),
        )
        self.assertFalse(
            torch.allclose(
                smaller["refined_candidate_positions"],
                larger["refined_candidate_positions"],
            )
        )
        for output in (smaller, larger):
            selected = output["refined_candidate_positions"][
                output["final_hard_keep_mask"]
            ]
            if selected.numel() > 1:
                self.assertTrue(torch.all(selected[1:] - selected[:-1] >= 0.01))

    def test_v11_position_proposal_feeds_back_into_final_keep(self) -> None:
        torch.manual_seed(101)
        hidden_dim = 16
        head = InteractivePruningHead(
            hidden_dim=hidden_dim,
            residual_feature_dim=1,
            attention_heads=4,
            min_gap=0.01,
            one_shot_adaptive=True,
            one_shot_fixed_proposal_geometry=True,
            one_shot_selection_policy="mass_topk",
            one_shot_joint_position_refinement=True,
        ).eval()
        with torch.no_grad():
            feedback_in = head.joint_position_to_keep_feedback[0]
            feedback_out = head.joint_position_to_keep_feedback[-1]
            feedback_in.weight.zero_()
            feedback_in.bias.zero_()
            feedback_in.weight[:, hidden_dim : 2 * hidden_dim].copy_(
                torch.eye(hidden_dim)
            )
            feedback_out.weight.copy_(torch.eye(hidden_dim))
            feedback_out.bias.zero_()
            head.keep_head.weight.zero_()
            head.keep_head.weight[0, 0] = 1.0
            head.keep_head.bias.zero_()
            head.adaptive_threshold_head[-1].weight.zero_()
            head.adaptive_threshold_head[-1].bias.zero_()
            head.joint_preliminary_position_head.weight.zero_()

        positions = torch.tensor([[0.08, 0.22, 0.40, 0.58, 0.76, 0.92]])
        tokens = torch.randn(1, 6, hidden_dim)
        inputs = {
            "coefficient_energy": torch.rand(1, 6),
            "deletion_delta": torch.rand(1, 6),
            "residual_features": torch.rand(1, 6),
        }
        with torch.no_grad():
            head.joint_preliminary_position_head.bias.fill_(-3.0)
            shifted_left = head(tokens, positions, **inputs)
            head.joint_preliminary_position_head.bias.fill_(3.0)
            shifted_right = head(tokens, positions, **inputs)

        torch.testing.assert_close(
            shifted_left["preliminary_keep_probability"],
            shifted_right["preliminary_keep_probability"],
        )
        self.assertFalse(
            torch.allclose(
                shifted_left["final_keep_probability"],
                shifted_right["final_keep_probability"],
            )
        )

    def test_v11_two_stage_motion_respects_total_shift_budget(self) -> None:
        torch.manual_seed(103)
        max_shift = 0.05
        head = InteractivePruningHead(
            hidden_dim=16,
            residual_feature_dim=1,
            attention_heads=4,
            min_gap=0.01,
            one_shot_adaptive=True,
            one_shot_fixed_proposal_geometry=True,
            one_shot_selection_policy="mass_topk",
            one_shot_joint_position_refinement=True,
            one_shot_max_position_shift=max_shift,
        ).eval()
        with torch.no_grad():
            # Make the fixed proposal equal the supplied positions, retain all
            # slots, and ask both refinement stages for the same maximal move.
            head.position_residual_head.weight.zero_()
            head.position_residual_head.bias.zero_()
            head.keep_head.weight.zero_()
            head.keep_head.bias.fill_(8.0)
            head.adaptive_threshold_head[-1].weight.zero_()
            head.adaptive_threshold_head[-1].bias.zero_()
            head.joint_preliminary_position_head.weight.zero_()
            head.joint_final_position_head.weight.zero_()

        positions = torch.tensor([[0.10, 0.26, 0.42, 0.58, 0.74, 0.90]])
        tokens = torch.randn(1, 6, 16)
        inputs = {
            "coefficient_energy": torch.rand(1, 6),
            "deletion_delta": torch.rand(1, 6),
            "residual_features": torch.rand(1, 6),
        }
        for direction in (-10.0, 10.0):
            with self.subTest(direction=direction), torch.no_grad():
                head.joint_preliminary_position_head.bias.fill_(direction)
                head.joint_final_position_head.bias.fill_(direction)
                output = head(tokens, positions, **inputs)

            self.assertTrue(torch.all(output["final_hard_keep_mask"]))
            self.assertGreater(
                float(output["provisional_position_residual"].abs().max()),
                0.0,
            )
            self.assertGreater(
                float(output["final_position_residual"].abs().max()),
                0.0,
            )
            total_shift = (
                output["refined_candidate_positions"]
                - output["proposal_candidate_positions"]
            )
            self.assertTrue(torch.all(total_shift.abs() <= max_shift + 1e-6))

    def test_mass_topk_score_below_half_stably_selects_zero_knots(self) -> None:
        head = InteractivePruningHead(
            hidden_dim=16,
            attention_heads=4,
            one_shot_adaptive=True,
            one_shot_fixed_proposal_geometry=True,
            one_shot_selection_policy="mass_topk",
            one_shot_safety_sigma=0.25,
        )
        probability = torch.full((2, 4), 0.01)
        candidate_mask = torch.ones_like(probability, dtype=torch.bool)
        positions = torch.tensor([[0.1, 0.3, 0.6, 0.9], [0.1, 0.3, 0.6, 0.9]])

        first_mask, first_count, uncertainty = head._select_hard_keep_mask(
            probability,
            candidate_mask,
            positions,
        )
        second_mask, second_count, _ = head._select_hard_keep_mask(
            probability,
            candidate_mask,
            positions,
        )

        requested_score = probability.sum(dim=-1) + 0.25 * uncertainty
        self.assertTrue(torch.all(requested_score < 0.5))
        self.assertEqual(first_count.tolist(), [0, 0])
        self.assertFalse(torch.any(first_mask))
        self.assertTrue(torch.equal(second_count, first_count))
        self.assertTrue(torch.equal(second_mask, first_mask))

    def test_mass_topk_uses_probability_mass_as_structured_cardinality(self) -> None:
        head = InteractivePruningHead(
            hidden_dim=16,
            attention_heads=4,
            one_shot_adaptive=True,
            one_shot_fixed_proposal_geometry=True,
            one_shot_selection_policy="mass_topk",
        )
        probability = torch.tensor([[0.90, 0.80, 0.40, 0.30]])
        mask, count, uncertainty = head._select_hard_keep_mask(
            probability,
            torch.ones_like(probability, dtype=torch.bool),
            torch.tensor([[0.1, 0.3, 0.6, 0.9]]),
        )

        self.assertEqual(count.tolist(), [3])
        self.assertEqual(mask.tolist(), [[True, True, True, False]])
        self.assertGreater(float(uncertainty[0]), 0.0)

    def test_mass_topk_coverage_anchors_prevent_parameter_domain_collapse(self) -> None:
        head = InteractivePruningHead(
            hidden_dim=16,
            attention_heads=4,
            one_shot_adaptive=True,
            one_shot_fixed_proposal_geometry=True,
            one_shot_selection_policy="mass_topk",
            one_shot_coverage_bins=4,
        )
        positions = torch.tensor([[0.10, 0.20, 0.30, 0.45, 0.60, 0.70, 0.80, 0.90]])
        probability = torch.tensor([[0.80, 0.75, 0.70, 0.65, 0.30, 0.25, 0.20, 0.15]])
        mask, count, _ = head._select_hard_keep_mask(
            probability,
            torch.ones_like(probability, dtype=torch.bool),
            positions,
        )

        self.assertEqual(count.tolist(), [4])
        selected_positions = positions[mask]
        self.assertTrue(torch.all(selected_positions[:-1] < selected_positions[1:]))
        self.assertEqual(
            torch.floor(selected_positions * 4).to(torch.long).tolist(),
            [0, 1, 2, 3],
        )

    def test_final_keep_context_is_hard_forward_and_st_backward(self) -> None:
        torch.manual_seed(67)
        head = InteractivePruningHead(
            hidden_dim=16,
            residual_feature_dim=1,
            attention_heads=4,
            min_gap=0.01,
            one_shot_adaptive=True,
        )
        with torch.no_grad():
            head.keep_head.weight.zero_()
            head.keep_head.weight[0, 0] = 8.0
            head.keep_head.bias.zero_()
            head.adaptive_threshold_head[-1].weight.zero_()
            head.adaptive_threshold_head[-1].bias.zero_()

        positions = torch.tensor([[0.08, 0.22, 0.40, 0.58, 0.76, 0.92]]).expand(2, -1)
        output = head(
            torch.randn(2, 6, 16),
            positions,
            coefficient_energy=torch.rand(2, 6),
            deletion_delta=torch.rand(2, 6),
            residual_features=torch.rand(2, 6),
        )
        hard_mask = output["final_hard_keep_mask"]
        self.assertTrue(torch.all(hard_mask.any(dim=-1)))
        self.assertTrue(torch.all((~hard_mask).any(dim=-1)))
        expected_context = torch.stack(
            [
                output["final_decision_tokens"][index, hard_mask[index]].mean(dim=0)
                for index in range(hard_mask.shape[0])
            ]
        )
        torch.testing.assert_close(
            output["final_hard_st_keep_context"], expected_context
        )
        torch.testing.assert_close(
            output["final_hard_st_keep_gate"], hard_mask.to(torch.float32)
        )

        output["final_hard_st_keep_context"].square().mean().backward()
        self.assertGreater(
            float(head.adaptive_threshold_head[-1].bias.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(float(head.keep_head.weight.grad.abs().sum()), 0.0)

    def test_all_false_keep_mask_is_finite_and_keeps_original_positions(self) -> None:
        torch.manual_seed(71)
        head = InteractivePruningHead(
            hidden_dim=16,
            residual_feature_dim=1,
            attention_heads=4,
            min_gap=0.01,
            one_shot_adaptive=True,
        )
        with torch.no_grad():
            head.keep_head.weight.zero_()
            head.keep_head.bias.zero_()
            head.adaptive_threshold_head[-1].weight.zero_()
            head.adaptive_threshold_head[-1].bias.fill_(4.0)
            head.position_residual_head.weight.zero_()
            head.position_residual_head.bias.fill_(1.0)

        positions = torch.tensor([[0.10, 0.28, 0.47, 0.69, 0.90]])
        output = head(
            torch.randn(1, 5, 16),
            positions,
            coefficient_energy=torch.rand(1, 5),
            deletion_delta=torch.rand(1, 5),
            residual_features=torch.rand(1, 5),
        )
        self.assertFalse(bool(output["final_hard_keep_mask"].any()))
        torch.testing.assert_close(
            output["final_hard_st_keep_gate"],
            torch.zeros_like(output["final_hard_st_keep_gate"]),
        )
        torch.testing.assert_close(
            output["final_hard_st_keep_context"],
            torch.zeros_like(output["final_hard_st_keep_context"]),
        )
        torch.testing.assert_close(output["refined_candidate_positions"], positions)
        for name in (
            "final_hard_st_keep_context",
            "position_refinement_tokens",
            "position_residual",
            "refined_candidate_positions",
        ):
            self.assertTrue(torch.all(torch.isfinite(output[name])), name)

        output["refined_candidate_positions"].sum().backward()
        self.assertGreater(
            float(head.adaptive_threshold_head[-1].bias.grad.abs().sum()),
            0.0,
        )

    def test_soft_keep_state_changes_position_update(self) -> None:
        torch.manual_seed(29)
        head = InteractivePruningHead(
            hidden_dim=16,
            residual_feature_dim=1,
            attention_heads=4,
            min_gap=0.01,
            one_shot_adaptive=True,
        ).eval()
        with torch.no_grad():
            # A nonzero residual makes the keep gate's influence observable
            # independently of random initialization.
            head.position_residual_head.weight.zero_()
            head.position_residual_head.bias.fill_(1.0)
        tokens = torch.randn(1, 5, 16)
        positions = torch.tensor([[0.1, 0.28, 0.47, 0.69, 0.9]])
        keyword_inputs = {
            "coefficient_energy": torch.rand(1, 5),
            "deletion_delta": torch.rand(1, 5),
            "residual_features": torch.rand(1, 5),
        }
        with torch.no_grad():
            head.adaptive_threshold_head[-1].weight.zero_()
            head.adaptive_threshold_head[-1].bias.fill_(-8.0)
            mostly_kept = head(tokens, positions, **keyword_inputs)
            head.adaptive_threshold_head[-1].bias.fill_(8.0)
            mostly_removed = head(tokens, positions, **keyword_inputs)

        self.assertGreater(
            float(mostly_kept["keep_probability"].mean()),
            float(mostly_removed["keep_probability"].mean()),
        )
        self.assertFalse(
            torch.allclose(
                mostly_kept["position_residual"],
                mostly_removed["position_residual"],
            )
        )
        self.assertFalse(
            torch.allclose(
                mostly_kept["refined_candidate_positions"],
                mostly_removed["refined_candidate_positions"],
            )
        )

    def test_legacy_mode_has_exact_v7_parameter_layout(self) -> None:
        legacy = InteractivePruningHead(
            hidden_dim=16,
            residual_feature_dim=1,
            attention_heads=4,
            one_shot_adaptive=False,
        )
        state = legacy.state_dict()
        self.assertFalse(any("adaptive_threshold" in key for key in state))
        self.assertFalse(any("keep_context" in key for key in state))
        self.assertFalse(any("position_to_keep_feedback" in key for key in state))

        restored = InteractivePruningHead(
            hidden_dim=16,
            residual_feature_dim=1,
            attention_heads=4,
        )
        restored.load_state_dict(state, strict=True)

    def test_early_one_shot_v8_state_strictly_loads_without_feedback_keys(
        self,
    ) -> None:
        torch.manual_seed(83)
        source = InteractivePruningHead(
            hidden_dim=16,
            residual_feature_dim=1,
            attention_heads=4,
            one_shot_adaptive=True,
        )
        early_v8_state = {
            key: value.clone()
            for key, value in source.state_dict().items()
            if "position_to_keep_feedback" not in key
        }
        torch.manual_seed(89)
        restored = InteractivePruningHead(
            hidden_dim=16,
            residual_feature_dim=1,
            attention_heads=4,
            one_shot_adaptive=True,
        )
        expected_feedback = {
            key: value.clone()
            for key, value in restored.position_to_keep_feedback.state_dict().items()
        }
        restored.load_state_dict(early_v8_state, strict=True)

        for key, expected in source.state_dict().items():
            if "position_to_keep_feedback" not in key:
                torch.testing.assert_close(restored.state_dict()[key], expected)
        for key, expected in expected_feedback.items():
            torch.testing.assert_close(
                restored.position_to_keep_feedback.state_dict()[key], expected
            )

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

    def test_one_shot_random_extreme_updates_remain_strictly_ordered(self) -> None:
        hidden_dim = 16
        num_candidates = 12
        min_gap = 0.005
        available_interval_mass = 1.0 - (num_candidates + 1) * min_gap
        for seed in range(10):
            torch.manual_seed(100 + seed)
            head = InteractivePruningHead(
                hidden_dim=hidden_dim,
                residual_feature_dim=1,
                attention_heads=4,
                min_gap=min_gap,
                max_position_fraction=0.45,
                one_shot_adaptive=True,
            ).eval()
            with torch.no_grad():
                # Exercise saturated keep decisions and maximally bounded
                # moves in both directions under random extreme weights.
                head.keep_head.weight.normal_(std=100.0)
                head.keep_head.bias.uniform_(-100.0, 100.0)
                head.adaptive_threshold_head[-1].weight.normal_(std=100.0)
                head.adaptive_threshold_head[-1].bias.uniform_(-100.0, 100.0)
                head.position_to_keep_feedback[-1].weight.normal_(std=100.0)
                head.position_to_keep_feedback[-1].bias.uniform_(-100.0, 100.0)
                head.position_residual_head.weight.normal_(std=1000.0)
                head.position_residual_head.bias.uniform_(-1000.0, 1000.0)

            interval_weights = torch.softmax(
                8.0 * torch.randn(3, num_candidates + 1), dim=-1
            )
            intervals = min_gap + available_interval_mass * interval_weights
            positions = intervals[:, :-1].cumsum(dim=-1)
            output = head(
                torch.randn(3, num_candidates, hidden_dim),
                positions,
                coefficient_energy=1e6 * torch.rand(3, num_candidates),
                deletion_delta=1e6 * torch.randn(3, num_candidates),
                residual_features=1e6 * torch.randn(3, num_candidates),
            )
            for name in (
                "provisional_candidate_positions",
                "refined_candidate_positions",
            ):
                refined = output[name]
                self.assertTrue(torch.all(torch.isfinite(refined)), (seed, name))
                self.assertTrue(torch.all(refined > 0.0), (seed, name))
                self.assertTrue(torch.all(refined < 1.0), (seed, name))
                self.assertTrue(
                    torch.all(refined[:, 1:] - refined[:, :-1] >= min_gap - 1e-6),
                    (seed, name),
                )

    def test_extreme_analytic_features_remain_finite_and_mask_is_respected(
        self,
    ) -> None:
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
        self.assertEqual(float(output["predicted_deletion_cost"][0, 2].detach()), 0.0)
        self.assertEqual(float(output["position_residual"][0, 2].detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
