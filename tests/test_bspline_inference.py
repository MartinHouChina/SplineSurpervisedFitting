from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.synthetic import evaluate_bspline_curve
from spline_fitting.evaluation.bspline_inference import (
    hard_gate_mask,
    refit_bspline_control_points,
    refit_hard_gated_bspline_batch,
    refit_model_output_as_bsplines,
    second_difference_matrix,
)
from spline_fitting.evaluation.knot_diagnostics import build_open_knot_vector


class BSplineInferenceTests(unittest.TestCase):
    dtype = torch.float64

    @staticmethod
    def _control_points(count: int) -> torch.Tensor:
        x = torch.linspace(-0.8, 0.9, count, dtype=BSplineInferenceTests.dtype)
        return torch.stack([x, 0.35 * torch.sin(2.3 * x) + x.square()], dim=-1)

    def _sample_curve(
        self,
        parameters: torch.Tensor,
        internal_knots: torch.Tensor,
    ) -> torch.Tensor:
        controls = self._control_points(internal_knots.numel() + 4)
        knots = build_open_knot_vector(internal_knots, degree=3)
        return evaluate_bspline_curve(parameters, controls, knots, degree=3)

    def test_k_zero_refits_open_cubic_bezier(self) -> None:
        parameters = torch.linspace(0.0, 1.0, 48, dtype=self.dtype)
        internal_knots = torch.empty(0, dtype=self.dtype)
        points = self._sample_curve(parameters, internal_knots)

        result = refit_bspline_control_points(
            parameters,
            points,
            internal_knots,
            degree=3,
            smoothness_weight=0.0,
        )

        self.assertEqual(result.control_points.shape, (4, 2))
        self.assertEqual(result.knot_vector.numel(), 8)
        self.assertEqual(result.solver_rank, 4)
        self.assertLess(float(result.fit_mse), 1e-24)
        dense = result.evaluate(torch.linspace(0.0, 1.0, 101, dtype=self.dtype))
        self.assertEqual(dense.shape, (101, 2))

        batched = refit_hard_gated_bspline_batch(
            parameters.unsqueeze(0),
            torch.empty((1, 0), dtype=self.dtype),
            torch.empty((1, 0), dtype=torch.bool),
            points.unsqueeze(0),
            smoothness_weight=0.0,
        )[0]
        self.assertEqual(batched.candidate_count, 0)
        self.assertEqual(batched.retained_count, 0)
        self.assertEqual(batched.control_points.shape[0], 4)

    def test_batch_supports_different_retained_counts(self) -> None:
        parameters = torch.linspace(0.0, 1.0, 64, dtype=self.dtype)
        candidates = torch.tensor([0.31, 0.72], dtype=self.dtype)
        gates = torch.tensor(
            [[False, False], [True, False], [True, True]],
        )
        points = torch.stack(
            [
                self._sample_curve(parameters, candidates[gates[index]])
                for index in range(gates.shape[0])
            ],
            dim=0,
        )

        results = refit_hard_gated_bspline_batch(
            parameters.expand(3, -1),
            candidates.expand(3, -1),
            gates,
            points,
            degree=3,
            smoothness_weight=0.0,
        )

        self.assertEqual([result.retained_count for result in results], [0, 1, 2])
        self.assertEqual(
            [result.control_points.shape[0] for result in results],
            [4, 5, 6],
        )
        self.assertEqual(
            [result.knot_vector.numel() for result in results],
            [8, 9, 10],
        )
        for result in results:
            self.assertLess(float(result.fit_mse), 1e-24)

    def test_soft_probability_is_rejected_as_a_hard_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "model.eval"):
            hard_gate_mask(torch.tensor([0.0, 0.37, 1.0]))

        mask = hard_gate_mask(torch.tensor([0.0, 1.0, 1.0]))
        torch.testing.assert_close(mask, torch.tensor([False, True, True]))

    def test_model_output_adapter_uses_activity_gate(self) -> None:
        parameters = torch.linspace(0.0, 1.0, 40, dtype=self.dtype)
        candidates = torch.tensor([[0.4, 0.75]], dtype=self.dtype)
        hard_gate = torch.tensor([[1.0, 0.0]], dtype=self.dtype)
        points = self._sample_curve(parameters, candidates[0, :1]).unsqueeze(0)
        output = {
            "params": parameters.unsqueeze(0),
            "internal_knots": candidates,
            "activity_gate": hard_gate,
            # A contradictory probability verifies that inference does not use it.
            "activity": torch.tensor([[0.01, 0.99]], dtype=self.dtype),
        }

        result = refit_model_output_as_bsplines(
            output,
            points,
            smoothness_weight=0.0,
        )[0]

        self.assertEqual(result.retained_count, 1)
        torch.testing.assert_close(
            result.retained_internal_knots,
            torch.tensor([0.4], dtype=self.dtype),
        )
        self.assertLess(float(result.fit_mse), 1e-24)

    def test_augmented_objective_matches_reported_terms(self) -> None:
        parameters = torch.linspace(0.0, 1.0, 32, dtype=self.dtype)
        internal_knots = torch.tensor([0.25, 0.6], dtype=self.dtype)
        points = self._sample_curve(parameters, internal_knots)
        smoothness_weight = 2e-3
        control_ridge = 3e-4

        result = refit_bspline_control_points(
            parameters,
            points,
            internal_knots,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
        )

        expected = (
            result.data_squared_error
            + smoothness_weight * result.smoothness_squared
            + control_ridge * result.control_squared
        )
        torch.testing.assert_close(result.augmented_objective, expected)

        difference = second_difference_matrix(
            6,
            device=parameters.device,
            dtype=parameters.dtype,
        )
        self.assertEqual(difference.shape, (4, 6))
        torch.testing.assert_close(
            difference[0],
            torch.tensor([1.0, -2.0, 1.0, 0.0, 0.0, 0.0], dtype=self.dtype),
        )


if __name__ == "__main__":
    unittest.main()
