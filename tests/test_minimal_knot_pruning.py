from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.synthetic import evaluate_bspline_curve
from spline_fitting.evaluation.knot_diagnostics import build_open_knot_vector
from spline_fitting.evaluation.minimal_knot_pruning import (
    prune_knots_to_rms_tolerance,
)


class MinimalKnotPruningTests(unittest.TestCase):
    dtype = torch.float64

    def _sample_curve(
        self,
        parameters: torch.Tensor,
        internal_knots: torch.Tensor,
    ) -> torch.Tensor:
        control_count = int(internal_knots.numel()) + 4
        x = torch.linspace(-0.8, 0.9, control_count, dtype=self.dtype)
        controls = torch.stack(
            [x, 0.45 * torch.sin(4.1 * x) + 0.2 * x.square()], dim=-1
        )
        return evaluate_bspline_curve(
            parameters,
            controls,
            build_open_knot_vector(internal_knots, degree=3),
            degree=3,
        )

    def test_removes_inserted_redundant_knot_and_hard_stops(self) -> None:
        parameters = torch.linspace(0.0, 1.0, 129, dtype=self.dtype)
        source_knots = torch.tensor([0.28, 0.71], dtype=self.dtype)
        # Adding 0.5 only refines the same spline space. A full control-point
        # refit can therefore remove it with numerical-zero geometric error.
        candidates = torch.tensor([0.28, 0.5, 0.71], dtype=self.dtype)
        points = self._sample_curve(parameters, source_knots)

        result = prune_knots_to_rms_tolerance(
            parameters,
            points,
            candidates,
            error_tolerance=1e-10,
            smoothness_weight=0.0,
            control_ridge=0.0,
        )

        self.assertTrue(result.threshold_satisfied)
        self.assertEqual(result.initial_count, 3)
        self.assertEqual(result.final_count, 2)
        self.assertEqual(len(result.accepted_steps), 1)
        self.assertEqual(len(result.steps), 2)
        self.assertFalse(result.steps[-1].accepted)
        torch.testing.assert_close(result.removed_knots, candidates[1:2])
        torch.testing.assert_close(result.final_internal_knots, source_knots)
        self.assertLessEqual(float(result.final_fit.fit_rmse), 1e-10)
        self.assertGreater(
            float(result.steps[-1].candidate_rmse), result.error_tolerance
        )
        self.assertEqual(result.steps[0].all_candidate_rmse.shape, (3,))
        self.assertEqual(result.rms_trajectory.shape, (2,))

    def test_larger_tolerance_cannot_retain_more_knots(self) -> None:
        parameters = torch.linspace(0.0, 1.0, 129, dtype=self.dtype)
        source_knots = torch.tensor([0.24, 0.47, 0.76], dtype=self.dtype)
        candidates = torch.tensor(
            [0.16, 0.24, 0.34, 0.47, 0.62, 0.76, 0.88], dtype=self.dtype
        )
        points = self._sample_curve(parameters, source_knots)

        strict = prune_knots_to_rms_tolerance(
            parameters,
            points,
            candidates,
            error_tolerance=1e-9,
            smoothness_weight=0.0,
        )
        relaxed = prune_knots_to_rms_tolerance(
            parameters,
            points,
            candidates,
            error_tolerance=3e-2,
            smoothness_weight=0.0,
        )

        self.assertLessEqual(relaxed.final_count, strict.final_count)
        self.assertLessEqual(float(relaxed.final_fit.fit_rmse), 3e-2)
        self.assertLessEqual(float(strict.final_fit.fit_rmse), 1e-9)

    def test_minimum_count_and_endpoint_interpolation_are_respected(self) -> None:
        parameters = torch.linspace(0.0, 1.0, 81, dtype=self.dtype)
        candidates = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=self.dtype)
        points = torch.stack(
            [parameters, torch.sin(3.0 * torch.pi * parameters)], dim=-1
        )

        result = prune_knots_to_rms_tolerance(
            parameters,
            points,
            candidates,
            error_tolerance=10.0,
            min_internal_knots=2,
            smoothness_weight=2e-3,
            control_ridge=3e-4,
            interpolate_endpoints=True,
        )

        self.assertEqual(result.final_count, 2)
        self.assertEqual(len(result.accepted_steps), 2)
        self.assertTrue(all(step.accepted for step in result.steps))
        torch.testing.assert_close(result.final_fit.reconstructed_points[0], points[0])
        torch.testing.assert_close(
            result.final_fit.reconstructed_points[-1], points[-1]
        )

    def test_invalid_minimum_count_is_rejected(self) -> None:
        parameters = torch.linspace(0.0, 1.0, 16, dtype=self.dtype)
        points = torch.stack([parameters, parameters.square()], dim=-1)
        candidates = torch.tensor([0.5], dtype=self.dtype)

        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            prune_knots_to_rms_tolerance(
                parameters,
                points,
                candidates,
                error_tolerance=0.1,
                min_internal_knots=2,
            )


if __name__ == "__main__":
    unittest.main()
