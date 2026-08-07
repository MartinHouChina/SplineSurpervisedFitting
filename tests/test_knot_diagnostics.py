from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.evaluation.knot_diagnostics import (
    activity_statistics,
    build_open_knot_vector,
    hard_prune_and_refit,
    match_internal_knots,
    point_fit_statistics,
)


class KnotDiagnosticsTests(unittest.TestCase):
    def test_activity_statistics_uses_greater_equal_threshold(self) -> None:
        activity = torch.tensor([[0.2, 0.5, 0.8], [0.0, 0.1, 0.2]])
        result = activity_statistics(activity, threshold=0.5)

        torch.testing.assert_close(result["activity_mass"], torch.tensor([1.5, 0.3]))
        torch.testing.assert_close(result["hard_active_count"], torch.tensor([2.0, 0.0]))
        torch.testing.assert_close(
            result["candidate_knot_count"], torch.tensor([3.0, 3.0])
        )

    def test_open_knot_vector_handles_no_internal_knots(self) -> None:
        result = build_open_knot_vector(torch.empty(0), degree=3)
        torch.testing.assert_close(
            result, torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        )

    def test_hard_prune_and_refit_supports_different_counts_in_batch(self) -> None:
        params = torch.linspace(0.0, 1.0, 32).expand(3, -1)
        points = torch.stack(
            [
                torch.stack([params[0], params[0].pow(2)], dim=-1),
                torch.stack([params[1], params[1].pow(2)], dim=-1),
                torch.stack([params[2], params[2].pow(2)], dim=-1),
            ]
        )
        knots = torch.tensor(
            [[0.25, 0.75], [0.25, 0.75], [0.25, 0.75]], dtype=params.dtype
        )
        activity = torch.tensor(
            [[0.1, 0.2], [0.6, 0.2], [0.6, 0.8]], dtype=params.dtype
        )

        reports = hard_prune_and_refit(
            params,
            knots,
            activity,
            points,
            degree=3,
            threshold=0.5,
            lambda_poly=1e-8,
            lambda_knot=1e-8,
        )

        self.assertEqual([item.retained_count for item in reports], [0, 1, 2])
        for item in reports:
            self.assertTrue(torch.isfinite(item.fit_mse))
            self.assertLess(float(item.fit_mse), 1e-8)
            self.assertEqual(item.open_knot_vector.numel(), item.retained_count + 8)

    def test_point_fit_statistics_matches_training_definition(self) -> None:
        points = torch.zeros(1, 2, 2)
        reconstructed = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
        result = point_fit_statistics(reconstructed, points)
        self.assertAlmostEqual(float(result["fit_mse"]), 2.5)
        self.assertAlmostEqual(float(result["coordinate_mse"]), 1.25)

    def test_ordered_knot_matching(self) -> None:
        result = match_internal_knots(
            torch.tensor([0.19, 0.51, 0.9]),
            torch.tensor([0.2, 0.5]),
            tolerance=0.02,
        )
        self.assertEqual(result.matched_count, 2)
        self.assertAlmostEqual(result.precision, 2.0 / 3.0)
        self.assertAlmostEqual(result.recall, 1.0)
        self.assertAlmostEqual(result.matched_mae, 0.01, places=6)

    def test_invalid_activity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            activity_statistics(torch.tensor([[float("nan")]]))
        with self.assertRaises(ValueError):
            activity_statistics(torch.tensor([[1.1]]))


if __name__ == "__main__":
    unittest.main()
