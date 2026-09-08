from __future__ import annotations

# ruff: noqa: E402

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.synthetic import (
    evaluate_bspline_curve,
    generate_cubic_bspline_sample,
)
from spline_fitting.evaluation.knot_diagnostics import build_open_knot_vector
from spline_fitting.evaluation.sparse_knot_paper import (
    _relocate_clusters,
    fit_sparse_knots_paper,
)


class SparseKnotPaperTests(unittest.TestCase):
    dtype = torch.float64

    def test_polynomial_has_no_active_knots(self) -> None:
        parameters = torch.linspace(0.0, 1.0, 49, dtype=self.dtype)
        points = torch.stack(
            [
                0.2 + parameters - 0.7 * parameters.square(),
                -0.4 + 0.3 * parameters + parameters.pow(3),
            ],
            dim=-1,
        )
        result = fit_sparse_knots_paper(
            parameters,
            points,
            degree=3,
            initial_internal_knot_count=20,
            data_tolerance=1e-12,
            jump_threshold=1e-8,
        )

        self.assertEqual(result.active_count, 0)
        self.assertEqual(result.knots.numel(), 0)
        self.assertLess(float(result.fit_mse), 1e-20)
        self.assertIn("group-L1", result.method)

    def test_zero_dense_internal_knots_accepts_relative_jump_floor(self) -> None:
        parameters = torch.linspace(0.0, 1.0, 33, dtype=self.dtype)
        points = torch.stack([parameters, parameters.square()], dim=-1)

        result = fit_sparse_knots_paper(
            parameters,
            points,
            initial_internal_knot_count=0,
            data_tolerance=1e-12,
            relative_jump_threshold=1e-3,
        )

        self.assertEqual(result.active_count, 0)
        self.assertEqual(result.effective_jump_threshold, result.jump_threshold)
        self.assertEqual(result.knots.numel(), 0)

    def test_infeasible_dense_start_is_reported_instead_of_aborting_batch(self) -> None:
        parameters = torch.linspace(0.0, 1.0, 33, dtype=self.dtype)
        points = torch.stack(
            [parameters, torch.sin(5.0 * torch.pi * parameters)], dim=-1
        )

        result = fit_sparse_knots_paper(
            parameters,
            points,
            initial_internal_knot_count=0,
            data_tolerance=1e-12,
            relative_jump_threshold=1e-3,
        )

        self.assertFalse(result.dense_initial_threshold_satisfied)
        self.assertGreater(float(result.dense_initial_fit_mse), 1e-12)
        self.assertEqual(result.sparse_iterations, 0)
        self.assertFalse(result.final_threshold_satisfied)

    def test_known_knot_curve_is_localized_and_refit(self) -> None:
        parameters = torch.linspace(0.0, 1.0, 81, dtype=self.dtype)
        true_knots = torch.tensor([0.31, 0.72], dtype=self.dtype)
        controls = torch.tensor(
            [
                [0.0, 0.0],
                [0.15, 0.7],
                [0.35, -0.55],
                [0.68, 0.8],
                [0.86, -0.35],
                [1.0, 0.05],
            ],
            dtype=self.dtype,
        )
        points = evaluate_bspline_curve(
            parameters,
            controls,
            build_open_knot_vector(true_knots, degree=3),
            degree=3,
        )
        result = fit_sparse_knots_paper(
            parameters,
            points,
            initial_internal_knot_count=24,
            data_tolerance=2e-6,
            jump_threshold=1e-5,
            admm_max_iterations=1000,
            relocation_max_iterations=8,
            feasibility_repair=True,
        )

        self.assertGreaterEqual(result.active_count, 2)
        self.assertLessEqual(result.knots.numel(), result.active_count)
        self.assertLess(float(result.fit_mse), 2e-5)
        for knot in true_knots:
            self.assertLess(float((result.knots - knot).abs().min()), 0.08)
        self.assertGreaterEqual(result.final_refit_count, 1)
        self.assertTrue(result.final_threshold_satisfied)
        self.assertGreaterEqual(result.local_refit_count, 0)

    def test_singleton_active_cluster_is_locally_relocated(self) -> None:
        parameters = torch.linspace(0.0, 1.0, 101, dtype=self.dtype)
        true_knot = torch.tensor([0.31], dtype=self.dtype)
        controls = torch.tensor(
            [[0.0, 0.0], [0.15, 0.8], [0.45, -0.6], [0.75, 0.7], [1.0, 0.0]],
            dtype=self.dtype,
        )
        points = evaluate_bspline_curve(
            parameters,
            controls,
            build_open_knot_vector(true_knot, degree=3),
            degree=3,
        )
        starting = torch.tensor([0.5], dtype=self.dtype)

        relocated, refits = _relocate_clusters(
            parameters,
            points,
            [starting],
            degree=3,
            initial_spacing=0.25,
            tolerance=1e-4,
            max_iterations=12,
        )

        self.assertGreater(refits, 0)
        self.assertLess(
            float((relocated - true_knot).abs()),
            float((starting - true_knot).abs()),
        )

    def test_knots_are_ordered_bounded_and_thresholded(self) -> None:
        parameters = torch.linspace(0.0, 1.0, 65, dtype=self.dtype)
        points = torch.stack(
            [parameters, torch.sin(5.0 * torch.pi * parameters)], dim=-1
        )
        base = dict(
            initial_internal_knot_count=18,
            data_tolerance=2e-4,
            admm_max_iterations=700,
            relocation_max_iterations=5,
        )
        low = fit_sparse_knots_paper(
            parameters, points, jump_threshold=1e-9, **base
        )
        high = fit_sparse_knots_paper(
            parameters, points, jump_threshold=1e9, **base
        )

        self.assertGreaterEqual(low.active_count, high.active_count)
        self.assertEqual(high.active_count, 0)
        self.assertTrue(torch.all(low.knots > 0.0))
        self.assertTrue(torch.all(low.knots < 1.0))
        self.assertTrue(torch.all(low.knots[1:] >= low.knots[:-1]))
        torch.testing.assert_close(
            low.active_mask,
            low.jump_norms > low.effective_jump_threshold,
        )

    def test_numerical_jump_floor_and_post_merge_repair(self) -> None:
        """Exercise the dense-active-cluster failure seen in comparison runs."""
        sample = generate_cubic_bspline_sample(
            num_points=48,
            min_control_points=8,  # source K = 4
            max_control_points=8,
            noise_std=0.0,
            generator=torch.Generator().manual_seed(20_000),
            dtype=self.dtype,
        )
        common = dict(
            initial_internal_knot_count=12,
            data_tolerance=2.5e-5,
            admm_max_iterations=400,
            relocation_max_iterations=6,
        )
        absolute_only = fit_sparse_knots_paper(
            sample.parameters,
            sample.points,
            feasibility_repair=True,
            **common,
        )
        relative = fit_sparse_knots_paper(
            sample.parameters,
            sample.points,
            relative_jump_threshold=1e-3,
            feasibility_repair=True,
            **common,
        )

        # Tiny ADMM residual jumps make every candidate look active under the
        # legacy absolute threshold.  The repair must still restore the final
        # exact-refit bound after that cluster is merged.
        self.assertEqual(absolute_only.active_count, 12)
        self.assertTrue(absolute_only.repair_used)
        self.assertGreater(absolute_only.repair_refit_count, 1)
        self.assertTrue(absolute_only.final_threshold_satisfied)
        self.assertEqual(absolute_only.effective_jump_threshold, 1e-7)
        self.assertGreater(relative.effective_jump_threshold, 1e-7)
        self.assertLess(relative.active_count, absolute_only.active_count)
        self.assertTrue(relative.final_threshold_satisfied)

    def test_faithful_default_does_not_add_back_knots(self) -> None:
        parameters = torch.linspace(0.0, 1.0, 49, dtype=self.dtype)
        points = torch.stack(
            [parameters, torch.sin(5.0 * torch.pi * parameters)], dim=-1
        )
        result = fit_sparse_knots_paper(
            parameters,
            points,
            initial_internal_knot_count=12,
            data_tolerance=2.5e-5,
            admm_max_iterations=200,
            relocation_max_iterations=3,
        )

        self.assertFalse(result.repair_used)
        self.assertEqual(result.repair_added_knots.numel(), 0)
        self.assertIn("Disabled", result.repair_method)


if __name__ == "__main__":
    unittest.main()
