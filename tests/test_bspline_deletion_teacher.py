from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.evaluation.bspline_inference import (  # noqa: E402
    refit_bspline_control_points,
)
from spline_fitting.spline.bspline_deletion_teacher import (  # noqa: E402
    single_knot_deletion_mse_batch,
    single_knot_deletion_rmse_batch,
)


class BSplineDeletionTeacherTests(unittest.TestCase):
    dtype = torch.float64

    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parameters = torch.stack(
            [
                torch.linspace(0.0, 1.0, 67, dtype=self.dtype),
                torch.linspace(0.0, 1.0, 67, dtype=self.dtype).pow(1.35),
            ]
        )
        x = parameters
        points = torch.stack(
            [
                torch.stack(
                    [x[0], 0.17 * x[0] + torch.sin(4.7 * torch.pi * x[0])],
                    dim=-1,
                ),
                torch.stack(
                    [
                        torch.cos(1.6 * torch.pi * x[1]),
                        x[1] + 0.3 * torch.sin(6.1 * torch.pi * x[1]),
                    ],
                    dim=-1,
                ),
            ]
        )
        knots = torch.tensor(
            [[0.13, 0.31, 0.56, 0.82], [0.08, 0.38, 0.63, 0.91]],
            dtype=self.dtype,
        )
        return parameters, points, knots

    def _reference(
        self,
        parameters: torch.Tensor,
        points: torch.Tensor,
        knots: torch.Tensor,
        *,
        smoothness_weight: float,
        control_ridge: float,
        interpolate_endpoints: bool,
    ) -> torch.Tensor:
        rows = []
        for batch_index in range(knots.shape[0]):
            values = []
            for deleted_index in range(knots.shape[1]):
                retained = torch.cat(
                    [
                        knots[batch_index, :deleted_index],
                        knots[batch_index, deleted_index + 1 :],
                    ]
                )
                fit = refit_bspline_control_points(
                    parameters[batch_index],
                    points[batch_index],
                    retained,
                    degree=3,
                    smoothness_weight=smoothness_weight,
                    control_ridge=control_ridge,
                    interpolate_endpoints=interpolate_endpoints,
                )
                values.append(fit.fit_rmse)
            rows.append(torch.stack(values))
        return torch.stack(rows)

    def test_matches_individual_deployment_refits(self) -> None:
        parameters, points, knots = self._inputs()
        actual = single_knot_deletion_rmse_batch(parameters, points, knots)
        expected = self._reference(
            parameters,
            points,
            knots,
            smoothness_weight=1e-6,
            control_ridge=0.0,
            interpolate_endpoints=True,
        )

        self.assertEqual(actual.shape, knots.shape)
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-12)

    def test_matches_ridge_refit_without_endpoint_constraints(self) -> None:
        parameters, points, knots = self._inputs()
        actual = single_knot_deletion_rmse_batch(
            parameters,
            points,
            knots,
            smoothness_weight=2e-3,
            control_ridge=4e-4,
            interpolate_endpoints=False,
        )
        expected = self._reference(
            parameters,
            points,
            knots,
            smoothness_weight=2e-3,
            control_ridge=4e-4,
            interpolate_endpoints=False,
        )

        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-12)

    def test_mse_batch_is_finite_rmse_square_and_preserves_legacy_result(self) -> None:
        parameters, points, knots = self._inputs()
        options = {
            "smoothness_weight": 2e-3,
            "control_ridge": 4e-4,
            "interpolate_endpoints": False,
        }

        legacy_before = single_knot_deletion_rmse_batch(
            parameters,
            points,
            knots,
            **options,
        )
        mse = single_knot_deletion_mse_batch(
            parameters,
            points,
            knots,
            **options,
        )
        legacy_after = single_knot_deletion_rmse_batch(
            parameters,
            points,
            knots,
            **options,
        )

        self.assertEqual(mse.shape, knots.shape)
        self.assertTrue(bool(torch.isfinite(mse).all()))
        self.assertTrue(bool((mse >= 0.0).all()))
        torch.testing.assert_close(
            mse,
            legacy_before.square(),
            rtol=5e-14,
            atol=5e-16,
        )
        torch.testing.assert_close(
            legacy_after,
            legacy_before,
            rtol=5e-14,
            atol=5e-16,
        )

        expected_legacy = self._reference(
            parameters,
            points,
            knots,
            smoothness_weight=options["smoothness_weight"],
            control_ridge=options["control_ridge"],
            interpolate_endpoints=options["interpolate_endpoints"],
        )
        torch.testing.assert_close(
            legacy_after,
            expected_legacy,
            rtol=2e-10,
            atol=2e-12,
        )

    def test_one_candidate_produces_bezier_deletion_state(self) -> None:
        parameters = torch.linspace(0.0, 1.0, 45, dtype=self.dtype).unsqueeze(0)
        points = torch.stack(
            [parameters[0], torch.sin(3.0 * torch.pi * parameters[0])], dim=-1
        ).unsqueeze(0)
        knots = torch.tensor([[0.47]], dtype=self.dtype)

        actual = single_knot_deletion_rmse_batch(parameters, points, knots)
        expected = refit_bspline_control_points(
            parameters[0],
            points[0],
            torch.empty(0, dtype=self.dtype),
        ).fit_rmse

        self.assertEqual(actual.shape, (1, 1))
        torch.testing.assert_close(actual[0, 0], expected, rtol=2e-10, atol=2e-12)

    def test_rejects_zero_candidates(self) -> None:
        with self.assertRaisesRegex(ValueError, "K >= 1"):
            single_knot_deletion_rmse_batch(
                torch.linspace(0.0, 1.0, 8).unsqueeze(0),
                torch.zeros(1, 8, 2),
                torch.empty(1, 0),
            )


if __name__ == "__main__":
    unittest.main()
