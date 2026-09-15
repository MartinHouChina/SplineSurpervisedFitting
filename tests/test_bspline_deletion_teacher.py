from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.evaluation.bspline_inference import (  # noqa: E402
    refit_bspline_control_points,
)
from spline_fitting.spline.bspline_deletion_teacher import (  # noqa: E402
    _rank_aware_batched_lstsq,
    single_knot_deletion_mse_batch,
    single_knot_deletion_rmse_batch,
)
from spline_fitting.training.one_shot_teacher import (  # noqa: E402
    OneShotTeacherConfig,
    build_one_shot_teacher_batch,
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

    def test_rank_deficient_cpu_batch_matches_standard_svd_refit(self) -> None:
        parameters = torch.linspace(0.0, 1.0, 64, dtype=self.dtype)[None, :]
        points = torch.stack(
            [parameters[0], torch.sin(3.1 * torch.pi * parameters[0])], dim=-1
        )[None, :, :]
        knots = torch.full((1, 8), 0.5, dtype=self.dtype)
        actual = single_knot_deletion_mse_batch(
            parameters,
            points,
            knots,
            smoothness_weight=0.0,
            control_ridge=0.0,
        )
        retained = knots[0, 1:]
        expected = refit_bspline_control_points(
            parameters[0],
            points[0],
            retained,
            smoothness_weight=0.0,
            control_ridge=0.0,
        ).fit_rmse.square()
        self.assertTrue(bool(torch.isfinite(actual).all()))
        torch.testing.assert_close(
            actual, expected.expand_as(actual), rtol=2e-10, atol=2e-12
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_k72_rank_deficient_cuda_deletions_match_cpu_svd_refits(self) -> None:
        """Dense candidate proposals may contain unsupported basis columns."""
        device = torch.device("cuda:0")
        parameters = torch.linspace(
            0.0, 1.0, 192, dtype=self.dtype, device=device
        ).expand(2, -1)
        points = torch.stack(
            [
                parameters,
                0.2 * parameters + torch.sin(5.3 * torch.pi * parameters),
            ],
            dim=-1,
        )
        generator = torch.Generator(device=device).manual_seed(915)
        knots = torch.rand(
            2, 72, dtype=self.dtype, device=device, generator=generator
        ).sort(dim=-1).values.clamp(1e-4, 1.0 - 1e-4)

        actual = single_knot_deletion_mse_batch(
            parameters,
            points,
            knots,
            smoothness_weight=0.0,
            control_ridge=0.0,
        )
        self.assertEqual(actual.shape, (2, 72))
        self.assertTrue(bool(torch.isfinite(actual).all()))

        for curve_index, deletion_index in ((0, 0), (0, 35), (1, 71)):
            retained = torch.cat(
                [
                    knots[curve_index, :deletion_index],
                    knots[curve_index, deletion_index + 1 :],
                ]
            ).cpu()
            expected = refit_bspline_control_points(
                parameters[curve_index].cpu(),
                points[curve_index].cpu(),
                retained,
                smoothness_weight=0.0,
                control_ridge=0.0,
            ).fit_rmse.square()
            torch.testing.assert_close(
                actual[curve_index, deletion_index].cpu(),
                expected,
                rtol=2e-8,
                atol=1e-12,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_k72_offline_teacher_completes_one_greedy_round(self) -> None:
        """Exercise deletion, scalar refit and final leave-one-out together."""
        device = torch.device("cuda:0")
        parameters = torch.linspace(
            0.0, 1.0, 192, dtype=self.dtype, device=device,
        )[None, :]
        points = torch.stack(
            [parameters, torch.sin(5.3 * torch.pi * parameters)], dim=-1,
        )
        generator = torch.Generator(device=device).manual_seed(915)
        knots = torch.rand(
            1, 72, dtype=self.dtype, device=device, generator=generator,
        ).sort(dim=-1).values.clamp(1e-4, 1.0 - 1e-4)
        teacher = build_one_shot_teacher_batch(
            parameters, points, knots,
            sample_indices=[0],
            config=OneShotTeacherConfig(
                error_tolerance=10.0,
                min_internal_knots=71,
                smoothness_weight=0.0,
                control_ridge=0.0,
            ),
        )
        self.assertEqual(int(teacher.teacher_count[0]), 71)
        self.assertTrue(bool(teacher.teacher_threshold_satisfied[0]))
        self.assertTrue(bool(torch.isfinite(teacher.teacher_fit_mse).all()))
        self.assertTrue(bool(torch.isfinite(teacher.teacher_soft_keep_risk).all()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_nonfinite_cuda_qr_solution_retries_only_affected_state(self) -> None:
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(361)
        design = torch.randn(
            2, 3, 12, 5, dtype=self.dtype, device=device, generator=generator
        )
        target = torch.randn(
            2, 3, 12, 2, dtype=self.dtype, device=device, generator=generator
        )
        expected = torch.linalg.lstsq(
            design.cpu(), target.cpu(), driver="gelsd"
        ).solution.to(device)
        original_lstsq = torch.linalg.lstsq

        def nonfinite_cuda_lstsq(a, b, **kwargs):
            result = original_lstsq(a, b, **kwargs)
            if a.device.type == "cuda":
                solution = result.solution.clone()
                solution[0, 1] = float("nan")
                return SimpleNamespace(solution=solution)
            return result

        with patch.object(torch.linalg, "lstsq", side_effect=nonfinite_cuda_lstsq):
            actual = _rank_aware_batched_lstsq(design, target, rcond=None)

        self.assertTrue(bool(torch.isfinite(actual).all()))
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-12)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_finite_explosive_cuda_coefficients_retry_invalid_mse(self) -> None:
        parameters, points, knots = self._inputs()
        expected = single_knot_deletion_mse_batch(parameters, points, knots)
        parameters = parameters.cuda()
        points = points.cuda()
        knots = knots.cuda()
        original_lstsq = torch.linalg.lstsq

        def explosive_cuda_lstsq(a, b, **kwargs):
            result = original_lstsq(a, b, **kwargs)
            if a.device.type == "cuda":
                return SimpleNamespace(
                    solution=torch.full_like(result.solution, 1e200)
                )
            return result

        with patch.object(torch.linalg, "lstsq", side_effect=explosive_cuda_lstsq):
            actual = single_knot_deletion_mse_batch(parameters, points, knots)

        self.assertTrue(bool(torch.isfinite(actual).all()))
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-10, atol=2e-12)


if __name__ == "__main__":
    unittest.main()
