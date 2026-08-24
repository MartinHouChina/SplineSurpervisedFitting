from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.synthetic import evaluate_bspline_curve  # noqa: E402
from spline_fitting.evaluation.bspline_inference import (  # noqa: E402
    refit_bspline_control_points,
)
from spline_fitting.evaluation.knot_diagnostics import (  # noqa: E402
    build_open_knot_vector,
)
from spline_fitting.spline.bspline_deletion_teacher import (  # noqa: E402
    single_knot_deletion_rmse_batch,
)
from spline_fitting.training.one_shot_teacher import (  # noqa: E402
    OneShotTeacherConfig,
    TeacherAugmentedDataset,
    build_one_shot_teacher_batch,
    load_one_shot_teacher_cache,
    save_one_shot_teacher_cache,
)


class _FixedDataset(Dataset):
    def __init__(self, points: torch.Tensor, ids: torch.Tensor) -> None:
        self.points = points
        self.ids = ids
        self.resample_each_epoch = False
        self.last_epoch: int | None = None

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        return {"points": self.points[index], "sample_id": int(self.ids[index])}

    def set_epoch(self, epoch: int) -> None:
        self.last_epoch = epoch


class OneShotTeacherTests(unittest.TestCase):
    dtype = torch.float64

    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        parameters = torch.linspace(0.0, 1.0, 81, dtype=self.dtype).expand(2, -1)
        true_knots = (
            torch.tensor([0.5], dtype=self.dtype),
            torch.tensor([0.35, 0.7], dtype=self.dtype),
        )
        candidates = torch.tensor(
            [[0.2, 0.5, 0.72, 0.88], [0.18, 0.35, 0.55, 0.7]],
            dtype=self.dtype,
        )
        curves = []
        for batch_index, knots in enumerate(true_knots):
            count = knots.numel() + 4
            x = torch.linspace(-0.8, 0.9, count, dtype=self.dtype)
            controls = torch.stack(
                [x, x.square() + (0.2 + 0.1 * batch_index) * torch.sin(3.1 * x)],
                dim=-1,
            )
            curves.append(
                evaluate_bspline_curve(
                    parameters[batch_index],
                    controls,
                    build_open_knot_vector(knots, degree=3),
                    degree=3,
                )
            )
        return parameters, torch.stack(curves), candidates, torch.tensor([17, 29])

    def _build(self):
        parameters, points, candidates, sample_ids = self._inputs()
        config = OneShotTeacherConfig(
            error_tolerance=1e-8,
            temperature=0.2,
            smoothness_weight=0.0,
            dataset_fingerprint="fixed-two-curve-test-v1",
        )
        batch = build_one_shot_teacher_batch(
            parameters,
            points,
            candidates,
            sample_indices=sample_ids,
            config=config,
        )
        return parameters, points, candidates, sample_ids, config, batch

    def test_hard_labels_map_back_to_original_candidate_slots(self) -> None:
        parameters, points, candidates, _, config, batch = self._build()

        expected_masks = torch.tensor(
            [[False, True, False, False], [False, True, False, True]]
        )
        torch.testing.assert_close(batch.teacher_retained_mask, expected_masks)
        torch.testing.assert_close(batch.teacher_count, torch.tensor([1, 2]))
        self.assertTrue(bool(batch.teacher_threshold_satisfied.all()))
        for batch_index in range(2):
            count = int(batch.teacher_count[batch_index])
            retained = candidates[batch_index, expected_masks[batch_index]]
            torch.testing.assert_close(
                batch.teacher_internal_knots[batch_index, :count], retained
            )
            fit = refit_bspline_control_points(
                parameters[batch_index],
                points[batch_index],
                retained,
                smoothness_weight=config.smoothness_weight,
            )
            torch.testing.assert_close(batch.teacher_fit_rms[batch_index], fit.fit_rmse)

            removed = batch.teacher_deletion_order[batch_index]
            removed = removed[removed >= 0]
            self.assertEqual(removed.numel(), candidates.shape[1] - count)
            self.assertEqual(torch.unique(removed).numel(), removed.numel())
            self.assertTrue(bool((~expected_masks[batch_index, removed]).all()))

    def test_soft_keep_risk_uses_final_stopping_state_leave_one_out(self) -> None:
        parameters, points, candidates, _, config, batch = self._build()
        expected = torch.zeros_like(candidates)
        for batch_index in range(candidates.shape[0]):
            retained_mask = batch.teacher_retained_mask[batch_index]
            retained = candidates[batch_index, retained_mask]
            if retained.numel():
                final_leave_one_out = single_knot_deletion_rmse_batch(
                    parameters[batch_index : batch_index + 1],
                    points[batch_index : batch_index + 1],
                    retained.unsqueeze(0),
                    smoothness_weight=config.smoothness_weight,
                )[0]
                expected[batch_index, retained_mask] = torch.sigmoid(
                    (final_leave_one_out / config.error_tolerance - 1.0)
                    / config.temperature
                ).clamp_min(0.5)

        torch.testing.assert_close(batch.teacher_soft_keep_risk, expected)
        self.assertTrue(
            bool(
                (batch.teacher_soft_keep_risk[batch.teacher_retained_mask] >= 0.5).all()
            )
        )
        torch.testing.assert_close(
            batch.teacher_soft_keep_risk[~batch.teacher_retained_mask],
            torch.zeros_like(
                batch.teacher_soft_keep_risk[~batch.teacher_retained_mask]
            ),
        )
        self.assertEqual(
            set(batch.as_loss_kwargs()),
            {
                "teacher_retained_mask",
                "teacher_soft_keep_risk",
                "teacher_internal_knots",
                "teacher_internal_knot_mask",
                "teacher_count",
                "teacher_fit_rms",
                "teacher_threshold_satisfied",
                "teacher_deletion_order",
                "teacher_single_deletion_rms",
            },
        )

    def test_final_risk_does_not_conflict_with_redundant_hard_subset(self) -> None:
        parameters = torch.linspace(0.0, 1.0, 81, dtype=self.dtype).unsqueeze(0)
        true_knots = torch.tensor([0.5], dtype=self.dtype)
        x = torch.linspace(-0.8, 0.9, 5, dtype=self.dtype)
        controls = torch.stack([x, x.square() + 0.7 * torch.sin(3.1 * x)], dim=-1)
        points = evaluate_bspline_curve(
            parameters[0],
            controls,
            build_open_knot_vector(true_knots, degree=3),
            degree=3,
        ).unsqueeze(0)
        candidates = torch.tensor([[0.2, 0.49, 0.5, 0.51, 0.8]], dtype=self.dtype)
        config = OneShotTeacherConfig(
            error_tolerance=5e-3,
            temperature=0.2,
            smoothness_weight=0.0,
            dataset_fingerprint="near-duplicate-risk-regression-v1",
        )
        batch = build_one_shot_teacher_batch(
            parameters,
            points,
            candidates,
            sample_indices=[0],
            config=config,
        )

        retained = batch.teacher_retained_mask
        self.assertEqual(int(retained.sum()), 1)
        # In the original redundant set, even the ultimately retained slot can
        # be removed safely. This diagnostic must not become its final target.
        self.assertLess(
            float(batch.teacher_single_deletion_rms[retained]),
            config.error_tolerance,
        )
        self.assertGreaterEqual(float(batch.teacher_soft_keep_risk[retained]), 0.5)
        torch.testing.assert_close(
            batch.teacher_soft_keep_risk[~retained],
            torch.zeros_like(batch.teacher_soft_keep_risk[~retained]),
        )

    def test_cache_round_trip_and_strict_mismatch_checks(self) -> None:
        parameters, points, candidates, sample_ids, config, batch = self._build()
        shapes = {
            "parameters": parameters.shape,
            "points": points.shape,
            "candidate_knots": candidates.shape,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pt"
            save_one_shot_teacher_cache(path, batch)
            loaded = load_one_shot_teacher_cache(
                path,
                expected_config=config,
                expected_sample_indices=sample_ids,
                expected_input_shapes=shapes,
            )
            for name, expected in batch.as_loss_kwargs().items():
                torch.testing.assert_close(getattr(loaded, name), expected)

            with self.assertRaisesRegex(ValueError, "sample_indices"):
                load_one_shot_teacher_cache(
                    path,
                    expected_config=config,
                    expected_sample_indices=torch.tensor([17, 30]),
                    expected_input_shapes=shapes,
                )
            wrong_config = OneShotTeacherConfig(
                error_tolerance=2e-8,
                temperature=config.temperature,
                smoothness_weight=0.0,
                dataset_fingerprint=config.dataset_fingerprint,
            )
            with self.assertRaisesRegex(ValueError, "config"):
                load_one_shot_teacher_cache(
                    path,
                    expected_config=wrong_config,
                    expected_sample_indices=sample_ids,
                    expected_input_shapes=shapes,
                )
            wrong_dataset = OneShotTeacherConfig(
                error_tolerance=config.error_tolerance,
                temperature=config.temperature,
                smoothness_weight=0.0,
                dataset_fingerprint="different-curve-population-v1",
            )
            with self.assertRaisesRegex(ValueError, "config"):
                load_one_shot_teacher_cache(
                    path,
                    expected_config=wrong_dataset,
                    expected_sample_indices=sample_ids,
                    expected_input_shapes=shapes,
                )
            wrong_shapes = dict(shapes)
            wrong_shapes["points"] = (2, 80, 2)
            with self.assertRaisesRegex(ValueError, "shapes"):
                load_one_shot_teacher_cache(
                    path,
                    expected_config=config,
                    expected_sample_indices=sample_ids,
                    expected_input_shapes=wrong_shapes,
                )

    def test_dataset_merges_rows_and_rejects_resampling(self) -> None:
        _, points, _, sample_ids, _, batch = self._build()
        base = _FixedDataset(points, sample_ids)
        dataset = TeacherAugmentedDataset(base, batch)
        sample = dataset[1]
        torch.testing.assert_close(
            sample["teacher_retained_mask"], batch.teacher_retained_mask[1]
        )
        dataset.set_epoch(4)
        self.assertEqual(base.last_epoch, 4)

        base.ids[1] = 31
        with self.assertRaisesRegex(ValueError, "sample_id"):
            _ = dataset[1]
        base.resample_each_epoch = True
        with self.assertRaisesRegex(ValueError, "resample_each_epoch"):
            TeacherAugmentedDataset(base, batch)

    def test_endpoint_interpolation_cannot_be_disabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "interpolate_endpoints"):
            OneShotTeacherConfig(
                error_tolerance=1e-3,
                interpolate_endpoints=False,
            )


if __name__ == "__main__":
    unittest.main()
