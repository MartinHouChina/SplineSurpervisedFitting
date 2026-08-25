from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.synthetic import SyntheticCubicBSplineDataset


class CanonicalLabelTests(unittest.TestCase):
    def test_canonical_labels_are_consistent_and_deterministic(self) -> None:
        dataset = SyntheticCubicBSplineDataset(
            size=8,
            seed=321,
            canonical_knot_tolerance=5e-3,
        )
        first = dataset[3]
        repeated = dataset[3]
        canonical_count = int(first["true_internal_knot_mask"].sum())

        self.assertLessEqual(canonical_count, first["source_internal_knot_count"])
        self.assertEqual(first["num_control_points"], canonical_count + 4)
        self.assertLessEqual(float(first["canonical_fit_rms"]), 5e-3 + 1e-7)
        torch.testing.assert_close(
            first["true_internal_knots"], repeated["true_internal_knots"]
        )
        torch.testing.assert_close(
            first["true_control_points"], repeated["true_control_points"]
        )

    def test_source_spline_is_preserved_in_fixed_size_batch_fields(self) -> None:
        dataset = SyntheticCubicBSplineDataset(
            size=4,
            seed=4321,
            min_control_points=6,
            max_control_points=12,
            canonical_knot_tolerance=5e-3,
        )
        batch = next(iter(DataLoader(dataset, batch_size=4)))

        self.assertEqual(tuple(batch["source_control_points"].shape), (4, 12, 2))
        self.assertEqual(tuple(batch["source_control_mask"].shape), (4, 12))
        self.assertEqual(tuple(batch["source_knot_vector"].shape), (4, 16))
        self.assertEqual(tuple(batch["source_knot_mask"].shape), (4, 16))
        torch.testing.assert_close(
            batch["source_control_mask"].sum(dim=1),
            batch["source_num_control_points"],
        )
        torch.testing.assert_close(
            batch["source_knot_mask"].sum(dim=1),
            batch["source_num_control_points"] + 4,
        )

    def test_zero_tolerance_preserves_source_knot_count(self) -> None:
        sample = SyntheticCubicBSplineDataset(
            size=1,
            seed=111,
            canonical_knot_tolerance=0.0,
        )[0]
        self.assertEqual(
            int(sample["true_internal_knot_mask"].sum()),
            sample["source_internal_knot_count"],
        )

    def test_training_population_can_resample_without_changing_validation_mode(
        self,
    ) -> None:
        resampled = SyntheticCubicBSplineDataset(
            size=2,
            seed=123,
            resample_each_epoch=True,
        )
        epoch_zero = resampled[0]["points"].clone()
        resampled.set_epoch(1)
        epoch_one = resampled[0]["points"].clone()
        self.assertFalse(torch.equal(epoch_zero, epoch_one))
        self.assertEqual(resampled[0]["sample_epoch"], 1)

        fixed = SyntheticCubicBSplineDataset(size=2, seed=123)
        fixed_zero = fixed[0]["points"].clone()
        fixed.set_epoch(3)
        torch.testing.assert_close(fixed[0]["points"], fixed_zero)
        self.assertEqual(fixed[0]["sample_epoch"], 0)


if __name__ == "__main__":
    unittest.main()
