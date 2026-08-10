from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

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


if __name__ == "__main__":
    unittest.main()
