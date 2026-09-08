from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.synthetic import (  # noqa: E402
    SyntheticCubicBSplineDataset,
    certify_source_knot_minimality,
)


class CanonicalLabelTests(unittest.TestCase):
    def test_minimality_certificate_rejects_redundant_polynomial_knots(self) -> None:
        parameters = torch.linspace(0.0, 1.0, 129, dtype=torch.float64)
        polynomial_curve = torch.stack(
            [parameters, parameters.square() - 0.25 * parameters], dim=-1
        )
        certificate = certify_source_knot_minimality(
            parameters,
            polynomial_curve,
            torch.tensor([0.3, 0.7], dtype=torch.float64),
            error_tolerance=5e-3,
            margin=0.2,
        )

        self.assertFalse(certificate.certified)
        self.assertLess(float(certificate.full_fit_rms), 1e-10)
        self.assertLess(float(certificate.minimum_single_deletion_rms), 1e-10)

    def test_certified_dataset_has_noise_invariant_minimal_labels(self) -> None:
        common = {
            "size": 4,
            "num_points": 96,
            "min_control_points": 8,
            "max_control_points": 24,
            "canonical_knot_tolerance": 5e-3,
            "certified_minimal_source": True,
            "minimality_margin": 0.2,
            "minimality_audit_points": 128,
            "seed": 7654,
            "dtype": torch.float64,
        }
        clean_dataset = SyntheticCubicBSplineDataset(noise_std=0.0, **common)
        noisy_dataset = SyntheticCubicBSplineDataset(noise_std=0.01, **common)

        for index in range(4):
            clean = clean_dataset[index]
            noisy = noisy_dataset[index]
            clean_count = int(clean["true_internal_knot_mask"].sum())
            noisy_count = int(noisy["true_internal_knot_mask"].sum())

            self.assertTrue(clean["source_minimality_certified"])
            self.assertTrue(noisy["source_minimality_certified"])
            self.assertEqual(clean_count, clean["source_internal_knot_count"])
            self.assertEqual(noisy_count, noisy["source_internal_knot_count"])
            self.assertGreater(
                float(noisy["source_min_single_deletion_rms"]),
                float(noisy["source_minimality_required_rms"]),
            )
            torch.testing.assert_close(
                clean["true_internal_knots"], noisy["true_internal_knots"]
            )
            torch.testing.assert_close(clean["clean_points"], noisy["clean_points"])
            self.assertFalse(torch.equal(noisy["points"], noisy["clean_points"]))

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
