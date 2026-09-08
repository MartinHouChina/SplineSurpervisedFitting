from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.real_world import (  # noqa: E402
    RealWorldCurveDataset,
    read_curve_manifest,
    write_curve_manifest,
)


def _record(sample_id: str, points_path: str, split: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "source_dataset": "fixture",
        "group_id": f"writer-{sample_id}",
        "split": split,
        "points_path": points_path,
        "num_points": 5,
        "point_dim": 2,
        "has_knot_labels": False,
        "metadata": {"fixture": True},
    }


class RealWorldDataTests(unittest.TestCase):
    def test_manifest_dataset_filters_resamples_and_never_adds_knot_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points_dir = root / "curves"
            points_dir.mkdir()
            values = np.asarray(
                [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0], [4.0, 2.0]],
                dtype=np.float32,
            )
            np.save(points_dir / "train.npy", values, allow_pickle=False)
            np.save(points_dir / "test.npy", values + 2.0, allow_pickle=False)
            manifest = root / "curves.jsonl"
            write_curve_manifest(
                [
                    _record("train", "curves/train.npy", "train"),
                    _record("test", "curves/test.npy", "test"),
                ],
                manifest,
            )

            dataset = RealWorldCurveDataset(
                manifest,
                split="train",
                num_points=9,
            )
            sample = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(sample["points"].shape, (9, 2))
            self.assertFalse(sample["has_knot_labels"])
            self.assertNotIn("true_knot_vector", sample)
            self.assertNotIn("canonical_internal_knots", sample)
            self.assertEqual(dataset.load_reference_points(0).shape, (5, 2))
            torch.testing.assert_close(sample["points"].mean(dim=0), torch.zeros(2))
            self.assertAlmostEqual(float(sample["chord_params"][-1]), 1.0, places=6)

    def test_manifest_rejects_duplicate_ids_and_bad_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "duplicate.jsonl"
            with self.assertRaisesRegex(ValueError, "duplicate manifest sample_id"):
                write_curve_manifest(
                    [
                        _record("same", "a.npy", "train"),
                        _record("same", "b.npy", "val"),
                    ],
                    manifest,
                )

            manifest.write_text("{bad json}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                read_curve_manifest(manifest)

    def test_manifest_requires_explicit_knot_label_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "missing.jsonl"
            record = _record("x", "x.npy", "train")
            del record["has_knot_labels"]
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "has_knot_labels"):
                read_curve_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
