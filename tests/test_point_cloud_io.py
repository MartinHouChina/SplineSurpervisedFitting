from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.point_cloud_io import (
    load_ordered_point_cloud,
    normalize_ordered_point_cloud,
    resample_ordered_point_cloud,
)


class PointCloudIOTests(unittest.TestCase):
    def test_json_point_cloud_loads_and_uses_training_normalization(self) -> None:
        values = [[0.0, 0.0], [1.0, 0.2], [2.0, 0.8], [3.0, 1.0]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curve.json"
            path.write_text(json.dumps({"points": values}), encoding="utf-8")
            points = load_ordered_point_cloud(path, point_dim=2)

        normalized = normalize_ordered_point_cloud(points)
        self.assertEqual(points.shape, (4, 2))
        torch.testing.assert_close(normalized["points"].mean(dim=0), torch.zeros(2))
        self.assertAlmostEqual(
            float(normalized["points"].norm(dim=-1).max()), 1.0, places=6
        )
        self.assertEqual(float(normalized["chord_params"][0]), 0.0)
        self.assertAlmostEqual(
            float(normalized["chord_params"][-1]), 1.0, places=6
        )
        self.assertTrue(
            torch.all(normalized["chord_params"][1:] > normalized["chord_params"][:-1])
        )

    def test_dimension_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curve.json"
            path.write_text(
                json.dumps([[0, 0], [1, 0], [2, 0], [3, 0]]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "expects point dimension"):
                load_ordered_point_cloud(path, point_dim=3)

    def test_resampling_preserves_endpoints_and_requested_length(self) -> None:
        points = torch.tensor(
            [[0.0, 0.0], [0.5, 0.0], [0.5, 0.0], [2.0, 1.0]]
        )
        resampled = resample_ordered_point_cloud(points, 9)
        self.assertEqual(resampled.shape, (9, 2))
        torch.testing.assert_close(resampled[0], points[0])
        torch.testing.assert_close(resampled[-1], points[-1])


if __name__ == "__main__":
    unittest.main()
