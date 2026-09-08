from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.real_world import load_manifest_points, read_curve_manifest  # noqa: E402
from spline_fitting.data.uji_pen import (  # noqa: E402
    parse_uji_pen_v2,
    prepare_uji_pen_v2,
    remove_consecutive_duplicate_points,
    select_validation_writers,
)


FIXTURE = """\
// UJI: 100 units per millimetre
// ASCII char: ;
WORD ; trn_UJI_W03-01
NUMSTROKES 2
POINTS 6 # 0 0 0 0 100 0 200 100 300 100 400 200
POINTS 3 # 20 20 20 20 20 20
// Non-ASCII char: aacute
WORD á trn_UPV_W12-02
NUMSTROKES 1
POINTS 4 # 0 0 152 0 304 152 456 304
WORD U trn_UPV_W13-01
NUMSTROKES 1
POINTS 4 # 0 0 100 0 100 100 200 100
WORD ? tst_UJI_W42-02
NUMSTROKES 1
POINTS 5 # -100 0 0 0 100 100 200 100 300 200
"""


class UJIPenDataTests(unittest.TestCase):
    def test_parser_preserves_strokes_unicode_and_writer_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "ujipenchars2.txt"
            source.write_text(FIXTURE, encoding="utf-8")
            samples = list(parse_uji_pen_v2(source))

        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0].character, ";")
        self.assertEqual(samples[0].writer_id, "trn_UJI_W03")
        self.assertEqual(len(samples[0].strokes), 2)
        self.assertEqual(samples[1].character, "á")
        self.assertEqual(samples[-1].official_partition, "tst")
        self.assertEqual(samples[-1].site, "UJI")

    def test_preparation_is_writer_disjoint_metric_and_unlabeled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "ujipenchars2.txt"
            source.write_text(FIXTURE, encoding="utf-8")
            manifest = root / "splits" / "uji.jsonl"
            summary = root / "splits" / "uji_split.json"
            result = prepare_uji_pen_v2(
                source,
                root / "processed",
                manifest,
                summary,
                validation_writer_count=1,
                split_seed=7,
                strict_official_counts=False,
            )

            records = read_curve_manifest(manifest)
            split_payload = json.loads(summary.read_text(encoding="utf-8"))
            upv_record = next(
                item for item in records if item["metadata"]["site"] == "UPV"
            )
            upv_points = load_manifest_points(manifest, upv_record).numpy()

        self.assertEqual(result.source_sample_count, 4)
        self.assertEqual(result.curve_count, 4)
        self.assertEqual(result.skipped_stroke_count, 1)
        self.assertEqual(
            {record["split"] for record in records}, {"train", "val", "test"}
        )
        self.assertTrue(split_payload["writer_disjoint"])
        self.assertTrue(split_payload["official_test_partition_preserved"])
        self.assertFalse(split_payload["has_knot_labels"])
        self.assertTrue(all(not record["has_knot_labels"] for record in records))
        self.assertEqual(
            {record["group_id"] for record in records if record["split"] == "test"},
            {"tst_UJI_W42"},
        )
        np.testing.assert_allclose(
            upv_points,
            np.asarray([[0, 0], [1, 0], [2, 1], [3, 2]], dtype=np.float32),
        )

    def test_duplicate_removal_only_removes_consecutive_runs(self) -> None:
        points = np.asarray([[0, 0], [0, 0], [1, 1], [0, 0]], dtype=np.int32)
        result = remove_consecutive_duplicate_points(points)
        np.testing.assert_array_equal(result, np.asarray([[0, 0], [1, 1], [0, 0]]))

    def test_validation_writer_selection_is_stable(self) -> None:
        writers = {"trn_UJI_W01", "trn_UJI_W02", "trn_UPV_W12"}
        first = select_validation_writers(writers, count=1, seed=123)
        second = select_validation_writers(writers, count=1, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)

    def test_malformed_declared_point_count_is_rejected(self) -> None:
        malformed = """\
WORD A trn_UJI_W01-01
NUMSTROKES 1
POINTS 4 # 0 0 1 1 2 2
"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.txt"
            source.write_text(malformed, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "declares 4 points"):
                list(parse_uji_pen_v2(source))


if __name__ == "__main__":
    unittest.main()
