from __future__ import annotations

# Tests import the local ``src`` tree without requiring package installation.
# ruff: noqa: E402

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.geospatial_curves import (
    LineFeature,
    canonical_geometry_hash,
    deterministic_group_split,
    fetch_usgs_contours_region,
    line_features_from_json,
    natural_earth_geojson_url,
    prepare_geospatial_curves,
    project_lonlat_local_azimuthal,
    spatial_tile_group,
)
from spline_fitting.data.real_world import RealWorldCurveDataset, read_curve_manifest


class GeospatialCurveTests(unittest.TestCase):
    def test_geojson_and_arcgis_line_parts_are_extracted(self) -> None:
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "coast-a",
                    "properties": {"name": "A"},
                    "geometry": {
                        "type": "MultiLineString",
                        "coordinates": [
                            [[-1.0, 50.0], [-0.9, 50.1]],
                            [[-0.8, 50.2], [-0.7, 50.3]],
                        ],
                    },
                }
            ],
        }
        parts = line_features_from_json(geojson)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].feature_id, "coast-a")
        self.assertEqual(parts[1].part_index, 1)
        self.assertEqual(parts[0].coordinates.shape, (2, 2))

        arcgis = {
            "features": [
                {
                    "attributes": {"objectid": 17, "contourelevation": 2200.0},
                    "geometry": {
                        "paths": [[[-105.3, 39.7, 2200.0], [-105.29, 39.71, 2200.0]]]
                    },
                }
            ]
        }
        contour = line_features_from_json(arcgis)
        self.assertEqual(len(contour), 1)
        self.assertEqual(contour[0].feature_id, "17")
        self.assertEqual(contour[0].coordinates.shape, (2, 2))

    def test_projection_handles_an_antimeridian_crossing_locally(self) -> None:
        lonlat = np.asarray(
            [[179.90, 10.0], [179.98, 10.02], [-179.98, 10.04], [-179.90, 10.06]],
            dtype=np.float64,
        )
        points, metadata = project_lonlat_local_azimuthal(lonlat)
        length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        self.assertTrue(np.isfinite(points).all())
        self.assertLess(float(length), 50_000.0)
        self.assertEqual(metadata["unit"], "meter")

    def test_geometry_hash_is_direction_invariant(self) -> None:
        points = np.asarray([[0.0, 0.0], [0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(
            canonical_geometry_hash(points),
            canonical_geometry_hash(points[::-1]),
        )

    def test_group_split_is_deterministic_and_group_safe(self) -> None:
        first = np.asarray([[-105.40, 39.91], [-105.35, 39.96]])
        second = np.asarray([[-105.33, 39.97], [-105.31, 39.99]])
        first_group = spatial_tile_group(
            first,
            source_dataset="usgs",
            tile_degrees=1.0,
        )
        second_group = spatial_tile_group(
            second,
            source_dataset="usgs",
            tile_degrees=1.0,
        )
        self.assertEqual(first_group, second_group)
        self.assertEqual(
            deterministic_group_split(first_group, seed=42),
            deterministic_group_split(second_group, seed=42),
        )

    def test_preparation_writes_loadable_unlabeled_manifest(self) -> None:
        longitude = np.linspace(-105.45, -105.25, 12)
        latitude = 39.9 + 0.03 * np.sin(np.linspace(0.0, 2.0, 12))
        coordinates = np.stack((longitude, latitude), axis=1)
        features = [
            LineFeature("contour-a", 0, coordinates, {"elevation": 2200}),
            LineFeature("duplicate", 0, coordinates[::-1], {"elevation": 2200}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepared"
            result = prepare_geospatial_curves(
                features,
                output,
                source_dataset="usgs_test",
                source_version="snapshot",
                source_url="local-test",
                source_sha256="0" * 64,
                reference_points=32,
                min_source_vertices=4,
                max_source_vertices=8,
                min_length_m=0.0,
                tile_degrees=5.0,
            )
            records = read_curve_manifest(result.manifest_path)
            dataset = RealWorldCurveDataset(result.manifest_path, num_points=16)
            sample = dataset[0]

            self.assertEqual(result.sample_count, 2)
            self.assertEqual(result.duplicate_count, 1)
            self.assertEqual(len(records), 2)
            self.assertTrue(all(not record["has_knot_labels"] for record in records))
            self.assertEqual(tuple(sample["points"].shape), (16, 2))
            self.assertAlmostEqual(float(sample["points"].mean()), 0.0, places=5)
            self.assertTrue((output / "dataset_metadata.json").is_file())

    def test_usgs_fetch_is_sorted_cached_and_hash_checked(self) -> None:
        calls: list[tuple[str, dict[str, str]]] = []

        def requester(
            url: str,
            parameters: Mapping[str, str],
            timeout: float,
        ) -> Mapping[str, Any]:
            del timeout
            calls.append((url, dict(parameters)))
            if not url.endswith("/query"):
                return {"objectIdField": "objectid", "name": "contours"}
            if parameters.get("returnIdsOnly") == "true":
                return {"objectIds": [3, 1, 2], "objectIdFieldName": "objectid"}
            identifiers = [int(value) for value in parameters["objectIds"].split(",")]
            return {
                "features": [
                    {
                        "attributes": {"objectid": identifier},
                        "geometry": {
                            "paths": [
                                [
                                    [-105.0, 39.0],
                                    [-104.99 + identifier * 1e-5, 39.01],
                                ]
                            ]
                        },
                    }
                    for identifier in identifiers[::-1]
                ]
            }

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "region.json"
            path, digest = fetch_usgs_contours_region(
                bbox=(-105.1, 38.9, -104.9, 39.1),
                destination=destination,
                region_id="test-region",
                max_features=2,
                request_page_size=1,
                request_json=requester,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            initial_call_count = len(calls)
            cached_path, cached_digest = fetch_usgs_contours_region(
                bbox=(-105.1, 38.9, -104.9, 39.1),
                destination=destination,
                region_id="test-region",
                max_features=2,
                request_json=lambda *_: self.fail("cache should avoid network access"),
            )
            with self.assertRaisesRegex(ValueError, "different settings"):
                fetch_usgs_contours_region(
                    bbox=(-105.2, 38.9, -104.9, 39.1),
                    destination=destination,
                    region_id="test-region",
                    max_features=2,
                    request_json=lambda *_: self.fail(
                        "a mismatched cache must fail before network access"
                    ),
                )

        self.assertEqual(payload["selected_feature_count"], 2)
        object_ids = [item["attributes"]["objectid"] for item in payload["features"]]
        self.assertEqual(object_ids, sorted(object_ids))
        self.assertGreaterEqual(initial_call_count, 4)
        self.assertEqual(cached_path, path)
        self.assertEqual(cached_digest, digest)

    def test_natural_earth_default_is_version_pinned(self) -> None:
        url = natural_earth_geojson_url(
            resolution="10m",
            layer="coastline",
            version="v5.1.2",
        )
        self.assertIn("/v5.1.2/geojson/ne_10m_coastline.geojson", url)


if __name__ == "__main__":
    unittest.main()
