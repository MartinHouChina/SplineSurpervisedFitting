from __future__ import annotations

# Tests import the local ``src`` tree without requiring package installation.
# ruff: noqa: E402

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.industrial_offsets import (
    INDUSTRIAL_OFFSET_DATASET_NAME,
    INDUSTRIAL_PROFILE_FAMILIES,
    OffsetCurveError,
    closed_polyline_self_intersects,
    generate_industrial_profile,
    offset_closed_polyline,
    prepare_industrial_offset_curves,
)
from spline_fitting.data.real_world import RealWorldCurveDataset, read_curve_manifest


def test_all_profile_families_are_deterministic_finite_simple_contours() -> None:
    for family in INDUSTRIAL_PROFILE_FAMILIES:
        first, first_parameters = generate_industrial_profile(
            family,
            2,
            source_points=96,
            seed=17,
        )
        second, second_parameters = generate_industrial_profile(
            family,
            2,
            source_points=96,
            seed=17,
        )
        np.testing.assert_array_equal(first, second)
        assert first_parameters == second_parameters
        assert first.shape[1] == 2
        assert len(first) >= 65
        assert np.isfinite(first).all()
        np.testing.assert_allclose(first[0], first[-1])
        assert not closed_polyline_self_intersects(first)


def test_sharp_square_uses_valid_outward_miter_and_rejects_collapse() -> None:
    square = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    outward = offset_closed_polyline(square, 0.25)
    np.testing.assert_allclose(outward.min(axis=0), [-0.25, -0.25])
    np.testing.assert_allclose(outward.max(axis=0), [1.25, 1.25])
    assert not closed_polyline_self_intersects(outward)

    with pytest.raises(OffsetCurveError, match="medial axis|collapsed|reversed") as captured:
        offset_closed_polyline(square, -0.6)
    assert captured.value.reason == "topology_change"


def test_self_intersection_detector_rejects_bow_tie_source() -> None:
    bow_tie = np.asarray(
        [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    assert closed_polyline_self_intersects(bow_tie)
    with pytest.raises(OffsetCurveError) as captured:
        offset_closed_polyline(bow_tie, 0.1)
    assert captured.value.reason in {"degenerate", "self_intersection"}


def test_preparation_writes_group_safe_unlabelled_manifest(tmp_path: Path) -> None:
    output = tmp_path / "industrial"
    result = prepare_industrial_offset_curves(
        output,
        families=("elliptic_bore", "keyway_bore"),
        variants_per_family=3,
        offset_fractions=(-0.04, 0.02, 0.06),
        source_points=96,
        reference_points=64,
        generation_seed=9,
        split_seed=11,
        train_fraction=0.5,
        val_fraction=0.25,
    )
    records = read_curve_manifest(result.manifest_path)
    dataset = RealWorldCurveDataset(result.manifest_path, num_points=32)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert result.sample_count == len(records) == len(dataset)
    assert result.group_count == 6
    assert sum(result.split_counts.values()) == len(records)
    assert all(
        record["source_dataset"] == INDUSTRIAL_OFFSET_DATASET_NAME for record in records
    )
    assert all(record["has_knot_labels"] is False for record in records)
    assert all(record["metadata"]["offset_distance_mm"] != 0.0 for record in records)
    assert metadata["dataset_class"].startswith("CAD-driven semi-synthetic")
    assert metadata["training_role"].startswith("external validation/test only")

    split_by_group: dict[str, str] = {}
    for record in records:
        previous = split_by_group.setdefault(record["group_id"], record["split"])
        assert previous == record["split"]
    sample = dataset[0]
    assert tuple(sample["points"].shape) == (32, 2)
    assert bool(sample["has_knot_labels"]) is False


def test_preparation_refuses_silent_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "industrial"
    arguments = dict(
        families=("elliptic_bore",),
        variants_per_family=1,
        offset_fractions=(0.02,),
        source_points=64,
        reference_points=32,
    )
    prepare_industrial_offset_curves(output, **arguments)
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_industrial_offset_curves(output, **arguments)
