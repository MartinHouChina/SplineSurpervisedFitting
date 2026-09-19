from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.v16_mixed import (  # noqa: E402
    MixedTrainingCurves,
    ValidationCurves,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def certified_config(point_dim: int = 2) -> dict:
    return dict(
        num_points=64,
        point_dim=point_dim,
        min_control_points=8,
        max_control_points=28,
        noise_std=0.001,
        normalize=True,
        canonical_knot_tolerance=5e-5**0.5,
        certified_minimal_source=True,
        minimality_margin=0.2,
        minimality_max_attempts=16,
        minimality_audit_points=64,
        oscillation_amplitude=0.3,
    )


def test_zero_augmentation_fractions_preserve_the_legacy_sample_exactly():
    config = dict(
        num_points=24,
        point_dim=2,
        min_control_points=8,
        max_control_points=12,
        noise_std=0.001,
        normalize=True,
        canonical_knot_tolerance=0.0,
    )
    legacy = MixedTrainingCurves(config, size=4, seed=319, real_fraction=0.0)
    explicit_zero = MixedTrainingCurves(
        config,
        size=4,
        seed=319,
        real_fraction=0.0,
        synthetic_simple_fraction=0.0,
        synthetic_shape_fraction=0.0,
    )

    for index in range(4):
        expected = legacy[index]
        actual = explicit_zero[index]
        assert set(actual) == set(expected)
        assert "synthetic_family" not in actual
        for key, expected_value in expected.items():
            actual_value = actual[key]
            if torch.is_tensor(expected_value):
                torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
            else:
                assert actual_value == expected_value


def test_validation_only_sources_do_not_perturb_enhanced_training_sequence():
    without_manifests = MixedTrainingCurves(
        certified_config(),
        size=16,
        seed=42,
        real_fraction=0.0,
        synthetic_simple_fraction=0.35,
        synthetic_shape_fraction=0.25,
    )
    with_manifests = MixedTrainingCurves(
        certified_config(),
        real_sources=(("HeldOut", object(), object()),),
        size=16,
        seed=42,
        real_fraction=0.0,
        synthetic_simple_fraction=0.35,
        synthetic_shape_fraction=0.25,
    )

    for index in range(16):
        expected = without_manifests[index]
        actual = with_manifests[index]
        assert set(actual) == set(expected)
        for key, expected_value in expected.items():
            actual_value = actual[key]
            if torch.is_tensor(expected_value):
                torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
            else:
                assert actual_value == expected_value


@pytest.mark.parametrize(
    ("simple", "shape"),
    [(-0.1, 0.0), (0.0, 1.1), (float("nan"), 0.0), (0.6, 0.5)],
)
def test_augmentation_fraction_validation(simple, shape):
    with pytest.raises(ValueError, match="fraction"):
        MixedTrainingCurves(
            certified_config(),
            size=2,
            synthetic_simple_fraction=simple,
            synthetic_shape_fraction=shape,
        )


def test_simple_family_is_certified_low_k_with_full_width_targets():
    first = MixedTrainingCurves(
        certified_config(),
        size=8,
        seed=9182,
        real_fraction=0.0,
        synthetic_simple_fraction=1.0,
    )
    repeated = MixedTrainingCurves(
        certified_config(),
        size=8,
        seed=9182,
        real_fraction=0.0,
        synthetic_simple_fraction=1.0,
    )

    for index in range(8):
        sample = first[index]
        assert sample["source"] == "Synthetic"
        assert sample["synthetic_family"] == "simple"
        assert bool(sample["target_internal_knot_count_valid"])
        assert bool(sample["target_geometry_valid"])
        assert 4 <= int(sample["target_internal_knot_count"]) <= 8
        assert sample["target_params"].shape == (64,)
        assert sample["target_internal_knots"].shape == (24,)
        assert sample["target_internal_knot_mask"].shape == (24,)
        assert int(sample["target_internal_knot_mask"].sum()) == int(
            sample["target_internal_knot_count"]
        )
        torch.testing.assert_close(
            sample["points"], repeated[index]["points"], rtol=0, atol=0
        )


def test_simple_family_directly_generates_a_narrow_low_k_overlap():
    config = certified_config()
    config["min_control_points"] = 12
    dataset = MixedTrainingCurves(
        config,
        size=191,
        seed=9182,
        real_fraction=0.0,
        synthetic_simple_fraction=1.0,
    )

    assert dataset.simple_synthetic.dataset.min_control_points == 12
    assert dataset.simple_synthetic.dataset.max_control_points == 12
    for index in (100, 121, 175, 190):
        sample = dataset[index]
        assert sample["synthetic_family"] == "simple"
        assert int(sample["target_internal_knot_count"]) == 8
        assert sample["target_internal_knots"].shape == (24,)
        assert sample["target_internal_knot_mask"].shape == (24,)
        assert int(sample["target_internal_knot_mask"].sum()) == 8


@pytest.mark.parametrize("point_dim", [2, 3])
def test_shape_families_are_deterministic_normalized_and_never_claim_labels(
    point_dim,
):
    dataset = MixedTrainingCurves(
        certified_config(point_dim),
        size=6,
        seed=771,
        real_fraction=0.0,
        synthetic_shape_fraction=1.0,
    )
    repeated = MixedTrainingCurves(
        certified_config(point_dim),
        size=6,
        seed=771,
        real_fraction=0.0,
        synthetic_shape_fraction=1.0,
    )
    expected_families = [
        "shape_industrial",
        "shape_terrain",
        "shape_handwriting",
    ] * 2

    for index, expected_family in enumerate(expected_families):
        sample = dataset[index]
        assert sample["source"] == "Synthetic"
        assert sample["synthetic_family"] == expected_family
        assert sample["points"].shape == (64, point_dim)
        assert torch.isfinite(sample["points"]).all()
        torch.testing.assert_close(
            sample["points"].mean(dim=0),
            torch.zeros(point_dim),
            atol=2e-6,
            rtol=0,
        )
        assert float(sample["points"].norm(dim=-1).amax()) == pytest.approx(
            1.0, abs=2e-6
        )
        assert not torch.equal(sample["points"][0], sample["points"][-1])
        assert not bool(sample["target_internal_knot_count_valid"])
        assert not bool(sample["target_geometry_valid"])
        assert int(sample["target_internal_knot_count"]) == 0
        assert torch.count_nonzero(sample["target_params"]) == 0
        assert torch.count_nonzero(sample["target_internal_knots"]) == 0
        assert not sample["target_internal_knot_mask"].any()
        torch.testing.assert_close(
            sample["points"], repeated[index]["points"], rtol=0, atol=0
        )

    batch = next(iter(DataLoader(dataset, batch_size=6, shuffle=False)))
    assert batch["synthetic_family"] == expected_families
    assert not batch["target_internal_knot_count_valid"].any()
    assert not batch["target_geometry_valid"].any()


def test_enhanced_training_adds_family_field_without_changing_validation():
    training = MixedTrainingCurves(
        certified_config(),
        size=12,
        seed=412,
        real_fraction=0.0,
        synthetic_simple_fraction=0.35,
        synthetic_shape_fraction=0.25,
    )
    batch = next(iter(DataLoader(training, batch_size=12, shuffle=False)))
    assert len(batch["synthetic_family"]) == 12
    assert set(batch["synthetic_family"]).issubset(
        {
            "historical",
            "simple",
            "shape_industrial",
            "shape_terrain",
            "shape_handwriting",
        }
    )

    validation = ValidationCurves(certified_config(), size=2, seed=412)
    assert "synthetic_family" not in validation[0]


def test_simple_family_rejects_uncertified_configuration():
    config = certified_config()
    config["certified_minimal_source"] = False
    with pytest.raises(ValueError, match="certified_minimal_source"):
        MixedTrainingCurves(
            config,
            size=2,
            synthetic_simple_fraction=1.0,
        )
