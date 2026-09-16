"""Fixed high-complexity strata for the Proposal/Joint v16 experiment."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.v16_mixed import MixedTrainingCurves, ValidationCurves
from spline_fitting.training.v16_feasible_teacher import _fixed_dataset_points


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def synthetic_config():
    return dict(
        num_points=24,
        point_dim=2,
        min_control_points=8,  # source K = 4..8
        max_control_points=12,
        noise_std=0.001,
        normalize=True,
        canonical_knot_tolerance=0.0,
    )


def test_fixed_joint_high_k_rows_keep_exact_allocation_and_cache_identity(
    synthetic_config,
):
    options = dict(
        size=12,
        seed=73,
        real_fraction=0,
        resample=False,
        synthetic_high_k_fraction=0.5,
        synthetic_high_k_min_knots=7,
    )
    first = MixedTrainingCurves(synthetic_config, epoch=0, **options)
    resumed = MixedTrainingCurves(synthetic_config, epoch=59, **options)
    assert len(first.synthetic_high_k_indices) == 6
    assert first.synthetic_high_k_indices == resumed.synthetic_high_k_indices
    assert first.synthetic_high is not None
    assert first.synthetic_low is not None
    assert first.synthetic_high.min_control_points == 11
    assert first.synthetic_low.max_control_points == 10
    first.synthetic_high.return_ground_truth = True
    first.synthetic_low.return_ground_truth = True
    for index in range(len(first)):
        source = first.synthetic_high if index in first.synthetic_high_k_indices else first.synthetic_low
        source_k = source[index]["source_internal_knot_count"]
        assert (7 <= source_k <= 8) if index in first.synthetic_high_k_indices else (4 <= source_k <= 6)
        torch.testing.assert_close(first[index]["points"], resumed[index]["points"], rtol=0, atol=0)

    _, first_hash, _ = _fixed_dataset_points(first)
    _, resumed_hash, _ = _fixed_dataset_points(resumed)
    assert first_hash == resumed_hash
    different = MixedTrainingCurves(
        synthetic_config,
        epoch=0,
        **{**options, "synthetic_high_k_fraction": 0.0},
    )
    _, different_hash, _ = _fixed_dataset_points(different)
    assert different_hash != first_hash


def test_validation_has_separate_reproducible_high_k_and_boundary_strata(
    synthetic_config,
):
    options = dict(
        size=12,
        seed=91,
        synthetic_boundary_samples=2,
        synthetic_high_k_samples=4,
        synthetic_high_k_min_knots=7,
    )
    first = ValidationCurves(synthetic_config, **options)
    repeated = ValidationCurves(synthetic_config, **options)
    assert len(first) == 12
    assert len(first.entries) == 12
    assert sum(dataset is first.synthetic_boundary for _, dataset, _ in first.entries) == 2
    assert sum(dataset is first.synthetic_high_k for _, dataset, _ in first.entries) == 4
    assert sum(dataset is first.synthetic for _, dataset, _ in first.entries) == 6
    assert first.synthetic_high_k.min_control_points == 11
    first.synthetic.return_ground_truth = True
    first.synthetic_high_k.return_ground_truth = True
    first.synthetic_boundary.return_ground_truth = True
    for index, (_, dataset, sample_index) in enumerate(first.entries):
        source_k = dataset[sample_index]["source_internal_knot_count"]
        if dataset is first.synthetic_boundary:
            assert source_k == 8
        elif dataset is first.synthetic_high_k:
            assert 7 <= source_k <= 8
        torch.testing.assert_close(first[index]["points"], repeated[index]["points"], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"synthetic_high_k_samples": -1}, "non-negative"),
        ({"synthetic_high_k_samples": 13}, "exceed size"),
        ({"synthetic_high_k_samples": 1}, "must be an integer"),
        ({"synthetic_high_k_samples": 1, "synthetic_high_k_min_knots": 3}, "inside"),
    ],
)
def test_validation_rejects_invalid_high_k_strata(synthetic_config, extra, message):
    with pytest.raises(ValueError, match=message):
        ValidationCurves(synthetic_config, size=12, **extra)
