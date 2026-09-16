from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spline_fitting.training.v16_feasible_teacher import (
    _devices_equivalent,
    _ordered_anchor_slots,
    build_or_load_v16_feasible_teacher_cache,
    _local_swap_counterfactual_labels,
)
from spline_fitting.data.synthetic import bspline_basis_matrix
from spline_fitting.evaluation.knot_diagnostics import build_open_knot_vector


@pytest.fixture(autouse=True)
def single_threaded():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class _FixedSynthetic(Dataset):
    def __init__(self) -> None:
        t = torch.linspace(0.0, 1.0, 28)
        self.points = [
            torch.stack((t, 0.1 * (i + 1) * torch.sin(6.0 * t)), -1)
            for i in range(2)
        ]

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index: int) -> dict:
        return {"points": self.points[index], "source": "Synthetic"}


class _FixedProposal(nn.Module):
    max_internal_knots = 4
    degree = 3

    def __init__(self) -> None:
        super().__init__()
        self.candidate_head = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.candidate_head.weight, 0.0)

    def encode_candidates(self, points: torch.Tensor, mse_tolerance=None) -> dict:
        batch, count, _ = points.shape
        parameters = torch.linspace(
            0.0, 1.0, count, dtype=points.dtype, device=points.device
        ).expand(batch, -1)
        knots = torch.tensor(
            [0.2, 0.4, 0.6, 0.8], dtype=points.dtype, device=points.device
        ).expand(batch, -1)
        shift = self.candidate_head.weight.flatten()[0] * 0.01
        return {
            "proposal_params": parameters,
            "proposal_internal_knots": knots + shift,
        }


def _cache(model, dataset, path):
    return build_or_load_v16_feasible_teacher_cache(
        model,
        dataset,
        path,
        mse_tolerance=1e-4,
        device="cpu",
        batch_size=1,
    )


def test_feasible_teacher_build_load_and_indexed_gather(tmp_path):
    model = _FixedProposal()
    dataset = _FixedSynthetic()
    path = tmp_path / "teacher.pt"
    created = _cache(model, dataset, path)
    assert not created.loaded
    assert created.sample_count == 2
    assert created.candidate_count == 4
    assert created.batch.config.error_tolerance == pytest.approx(0.01)
    assert created.batch.config.smoothness_weight == 0.0
    assert torch.equal(
        created.labels["teacher_count"],
        created.labels["teacher_retained_mask"].sum(-1).long(),
    )
    assert torch.allclose(
        created.labels["teacher_fit_mse"],
        created.labels["teacher_fit_rms"].square(),
    )
    assert torch.equal(
        created.labels["teacher_threshold_satisfied"],
        created.labels["teacher_fit_mse"] <= 1e-4,
    )
    gathered = created.labels_for_indices(torch.tensor([1, 0]))
    assert torch.equal(
        gathered["teacher_retained_mask"],
        created.labels["teacher_retained_mask"].flip(0),
    )
    loaded = _cache(model, dataset, path)
    assert loaded.loaded
    loaded.assert_proposal_unchanged(model)
    assert torch.equal(
        loaded.labels["teacher_retained_mask"],
        created.labels["teacher_retained_mask"],
    )


def test_feasible_teacher_rejects_changed_proposal_and_points(tmp_path):
    model = _FixedProposal()
    dataset = _FixedSynthetic()
    path = tmp_path / "teacher.pt"
    cache = _cache(model, dataset, path)
    with torch.no_grad():
        model.candidate_head.weight.fill_(1.0)
    with pytest.raises(RuntimeError, match="frozen Proposal changed"):
        cache.assert_proposal_unchanged(model)
    with pytest.raises(ValueError, match="does not match frozen Proposal"):
        _cache(model, dataset, path)
    with torch.no_grad():
        model.candidate_head.weight.zero_()
    dataset.points[0][5, 1] += 0.01
    with pytest.raises(ValueError, match="does not match frozen Proposal"):
        _cache(model, dataset, path)


def test_feasible_teacher_is_synthetic_only_and_index_checked(tmp_path):
    model = _FixedProposal()
    dataset = _FixedSynthetic()
    path = tmp_path / "teacher.pt"
    cache = _cache(model, dataset, path)
    with pytest.raises(IndexError):
        cache.labels_for_indices([2])
    with pytest.raises(ValueError, match="one-dimensional"):
        cache.labels_for_indices(torch.tensor([[0]]))

    class _RealRow(_FixedSynthetic):
        def __getitem__(self, index: int) -> dict:
            row = super().__getitem__(index)
            row["source"] = "natural_earth"
            return row

    with pytest.raises(ValueError, match="synthetic-only"):
        _cache(model, _RealRow(), tmp_path / "real.pt")


class _CertifiedSynthetic(Dataset):
    def __init__(self) -> None:
        self.params = torch.linspace(0.0, 1.0, 28)
        self.true_knots = torch.tensor([0.4, 0.7])
        basis = bspline_basis_matrix(
            self.params,
            build_open_knot_vector(self.true_knots, degree=3),
            degree=3,
            num_control_points=6,
        )
        controls = torch.tensor([
            [0.0, 0.0], [0.1, 0.3], [0.3, -0.4],
            [0.7, 0.5], [0.9, -0.2], [1.0, 0.0],
        ])
        self.points = basis @ controls

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        if index:
            raise IndexError(index)
        return {
            "points": self.points,
            "source": "Synthetic",
            "target_geometry_valid": torch.tensor(True),
            "target_params": self.params,
            "target_internal_knots": torch.tensor([0.4, 0.7, 0.0, 0.0, 0.0]),
            "target_internal_knot_mask": torch.tensor([True, True, False, False, False]),
        }


class _CertifiedProposal(_FixedProposal):
    max_internal_knots = 5

    def encode_candidates(self, points: torch.Tensor, mse_tolerance=None) -> dict:
        batch, count, _ = points.shape
        parameters = torch.linspace(
            0.0, 1.0, count, dtype=points.dtype, device=points.device,
        ).expand(batch, -1)
        knots = torch.tensor(
            [0.2, 0.4, 0.6, 0.7, 0.8], dtype=points.dtype, device=points.device,
        ).expand(batch, -1)
        shift = self.candidate_head.weight.flatten()[0] * 0.01
        return {
            "proposal_params": parameters,
            "proposal_internal_knots": knots + shift,
        }


def test_ordered_anchor_slots_do_not_reuse_one_candidate():
    slots = _ordered_anchor_slots(
        torch.tensor([0.2, 0.4, 0.6, 0.8]),
        torch.tensor([0.39, 0.41]),
        max_distance=0.21,
    )
    assert slots is not None
    assert len(slots) == 2
    assert slots == sorted(set(slots))
    assert _ordered_anchor_slots(
        torch.tensor([0.2, 0.4, 0.6, 0.8]),
        torch.tensor([0.39, 0.41]),
        max_distance=0.02,
    ) is None


def test_anchor_first_cache_keeps_exact_source_slots_and_rejects_old_strategy(tmp_path):
    model = _CertifiedProposal()
    dataset = _CertifiedSynthetic()
    path = tmp_path / "anchor.pt"
    options = dict(
        mse_tolerance=1e-8,
        device="cpu",
        batch_size=1,
        teacher_strategy="synthetic_anchor_first",
        anchor_match_tolerance=0.02,
    )
    anchor = build_or_load_v16_feasible_teacher_cache(model, dataset, path, **options)
    assert anchor.teacher_strategy == "synthetic_anchor_first"
    assert not anchor.loaded
    assert anchor.batch.teacher_count.tolist() == [2]
    assert anchor.batch.teacher_retained_mask.tolist() == [[False, True, False, True, False]]
    assert anchor.feasible_fraction == 1.0
    assert float(anchor.batch.teacher_fit_mse[0]) <= 1e-8
    assert torch.equal(
        anchor.batch.teacher_deletion_order,
        torch.full((1, 5), -1, dtype=torch.long),
    )
    assert build_or_load_v16_feasible_teacher_cache(
        model, dataset, path, **options,
    ).loaded
    with pytest.raises(ValueError, match="does not match frozen Proposal"):
        build_or_load_v16_feasible_teacher_cache(
            model, dataset, path,
            mse_tolerance=1e-8, device="cpu", batch_size=1,
            teacher_strategy="full_greedy",
        )
    # A changed certified label must not silently reuse the old anchor cache.
    class _Changed(_CertifiedSynthetic):
        def __getitem__(self, index):
            result = super().__getitem__(index)
            result["target_internal_knots"][0] = 0.41
            return result
    with pytest.raises(ValueError, match="does not match frozen Proposal"):
        build_or_load_v16_feasible_teacher_cache(model, _Changed(), path, **options)


def test_anchor_first_rejects_uncertified_rows(tmp_path):
    with pytest.raises(ValueError, match="requires certified synthetic"):
        build_or_load_v16_feasible_teacher_cache(
            _FixedProposal(), _FixedSynthetic(), tmp_path / "uncertified.pt",
            mse_tolerance=1e-4, device="cpu",
            teacher_strategy="synthetic_anchor_first",
        )


def test_anchor_first_unmatched_anchor_falls_back_or_fails_explicitly(tmp_path):
    class _MissingAnchorProposal(_CertifiedProposal):
        def encode_candidates(self, points, mse_tolerance=None):
            output = super().encode_candidates(points, mse_tolerance)
            output["proposal_internal_knots"] = torch.tensor(
                [0.1, 0.2, 0.3, 0.5, 0.9],
                device=points.device, dtype=points.dtype,
            ).expand(points.shape[0], -1)
            return output

    model = _MissingAnchorProposal()
    dataset = _CertifiedSynthetic()
    options = dict(
        mse_tolerance=1e-4, device="cpu", batch_size=1,
        teacher_strategy="synthetic_anchor_first",
        anchor_match_tolerance=0.01,
    )
    fallback = build_or_load_v16_feasible_teacher_cache(
        model, dataset, tmp_path / "fallback.pt",
        anchor_fallback_to_greedy=True, **options,
    )
    assert fallback.batch.teacher_count.item() >= 0
    assert bool(fallback.batch.teacher_threshold_satisfied.item()) == (
        float(fallback.batch.teacher_fit_mse.item()) <= 1e-4
    )
    with pytest.raises(ValueError, match="cannot uniquely match"):
        build_or_load_v16_feasible_teacher_cache(
            model, dataset, tmp_path / "no_fallback.pt",
            anchor_fallback_to_greedy=False, **options,
        )


def test_anchor_first_adds_back_until_threshold_or_all_candidates(tmp_path):
    model = _CertifiedProposal()
    with torch.no_grad():
        model.candidate_head.weight.fill_(1.5)
    # The source anchors shift by 0.015 in the proposal frame. A stringent
    # threshold forces a fitted add-back rather than accepting source count.
    cache = build_or_load_v16_feasible_teacher_cache(
        model, _CertifiedSynthetic(), tmp_path / "addback.pt",
        mse_tolerance=1e-12, device="cpu", batch_size=1,
        teacher_strategy="synthetic_anchor_first",
        anchor_match_tolerance=0.02,
        anchor_fallback_to_greedy=False,
    )
    assert cache.batch.teacher_count.item() > 2
    assert cache.batch.teacher_count.item() <= 5
    assert bool(cache.batch.teacher_threshold_satisfied.item()) == (
        float(cache.batch.teacher_fit_rms.item()) <= 1e-6
    )


def test_counterfactual_swap_softens_feasible_alternative_and_bounds_work(monkeypatch):
    calls = []

    def fake_refit(parameters, points, knots, **kwargs):
        calls.append(knots.clone())
        mse = 5e-5 if float(knots[0]) < 0.35 else 2e-4
        return SimpleNamespace(fit_mse=torch.tensor(mse, dtype=knots.dtype))

    monkeypatch.setattr(
        "spline_fitting.training.v16_feasible_teacher.refit_bspline_control_points",
        fake_refit,
    )
    teacher = SimpleNamespace(
        teacher_retained_mask=torch.tensor([[False, True, False, True, False]]),
        teacher_threshold_satisfied=torch.tensor([True]),
        config=SimpleNamespace(
            error_tolerance=0.01, degree=3, smoothness_weight=0.0,
            control_ridge=0.0, rcond=None,
        ),
    )
    candidates = torch.tensor([[0.3, 0.4, 0.6, 0.7, 0.8]], dtype=torch.float64)
    labels = _local_swap_counterfactual_labels(
        torch.linspace(0, 1, 12, dtype=torch.float64).unsqueeze(0),
        torch.zeros(1, 12, 2, dtype=torch.float64),
        candidates, teacher, max_probes=2,
    )
    assert len(calls) == 2
    assert labels["teacher_counterfactual_probe_count"].tolist() == [2]
    assert labels["teacher_counterfactual_swap_feasible"].tolist() == [
        [False, True, False, False, False],
    ]
    weights = labels["teacher_counterfactual_slot_weight"][0]
    assert weights[1].item() == 0.0
    assert weights[0].item() == 0.0  # feasible pair gets OR, no hard BCE
    assert weights[3].item() == 1.5  # measured local swap breaches tolerance
    assert weights[2].item() == 1.0


def test_counterfactual_cache_is_opt_in_versioned_and_probe_fingerprinted(tmp_path):
    model = _CertifiedProposal()
    dataset = _CertifiedSynthetic()
    path = tmp_path / "counterfactual.pt"
    options = dict(
        mse_tolerance=1e-8, device="cpu", batch_size=1,
        teacher_strategy="synthetic_anchor_counterfactual",
        anchor_match_tolerance=0.02, counterfactual_max_probes=2,
    )
    cache = build_or_load_v16_feasible_teacher_cache(model, dataset, path, **options)
    assert cache.batch.teacher_counterfactual_probe_count.item() <= 2
    assert cache.batch.teacher_counterfactual_slot_weight.shape == (1, 5)
    assert torch.load(path, weights_only=True)["version"] == 4
    loaded = build_or_load_v16_feasible_teacher_cache(model, dataset, path, **options)
    assert loaded.loaded
    assert torch.equal(
        loaded.batch.teacher_counterfactual_slot_weight,
        cache.batch.teacher_counterfactual_slot_weight,
    )
    with pytest.raises(ValueError, match="does not match frozen Proposal"):
        build_or_load_v16_feasible_teacher_cache(
            model, dataset, path, **{**options, "counterfactual_max_probes": 3},
        )
    with pytest.raises(ValueError, match="does not match frozen Proposal"):
        build_or_load_v16_feasible_teacher_cache(
            model, dataset, path,
            **{**options, "teacher_strategy": "synthetic_anchor_first"},
        )


@pytest.mark.parametrize(
    ("current_index", "actual", "requested", "expected"),
    [
        (0, "cuda:0", "cuda", True),
        (0, "cuda", "cuda:0", True),
        (0, "cuda:1", "cuda", False),
        (1, "cuda:1", "cuda", True),
        (1, "cuda:0", "cuda", False),
        (0, "cpu", "cuda", False),
    ],
)
def test_device_alias_uses_current_cuda_index_without_gpu(
    monkeypatch, current_index, actual, requested, expected,
):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_index)
    assert _devices_equivalent(
        torch.device(actual), torch.device(requested),
    ) is expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_feasible_teacher_accepts_unindexed_cuda_alias_for_cuda_zero(tmp_path):
    """The Linux CLI's ``cuda`` device must match a model placed on ``cuda:0``."""
    previous_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        model = _FixedProposal().to(torch.device("cuda:0"))
        cache = build_or_load_v16_feasible_teacher_cache(
            model,
            _FixedSynthetic(),
            tmp_path / "cuda_teacher.pt",
            mse_tolerance=1e-4,
            device=torch.device("cuda"),
            batch_size=1,
        )
        assert cache.sample_count == 2
        assert cache.batch.teacher_retained_mask.device.type == "cpu"
    finally:
        torch.cuda.set_device(previous_device)
