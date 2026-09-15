from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spline_fitting.training.v16_feasible_teacher import (
    _devices_equivalent,
    build_or_load_v16_feasible_teacher_cache,
)


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
