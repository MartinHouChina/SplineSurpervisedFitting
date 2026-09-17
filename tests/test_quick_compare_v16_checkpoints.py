"""Focused checks for the native, paired v16 checkpoint diagnostic."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/quick_compare_v16_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("quick_compare_v16_checkpoints", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summary_uses_paired_same_case_subsets_and_unrepaired_mse():
    rows = [
        {"dataset": "Synthetic", "source_k": 4,
         "old": {"mse": 0.05, "k": 6}, "new": {"mse": 0.2, "k": 4}},
        {"dataset": "Synthetic", "source_k": 45,
         "old": {"mse": 0.2, "k": 52}, "new": {"mse": 0.05, "k": 46}},
        {"dataset": "Synthetic", "source_k": 56,
         "old": {"mse": 0.2, "k": 56}, "new": {"mse": 0.05, "k": 55}},
        {"dataset": "UJI", "source_k": None,
         "old": {"mse": 0.01, "k": 5}, "new": {"mse": 0.02, "k": 7}},
    ]
    result = MODULE.summarize(rows, tolerance=0.1)
    assert result["synthetic_overall"]["n"] == 3
    assert result["synthetic_overall"]["old"]["pass_rate"] == pytest.approx(1 / 3)
    assert result["synthetic_high_k_ge_45"]["new"]["pass_rate"] == 1
    assert result["synthetic_k_56"]["old"]["k_mean"] == 56
    assert result["real_by_dataset"]["UJI"]["n"] == 1


def test_native_refit_calls_standard_solver_once_with_network_mask(monkeypatch):
    class DummyModel:
        degree = 3

        def forward_deployment(self, points, *, mse_tolerance):
            assert points.shape == (1, 8, 2)
            assert mse_tolerance == 1e-4
            return {
                "params": torch.linspace(0, 1, 8).unsqueeze(0),
                "internal_knots": torch.tensor([[0.75, 0.25, 0.50]]),
                "learned_keep_mask": torch.tensor([[True, False, True]]),
            }

    calls = []

    def fake_refit(params, points, selected, **kwargs):
        calls.append((selected.tolist(), kwargs))
        return SimpleNamespace(fit_mse=torch.tensor(8e-5))

    monkeypatch.setattr(MODULE, "refit_bspline_control_points", fake_refit)
    result = MODULE._native_refit(
        DummyModel(), torch.zeros(8, 2), torch.device("cpu"), 1e-4
    )
    assert result["k"] == 2
    assert result["mse"] == pytest.approx(8e-5)
    assert len(calls) == 1
    assert calls[0][0] == pytest.approx([0.5, 0.75])
    assert calls[0][1]["smoothness_weight"] == 0.0
    assert calls[0][1]["control_ridge"] == 0.0
    assert calls[0][1]["interpolate_endpoints"] is True


def test_seed_overlap_is_rejected(monkeypatch):
    monkeypatch.setattr(
        MODULE, "known_synthetic_seed_ranges",
        lambda _: [{"start": 100, "stop": 105, "split": "train", "epoch": 2}],
    )
    with pytest.raises(ValueError, match="overlaps"):
        MODULE._check_seed_not_trained(101, ({}, {}))
    MODULE._check_seed_not_trained(105, ({}, {}))
