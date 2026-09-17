"""Fast contract checks for the CPU-only v16 Teacher cache audit script."""
from __future__ import annotations

# ruff: noqa: E402

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_v16_teacher_cache", ROOT / "scripts/audit_v16_teacher_cache.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

from spline_fitting.training.v16_feasible_teacher import _proposal_fingerprint


def test_indices_support_slice_and_explicit_ids_and_reject_invalid() -> None:
    assert MODULE.parse_indices("0:3", 5) == [0, 1, 2]
    assert MODULE.parse_indices("3,1", 5) == [3, 1]
    for bad in ("", "2:2", "1,1", "-1", "0:6", "1:2:3"):
        with pytest.raises(ValueError):
            MODULE.parse_indices(bad, 5)


def test_fingerprints_reject_changed_proposal_or_dataset() -> None:
    model = nn.Linear(1, 1, bias=False)
    proposal = _proposal_fingerprint(model)
    MODULE._require_matching_fingerprints(model, proposal, "same", "same")
    with pytest.raises(ValueError, match="dataset point/anchor fingerprint"):
        MODULE._require_matching_fingerprints(model, proposal, "changed", "same")
    with torch.no_grad():
        model.weight.add_(1.0)
    with pytest.raises(ValueError, match="Proposal checkpoint weights"):
        MODULE._require_matching_fingerprints(model, proposal, "same", "same")


def test_topk_feasible_denominator_excludes_infeasible_all_keep_rows() -> None:
    def diagnostics(passed: bool) -> dict:
        return {
            "count": 2, "proposal_mse": 1e-5, "proposal_pass": passed,
            "decoded_mse": 1e-5, "decoded_pass": passed,
            "decoded_minus_proposal_mse": 0.0,
        }

    rows = [
        {
            "cached_pass": True, "cpu_pass": True,
            "cpu_minus_cached_mse": 0.0,
            "max_abs_selected_knot_difference": 0.0,
            "joint": {
                "teacher_mask": diagnostics(True),
                "teacher_count_topk": diagnostics(False),
                "free_mask": diagnostics(True),
                "teacher_count_topk_recall": 0.5,
                "teacher_count_topk_exact": False,
            },
        },
        {
            "cached_pass": False, "cpu_pass": False,
            "cpu_minus_cached_mse": 0.0,
            "max_abs_selected_knot_difference": 0.0,
            "joint": {
                "teacher_mask": diagnostics(False),
                "teacher_count_topk": diagnostics(False),
                "free_mask": diagnostics(True),
                "teacher_count_topk_recall": 1.0,
                "teacher_count_topk_exact": True,
            },
        },
    ]
    summary = MODULE._summarize(rows, tolerance=1e-4)
    assert summary["teacher_count_topk_recall"] == pytest.approx(0.75)
    assert summary["teacher_count_topk_exact_fraction"] == pytest.approx(0.5)
    feasible = summary["teacher_feasible_subset"]
    assert feasible["sample_count"] == 1
    assert feasible["teacher_count_topk_recall"] == pytest.approx(0.5)
    assert feasible["teacher_count_topk_exact_fraction"] == 0.0
    assert feasible["teacher_mask_decoded_pass_fraction"] == 1.0
    assert feasible["teacher_count_topk_decoded_pass_fraction"] == 0.0

