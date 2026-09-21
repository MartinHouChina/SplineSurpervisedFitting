"""Validation peaks are actual point residuals and retain their aggregation scope."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_v16 as entry


class GeometryFixture:
    degree = 3

    def eval(self):
        return self

    def encode_candidates(self, points, mse_tolerance):
        return {
            "proposal_params": torch.linspace(0, 1, points.shape[1]).expand(points.shape[0], -1),
            "proposal_internal_knots": points.new_tensor([[.3, .7]]).expand(points.shape[0], -1),
            "one_shot_probability_mass": points.new_full((points.shape[0],), 1.),
            "adaptive_keep_threshold": points.new_zeros(points.shape[0]),
        }

    def select_mask(self, context):
        mask = torch.zeros_like(context["proposal_internal_knots"], dtype=torch.bool)
        mask[:, 0] = True
        return mask

    def decode_subset(self, context, mask):
        return {"params": context["proposal_params"],
                "internal_knots": context["proposal_internal_knots"],
                "learned_keep_mask": mask}


@pytest.mark.parametrize("stage", ["proposal", "joint"])
def test_validation_peak_mean_p95_worst_use_curves_not_means_of_batch_maxima(monkeypatch, stage):
    points = torch.zeros(3, 6, 2)
    points[:, :, 0] = torch.linspace(0, 1, 6)
    points[:, :, 1] = torch.arange(3)[:, None]
    loader = [dict(points=points[:2], source=["A", "A"]),
              dict(points=points[2:], source=["B"])]
    calls, evaluations = [], []

    def fake_refit(params, observed, knots, **options):
        sample = int(observed[0, 1])
        dense = knots.numel() == 2
        peak = (sample + 1) * 1e-4 if dense else (sample + 1) ** 2 * 2e-4
        residual = torch.zeros_like(observed)
        residual[1, 0] = math.sqrt(peak)
        calls.append((sample, dense))

        def evaluate(requested):
            torch.testing.assert_close(requested, params, atol=0, rtol=0)
            evaluations.append((sample, dense))
            return observed + residual

        return SimpleNamespace(fit_mse=peak / len(observed), evaluate=evaluate)

    monkeypatch.setattr(entry, "refit_bspline_control_points", fake_refit)
    monkeypatch.setattr(entry, "progress", lambda *args, **kwargs: None)
    result = entry.validate(GeometryFixture(), loader, torch.device("cpu"), 5e-5, stage=stage)
    assert len(calls) == len(evaluations) == (3 if stage == "proposal" else 6)
    dense_peaks = [1e-4, 2e-4, 3e-4]
    deployed_peaks = dense_peaks if stage == "proposal" else [2e-4, 8e-4, 18e-4]
    scopes = [(result, slice(None)), (result["by_source"]["A"], slice(0, 2)),
              (result["by_source"]["B"], slice(2, 3))]
    for summary, selected in scopes:
        for prefix, peaks in (("dense", dense_peaks), ("deployment", deployed_peaks)):
            chosen = peaks[selected]
            field = prefix + "_max_point_squared_error"
            assert summary[field + "_mean"] == pytest.approx(np.mean(chosen))
            assert summary[field + "_p95"] == pytest.approx(np.quantile(chosen, .95))
            assert summary[field + "_max"] == pytest.approx(max(chosen))
            assert summary[prefix + "_pass_rate"] == pytest.approx(
                np.mean([peak / 6 <= 5e-5 for peak in chosen])
            )
    assert result["deployment_max_point_squared_error_max"] > result["deployment_mse"]


def test_validation_rejects_nonfinite_point_error_even_when_mse_field_is_finite(monkeypatch):
    points = torch.zeros(1, 6, 2)
    monkeypatch.setattr(entry, "refit_bspline_control_points", lambda params, q, *a, **kw:
                        SimpleNamespace(fit_mse=0., evaluate=lambda _: torch.full_like(q, float("nan"))))
    with pytest.raises(RuntimeError, match="non-finite validation pointwise"):
        entry.validate(GeometryFixture(), [dict(points=points, source=["A"])],
                       torch.device("cpu"), 5e-5, stage="proposal")
