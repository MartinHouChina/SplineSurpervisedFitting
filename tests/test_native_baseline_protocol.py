from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.evaluation import published_baselines as baselines  # noqa: E402
from spline_fitting.evaluation.native_protocol import (  # noqa: E402
    NativeBaselineUnavailableError,
    baseline_provenance,
    validate_baseline_protocol,
)


def _points() -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, 25, dtype=torch.float64)
    return torch.stack((t, 0.15 * torch.sin(2 * torch.pi * t)), dim=-1)


@pytest.mark.parametrize("method", baselines.COMPARISON_BASELINE_METHODS)
def test_registry_covers_all_baselines_without_claiming_source_native(method: str) -> None:
    record = baseline_provenance(method)
    assert record["method"] == method
    assert record["native_comparison_available"] is False
    assert record["native_available"] is False
    assert record["author_code_executed"] is False
    assert record["known_departures"]
    expected_status = (
        "repository_control" if method in baselines.NUMERICAL_BASELINE_METHODS
        else "reproduction_unverified"
    )
    assert record["reproduction_status"] == expected_status
    assert json.loads(json.dumps(record)) == record


@pytest.mark.parametrize("method", baselines.COMPARISON_BASELINE_METHODS)
@pytest.mark.parametrize("safeguard", (False, True))
def test_native_unavailable_is_rejected_before_any_numerical_work(
    monkeypatch: pytest.MonkeyPatch, method: str, safeguard: bool
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("an unavailable native method must not execute adaptation code")

    for name in (
        "chord_length_parameters", "fit_park_dominant_points",
        "fit_liang_feature_iki", "fit_dung_direct_knots",
        "fit_sparse_knots_paper", "fit_luo_linf_de",
        "fit_feature_cdf_to_tolerance", "gradient_knot_pruning_baseline",
        "refit_bspline_control_points", "_common_mse_feasibility_safeguard",
    ):
        monkeypatch.setattr(baselines, name, forbidden)
    with pytest.raises(NativeBaselineUnavailableError, match="Native comparison unavailable") as exc:
        baselines.run_published_baseline(
            method, _points(), mse_tolerance=1e-4,
            baseline_protocol="native",
            published_feasibility_safeguard=safeguard,
        )
    assert exc.value.method == method
    assert exc.value.provenance == baseline_provenance(method)


def test_provenance_records_are_independent_copies() -> None:
    method = "kang_sparse_2015_adaptation"
    first = baseline_provenance(method)
    first["native_comparison_available"] = True
    first["known_departures"].clear()
    second = baseline_provenance(method)
    assert second["native_comparison_available"] is False
    assert second["known_departures"]


@pytest.mark.parametrize("protocol", ("", "original", "Native", None, True))
def test_unknown_protocol_is_not_silently_downgraded(protocol: object) -> None:
    with pytest.raises(ValueError, match="baseline_protocol"):
        validate_baseline_protocol("kang_sparse_2015_adaptation", protocol)


def test_unknown_baseline_has_no_implicit_native_certification() -> None:
    with pytest.raises(ValueError, match="unknown baseline"):
        baseline_provenance("kang_sparse_2015_original")


def test_historical_adaptation_defaults_are_preserved() -> None:
    parameters = inspect.signature(baselines.run_published_baseline).parameters
    assert parameters["baseline_protocol"].default == "adaptation"
    assert parameters["published_feasibility_safeguard"].default is True


@pytest.mark.parametrize("method", baselines.PUBLISHED_ADAPTATION_METHODS)
def test_disabling_safeguards_does_not_claim_native_reproduction(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("explicitly disabled common-MSE safeguard was executed")

    monkeypatch.setattr(baselines, "_common_mse_feasibility_safeguard", forbidden)
    result = baselines.run_published_baseline(
        method, _points(), mse_tolerance=1e-4,
        max_internal_knots=2, paper_initial_knots=3,
        paper_admm_iterations=5, paper_lambda_bisections=1,
        paper_relocation_iterations=1,
        liang_dense_knots=4, liang_initial_knots=1, liang_feature_samples=33,
        dung_scan_intervals=2, dung_optimization_iterations=1,
        luo_de_population=5, luo_de_iterations=1,
        baseline_protocol="adaptation", published_feasibility_safeguard=False,
    )
    assert result.diagnostics["reproduction_status"] == "reproduction_unverified"
    assert result.diagnostics["baseline_protocol"] == "adaptation"
    assert result.diagnostics["native_comparison_available"] is False
    assert result.diagnostics["source_native_algorithm_executed"] is False
    assert result.diagnostics["baseline_provenance"] == baseline_provenance(method)
    if method in baselines._SAFEGUARD_METHODS:
        assert result.diagnostics["comparison_method_label"] == "unrepaired disclosed adaptation"
        assert result.diagnostics["comparison_feasibility_refit_count"] == 0


def test_official_dung_code_is_not_confused_with_executed_adapter() -> None:
    record = baseline_provenance("dung_direct_knot_2017_adaptation")
    assert "pone.0173857.s006" in record["author_code_url"]
    assert len(record["author_code_sha256"]) == 64
    assert record["author_code_executed"] is False
    assert record["native_comparison_available"] is False
