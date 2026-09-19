from __future__ import annotations

# ruff: noqa: E402

import importlib.util
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from spline_fitting.checkpointing import (
    V16_ADAPTIVE_SELECTION_REVISION,
    V16_SIMPLIFICATION_CONTRACT,
)
spec = importlib.util.spec_from_file_location("benchmark_v15", ROOT / "scripts/benchmark_v15_datasets.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def v16_checkpoint(*, target=0.90, observed=0.92, stage="joint"):
    accepted = stage == "joint" and observed >= target
    return {
        "objective_version": benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        "architecture_revision": V16_ADAPTIVE_SELECTION_REVISION,
        "simplification_contract": V16_SIMPLIFICATION_CONTRACT,
        "simplification_ready": True,
        "synthetic_data_contract": "source_subset_threshold_minimal_v1",
        "dataset_config": {
            "certified_minimal_source": True,
            "canonical_knot_tolerance": 0.005,
            "minimality_margin": 0.2,
            "minimality_audit_points": 512,
        },
        "model_config": {
            "one_shot_selection_policy": "mass_topk",
            "one_shot_adaptive_threshold": True,
            "one_shot_safety_sigma": 0.05,
            "one_shot_safety_knots": 0,
            "max_internal_knots": 16,
        },
        "stage": stage,
        "training_config": {
            "proposal_pass_target": target,
            "deployment_pass_target": target,
            "mse_tolerance": 2.5e-5,
            "knot_match_tolerance": 0.01,
            "allow_infeasible_proposals": False,
            "final_safety_sigma": 0.05,
            "final_safety_knots": 0,
        },
        "validation_metrics": {
            "worst_dense_pass_rate": 0.98,
            "worst_deployment_pass_rate": observed,
            "keep_count": 8.0,
            "synthetic_count_mae": 0.75,
            "synthetic_knot_match_f1": 0.8,
            "synthetic_knot_matched_mae": 0.004,
        },
        "deployment_config": {
            "mse_tolerance": 2.5e-5,
            "knot_match_tolerance": 0.01,
            "one_shot_selection_policy": "mass_topk",
            "adaptive_keep_threshold": True,
            "one_shot_safety_sigma": 0.05,
            "one_shot_safety_knots": 0,
        },
        "loss_config": {
            "ranked_prefix_teacher": True,
            "weights": {
                "count_weight": 2.0,
                "supervised_count_weight": 1.0,
                "supervised_over_count_weight": 1.0,
                "complexity_weight": 0.05,
                "true_parameter_weight": 0.1,
                "proposal_knot_coverage_weight": 1.0,
                "selected_knot_position_weight": 1.0,
            },
        },
        "proposal_ready": True,
        "best_deployment_pass_constraint_satisfied": accepted,
        "checkpoint_quality": "deployment_target_met" if accepted else "target_not_met",
    }


def test_balanced_sampling_visits_each_group_and_is_reproducible():
    records = [{"group_id": f"g{i // 4}"} for i in range(12)]
    first = benchmark.balanced_indices(records, 6, 42)
    assert first == benchmark.balanced_indices(records, 6, 42)
    assert len(set(first)) == 6
    assert len({records[i]["group_id"] for i in first[:3]}) == 3
    assert len(benchmark.balanced_indices(records, 100, 42)) == 12


def test_failures_remain_in_pass_rate_denominator():
    row = {"dataset": "UJI", "method": "ours", "status": "ok",
           "mse": 1e-6, "reference_mse": 2e-6, "fit_pass": True,
           "reference_pass": True, "final_k": 2, "total_ms": 3, "network_ms": 1}
    failure = {**row, "status": "failed", "fit_pass": False, "reference_pass": None,
               "mse": None, "reference_mse": None}
    summary = benchmark.summarize([row, failure])[0]
    assert summary["failed"] == 1
    assert summary["n"] == 2
    assert summary["fit_pass_rate"] == .5
    assert summary["reference_pass_rate"] == .5
    assert summary["mse_mean"] == 1e-6


def test_synthetic_summary_reports_canonical_count_accuracy():
    rows = [
        {
            "dataset": "Synthetic", "method": "ours", "status": "ok",
            "mse": 1e-6, "reference_mse": None, "fit_pass": True,
            "reference_pass": None, "final_k": 9, "canonical_k": 8,
            "total_ms": 3, "network_ms": 1,
        },
        {
            "dataset": "Synthetic", "method": "ours", "status": "ok",
            "mse": 2e-6, "reference_mse": None, "fit_pass": True,
            "reference_pass": None, "final_k": 12, "canonical_k": 12,
            "total_ms": 4, "network_ms": 1.5,
        },
        {
            "dataset": "Synthetic", "method": "ours", "status": "ok",
            "mse": 3e-6, "reference_mse": None, "fit_pass": True,
            "reference_pass": None, "final_k": 12, "canonical_k": 14,
            "total_ms": 5, "network_ms": 2,
        },
    ]

    summary = benchmark.summarize(rows)[0]

    assert summary["canonical_k_mean"] == pytest.approx(34 / 3)
    assert summary["canonical_count_mae"] == pytest.approx(1.0)
    assert summary["canonical_count_bias"] == pytest.approx(-1 / 3)
    assert summary["canonical_count_exact_rate"] == pytest.approx(1 / 3)
    assert summary["canonical_count_within_one_rate"] == pytest.approx(2 / 3)


def test_summary_without_canonical_labels_and_empty_input_are_safe():
    row = {
        "dataset": "UJI", "method": "ours", "status": "ok",
        "mse": 1e-6, "reference_mse": 2e-6, "fit_pass": True,
        "reference_pass": True, "final_k": 7, "canonical_k": None,
        "total_ms": 3, "network_ms": 1,
    }

    summary = benchmark.summarize([row])[0]

    assert summary["canonical_k_mean"] is None
    assert summary["canonical_count_mae"] is None
    assert summary["canonical_count_bias"] is None
    assert summary["canonical_count_exact_rate"] is None
    assert summary["canonical_count_within_one_rate"] is None
    assert benchmark.summarize([]) == []


def test_reference_alignment_uses_original_resampling_grid():
    class Fit:
        def evaluate(self, parameters):
            return parameters[:, None]
    parameters = torch.tensor([0., .2, 1.], dtype=torch.float64)
    case = {"reference_grid": torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64),
            "reference": torch.tensor([[0.], [.1], [.2], [.6], [1.]], dtype=torch.float64)}
    assert benchmark.reference_mse(Fit(), parameters, case) == pytest.approx(0, abs=1e-30)


def test_peak_error_is_squared_euclidean_and_uses_same_reference_mapping():
    class Fit:
        def evaluate(self, parameters):
            return torch.stack((parameters, 2 * parameters), dim=-1)

    parameters = torch.tensor([0., .2, 1.], dtype=torch.float64)
    case = {
        "points": torch.tensor([[0., 0.], [.5, .8], [1., 2.]], dtype=torch.float64),
        "reference_grid": torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64),
        "reference": torch.tensor([[0., 0.], [.1, .5], [.4, .4], [.6, 1.2], [1., 2.]], dtype=torch.float64),
    }
    metrics = benchmark.fit_error_metrics(Fit(), parameters, case)
    # Sum coordinate squares first, then mean/max over points. No sqrt, no
    # coordinate averaging, and no nearest-point or reparameterization search.
    assert metrics["mse"] == pytest.approx(.25 / 3)
    assert metrics["max_squared_error"] == pytest.approx(.25)
    assert metrics["reference_mse"] == pytest.approx(.13 / 5)
    assert metrics["reference_max_squared_error"] == pytest.approx(.09)
    assert benchmark.reference_mse(Fit(), parameters, case) == metrics["reference_mse"]


def test_peak_error_does_not_invent_synthetic_reference():
    class Fit:
        def evaluate(self, parameters):
            return parameters[:, None]
    params = torch.tensor([0., .5, 1.], dtype=torch.float64)
    case = {"points": torch.tensor([[0.], [.75], [1.]], dtype=torch.float64), "reference": None}
    result = benchmark.fit_error_metrics(Fit(), params, case)
    assert result["max_squared_error"] == pytest.approx(.0625)
    assert result["reference_mse"] is None
    assert result["reference_max_squared_error"] is None


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_peak_error_rejects_nonfinite_residuals(invalid):
    class Fit:
        def evaluate(self, parameters):
            return torch.full((parameters.numel(), 2), invalid)
    with pytest.raises(ValueError, match="non-finite fitted squared residuals"):
        benchmark.fit_error_metrics(Fit(), torch.tensor([0., 1.]),
                                    {"points": torch.zeros(2, 2), "reference": None})


def test_measured_peak_and_mse_agree_with_actual_bspline_refit():
    params = torch.linspace(0, 1, 21, dtype=torch.float64)
    points = torch.stack((params, torch.sin(2 * torch.pi * params)), dim=-1)
    fit = benchmark.refit_bspline_control_points(
        params, points, torch.empty(0, dtype=torch.float64), degree=3,
        smoothness_weight=0., control_ridge=0., interpolate_endpoints=True,
    )
    errors = benchmark.fit_error_metrics(fit, params, {"points": points, "reference": None})
    assert errors["mse"] == pytest.approx(float(fit.fit_mse), rel=1e-12, abs=1e-15)
    assert errors["max_squared_error"] > errors["mse"] > 0


def _peak_row(**overrides):
    return {
        "dataset": "UJI", "method": "ours", "status": "ok", "has_reference": True,
        "mse": 1e-6, "max_squared_error": 4e-6,
        "reference_mse": 2e-6, "reference_max_squared_error": 8e-6,
        "fit_pass": True, "reference_pass": True,
        "final_k": 2, "total_ms": 3, "network_ms": 1,
        **overrides,
    }


def test_peak_summary_distinguishes_max_residual_from_worst_curve_mse():
    rows = [_peak_row(), _peak_row(mse=3e-6, max_squared_error=12e-6,
                                 reference_mse=4e-6, reference_max_squared_error=24e-6)]
    result = benchmark.summarize(rows)[0]
    assert result["mse_max"] == 3e-6
    assert result["max_squared_error_mean"] == pytest.approx(8e-6)
    assert result["max_squared_error_p95"] == pytest.approx(11.6e-6)
    assert result["max_squared_error_max"] == 12e-6
    assert result["reference_mse_max"] == 4e-6
    assert result["reference_max_squared_error_mean"] == pytest.approx(16e-6)
    assert result["reference_max_squared_error_p95"] == pytest.approx(23.2e-6)
    assert result["reference_max_squared_error_max"] == 24e-6
    assert result["max_squared_error_valid_count"] == 2
    assert result["max_squared_error_missing_count"] == 0


def test_old_or_mixed_measurements_do_not_invent_or_understate_peak_errors():
    legacy = _peak_row()
    legacy.pop("max_squared_error")
    legacy.pop("reference_max_squared_error")
    for rows, valid in (([legacy], 0), ([legacy, _peak_row()], 1)):
        result = benchmark.summarize(rows)[0]
        for prefix in ("max_squared_error", "reference_max_squared_error"):
            assert result[prefix + "_mean"] is None
            assert result[prefix + "_p95"] is None
            assert result[prefix + "_max"] is None
            assert result[prefix + "_valid_count"] == valid
            assert result[prefix + "_missing_count"] == 1


def test_failed_rows_are_not_missing_successful_peak_measurements():
    failure = _peak_row(status="failed", mse=None, max_squared_error=None,
                        reference_mse=None, reference_max_squared_error=None,
                        fit_pass=False, reference_pass=None)
    result = benchmark.summarize([_peak_row(), failure])[0]
    assert result["max_squared_error_valid_count"] == 1
    assert result["max_squared_error_missing_count"] == 0
    assert result["reference_max_squared_error_missing_count"] == 0
    assert result["max_squared_error_max"] == 4e-6
    assert result["fit_pass_rate"] == .5


def test_peak_reports_explain_metrics_and_preserve_missing_legacy_values(tmp_path):
    metadata = {"checkpoint": "legacy.pt", "epoch": 1, "mse_tolerance": 1e-5,
                "objective_version": benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
                "hardware": {"device": "cpu"}, "num_points": 24}
    legacy = _peak_row()
    legacy.pop("max_squared_error")
    legacy.pop("reference_max_squared_error")
    benchmark.write_reports(tmp_path, metadata, [legacy, _peak_row()])
    report = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert report["measurements"][0]["max_squared_error"] is None
    assert report["summary"][0]["max_squared_error_max"] is None
    assert report["metric_definitions"]["max_squared_error"].startswith("max_i")
    assert "error_metric_version" not in report["metadata"]  # no fabricated historical protocol
    assert "max_squared_error" not in legacy  # source records were not changed
    with (tmp_path / "measurements.csv").open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["max_squared_error"] == ""
    assert float(csv_rows[1]["max_squared_error"]) == 4e-6
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "MaxSqErr" in markdown and "N/A" in markdown
    assert "不是 Hausdorff" in markdown
    assert "通过率仍由 MSE 阈值判断" in markdown
    with (tmp_path / "metric_definitions.csv").open(encoding="utf-8-sig", newline="") as handle:
        definitions = {item["metric"]: item["definition"] for item in csv.DictReader(handle)}
    assert definitions["max_squared_error"].startswith("max_i")


def test_six_method_benchmark_exports_peak_fields_and_versions_resume_protocol(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"mock checkpoint; torch.load is patched")
    checkpoint = {"objective_version": benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
                  "epoch": 1, "model_config": {"max_internal_knots": 32, "degree": 3}}

    class Model:
        degree = 3
        def to(self, device):
            return self
        def eval(self):
            return self

    parameters = torch.linspace(0, 1, 21, dtype=torch.float64)
    points = torch.stack((parameters, torch.sin(2 * torch.pi * parameters)), dim=-1)
    fit = benchmark.refit_bspline_control_points(
        parameters, points, torch.empty(0, dtype=torch.float64), degree=3,
        smoothness_weight=0., control_ridge=0., interpolate_endpoints=True,
    )
    case = {"dataset": "UJI", "sample_id": "test", "group_id": "writer-test",
            "source_k": None, "canonical_k": None, "points": points,
            "reference": points.clone(), "reference_grid": parameters.clone()}
    monkeypatch.setattr(benchmark.torch, "load", lambda *a, **k: checkpoint)
    monkeypatch.setattr(benchmark, "build_model_from_checkpoint", lambda *a: (
        Model(), {"structure_mode": "candidate_pruning_one_shot"}, None))
    monkeypatch.setattr(benchmark, "validate_checkpoint_for_benchmark", lambda *a, **k: (None, False))
    monkeypatch.setattr(benchmark, "prepare_cases", lambda *a: ([case], []))
    monkeypatch.setattr(benchmark, "run_published_baseline", lambda *a, **k: None)  # numerical warmup
    monkeypatch.setattr(benchmark, "measure_ours", lambda *a, **k: (fit, parameters, 3., 1., {}))
    monkeypatch.setattr(benchmark, "measure_numerical_baseline", lambda *a, **k: (fit, parameters, 4., None, {}))
    directory = tmp_path / "reports"
    argv = ["--checkpoint", str(checkpoint_path), "--output-dir", str(directory),
            "--method-set", "published", "--device", "cpu",
            "--max-internal-knots", "32", "--paper-initial-knots", "32", "--liang-dense-knots", "32"]
    benchmark.main(argv, expected_objective=benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION)
    report = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    assert len(report["measurements"]) == 6
    assert report["metadata"]["error_metric_version"] == benchmark.ERROR_METRIC_VERSION
    assert report["metadata"]["error_metric_definitions"] == benchmark.ERROR_METRIC_DEFINITIONS
    assert report["metadata"]["knot_capacities"]["equal_initial_capacity"]
    for row in report["measurements"]:
        assert row["status"] == "ok"
        assert row["max_squared_error"] > row["mse"] > 0
        assert row["reference_max_squared_error"] == pytest.approx(row["max_squared_error"])
        assert row["total_ms"] == (3. if row["method"] == "ours" else 4.)
    # A report made with the old metric protocol cannot silently accept new
    # measurements under --resume; there is no attempt to infer old peaks.
    old_metadata = report["metadata"].copy()
    old_metadata["fingerprint"] = "legacy-mean-error-only-protocol"
    (directory / "experiment.json").write_text(json.dumps(old_metadata), encoding="utf-8")
    with pytest.raises(SystemExit):
        benchmark.main([*argv, "--resume"], expected_objective=benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION)


def test_all_real_failures_have_zero_reference_pass_rate():
    row = {"dataset": "UJI", "method": "ours", "status": "failed", "has_reference": True,
           "mse": None, "reference_mse": None, "fit_pass": False, "reference_pass": None,
           "final_k": None, "total_ms": 10, "network_ms": None}
    result = benchmark.summarize([row])[0]
    assert result["reference_pass_rate"] == 0
    assert result["mse_mean"] is None


def test_recover_torn_journal_preserves_complete_records(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_bytes(b'{"a":1}\n{"unfinished"')
    assert benchmark.recover_journal(path) == [{"a": 1}]
    assert path.read_bytes() == b'{"a":1}\n'
    assert path.with_suffix(".interrupted_tail.bin").read_bytes() == b'{"unfinished"'
    path.write_bytes(b'{broken}\n{"a":1}\n')
    with pytest.raises(ValueError, match="Corrupt completed"):
        benchmark.recover_journal(path)


def test_version_gate_rejects_v15_as_v16_and_unknown_objectives():
    v15 = benchmark.V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION
    v16 = benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION
    assert benchmark.benchmark_version(v15, expected_objective=v15) == "v15"
    assert benchmark.benchmark_version(v16, expected_objective=v16) == "v16"
    with pytest.raises(ValueError, match="requires"):
        benchmark.benchmark_version(v15, expected_objective=v16)
    with pytest.raises(ValueError, match="requires"):
        benchmark.benchmark_version("invented", expected_objective=v16)
    with pytest.raises(ValueError, match="Unsupported"):
        benchmark.benchmark_version("invented", expected_objective="invented")


def test_missing_checkpoint_is_reported_before_torch_load(tmp_path, capsys):
    missing = tmp_path / "not_trained.pt"
    with pytest.raises(SystemExit) as error:
        benchmark.main([
            "--checkpoint", str(missing),
            "--output-dir", str(tmp_path / "comparison"),
        ])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "checkpoint does not exist" in message
    assert "run scripts/train_v16.py first" in message
    assert not (tmp_path / "comparison").exists()


def test_v16_benchmark_rejects_relaxed_or_proposal_checkpoint_by_default():
    for checkpoint in (
        v16_checkpoint(target=0.85, observed=0.92),
        v16_checkpoint(stage="proposal"),
    ):
        with pytest.raises(ValueError, match="allow-unqualified-diagnostic"):
            benchmark.validate_checkpoint_for_benchmark(
                checkpoint, mse_tolerance=2.5e-5,
                allow_unqualified_diagnostic=False,
            )
        qualification, diagnostic = benchmark.validate_checkpoint_for_benchmark(
            checkpoint, mse_tolerance=2.5e-5,
            allow_unqualified_diagnostic=True,
        )
        assert diagnostic
        assert not qualification["formal_reporting_eligible"]


def test_v16_benchmark_accepts_consistent_formal_checkpoint():
    qualification, diagnostic = benchmark.validate_checkpoint_for_benchmark(
        v16_checkpoint(), mse_tolerance=2.5e-5,
        allow_unqualified_diagnostic=False,
    )
    assert qualification["formal_reporting_eligible"]
    assert not diagnostic


def test_v16_forwards_reported_mse_budget_but_v15_has_no_new_argument():
    class V15Model:
        def forward_deployment(self, points):
            return points

    class V16Model:
        mse_tolerance = 9e-4

        def forward_deployment(self, points, *, mse_tolerance):
            return points, mse_tolerance

    points = torch.zeros(1, 4, 2)
    assert benchmark.deployment_forward(
        V15Model(), points,
        objective_version=benchmark.V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
        mse_tolerance=1e-5,
    ) is points
    result = benchmark.deployment_forward(
        V16Model(), points,
        objective_version=benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        mse_tolerance=1e-5,
    )
    assert result[0] is points
    assert result[1] == 1e-5
    with pytest.raises(ValueError, match="Unsupported"):
        benchmark.deployment_forward(V15Model(), points,
                                     objective_version="unknown", mse_tolerance=1e-5)


def test_numerical_timing_uses_same_complete_repeat_count_and_median(monkeypatch):
    elapsed = iter([9.0, 3.0, 6.0])
    fit = SimpleNamespace()

    def run(method, points, **kwargs):
        return SimpleNamespace(
            fit=fit,
            parameters=torch.linspace(0, 1, points.shape[0]),
            elapsed_ms=next(elapsed),
            diagnostics={"method": method},
        )

    monkeypatch.setattr(benchmark, "run_published_baseline", run)
    monkeypatch.setattr(benchmark, "published_baseline_kwargs", lambda *a, **k: {})
    args = SimpleNamespace(end_to_end_repeats=3)
    returned = benchmark.measure_numerical_baseline(
        "uniform_gradient_pruning", torch.zeros(8, 2), args, degree=3,
    )
    assert returned[0] is fit
    assert returned[2] == 6.0
    assert returned[3] is None
    assert returned[4]["repeated_complete_timing"]["samples_ms"] == [9.0, 3.0, 6.0]


def test_v16_wrapper_uses_explicit_objective_and_dedicated_paths(monkeypatch):
    import benchmark_v16_datasets as wrapper

    called = {}

    def capture(argv, **kwargs):
        called.update(kwargs)
        called["argv"] = argv

    monkeypatch.setattr(wrapper, "shared_main", capture)
    wrapper.main(["--skip-real"])
    assert called["argv"] == ["--skip-real"]
    assert called["expected_objective"] == benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION
    assert (
        called["default_checkpoint"].name
        == "candidate_selection_v16_simplified_certified_k96.pt"
    )
    assert called["default_output_dir"].name == "v16_multidata"


def test_v16_report_labels_cannot_claim_v15(tmp_path):
    metadata = {
        "checkpoint": "v16.pt", "epoch": 1, "mse_tolerance": 1e-5,
        "objective_version": benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        "hardware": {"device": "cpu"}, "num_points": 24,
    }
    row = {"dataset": "Synthetic", "method": "ours", "status": "ok",
           "mse": 1e-6, "reference_mse": None, "fit_pass": True,
           "reference_pass": None, "final_k": 2, "total_ms": 3, "network_ms": 1}
    benchmark.write_reports(tmp_path, metadata, [row])
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "# v16 " in report
    assert "Ours v16 learned" in report
    assert "Ours v15" not in report


def test_diagnostic_benchmark_report_has_unmissable_warning(tmp_path):
    metadata = {
        "checkpoint": "v16.pt", "epoch": 1, "mse_tolerance": 2.5e-5,
        "objective_version": benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        "hardware": {"device": "cpu"}, "num_points": 24,
        "diagnostic_not_final": True,
        "checkpoint_qualification": {"reasons": ["configured target below 97%"]},
    }
    row = {"dataset": "Synthetic", "method": "ours", "status": "ok",
           "mse": 1e-6, "reference_mse": None, "fit_pass": True,
           "reference_pass": None, "final_k": 2, "total_ms": 3, "network_ms": 1}
    benchmark.write_reports(tmp_path, metadata, [row])
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "DIAGNOSTIC NOT FINAL" in report
    assert "configured target below 97%" in report


@pytest.mark.parametrize("v16, expected", [(False, (28, 40)), (True, (64, 64))])
def test_comparison_caps_keep_v15_defaults_and_match_v16_checkpoint(v16, expected):
    args = SimpleNamespace(
        max_internal_knots=None, paper_initial_knots=None,
        liang_dense_knots=None,
    )
    checkpoint = {
        "objective_version": (benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION
                              if v16 else benchmark.V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION),
        "model_config": {"max_internal_knots": 64},
    }
    caps = benchmark.resolve_comparison_capacities(args, checkpoint)
    assert (args.max_internal_knots, args.paper_initial_knots) == expected
    assert caps["network_candidates"] == 64
    assert caps["equal_initial_capacity"] is v16


def test_explicit_numerical_capacities_are_retained_and_disclosed():
    args = SimpleNamespace(
        max_internal_knots=28, paper_initial_knots=40,
        liang_dense_knots=None,
    )
    checkpoint = {"objective_version": benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
                  "model_config": {"max_internal_knots": 64}}
    caps = benchmark.resolve_comparison_capacities(args, checkpoint)
    assert caps == {
        "network_candidates": 64,
        "greedy_initial_and_yeh_max": 28,
        "kang_dense_initial": 40,
        "liang_dense_initial": 40,
        "equal_initial_capacity": False,
        "degree": 3,
        "clamped_endpoint_entries": 8,
        "network_full_knot_vector_size_at_all_keep": 72,
        "numerical_full_knot_vector_cap": 36,
    }
    args.max_internal_knots = 0
    with pytest.raises(ValueError, match="positive"):
        benchmark.resolve_comparison_capacities(args, checkpoint)


def test_full_knot_vector_notation_maps_to_internal_capacity():
    args = SimpleNamespace(
        max_internal_knots=None,
        full_knot_vector_size=64,
        paper_initial_knots=None,
        liang_dense_knots=None,
    )
    checkpoint = {
        "objective_version": benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        "model_config": {"max_internal_knots": 56, "degree": 3},
    }
    caps = benchmark.resolve_comparison_capacities(args, checkpoint)
    assert args.max_internal_knots == 56
    assert args.paper_initial_knots == 56
    assert caps["network_full_knot_vector_size_at_all_keep"] == 64
    assert caps["numerical_full_knot_vector_cap"] == 64


def test_v16_seed_reservations_include_later_configured_and_resumed_epochs():
    checkpoint = {
        "objective_version": benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        "epoch": 2, "history": [{"epoch": 1}, {"epoch": 2}],
        "training_config": {"train_seed": 42, "train_size": 4000, "epochs": 5,
                            "train_seed_stride": 10_000_000, "resample_train_each_epoch": True,
                            "val_seed": 1_000_000, "val_size": 1000},
    }
    ranges = benchmark.known_synthetic_seed_ranges(checkpoint)
    assert len(ranges) == 6
    assert ranges[4] == {"split": "train", "epoch": 5, "start": 40_000_042, "stop": 40_004_042}
    assert ranges[-1] == {"split": "val", "epoch": 1, "start": 1_000_000, "stop": 1_001_000}
    checkpoint["history"].append({"epoch": 7})
    assert len(benchmark.known_synthetic_seed_ranges(checkpoint)) == 8
    checkpoint["training_config"]["resample_train_each_epoch"] = False
    assert len(benchmark.known_synthetic_seed_ranges(checkpoint)) == 2
    checkpoint["objective_version"] = benchmark.V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION
    checkpoint["training_config"]["resample_train_each_epoch"] = True
    assert len(benchmark.known_synthetic_seed_ranges(checkpoint)) == 2


def test_v16_rejects_test_seed_colliding_with_later_epoch(monkeypatch):
    checkpoint = {
        "objective_version": benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        "training_config": {"train_seed": 42, "train_size": 4000, "epochs": 5,
                            "val_seed": 1_000_000, "val_size": 1000},
    }
    monkeypatch.setattr(benchmark, "_dataset_config_from_checkpoint", lambda *a: {})
    monkeypatch.setattr(benchmark, "_select_indices", lambda *a, **k: [42])
    args = SimpleNamespace(skip_synthetic=False, seed=20_000_000, scan_size=100,
                           selection_seed=1, min_knot_count=4, max_knot_count=20,
                           samples_per_knot_count=1)
    with pytest.raises(ValueError, match="train epoch 3"):
        benchmark.prepare_cases(args, checkpoint, {})
