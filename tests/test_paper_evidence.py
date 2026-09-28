from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_paper_evidence as benchmark
import plot_paper_evidence as plotting


def test_common_cap_does_not_reduce_legacy_iterations():
    source = dict(metadata=dict(baseline_options=dict(max_internal_knots=32, paper_initial_knots=32,
                  liang_dense_knots=32, paper_admm_iterations=400, luo_de_iterations=50)))
    options = benchmark.baseline_options(source, 1e-5, 64)
    assert options["max_internal_knots"] == options["paper_initial_knots"] == options["liang_dense_knots"] == 64
    assert options["paper_admm_iterations"] == 400 and options["luo_de_iterations"] == 50
    assert not options["published_feasibility_safeguard"]
    assert source["metadata"]["baseline_options"]["max_internal_knots"] == 32


def geometry():
    t = torch.linspace(0, 1, 24, dtype=torch.float64)
    points = torch.stack((t, t.square()), -1)
    return dict(dataset="Synthetic", sample_id="fixture", points=points, reference=None), t


def test_uniform_and_mask_bypass_share_exact_parameters_and_charge_network():
    case, t = geometry()
    shared = dict(params=t, all_knots=torch.tensor([.2, .4, .6, .8], dtype=torch.float64), network_ms=5., transfer_ms=2.)
    for method in benchmark.VARIANTS:
        fit, params, components = benchmark.variant_initial(method, case, shared, 4, 3)
        assert len(fit.internal_knots) == 4
        if method != "uniform_chord":
            assert params is t
            assert components["network_ms"] == 5. and components["transfer_ms"] == 2.
        else:
            assert "network_ms" not in components
        if method == "all_candidates_shared_params":
            assert torch.equal(fit.internal_knots, shared["all_knots"])


@pytest.mark.parametrize("count", [0, 2, 4])
def test_matched_uniform_control_preserves_initial_count_and_forward_cost(count):
    from benchmark_matched_count_control import matched_initial
    case, params = geometry()
    fit, components = matched_initial(case, params, count, 3, dict(network_ms=5., transfer_ms=2.))
    assert len(fit.internal_knots) == count
    torch.testing.assert_close(fit.internal_knots, torch.linspace(0, 1, count + 2, dtype=torch.float64)[1:-1])
    assert components["network_ms"] == 5. and components["transfer_ms"] == 2.
    assert components["initializer_ms"] >= 0. and components["initial_refit_ms"] >= 0.


@pytest.mark.parametrize("method,endpoint", [("ours", True), ("kang_sparse_2015_adaptation", False)])
def test_stages_preserve_components_thresholds_and_endpoint_conventions(monkeypatch, method, endpoint):
    case, t = geometry()
    fit = benchmark.refit(t, case["points"], torch.tensor([.3, .7], dtype=torch.float64), 3)
    calls = []
    def repair(params, points, original, **kwargs):
        calls.append(("repair", kwargs))
        return SimpleNamespace(initial_fit=original, final_fit=original, elapsed_repair_ms=2., refit_count=3)
    def prune(params, points, original, **kwargs):
        calls.append(("pruning", kwargs))
        return SimpleNamespace(initial_fit=original, final_fit=original, elapsed_pruning_ms=4., refit_count=7,
                               before_pass=True, after_pass=True)
    monkeypatch.setattr(benchmark, "repair_knots_to_dual_tolerance", repair)
    monkeypatch.setattr(benchmark, "prune_knots_to_dual_tolerance", prune)
    stages, _ = benchmark.run_chain(case, method, fit, t, dict(network_ms=5., initial_refit_ms=1.), 5e-5, 10, 64)
    assert [r["stage"] for r, _ in stages] == list(benchmark.STAGES)
    assert [r["total_ms"] for r, _ in stages] == [6., 8., 12.]
    for row, _ in stages:
        assert row["total_ms"] == sum(row["timing_components"].values())
    assert all(options["mse_tolerance"] == 5e-5 and options["max_squared_error_tolerance"] == 5e-4
               and options["max_internal_knots"] == 64 and options["interpolate_endpoints"] == endpoint for _, options in calls)
    assert calls[-1][1]["max_deletions"] == 64


def report_fixture():
    meta = dict(cases=[dict(dataset="Synthetic", sample_id=str(i)) for i in range(2)],
                methods=list(benchmark.PUBLISHED_METHODS), variants=list(benchmark.VARIANTS),
                mse_tolerances=[5e-5], ablation_tolerance=5e-5, peak_ratio=10, max_internal_knots=64,
                hardware=dict(gpu="fixture CPU"))
    rows = []
    for case in meta["cases"]:
        for method in meta["methods"] + meta["variants"]:
            for stage in benchmark.STAGES:
                valid = case["sample_id"] == "0"
                rows.append(dict(**case, method=method, stage=stage, mse_tolerance=5e-5, status="ok" if valid else "failed",
                                 joint_pass=valid, mse=1e-5 if valid else None, max_squared_error=2e-5 if valid else None,
                                 final_k=8 if valid else None, total_ms=7., repair_refit_count=2, pruning_refit_count=3,
                                 timing_components={k: 1. for k in benchmark.COMPONENTS}))
    return dict(metadata=meta, measurements=rows, complete=True)


def test_summary_keeps_failed_cases_in_denominator():
    report = report_fixture()
    summary = plotting.summarize(report)
    assert all(r["pass_percent"] == 50. and r["n"] == 2 and r["n_finite"] == 1 for r in summary)
    assert all(r["mse_mean"] == 1e-5 and r["final_k_mean"] == 8 for r in summary)
    assert all(sum(r["timing_components_mean"].values()) == r["total_ms_mean"] for r in summary)
    assert all(r["timing_components_mean"]["failed_ms"] == 3.5 for r in summary)
    assert all(r["total_ms_pass_mean"] == 7. for r in summary)
    report["measurements"].pop()
    with pytest.raises(ValueError):
        plotting.summarize(report)


def test_geometric_misses_stay_in_error_count_and_time_averages():
    report = report_fixture()
    for row in report["measurements"]:
        if row["status"] == "failed":
            row.update(status="ok", mse=9e-5, max_squared_error=1e-3,
                       final_k=64, total_ms=14., timing_components={k: 2. for k in benchmark.COMPONENTS})
    summary = plotting.summarize(report)
    assert all(r["pass_percent"] == 50. and r["n_finite"] == 2 for r in summary)
    assert all(r["mse_mean"] == pytest.approx(5e-5) and r["final_k_mean"] == 36 for r in summary)
    assert all(r["total_ms_mean"] == 10.5 and r["total_ms_pass_mean"] == 7. for r in summary)
    assert all(r["timing_components_mean"]["failed_ms"] == 0. for r in summary)


def test_paired_success_reports_conditional_denominator():
    report = report_fixture()
    result = plotting.paired_success_summary(report)
    assert result
    assert all(r["n_all"] == 2 and r["both_pass"] == 1 for r in result)
    assert all(r["ours_pass"] == 1 and r["other_pass"] == 1 for r in result)
    assert all(r["same_count"] == 1 and r["ours_fewer"] == r["ours_more"] == 0 for r in result)
    assert all(r["ours_final_k_mean"] == r["other_final_k_mean"] == 8 for r in result)


def test_render_emits_all_three_comparison_protocols(tmp_path):
    report = report_fixture()
    plotting.render(report, tmp_path)
    for filename in ("ablation_stages.png", "six_method_tradeoff_common.png", "six_method_tradeoff_raw.png",
                     "six_method_tradeoff_historical.png", "six_method_tradeoff_by_source.png", "time_decomposition.png"):
        assert (tmp_path / filename).stat().st_size > 1000
