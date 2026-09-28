from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location("ours_thresholds", Path(__file__).resolve().parents[1] / "scripts/benchmark_ours_thresholds.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def report_fixture():
    cases = [dict(dataset="Synthetic", sample_id=str(i)) for i in range(2)]
    rows = []
    for eps in module.TOLERANCES:
        for i in range(2):
            rows.append(dict(dataset="Synthetic", sample_id=str(i), mse_tolerance=eps,
                             status="ok" if i == 0 else "failed", mse=eps / 2 if i == 0 else None,
                             max_squared_error=eps * 4 if i == 0 else None,
                             fit_pass=i == 0, joint_pass=i == 0, final_k=8 if i == 0 else None, total_ms=4))
    return dict(metadata=dict(mse_tolerances=list(module.TOLERANCES), cases=cases, peak_ratio=10, max_internal_knots=32), measurements=rows)


def test_all_four_tiers_and_failure_denominators():
    report = report_fixture()
    rows = module.summarize(report)
    overall = [r for r in rows if r["dataset"] == "All"]
    assert len(overall) == 4
    assert all(r["n"] == 2 and r["n_finite"] == 1 and r["dual_pass_percent"] == 50 for r in overall)
    assert overall[0]["mse_mean"] == 5e-6
    assert overall[0]["max_squared_error_max"] == 4e-5
    for damage in ("missing", "duplicate"):
        bad = deepcopy(report)
        if damage == "missing":
            bad["measurements"].pop()
        else:
            bad["measurements"].append(dict(bad["measurements"][0]))
        with pytest.raises(ValueError):
            module.summarize(bad)


@pytest.mark.parametrize("prune_fails", [False, True])
def test_fresh_predictions_and_thresholds_reach_both_solvers(monkeypatch, prune_fails):
    from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
    t = torch.linspace(0, 1, 20, dtype=torch.float64)
    points = torch.stack((t, t.square()), -1)
    fit = refit_bspline_control_points(t, points, torch.tensor([.3, .7], dtype=torch.float64),
                                     interpolate_endpoints=True, smoothness_weight=0, control_ridge=0)
    calls = []
    def raw(model, checkpoint, observed, device, epsilon):
        calls.append(("network", epsilon))
        return fit, t
    def repair(parameters, observed, initial, **kwargs):
        calls.append(("repair", kwargs["mse_tolerance"], kwargs["max_squared_error_tolerance"]))
        return SimpleNamespace(final_fit=initial, initial_fit=initial, elapsed_repair_ms=2.)
    def prune(parameters, observed, initial, **kwargs):
        assert kwargs["max_deletions"] == kwargs["max_internal_knots"] == 64
        calls.append(("pruning", kwargs["mse_tolerance"], kwargs["max_squared_error_tolerance"]))
        if prune_fails:
            raise RuntimeError("fixture failure")
        return SimpleNamespace(final_fit=initial, initial_fit=initial, elapsed_pruning_ms=3.)
    monkeypatch.setattr(module, "raw_network_fit", raw)
    monkeypatch.setattr(module, "repair_knots_to_dual_tolerance", repair)
    monkeypatch.setattr(module, "prune_knots_to_dual_tolerance", prune)
    case = dict(dataset="Synthetic", sample_id="fixture", points=points, reference=None)
    for eps in module.TOLERANCES:
        final, raw, before, retained, _ = module.run_case(None, {}, case, torch.device("cpu"), eps, eps * 10, 64)
        assert retained is fit and final["joint_pass"]
        assert final["total_ms"] == pytest.approx(final["raw_ms"] + final["repair_ms"] + final["pruning_ms"])
        assert final["repair_ms"] == 2
        assert final["mse_tolerance"] == raw["mse_tolerance"] == before["mse_tolerance"] == eps
        if prune_fails:
            assert final["diagnostics"]["pruning"]["termination"] == "failed_previous_fit_retained"
    assert [x[1] for x in calls if x[0] == "network"] == list(module.TOLERANCES)
    assert all(x[2] == 10 * x[1] for x in calls if x[0] != "network")


def test_render_preserves_report_and_emits_png_and_values(tmp_path):
    report = report_fixture()
    original = deepcopy(report)
    module.render(report, tmp_path)
    assert report == original
    assert (tmp_path / "ours_threshold_sweep.png").stat().st_size > 1000
    assert (tmp_path / "threshold_summary.json").exists()
    assert "50.0%" in (tmp_path / "threshold_summary.md").read_text(encoding="utf-8")


def test_atomic_output_retries_temporary_lock(monkeypatch, tmp_path):
    attempts = []
    def transient(path, value):
        attempts.append(value)
        if len(attempts) < 3:
            raise PermissionError("temporary Windows sharing violation")
    monkeypatch.setattr(module, "_write_json", transient)
    monkeypatch.setattr(module.time, "sleep", lambda delay: None)
    module.write_json(tmp_path / "result.json", {"complete": False})
    assert len(attempts) == 3


def test_atomic_output_persistent_lock_is_not_silently_ignored(monkeypatch, tmp_path):
    monkeypatch.setattr(module.time, "sleep", lambda delay: None)
    def locked(*args):
        raise PermissionError("permanent lock")
    monkeypatch.setattr(module, "_write_json", locked)
    with pytest.raises(PermissionError):
        module.write_json(tmp_path / "result.json", {})


def test_capacity_tracks_actual_checkpoint_not_source_report():
    config = dict(max_internal_knots=64)
    assert module.resolve_capacity(config, None, 32, True) == 64
    assert module.resolve_capacity(config, None, 32, False) == 32
    assert module.resolve_capacity(config, 64, 32, True) == 64
    for invalid in (0, 65):
        with pytest.raises(ValueError):
            module.resolve_capacity(config, invalid, 32, True)


def test_ten_tier_render(tmp_path):
    report = report_fixture()
    template = report["measurements"][:2]
    report["metadata"]["mse_tolerances"] = [i / 100000 for i in range(1, 11)]
    report["metadata"]["max_internal_knots"] = 64
    report["measurements"] = [{**r, "mse_tolerance": eps} for eps in report["metadata"]["mse_tolerances"] for r in template]
    rows = module.render(report, tmp_path)
    assert len([r for r in rows if r["dataset"] == "All"]) == 10
