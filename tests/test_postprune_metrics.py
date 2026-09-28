import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("postprune_metrics", Path(__file__).resolve().parents[1] / "scripts/export_postprune_metrics.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def report_fixture():
    methods = ["ours", "dung_direct_knot_2017_adaptation"]
    rows = []
    for method in methods:
        for i in range(3):
            failed = method != "ours" and i == 2
            rows.append(dict(dataset="Synthetic", sample_id=str(i), method=method,
                             status="failed" if failed else "ok", mse=None if failed else [1e-5, 8e-5, 1e-6][i],
                             max_squared_error=None if failed else [1e-4, 6e-4, 8e-6][i],
                             final_k=None if failed else [8, 10, 4][i], total_ms=10. + i,
                             network_ms=1., pruning_ms=5. if method == "ours" else 0.,
                             joint_pass=not failed and i != 1))
    return dict(metadata=dict(methods=methods, mse_tolerance=5e-5, max_squared_error_tolerance=5e-4,
                              timing_protocol="fixture", cases=[dict(dataset="Synthetic", sample_id=str(i)) for i in range(3)]),
                raw_measurements=[dict(row) for row in rows], measurements=rows)


def test_four_groups_keep_failure_and_miss_denominators():
    report = report_fixture()
    rows = module.summary(report)
    ours, dung = [r for r in rows if r["dataset"] == "All"]
    assert ours["dual_pass_percent"] == pytest.approx(200 / 3)
    assert dung["dual_pass_percent"] == pytest.approx(100 / 3)
    assert dung["n_finite"] == 2 and dung["n_requested"] == 3
    assert dung["final_k_mean"] == 9  # Tolerance miss is not filtered away.
    assert dung["total_ms_mean"] == 11  # Failed run time stays included.
    assert ours["total_ms_mean"] != ours["network_ms_mean"]
    assert ours["mse_mean"] == pytest.approx((1e-5 + 8e-5 + 1e-6) / 3)
    assert ours["max_squared_error_max"] == 6e-4  # Not root, not average peak.
    assert len(module.definitions(report["metadata"])["four_groups"]) == 4


def test_incomplete_method_case_set_is_rejected():
    report = report_fixture()
    report["measurements"].pop()
    with pytest.raises(ValueError):
        module.summary(report)


def test_table_uses_saved_final_metrics_without_modifying_source(tmp_path):
    import json
    report = tmp_path / "comparison.json"
    report.write_text(json.dumps(report_fixture()), encoding="utf-8")
    original = report.read_bytes()
    output = tmp_path / "figures"
    args = ["--report", str(report), "--output-dir", str(output), "--dpi", "72"]
    assert module.main(args) == 0
    assert report.read_bytes() == original
    assert (output / "core_metrics.png").stat().st_size > 1000
    assert (output / "core_metrics_by_dataset.png").stat().st_size > 1000
    saved = json.loads((output / "core_metrics.json").read_text(encoding="utf-8"))
    assert saved["summary"] == module.summary(report_fixture())
    with pytest.raises(SystemExit):
        module.main(args)
