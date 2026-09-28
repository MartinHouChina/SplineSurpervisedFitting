from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_dual_error_comparison import pair_measurements, summarize, write_tables, write_overall_table, plot_five_metrics


def _row(sample, mse=1e-5, peak=1e-4, status="ok"):
    return dict(dataset="UJI", sample_id=sample, method="ours", status=status,
                mse=mse, max_squared_error=peak, final_k=8, total_ms=2.,
                joint_pass=status == "ok" and mse <= 5e-5 and peak <= 5e-4)


def _report():
    raw = [_row("a"), _row("b", peak=.01), _row("c", status="failed")]
    repaired = deepcopy(raw)
    repaired[1].update(mse=2e-5, max_squared_error=2e-4, final_k=9, joint_pass=True)
    for a, b in zip(raw, repaired):
        b.update(raw_mse=a["mse"], raw_max_squared_error=a["max_squared_error"],
                 raw_final_k=a["final_k"], raw_ms=a["total_ms"], repair_ms=1., total_ms=3., added_knots_count=b["final_k"]-a["final_k"])
    return dict(metadata=dict(mse_tolerance=5e-5, max_squared_error_tolerance=5e-4,
                              max_internal_knots=32, methods=["ours"]), raw_measurements=raw, measurements=repaired)


def test_failures_remain_in_denominator_and_peak_is_squared():
    all_rows = summarize(_report())[0]
    assert all_rows["dataset"] == "All"
    assert all_rows["raw"]["dual_pass_percent"] == 100 / 3
    assert all_rows["repaired"]["dual_pass_percent"] == 200 / 3
    assert all_rows["raw"]["max_squared_error_max"] == .01
    assert all_rows["repaired"]["max_squared_error_max"] == .0002
    assert all_rows["raw"]["n_finite"] == 2
    assert all_rows["raw"]["n_requested"] == 3
    assert all_rows["repaired"]["total_ms_n"] == 3
    assert all_rows["shared_dual_feasible_n"] == 1
    assert all_rows["shared_dual_feasible_k_delta_mean"] == 0


def test_raw_repaired_identity_must_match_exactly():
    report = _report()
    report["measurements"][0]["sample_id"] = "unpaired"
    with pytest.raises(ValueError, match="identities must match"):
        pair_measurements(report)


def test_duplicate_identity_is_rejected():
    report = _report()
    report["measurements"].append(deepcopy(report["measurements"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        pair_measurements(report)


def test_peak_only_failure_cannot_be_reported_as_joint_pass():
    report = _report()
    report["raw_measurements"][1]["joint_pass"] = True
    with pytest.raises(ValueError, match="joint_pass"):
        pair_measurements(report)


def test_raw_statistic_mismatch_is_rejected():
    report = _report()
    report["measurements"][0]["raw_mse"] *= 2
    with pytest.raises(ValueError, match="raw_mse"):
        pair_measurements(report)


def test_missing_method_failure_row_cannot_be_dropped():
    report = _report()
    report["metadata"]["methods"].append("another_method")
    with pytest.raises(ValueError, match="all declared methods"):
        pair_measurements(report)


def test_missing_whole_case_is_rejected_when_inventory_is_declared():
    report = _report()
    report["metadata"]["cases"] = [{"dataset": "UJI", "sample_id": name} for name in ("a", "b", "c")]
    report["raw_measurements"] = report["raw_measurements"][:-1]
    report["measurements"] = report["measurements"][:-1]
    with pytest.raises(ValueError, match="metadata.cases exactly"):
        pair_measurements(report)


def test_table_and_metric_render_smoke(tmp_path):
    from PIL import Image
    report = _report()
    rows = summarize(report)
    definitions, table = write_tables(rows, report["metadata"], tmp_path, 50)
    overall = write_overall_table(rows, report["metadata"], tmp_path, 50)
    metrics = plot_five_metrics(rows, report["metadata"], tmp_path, 50)
    assert any("square-rooted" in text for text in definitions)
    assert "→" in (tmp_path / "dual_error_summary.md").read_text(encoding="utf-8")
    assert not any("cached" in text.lower() for text in definitions)
    for artifact in (table, overall, metrics):
        with Image.open(tmp_path / artifact["path"]) as picture:
            picture.verify()
