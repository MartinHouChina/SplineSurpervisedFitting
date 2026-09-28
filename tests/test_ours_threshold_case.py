import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("threshold_case", Path(__file__).resolve().parents[1] / "scripts/plot_ours_threshold_case.py")
module = importlib.util.module_from_spec(spec)
# The plotting script is normally imported with scripts on sys.path.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec.loader.exec_module(module)


def report_fixture():
    cases = [dict(dataset="USGS", sample_id=str(i)) for i in range(2)]
    thresholds = [i / 100000 for i in range(1, 11)]
    rows = [dict(dataset="USGS", sample_id=str(j), mse_tolerance=t,
                 status="ok", joint_pass=j == 0 or i != 4, fit_pass=True,
                 mse=t / 2, max_squared_error=t * 2, final_k=20 - i, total_ms=5)
            for i, t in enumerate(thresholds) for j in range(2)]
    return dict(complete=True, metadata=dict(cases=cases, mse_tolerances=thresholds, peak_ratio=10), measurements=rows)


def test_only_complete_ten_tier_successful_cases_are_eligible():
    report = report_fixture()
    chosen = module.candidates(report)
    assert len(chosen) == 1
    assert chosen[0][0] == ("USGS", "0")
    assert [row["mse_tolerance"] for row in chosen[0][1]] == report["metadata"]["mse_tolerances"]
    assert len(report["measurements"]) == 20


def test_panel_order_is_loose_to_strict_without_changing_source():
    rows = module.candidates(report_fixture())[0][1]
    original = list(rows)
    displayed = module.panel_order(rows)
    assert [r["mse_tolerance"] for r in displayed] == [i / 100000 for i in range(10, 0, -1)]
    assert rows == original
    assert displayed[0] is rows[-1] and displayed[-1] is rows[0]


@pytest.mark.parametrize("damage", ["incomplete", "missing_row", "duplicate_row", "nine_tiers"])
def test_invalid_reports_rejected(damage):
    report = report_fixture()
    if damage == "incomplete":
        report["complete"] = False
    elif damage == "missing_row":
        report["measurements"].pop()
    elif damage == "duplicate_row":
        report["measurements"].append(dict(report["measurements"][0]))
    else:
        report["metadata"]["mse_tolerances"].pop()
        report["measurements"] = report["measurements"][:-2]
    with pytest.raises(ValueError):
        module.candidates(report)
