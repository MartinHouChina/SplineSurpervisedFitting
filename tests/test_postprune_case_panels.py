from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plot_postprune_case_panels as plots


def report_fixture():
    rows = []
    for source in plots.SOURCES:
        for sample in range(6):
            for method in plots.METHODS:
                passed = sample != 0
                rows.append(dict(dataset=source, sample_id=str(sample), method=method,
                                 group_id=str(sample // 2), status="ok",
                                 mse=1e-5 * sample if passed else 9e-5,
                                 max_squared_error=5e-5 * sample if passed else 9e-4,
                                 joint_pass=passed, final_k=5, total_ms=7.,
                                 geometry_artifact=dict(path=f"{source}_{sample}_{method}.json")))
    return dict(metadata=dict(methods=list(plots.METHODS), mse_tolerance=5e-5,
                              max_squared_error_tolerance=5e-4),
                raw_measurements=deepcopy(rows), measurements=rows)


def fake_geometry(root, row):
    points = np.array([[0., 0.], [.5, .8], [1., 0.]])
    return {"dataset": row["dataset"], "sample_id": row["sample_id"]}, {
        "input_points_normalized": points, "source_parameters": np.array([0., .5, 1.]),
    }, {"dense_curve_normalized": points + [0., .01],
        "control_points_normalized": points + [0., .2],
        "knot_positions_normalized": points[1:2] + [0., .01],
        "internal_knots": np.array([.25, .5, .75]),
        "sample_parameters": np.array([0., .5, 1.])}


def test_selection_random_comparison_independent_of_quality_and_no_report_mutation():
    report = report_fixture()
    original = deepcopy(report)
    _, comparison, showcase = plots.select_cases(report)
    assert all(len(values) == 5 and "0" not in values for values in showcase.values())
    assert len(comparison) == 5
    assert report == original
    changed = deepcopy(report)
    for stage in ("raw_measurements", "measurements"):
        for row in changed[stage]:
            row.update(mse=1e-9, max_squared_error=1e-8, joint_pass=True)
    assert plots.select_cases(changed)[1] == comparison


@pytest.mark.parametrize("damage", ["source", "method", "insufficient", "unpaired"])
def test_incomplete_inputs_rejected(damage):
    report = report_fixture()
    if damage == "source":
        for stage in ("raw_measurements", "measurements"):
            report[stage] = [row for row in report[stage] if row["dataset"] != "UJI"]
    elif damage == "method":
        report["metadata"]["methods"] = list(reversed(plots.METHODS))
    elif damage == "insufficient":
        for stage in ("raw_measurements", "measurements"):
            report[stage] = [row for row in report[stage] if row["sample_id"] not in {"0", "1"}]
    else:
        report["measurements"].pop()
    with pytest.raises(ValueError):
        plots.select_cases(report)


def test_geometry_bounds_include_far_control_vertices():
    loaded = fake_geometry(None, dict(dataset="UJI", sample_id="x"))
    loaded[2]["control_points_normalized"][0] = [500., -350.]
    low, high = plots.geometry_bounds([loaded])
    assert high[0] > 500 and low[1] < -350


def test_clean_panels_single_legend_no_metrics_and_actual_geometry(tmp_path, monkeypatch):
    from matplotlib.text import Text
    report = report_fixture()
    index, comparison, showcase = plots.select_cases(report)
    loaded = {key: fake_geometry(tmp_path, row) for key, row in index.items()}
    figures = []
    def capture(figure, path, dpi):
        figures.append(figure)
        return dict(path=path.name, sha256="captured")
    monkeypatch.setattr(plots, "_save", capture)
    try:
        plots.draw_sheet(loaded, comparison, plots.METHODS, tmp_path / "six.png", 40, True)
        plots.draw_sheet(loaded, showcase, plots.METHODS, tmp_path / "ours.png", 40, False)
        for figure, count in zip(figures, (30, 25)):
            text = "\n".join(item.get_text() for item in figure.findobj(match=Text))
            for excluded in ("MSE", "MaxSE", "PASS", "MISS", "K=", "ms", "Total", "adaptation", "post-pruning"):
                assert excluded not in text
            assert len(figure.legends) == 1
            assert len(figure.axes) == count * 2
            assert len(figure.axes[0].lines) == 3
            assert len(figure.axes[1].collections[0].get_offsets()) == 3
            assert "Control vertices / polygon" in text
        six_labels = [item.get_text() for item in figures[0].texts]
        assert all(six_labels.count(label) == 1 for label in plots.PAPER_LABELS.values())
        assert "Ours" not in [item.get_text() for item in figures[1].texts]
    finally:
        for figure in figures:
            plots.plt.close(figure)


def test_render_exports_exact_data_full_metrics_and_refuses_overwrite(tmp_path, monkeypatch):
    report = report_fixture()
    source = tmp_path / "comparison.json"
    source.write_text(json.dumps(report), encoding="utf-8")
    before = source.read_bytes()
    monkeypatch.setattr(plots, "_geometry", fake_geometry)
    monkeypatch.setattr(plots, "draw_sheet", lambda loaded, selection, methods, path, dpi, comparison:
                        dict(path=path.name, sha256="figure_stub"))
    target = tmp_path / "figures"
    manifest = plots.render(source, target, dpi=40)
    assert source.read_bytes() == before
    assert len(manifest["panels"]) == 55
    for panel in manifest["panels"]:
        assert panel["measurement"]["total_ms"] == 7
        assert "mse" in panel["measurement"] and "max_squared_error" in panel["measurement"]
        with np.load(target / panel["exported_geometry"]["path"], allow_pickle=False) as saved:
            case = fake_geometry(None, panel)
            assert np.array_equal(saved["fit__control_points_normalized"], case[2]["control_points_normalized"])
            assert np.array_equal(saved["source__input_points_normalized"], case[1]["input_points_normalized"])
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        plots.render(source, target, dpi=40)


def test_paired_source_tamper_rejected_before_writing(tmp_path, monkeypatch):
    source = tmp_path / "comparison.json"
    source.write_text(json.dumps(report_fixture()), encoding="utf-8")
    def changed(root, row):
        case, values, fitted = fake_geometry(root, row)
        if row["method"].startswith("luo"):
            values["input_points_normalized"] += 1
        return case, values, fitted
    monkeypatch.setattr(plots, "_geometry", changed)
    with pytest.raises(ValueError, match="Paired source geometry differs"):
        plots.render(source, tmp_path / "figures", dpi=40)
    assert not (tmp_path / "figures").exists()


def test_explicit_comparison_override_changes_only_requested_row_and_records_selection(tmp_path, monkeypatch):
    report = report_fixture()
    original = deepcopy(report)
    _, old_comparison, old_showcase = plots.select_cases(report)
    chosen = "1" if old_comparison["USGS"] != "1" else "2"
    _, comparison, showcase = plots.select_cases(report, comparison_overrides={"USGS": chosen})
    assert comparison == {**old_comparison, "USGS": chosen}
    assert showcase == old_showcase and report == original
    for bad in ({"USGS": "nonexistent"}, {"unknown_source": "1"}):
        with pytest.raises(ValueError):
            plots.select_cases(report, comparison_overrides=bad)
    source = tmp_path / "comparison.json"
    source.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(plots, "_geometry", fake_geometry)
    monkeypatch.setattr(plots, "draw_sheet", lambda *args: {"path": "stub"})
    output = tmp_path / "rendered"
    manifest = plots.render(source, output, dpi=40, comparison_overrides={"USGS": chosen})
    assert manifest["comparison_overrides"] == {"USGS": chosen}
    assert "user-selected" in manifest["comparison_selection_rule"]
    panels = [p for p in manifest["panels"] if p["sheet"] == "six_methods_6x5" and p["dataset"] == "USGS"]
    assert len(panels) == 6 and {p["sample_id"] for p in panels} == {chosen}
    assert (output / "six_methods_case_values.md").exists()
    assert json.loads(source.read_text(encoding="utf-8")) == original
