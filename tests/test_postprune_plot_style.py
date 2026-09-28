"""Paper styling changes presentation, never cohort, metrics or saved audits."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plot_dual_error_comparison as plots


def _report():
    raw = []
    final = []
    for method in plots.LABELS:
        for sample in ("a", "b"):
            failed = method == "luo_linf_de_2022_adaptation" and sample == "a"
            row = dict(dataset="UJI", sample_id=sample, method=method,
                       status="failed" if failed else "ok", mse=None if failed else 1e-5,
                       max_squared_error=None if failed else 1e-4,
                       joint_pass=not failed, final_k=8, total_ms=2.,
                       geometry_artifact={"path": f"raw/{method}_{sample}.json"})
            raw.append(row)
            after = deepcopy(row)
            after.update(raw_mse=row["mse"], raw_max_squared_error=row["max_squared_error"],
                         raw_final_k=8, raw_ms=2., total_ms=3., repair_ms=1., added_knots_count=0,
                         pruning_ms=0., removed_knots_count=0,
                         geometry_artifact={"path": f"repaired/{method}_{sample}.json"})
            if method == "ours":
                after.update(final_k=3, mse=2e-5, max_squared_error=2e-4,
                             total_ms=7.5, pruning_ms=4.5, removed_knots_count=5)
            final.append(after)
    return dict(metadata=dict(mse_tolerance=5e-5, max_squared_error_tolerance=5e-4,
                              max_internal_knots=32, methods=list(plots.LABELS),
                              cases=[dict(dataset="UJI", sample_id=sample) for sample in ("a", "b")],
                              ours_post_pruning=dict(enabled=True, implementation="measured greedy deletion",
                                                     timing="original + repair + pruning")),
                raw_measurements=raw, measurements=final)


@pytest.fixture
def captured(monkeypatch):
    figures = {}

    def save(figure, path, dpi):
        figures[path.name] = figure
        return {"path": path.name, "sha256": "unit-test-captured"}

    monkeypatch.setattr(plots, "_save", save)
    yield figures
    for figure in figures.values():
        plots.plt.close(figure)


def _text(figure):
    from matplotlib.text import Text
    labels = [item.get_text() for item in figure.findobj(match=Text)]
    labels += [cell.get_text().get_text() for axis in figure.axes for table in axis.tables
               for cell in table.get_celld().values()]
    return "\n".join(labels)


def test_postprune_statistics_preserve_raw_and_failures():
    report = _report()
    before = deepcopy(report)
    summary = plots.summarize(report)
    ours = next(row for row in summary if row["method"] == "ours" and row["dataset"] == "All")
    assert ours["raw"]["final_k_mean"] == 8
    assert ours["repaired"]["final_k_mean"] == 3
    assert ours["repaired"]["removed_knots_count_mean"] == 5
    assert ours["repaired"]["pruning_ms_mean"] == 4.5
    assert ours["repaired"]["total_ms_mean"] == 7.5
    assert ours["shared_dual_feasible_k_delta_mean"] == -5
    luo = next(row for row in summary if row["method"].startswith("luo") and row["dataset"] == "All")
    assert luo["repaired"]["n_requested"] == 2
    assert luo["repaired"]["n_failed"] == 1
    assert luo["repaired"]["dual_pass_percent"] == 50
    assert report == before


def test_paper_tables_keep_full_audit_outside_figures(tmp_path, captured):
    report = _report()
    rows = plots.summarize(report)
    definitions, _ = plots.write_tables(rows, report["metadata"], tmp_path, 40, paper_style=True)
    plots.write_overall_table(rows, report["metadata"], tmp_path, 40, paper_style=True)
    plots.plot_five_metrics(rows, report["metadata"], tmp_path, 40, paper_style=True)
    for figure in captured.values():
        text = _text(figure)
        assert "Ours" in text
        for excluded in ("adaptation", "hybrid", "one-shot", "common insertion", "denominator", "metadata"):
            assert excluded not in text
        assert len(figure.texts) == 1  # Title only, no footer or watermark.
    assert any("post-pruning" in note for note in definitions)
    markdown = (tmp_path / "dual_error_summary.md").read_text(encoding="utf-8")
    assert "repository adaptations" in markdown
    assert "measured greedy deletion" in markdown
    assert "Mean removed K" in markdown
    assert "Mean prune ms" in markdown
    assert "8.00 → 3.00" in markdown
    assert "Ours V16" in markdown


def test_default_postprune_titles_do_not_claim_insertion_only(tmp_path, captured):
    report = _report()
    rows = plots.summarize(report)
    plots.write_tables(rows, report["metadata"], tmp_path, 40)
    plots.write_overall_table(rows, report["metadata"], tmp_path, 40)
    plots.plot_five_metrics(rows, report["metadata"], tmp_path, 40)
    for figure in captured.values():
        assert "common insertion" not in figure._suptitle.get_text()
        assert "post-pruning" in _text(figure)


def test_paper_contact_sheets_preserve_geometry_selection_failures_and_time(tmp_path, captured, monkeypatch):
    report = _report()
    before = deepcopy(report)
    points = np.array([[0., 0.], [.5, .75], [1., 0.]])

    def geometry(root, row):
        return {}, {"input_points_normalized": points}, {
            "dense_curve_normalized": points,
            "control_points_normalized": points,
            "knot_positions_normalized": points[1:2],
            "internal_knots": np.linspace(.1, .9, row["final_k"]),
        }

    monkeypatch.setattr(plots, "_geometry", geometry)
    # Choose a stable seed selecting the source case with the explicit failure.
    seed = next(seed for seed in range(100) if plots._source_random(seed, "UJI", {"a", "b"}) == "a")
    ordinary = plots.plot_contact_sheets(report, tmp_path, tmp_path, 40, seed)
    for figure in captured.values():
        plots.plt.close(figure)
    captured.clear()
    paper = plots.plot_contact_sheets(report, tmp_path, tmp_path, 40, seed, paper_style=True)
    assert paper["selected_sample_ids"] == ordinary["selected_sample_ids"]
    assert paper["panels"] == ordinary["panels"]
    figure = captured["six_methods_repaired_6x5.png"]
    text = _text(figure)
    assert figure._suptitle.get_text() == "Six-method comparison"
    assert "FAILED" in text
    assert "K=3 MSE=2.00e-05 MaxSE=2.00e-04" in text
    assert "Total=7.5 ms" in text
    assert "UJI" in text
    assert paper["selected_sample_ids"] == {"UJI": "a"}
    assert "Data" in text and "Control polygon" in text and "Internal knots" in text
    assert len(figure.texts) == 1
    for excluded in ("adaptation", "hybrid", "one-shot", "post-pruning", "common insertion", "Same seeded"):
        assert excluded not in text
    assert len(figure.axes[0].lines) == 3  # Source, fit and control polygon.
    assert len(figure.axes[1].collections[0].get_offsets()) == 3
    assert report == before


def test_cli_paper_flag_keeps_report_metadata_and_measured_totals(tmp_path, captured):
    report = _report()
    source = tmp_path / "report.json"
    source.write_text(json.dumps(report), encoding="utf-8")
    assert plots.main(["--report", str(source), "--paper-style", "--no-contact-sheets", "--dpi", "40"]) == 0
    result = json.loads((tmp_path / "figures" / "dual_error_metrics.json").read_text(encoding="utf-8"))
    assert result["figure_style"] == "paper"
    assert result["metadata"] == report["metadata"]
    assert result["summary"] == plots.summarize(report)


def test_paper_style_does_not_bypass_cohort_audit(tmp_path, captured):
    report = _report()
    report["raw_measurements"] = report["raw_measurements"][:-1]
    with pytest.raises(ValueError, match="identities must match"):
        plots.plot_contact_sheets(report, tmp_path, tmp_path, 40, 0, paper_style=True)
