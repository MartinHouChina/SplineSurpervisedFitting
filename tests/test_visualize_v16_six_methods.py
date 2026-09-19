"""Small CPU drawing fixtures; mocked latencies are not benchmark results."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest
import torch
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import visualize_v16_real_deployments as entry  # noqa: E402
import visualize_v16_ours_cases as ours_entry  # noqa: E402
from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points  # noqa: E402


@pytest.fixture
def cpu_case():
    parameters = torch.linspace(0, 1, 32, dtype=torch.float64)
    points = torch.stack((parameters, 0.18 * torch.sin(2 * torch.pi * parameters)), dim=-1)
    fit = refit_bspline_control_points(
        parameters, points, torch.tensor([0.3, 0.68], dtype=torch.float64),
        degree=3, smoothness_weight=0.0, control_ridge=0.0,
        interpolate_endpoints=True,
    )
    case = dict(
        dataset="TEST_FIXTURE", sample_id="curve-1", group_id="test-group",
        points=points.float(), reference=points, reference_grid=parameters,
    )
    return case, fit, parameters


@pytest.fixture
def harness(tmp_path, monkeypatch, cpu_case):
    """Exercise the real CLI/report/PNG code without training or dataset access."""
    case, fit, parameters = cpu_case
    checkpoint_path = tmp_path / "test-fixture.pt"
    checkpoint_path.write_bytes(b"MOCK CHECKPOINT FOR ORCHESTRATION TEST ONLY")
    args = entry.parser().parse_args([
        "--checkpoint", str(checkpoint_path), "--output-dir", str(tmp_path / "figures"),
        "--device", "cpu", "--torch-num-threads", "1", "--dpi", "50",
        "--force-diagnostic",
    ])
    config = {"structure_mode": "candidate_pruning_one_shot", "max_internal_knots": 16}
    checkpoint = dict(
        objective_version=entry.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        model_config=config, epoch=1, stage="joint",
    )
    model = SimpleNamespace(degree=3)
    model.to = lambda device: model
    model.eval = lambda: model
    monkeypatch.setattr(entry.torch, "load", lambda *a, **kw: checkpoint)
    qualification_calls = []

    def qualify(*args, **kwargs):
        qualification_calls.append(kwargs)
        return False

    monkeypatch.setattr(entry, "validate_checkpoint_for_visualization", qualify)
    monkeypatch.setattr(entry, "assess_v16_checkpoint", lambda *a, **kw: {
        "formal_reporting_eligible": True, "reasons": [],
    })
    monkeypatch.setattr(entry, "build_model_from_checkpoint", lambda cp: (model, config, None))
    cases = [case]
    monkeypatch.setattr(entry, "prepare_cases", lambda *a: (
        cases, [{"dataset": "TEST_FIXTURE", "selected_count": len(cases)}],
    ))
    calls = []

    def ours(model, points, device, supplied_args, **kwargs):
        calls.append(("ours", supplied_args, kwargs))
        return fit, parameters, 1.75, 1.25, {"test_fixture": True}

    def numerical(method, points, supplied_args, **kwargs):
        calls.append((method, supplied_args, kwargs))
        assert points.dtype == torch.float64
        assert points.device.type == "cpu"
        return fit, parameters, 3.5, None, {"test_fixture": True}

    monkeypatch.setattr(entry, "measure_ours", ours)
    monkeypatch.setattr(entry, "measure_numerical_baseline", numerical)
    figures = []
    original_savefig = Figure.savefig

    def capture(self, *args, **kwargs):
        figures.append(self)
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    return SimpleNamespace(
        args=args, model=model, case=case, cases=cases, fit=fit, parameters=parameters,
        calls=calls, figures=figures, numerical=numerical, ours=ours,
        qualification_calls=qualification_calls,
    )


def assert_png(path):
    data = Path(path).read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", data[16:24])
    assert width >= 300 and height >= 250


def test_legacy_four_method_cli_default_remains_unchanged():
    args = entry.parser().parse_args([])
    assert args.method_set == "legacy"
    assert args.ours_only is False
    assert args.force_diagnostic is False
    assert entry.METHODS == (
        "ours", "kang_sparse_2015_adaptation", "yeh_feature_cdf_2020",
        "uniform_gradient_pruning",
    )


@pytest.mark.parametrize("ours_only", [False, True])
def test_industrial_case_figures_and_json_disclose_procedural_origin(harness, ours_only):
    harness.case.update(dataset="IndustrialOffset", source_kind="procedural_cad_offset",
                        dataset_label="IndustrialOffset (procedural CAD; not measured)",
                        source_note="Procedurally generated; no ground-truth B-spline knots.")
    harness.args.method_set = "published"
    harness.args.ours_only = ours_only
    report = entry.run(harness.args)
    assert report["records"][0]["source_kind"] == "procedural_cad_offset"
    assert "not measured" in report["records"][0]["dataset_label"]
    titles = [text.get_text() for figure in harness.figures for text in figure.texts]
    assert any("not measured" in title for title in titles)


@pytest.mark.parametrize("method_set, expected_grid", [("legacy", (2, 2)), ("published", (2, 3))])
def test_selected_case_methods_render_complete_grid_and_record_each_fit(harness, method_set, expected_grid):
    harness.args.method_set = method_set
    report = entry.run(harness.args)
    expected = entry.PUBLISHED_METHODS if method_set == "published" else entry.METHODS
    assert report["method_order"] == list(expected)
    assert [call[0] for call in harness.calls] == list(expected)
    assert report["visualization_mode"] == ("six_method" if method_set == "published" else "four_method")
    assert report["diagnostic_not_final"]
    assert report["checkpoint_qualification"]["formal_reporting_eligible"]
    assert harness.qualification_calls[0]["allow_unqualified_diagnostic"] is False
    assert len(report["records"]) == 1
    record = report["records"][0]
    assert [result["method"] for result in record["methods"]] == list(expected)
    assert all(result["status"] == "ok" for result in record["methods"])
    assert all(result["final_k"] == 2 for result in record["methods"])
    assert_png(record["image"])
    assert len(harness.figures) == 1
    figure = harness.figures[0]
    assert len(figure.axes) == len(expected)
    for axis, method in zip(figure.axes, expected):
        grid = axis.get_subplotspec().get_gridspec()
        assert (grid.nrows, grid.ncols) == expected_grid
        assert entry.PLOT_LABELS[method] in axis.get_title()
        assert any(line.get_label() == "Final B-spline" for line in axis.lines)
    assert any("DIAGNOSTIC NOT FINAL" in text.get_text() for text in figure.texts)
    saved = json.loads((harness.args.output_dir / "deployment_visualizations.json").read_text(encoding="utf-8"))
    assert saved == report


@pytest.mark.parametrize("failed_method", ["ours", "luo_linf_de_2022_adaptation"])
def test_failed_method_is_preserved_in_six_panel_png_and_json(harness, monkeypatch, failed_method):
    harness.args.method_set = "published"
    if failed_method == "ours":
        def fail(*args, **kwargs):
            raise RuntimeError("deliberate test-fixture Ours failure")
        monkeypatch.setattr(entry, "measure_ours", fail)
    else:
        def numerical(method, *args, **kwargs):
            if method == failed_method:
                raise ValueError("deliberate test-fixture baseline failure")
            return harness.numerical(method, *args, **kwargs)
        monkeypatch.setattr(entry, "measure_numerical_baseline", numerical)
    report = entry.run(harness.args)
    record = report["records"][0]
    assert len(record["methods"]) == 6
    failed = next(result for result in record["methods"] if result["method"] == failed_method)
    assert failed["status"] == "failed"
    assert failed["fit_pass"] is False
    assert failed["reference_pass"] is False
    assert failed["mse"] is None and failed["final_k"] is None
    assert failed["control_points_normalized"] is None
    assert "deliberate test-fixture" in failed["error"]
    assert sum(result["status"] == "ok" for result in record["methods"]) == 5
    assert_png(record["image"])
    figure = harness.figures[0]
    assert len(figure.axes) == 6
    axis = figure.axes[list(entry.PUBLISHED_METHODS).index(failed_method)]
    assert "FAILED" in axis.get_title()
    assert "deliberate test-fixture" in axis.get_title()
    assert any(collection.get_label() == "Input sampled points" for collection in axis.collections)
    assert not any(line.get_label() == "Final B-spline" for line in axis.lines)


def test_all_published_baseline_arguments_and_repeat_budget_reach_shared_measurement(harness):
    # Distinct values catch accidentally forwarded defaults and swapped options.
    options = {
        "mse_tolerance": 4e-5, "max_internal_knots": 15, "gradient_steps": 7,
        "paper_initial_knots": 14, "paper_admm_iterations": 23,
        "paper_lambda_bisections": 3, "paper_relocation_iterations": 2,
        "park_shape_weight": 0.7, "liang_dense_knots": 13,
        "liang_initial_knots": 3, "liang_curvature_weight": 0.4,
        "liang_feature_samples": 129, "dung_max_error": 0.03,
        "dung_scan_intervals": 4, "dung_optimization_iterations": 5,
        "luo_eta": 0.6, "luo_de_population": 7, "luo_de_iterations": 9,
        "luo_seed": 321, "end_to_end_repeats": 4,
    }
    cli = [value for name, value in options.items() for value in (
        "--" + name.replace("_", "-"), str(value),
    )]
    args = entry.parser().parse_args(cli)
    for method in entry.PUBLISHED_METHODS[1:]:
        result = entry._run_method(
            method, model=harness.model, points=harness.case["points"],
            case=harness.case, device=torch.device("cpu"), args=args,
            objective_version=entry.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        )
        assert result["method"] == method
        assert result["network_ms"] is None
        assert result["total_ms"] == 3.5
        captured_method, captured_args, kwargs = harness.calls[-1]
        assert captured_method == method
        assert captured_args is args
        assert kwargs == {"degree": 3}
        assert {name: getattr(captured_args, name) for name in options} == options


def test_ours_wrapper_keeps_explicit_options_and_ours_only_wins_over_published(harness):
    supplied = [
        "--checkpoint=" + str(harness.args.checkpoint),
        "--output-dir=" + str(harness.args.output_dir),
        "--method-set", "published", "--device", "cpu", "--dpi", "50",
        "--force-diagnostic",
    ]
    expanded = ours_entry.with_ours_defaults(supplied)
    assert expanded[:len(supplied)] == supplied
    assert expanded.count("--ours-only") == 1
    assert str(ours_entry.DEFAULT_CHECKPOINT) not in expanded
    report = ours_entry.main(supplied)
    assert report["method_order"] == ["ours"]
    assert report["visualization_mode"] == "ours_only"
    assert [call[0] for call in harness.calls] == ["ours"]
    assert_png(report["records"][0]["image"])
    assert_png(report["overview_image"])
    assert len(harness.figures[0].axes) == 2  # geometry plus internal-knot strip


def test_ours_only_failed_case_does_not_abort_later_case_or_overview(harness, monkeypatch):
    harness.args.ours_only = True
    harness.cases.append(dict(harness.case, sample_id="curve-2"))
    call_count = 0

    def first_failure(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("deliberate first-case failure")
        return harness.ours(*args, **kwargs)

    monkeypatch.setattr(entry, "measure_ours", first_failure)
    report = entry.run(harness.args)
    assert len(report["records"]) == 2
    assert [record["methods"][0]["status"] for record in report["records"]] == ["failed", "ok"]
    for record in report["records"]:
        assert_png(record["image"])
    assert_png(report["overview_image"])
    overview = harness.figures[-1]
    assert len(overview.axes) == 2
    assert "FAILED" in overview.axes[0].get_title()
    assert "MSE=" in overview.axes[1].get_title()
