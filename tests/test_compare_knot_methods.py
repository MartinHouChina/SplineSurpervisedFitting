from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_knot_methods.py"


def _load_script():
    module_name = "test_compare_knot_methods_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


comparison = _load_script()


def _method(
    name: str,
    *,
    mse: float,
    count: int,
    elapsed_ms: float,
) -> object:
    return comparison.MethodMeasurement(
        method=name,
        mse=mse,
        final_internal_knot_count=count,
        elapsed_ms=elapsed_ms,
        timing_scope="test scope",
        parameterization="test",
        internal_knots=tuple(index / (count + 1) for index in range(1, count + 1)),
        threshold_satisfied=mse <= 2.5e-5,
    )


def _sample(sample_index: int, source_count: int, scale: float) -> object:
    return comparison.SampleComparison(
        sample_index=sample_index,
        source_internal_knot_count=source_count,
        canonical_internal_knot_count=source_count - 1,
        source_mse=0.0,
        mse_tolerance=2.5e-5,
        methods=(
            _method("ours", mse=1e-5 * scale, count=4, elapsed_ms=1.0 * scale),
            _method(
                "kang_sparse_2015_adaptation",
                mse=2e-5 * scale,
                count=5,
                elapsed_ms=10.0 * scale,
            ),
            _method(
                "yeh_feature_cdf_2020",
                mse=1.8e-5 * scale,
                count=5,
                elapsed_ms=5.0 * scale,
            ),
            _method(
                "uniform_gradient_pruning",
                mse=1.5e-5 * scale,
                count=4,
                elapsed_ms=100.0 * scale,
            ),
        ),
    )


def test_cli_defaults_cover_source_k_4_through_20_and_uniform_kmax_20() -> None:
    parser = comparison.build_parser()

    args = parser.parse_args(
        ["--checkpoint", "model.pt", "--output-dir", "comparison"]
    )

    assert args.min_knot_count == 4
    assert args.max_knot_count == 20
    assert args.samples_per_knot_count == 1
    assert args.max_internal_knots == 20
    assert args.sample_figures is False


def test_aggregate_uses_methodwise_means_and_pass_rates() -> None:
    samples = (_sample(1, 4, 1.0), _sample(2, 4, 2.0))

    aggregate = comparison.aggregate_by_source_count(samples)

    ours = aggregate[4]["methods"]["ours"]
    assert aggregate[4]["sample_count"] == 2
    assert ours["mse_mean"] == pytest.approx(1.5e-5)
    assert ours["elapsed_ms_mean"] == pytest.approx(1.5)
    assert ours["final_internal_knot_count_mean"] == pytest.approx(4.0)
    assert ours["pass_rate"] == pytest.approx(1.0)
    assert ours["canonical_count_bias"] == pytest.approx(1.0)
    assert ours["canonical_count_mae"] == pytest.approx(1.0)


def test_manifest_and_four_method_png_are_written(tmp_path: Path) -> None:
    samples = (_sample(1, 4, 1.0), _sample(2, 5, 1.1))

    png = tmp_path / "four_method_comparison.png"
    comparison.render_aggregate_figure(
        samples,
        png,
        mse_tolerance=2.5e-5,
        dpi=72,
    )
    json_path, csv_path = comparison._write_outputs(
        tmp_path,
        samples,
        metadata={"checkpoint": "model.pt"},
    )

    assert png.is_file() and png.stat().st_size > 0
    assert csv_path.is_file() and csv_path.stat().st_size > 0
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["mse_definition"].endswith("(no square root)")
    assert "network-forward latency only" in report["timing_warning"]
    assert len(report["samples"]) == 2
