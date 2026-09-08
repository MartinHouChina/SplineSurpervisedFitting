from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_stratified_three_metrics.py"


def _load_script():
    module_name = "test_stratified_three_metric_plot_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


plotter = _load_script()


def _interval(value: float) -> dict[str, object]:
    return {
        "mean": value,
        "median": value,
        "mean_ci95": {"low": 0.9 * value, "high": 1.1 * value},
        "median_ci95": {"low": 0.9 * value, "high": 1.1 * value},
    }


def test_render_report_reuses_saved_per_k_summary(tmp_path: Path) -> None:
    per_k = []
    for source_k in (4, 5):
        per_k.append(
            {
                "source_k": source_k,
                "canonical_knot_count": _interval(float(source_k - 1)),
                "methods": {
                    "ours": {
                        "fit_mse": _interval(2.3e-5),
                        "knot_count": _interval(float(source_k)),
                        "time_ms": _interval(25.0),
                    },
                    "hard": {
                        "fit_mse": _interval(2.0e-5),
                        "knot_count": _interval(float(source_k - 1)),
                        "time_ms": _interval(200.0),
                    },
                },
            }
        )
    report = {
        "mse_tolerance": 2.5e-5,
        "dataset": {"samples_per_source_k": 2},
        "ours_deployment": {
            "mode": "verified",
            "verified_config": {"parameterization_policy": "chord"},
        },
        "per_k_summary": per_k,
    }
    output = tmp_path / "three_metrics.png"

    plotter.render_report(report, output, dpi=50)

    assert output.is_file()
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
