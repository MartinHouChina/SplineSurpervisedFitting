from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_knot_count_strata.py"


def _load_script():
    module_name = "test_stratified_knot_count_evaluation_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


evaluation = _load_script()


class _StaticOneShotModel(torch.nn.Module):
    degree = 3

    def __init__(self, output: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.output = output

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = points.shape[0]
        return {
            key: value.expand(batch_size, *value.shape[1:]).clone()
            for key, value in self.output.items()
        }


def _row(
    *,
    source_k: int,
    canonical_k: int,
    ours_k: int,
    hard_k: int,
    ours_mse: float,
    hard_mse: float,
    ours_time: float,
    hard_time: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "source_knot_count": source_k,
        "canonical_knot_count": canonical_k,
        "parameter_rmse": 0.02,
        "ours_knot_count": ours_k,
        "ours_count_bias_canonical": ours_k - canonical_k,
        "ours_count_bias_source": ours_k - source_k,
        "ours_fit_mse": ours_mse,
        "ours_threshold_satisfied": float(ours_mse <= 2.5e-5),
        "ours_time_median_ms": ours_time,
        "hard_knot_count": hard_k,
        "hard_count_bias_canonical": hard_k - canonical_k,
        "hard_count_bias_source": hard_k - source_k,
        "hard_fit_mse": hard_mse,
        "hard_threshold_satisfied": float(hard_mse <= 2.5e-5),
        "hard_time_median_ms": hard_time,
    }
    for method, predicted in (("ours", ours_k), ("hard", hard_k)):
        matched = min(predicted, canonical_k)
        row[f"{method}_match_0.010_matched"] = matched
        row[f"{method}_match_0.010_predicted"] = predicted
        row[f"{method}_match_0.010_target"] = canonical_k
        row[f"{method}_match_0.010_error_sum"] = 0.001 * matched
    return row


def test_bootstrap_interval_is_deterministic_and_finite() -> None:
    first = evaluation.bootstrap_interval(
        [1.0, 2.0, 3.0, 4.0],
        statistic="median",
        replicates=100,
        seed=7,
    )
    second = evaluation.bootstrap_interval(
        [1.0, 2.0, 3.0, 4.0],
        statistic="median",
        replicates=100,
        seed=7,
    )

    assert first == second
    assert first["low"] <= 2.5 <= first["high"]


def test_deploy_ours_dispatches_learned_and_verified_paths() -> None:
    dtype = torch.float64
    parameters = torch.linspace(0.0, 1.0, 33, dtype=dtype)
    points = torch.stack([parameters, parameters.square()], dim=-1)
    proposal = torch.tensor([[0.30, 0.70]], dtype=dtype)
    model = _StaticOneShotModel(
        {
            "params": parameters.unsqueeze(0),
            "internal_knots": proposal,
            "proposal_internal_knots": proposal,
            "deployment_internal_knots": proposal,
            "keep_probability": torch.tensor([[0.90, 0.10]], dtype=dtype),
            "learned_keep_mask": torch.tensor([[True, False]]),
        }
    )

    learned = evaluation.deploy_ours(
        model,
        points.unsqueeze(0),
        points,
        deployment="learned",
        mse_tolerance=2.5e-5,
        smoothness_weight=0.0,
        control_ridge=0.0,
    )
    verified = evaluation.deploy_ours(
        model,
        points.unsqueeze(0),
        points,
        deployment="verified",
        mse_tolerance=2.5e-5,
        smoothness_weight=0.0,
        control_ridge=0.0,
        verified_compact=False,
        verified_hard_fallback=True,
        verified_residual_fallback=True,
        verified_max_residual_insertions=2,
        verified_residual_min_gap=1e-3,
        verified_refit_device="cpu",
    )

    assert learned.verified_repair is None
    assert verified.verified_repair is not None
    assert verified.verified_repair.final_source == "learned_verified"
    assert verified.verified_repair.direct_refit_count == 1
    torch.testing.assert_close(
        learned.final_internal_knots,
        verified.final_internal_knots,
    )


def test_summary_uses_canonical_count_and_micro_matching() -> None:
    rows = [
        _row(
            source_k=4,
            canonical_k=2,
            ours_k=3,
            hard_k=2,
            ours_mse=3e-5,
            hard_mse=2e-5,
            ours_time=20.0,
            hard_time=200.0,
        ),
        _row(
            source_k=4,
            canonical_k=3,
            ours_k=4,
            hard_k=3,
            ours_mse=2e-5,
            hard_mse=1e-5,
            ours_time=22.0,
            hard_time=220.0,
        ),
    ]

    summary = evaluation.summarize_rows(
        rows,
        source_k=4,
        knot_tolerances=(0.01,),
        bootstrap_replicates=100,
        bootstrap_seed=3,
    )

    assert summary["canonical_knot_count"]["mean"] == pytest.approx(2.5)
    assert summary["methods"]["ours"]["count_bias_canonical"]["mean"] == pytest.approx(
        1.0
    )
    assert summary["methods"]["hard"]["count_exact_canonical_rate"] == 1.0
    assert summary["methods"]["ours"]["pass_rate"]["mean"] == 0.5
    assert summary["methods"]["ours"]["deployment_mode"] == "learned"
    assert summary["methods"]["ours"]["time_scope"] == "network_forward_only"
    assert summary["methods"]["hard"]["time_scope"] == "greedy_search_only"
    assert summary["paired"][
        "hard_search_over_ours_network_forward_time_ratio"
    ]["median"] == pytest.approx(10.0)
    ours_match = summary["methods"]["ours"]["matching_warped_to_canonical"]["0.010"]
    assert ours_match["precision"] == pytest.approx(5.0 / 7.0)
    assert ours_match["recall"] == 1.0


def test_complete_ours_pipeline_timing_is_diagnostic_not_primary() -> None:
    rows = [
        _row(
            source_k=4,
            canonical_k=3,
            ours_k=3,
            hard_k=3,
            ours_mse=2e-5,
            hard_mse=2e-5,
            ours_time=11.0,
            hard_time=210.0,
        ),
        _row(
            source_k=4,
            canonical_k=3,
            ours_k=3,
            hard_k=3,
            ours_mse=2e-5,
            hard_mse=2e-5,
            ours_time=13.0,
            hard_time=220.0,
        ),
    ]
    rows[0]["ours_quality_pipeline_time_median_ms"] = 70.0
    rows[1]["ours_quality_pipeline_time_median_ms"] = 90.0

    summary = evaluation.summarize_rows(
        rows,
        source_k=4,
        knot_tolerances=(0.01,),
        bootstrap_replicates=20,
        bootstrap_seed=31,
    )
    ours = summary["methods"]["ours"]

    assert ours["time_ms"]["median"] == pytest.approx(12.0)
    assert ours["quality_pipeline_time_ms"]["median"] == pytest.approx(80.0)


def test_verified_summary_records_adaptive_repair_and_dynamic_insertion() -> None:
    rows = [
        _row(
            source_k=4,
            canonical_k=3,
            ours_k=3,
            hard_k=3,
            ours_mse=2e-5,
            hard_mse=2e-5,
            ours_time=25.0,
            hard_time=200.0,
        ),
        _row(
            source_k=4,
            canonical_k=3,
            ours_k=10,
            hard_k=4,
            ours_mse=2.4e-5,
            hard_mse=2.2e-5,
            ours_time=80.0,
            hard_time=210.0,
        ),
    ]
    telemetry = (
        {
            "fallback_used": False,
            "hard_fallback_used": False,
            "cleanup_used": False,
            "residual_fallback_used": False,
            "inserted_knot_count": 0,
            "residual_refit_count": 0,
            "direct_refit_count": 1,
            "fit_evaluation_count": 1,
            "prefix_count_evaluations": 0,
            "final_source": "learned_verified",
        },
        {
            "fallback_used": True,
            "hard_fallback_used": False,
            "cleanup_used": False,
            "residual_fallback_used": True,
            "inserted_knot_count": 7,
            "residual_refit_count": 7,
            "direct_refit_count": 10,
            "fit_evaluation_count": 10,
            "prefix_count_evaluations": 2,
            "final_source": "residual_guided_insertion",
        },
    )
    for row, values in zip(rows, telemetry, strict=True):
        row["ours_deployment_mode"] = "verified"
        for name, value in values.items():
            row[f"ours_verified_{name}"] = value

    summary = evaluation.summarize_rows(
        rows,
        source_k=4,
        knot_tolerances=(0.01,),
        bootstrap_replicates=50,
        bootstrap_seed=13,
    )
    verified = summary["methods"]["ours"]["verified_adaptive"]

    assert summary["methods"]["ours"]["deployment_mode"] == "verified"
    assert verified["fallback_used"]["mean"] == 0.5
    assert verified["residual_fallback_used"]["mean"] == 0.5
    assert verified["inserted_knot_count"]["mean"] == 3.5
    assert verified["final_source_histogram"] == {
        "learned_verified": 1,
        "residual_guided_insertion": 1,
    }

    markdown = evaluation.render_summary_markdown(
        {
            "balanced_overall_summary": summary,
            "per_k_summary": [summary],
            "mse_tolerance": 2.5e-5,
            "dataset": {"requested_source_k": [4]},
            "ours_deployment": {"mode": "verified"},
            "timing_protocol": {
                "ours_scope": "full network forward plus exact adaptive repair",
                "hard_scope": "materialized-proposal pruning stage only",
            },
        }
    )
    assert "Ours verified (adaptive repair)" in markdown
    assert "not pure one-shot" in markdown
    assert "Residual-guided dynamic insertion: `50.0%`" in markdown


def test_paired_timing_blocks_calls_each_method_once_per_block() -> None:
    counts = {"ours": 0, "hard": 0}

    def ours() -> str:
        counts["ours"] += 1
        return "ours-result"

    def hard() -> str:
        counts["hard"] += 1
        return "hard-result"

    ours_result, hard_result, ours_times, hard_times, orders = (
        evaluation.paired_timing_blocks(
            ours,
            hard,
            repeats=5,
            order_rng=random.Random(9),
        )
    )

    assert ours_result == "ours-result"
    assert hard_result == "hard-result"
    assert counts == {"ours": 5, "hard": 5}
    assert len(ours_times) == len(hard_times) == len(orders) == 5
    assert all(value >= 0.0 for value in ours_times + hard_times)


def test_render_summary_figure_creates_png(tmp_path: Path) -> None:
    summaries = []
    for source_k in (4, 5):
        rows = [
            _row(
                source_k=source_k,
                canonical_k=source_k - 1,
                ours_k=source_k,
                hard_k=source_k - 1,
                ours_mse=3e-5,
                hard_mse=2e-5,
                ours_time=20.0,
                hard_time=200.0,
            ),
            _row(
                source_k=source_k,
                canonical_k=source_k - 2,
                ours_k=source_k - 1,
                hard_k=source_k - 2,
                ours_mse=2e-5,
                hard_mse=1.5e-5,
                ours_time=22.0,
                hard_time=220.0,
            ),
        ]
        summaries.append(
            evaluation.summarize_rows(
                rows,
                source_k=source_k,
                knot_tolerances=(0.01,),
                bootstrap_replicates=50,
                bootstrap_seed=11,
            )
        )

    output_path = tmp_path / "stratified.png"
    evaluation.render_summary_figure(
        summaries,
        output_path,
        mse_tolerance=2.5e-5,
        samples_per_k=2,
        dpi=50,
    )

    assert output_path.is_file()
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    three_metric_path = tmp_path / "stratified_three_metrics.png"
    evaluation.render_three_metric_figure(
        summaries,
        three_metric_path,
        mse_tolerance=2.5e-5,
        samples_per_k=2,
        dpi=50,
        ours_deployment="verified",
        verified_parameterization="chord",
    )

    assert three_metric_path.is_file()
    assert three_metric_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
