from __future__ import annotations

# ruff: noqa: E402

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from spline_fitting.checkpointing import (
    V16_ADAPTIVE_SELECTION_REVISION,
    V16_CERTIFIED_SYNTHETIC_CONTRACT,
    V16_SIMPLIFICATION_CONTRACT,
)
spec = importlib.util.spec_from_file_location("benchmark_v15", ROOT / "scripts/benchmark_v15_datasets.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def v16_checkpoint(*, target=0.90, observed=0.92, stage="joint"):
    accepted = stage == "joint" and observed >= target
    return {
        "objective_version": benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        "architecture_revision": V16_ADAPTIVE_SELECTION_REVISION,
        "simplification_contract": V16_SIMPLIFICATION_CONTRACT,
        "simplification_ready": True,
        "synthetic_data_contract": V16_CERTIFIED_SYNTHETIC_CONTRACT,
        "dataset_config": {
            "certified_minimal_source": True,
            "canonical_knot_tolerance": 0.005,
            "minimality_margin": 0.2,
            "minimality_audit_points": 512,
            "min_control_points": 8,
            "max_control_points": 60,
            "knot_min_span": 0.01,
        },
        "model_config": {
            "one_shot_selection_policy": "mass_topk",
            "one_shot_adaptive_threshold": True,
            "one_shot_safety_sigma": 0.05,
            "one_shot_safety_knots": 0,
            "max_internal_knots": 56,
        },
        "stage": stage,
        "training_config": {
            "proposal_pass_target": target,
            "deployment_pass_target": target,
            "mse_tolerance": 2.5e-5,
            "knot_match_tolerance": 0.01,
            "allow_infeasible_proposals": False,
            "final_safety_sigma": 0.05,
            "final_safety_knots": 0,
            "proposal_high_k_fraction": 0.5,
            "proposal_high_k_min_knots": 40,
        },
        "validation_metrics": {
            "worst_dense_pass_rate": 0.98,
            "worst_deployment_pass_rate": observed,
            "qualification_dense_pass_rate": 0.96,
            "qualification_deployment_pass_rate": observed,
            "keep_count": 8.0,
            "synthetic_count_mae": 0.75,
            "synthetic_knot_match_f1": 0.8,
            "synthetic_knot_matched_mae": 0.004,
            "synthetic_boundary_knot_count": 56,
            "synthetic_boundary_sample_count": 32,
            "synthetic_boundary_dense_pass_rate": 0.96,
            "synthetic_boundary_deployment_pass_rate": observed,
        },
        "deployment_config": {
            "mse_tolerance": 2.5e-5,
            "knot_match_tolerance": 0.01,
            "one_shot_selection_policy": "mass_topk",
            "adaptive_keep_threshold": True,
            "one_shot_safety_sigma": 0.05,
            "one_shot_safety_knots": 0,
        },
        "loss_config": {
            "ranked_prefix_teacher": True,
            "weights": {
                "count_weight": 2.0,
                "supervised_count_weight": 1.0,
                "supervised_over_count_weight": 1.0,
                "complexity_weight": 0.05,
                "true_parameter_weight": 0.1,
                "proposal_knot_coverage_weight": 1.0,
                "proposal_knot_assignment_weight": 1.0,
                "selected_knot_position_weight": 1.0,
            },
        },
        "proposal_ready": True,
        "best_deployment_pass_constraint_satisfied": accepted,
        "checkpoint_quality": "deployment_target_met" if accepted else "target_not_met",
    }


def test_balanced_sampling_visits_each_group_and_is_reproducible():
    records = [{"group_id": f"g{i // 4}"} for i in range(12)]
    first = benchmark.balanced_indices(records, 6, 42)
    assert first == benchmark.balanced_indices(records, 6, 42)
    assert len(set(first)) == 6
    assert len({records[i]["group_id"] for i in first[:3]}) == 3
    assert len(benchmark.balanced_indices(records, 100, 42)) == 12


def test_published_method_set_is_exactly_the_requested_six_methods():
    assert benchmark.PUBLISHED_METHODS == (
        "ours",
        "park_dominant_point_2007_adaptation",
        "liang_feature_iki_2017_adaptation",
        "dung_direct_knot_2017_adaptation",
        "kang_sparse_2015_adaptation",
        "luo_linf_de_2022_adaptation",
    )
    arguments = benchmark.parser().parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--output-dir",
            "comparison",
            "--method-set",
            "published",
        ]
    )
    assert arguments.method_set == "published"


def test_failures_remain_in_pass_rate_denominator():
    row = {"dataset": "UJI", "method": "ours", "status": "ok",
           "mse": 1e-6, "reference_mse": 2e-6, "fit_pass": True,
           "reference_pass": True, "final_k": 2, "total_ms": 3, "network_ms": 1}
    failure = {**row, "status": "failed", "fit_pass": False, "reference_pass": None,
               "mse": None, "reference_mse": None}
    summary = benchmark.summarize([row, failure])[0]
    assert summary["failed"] == 1
    assert summary["n"] == 2
    assert summary["fit_pass_rate"] == .5
    assert summary["reference_pass_rate"] == .5
    assert summary["mse_mean"] == 1e-6


def test_synthetic_summary_reports_canonical_count_accuracy():
    rows = [
        {
            "dataset": "Synthetic", "method": "ours", "status": "ok",
            "mse": 1e-6, "reference_mse": None, "fit_pass": True,
            "reference_pass": None, "final_k": 9, "canonical_k": 8,
            "total_ms": 3, "network_ms": 1,
        },
        {
            "dataset": "Synthetic", "method": "ours", "status": "ok",
            "mse": 2e-6, "reference_mse": None, "fit_pass": True,
            "reference_pass": None, "final_k": 12, "canonical_k": 12,
            "total_ms": 4, "network_ms": 1.5,
        },
        {
            "dataset": "Synthetic", "method": "ours", "status": "ok",
            "mse": 3e-6, "reference_mse": None, "fit_pass": True,
            "reference_pass": None, "final_k": 12, "canonical_k": 14,
            "total_ms": 5, "network_ms": 2,
        },
    ]

    summary = benchmark.summarize(rows)[0]

    assert summary["canonical_k_mean"] == pytest.approx(34 / 3)
    assert summary["canonical_count_mae"] == pytest.approx(1.0)
    assert summary["canonical_count_bias"] == pytest.approx(-1 / 3)
    assert summary["canonical_count_exact_rate"] == pytest.approx(1 / 3)
    assert summary["canonical_count_within_one_rate"] == pytest.approx(2 / 3)


def test_summary_without_canonical_labels_and_empty_input_are_safe():
    row = {
        "dataset": "UJI", "method": "ours", "status": "ok",
        "mse": 1e-6, "reference_mse": 2e-6, "fit_pass": True,
        "reference_pass": True, "final_k": 7, "canonical_k": None,
        "total_ms": 3, "network_ms": 1,
    }

    summary = benchmark.summarize([row])[0]

    assert summary["canonical_k_mean"] is None
    assert summary["canonical_count_mae"] is None
    assert summary["canonical_count_bias"] is None
    assert summary["canonical_count_exact_rate"] is None
    assert summary["canonical_count_within_one_rate"] is None
    assert benchmark.summarize([]) == []


def test_reference_alignment_uses_original_resampling_grid():
    class Fit:
        def evaluate(self, parameters):
            return parameters[:, None]
    parameters = torch.tensor([0., .2, 1.], dtype=torch.float64)
    case = {"reference_grid": torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64),
            "reference": torch.tensor([[0.], [.1], [.2], [.6], [1.]], dtype=torch.float64)}
    assert benchmark.reference_mse(Fit(), parameters, case) == pytest.approx(0, abs=1e-30)


def test_all_real_failures_have_zero_reference_pass_rate():
    row = {"dataset": "UJI", "method": "ours", "status": "failed", "has_reference": True,
           "mse": None, "reference_mse": None, "fit_pass": False, "reference_pass": None,
           "final_k": None, "total_ms": 10, "network_ms": None}
    result = benchmark.summarize([row])[0]
    assert result["reference_pass_rate"] == 0
    assert result["mse_mean"] is None


def test_recover_torn_journal_preserves_complete_records(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_bytes(b'{"a":1}\n{"unfinished"')
    assert benchmark.recover_journal(path) == [{"a": 1}]
    assert path.read_bytes() == b'{"a":1}\n'
    assert path.with_suffix(".interrupted_tail.bin").read_bytes() == b'{"unfinished"'
    path.write_bytes(b'{broken}\n{"a":1}\n')
    with pytest.raises(ValueError, match="Corrupt completed"):
        benchmark.recover_journal(path)


def test_version_gate_rejects_v15_as_v16_and_unknown_objectives():
    v15 = benchmark.V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION
    v16 = benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION
    assert benchmark.benchmark_version(v15, expected_objective=v15) == "v15"
    assert benchmark.benchmark_version(v16, expected_objective=v16) == "v16"
    with pytest.raises(ValueError, match="requires"):
        benchmark.benchmark_version(v15, expected_objective=v16)
    with pytest.raises(ValueError, match="requires"):
        benchmark.benchmark_version("invented", expected_objective=v16)
    with pytest.raises(ValueError, match="Unsupported"):
        benchmark.benchmark_version("invented", expected_objective="invented")


def test_missing_checkpoint_is_reported_before_torch_load(tmp_path, capsys):
    missing = tmp_path / "not_trained.pt"
    with pytest.raises(SystemExit) as error:
        benchmark.main([
            "--checkpoint", str(missing),
            "--output-dir", str(tmp_path / "comparison"),
        ])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "checkpoint does not exist" in message
    assert "run scripts/train_v16.py first" in message
    assert not (tmp_path / "comparison").exists()


def test_v16_benchmark_rejects_relaxed_or_proposal_checkpoint_by_default():
    for checkpoint in (
        v16_checkpoint(target=0.85, observed=0.92),
        v16_checkpoint(stage="proposal"),
    ):
        with pytest.raises(ValueError, match="allow-unqualified-diagnostic"):
            benchmark.validate_checkpoint_for_benchmark(
                checkpoint, mse_tolerance=2.5e-5,
                allow_unqualified_diagnostic=False,
            )
        qualification, diagnostic = benchmark.validate_checkpoint_for_benchmark(
            checkpoint, mse_tolerance=2.5e-5,
            allow_unqualified_diagnostic=True,
        )
        assert diagnostic
        assert not qualification["formal_reporting_eligible"]


def test_v16_benchmark_accepts_consistent_formal_checkpoint():
    qualification, diagnostic = benchmark.validate_checkpoint_for_benchmark(
        v16_checkpoint(), mse_tolerance=2.5e-5,
        allow_unqualified_diagnostic=False,
    )
    assert qualification["formal_reporting_eligible"]
    assert not diagnostic


def test_v16_forwards_reported_mse_budget_but_v15_has_no_new_argument():
    class V15Model:
        def forward_deployment(self, points):
            return points

    class V16Model:
        mse_tolerance = 9e-4

        def forward_deployment(self, points, *, mse_tolerance):
            return points, mse_tolerance

    points = torch.zeros(1, 4, 2)
    assert benchmark.deployment_forward(
        V15Model(), points,
        objective_version=benchmark.V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
        mse_tolerance=1e-5,
    ) is points
    result = benchmark.deployment_forward(
        V16Model(), points,
        objective_version=benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        mse_tolerance=1e-5,
    )
    assert result[0] is points
    assert result[1] == 1e-5
    with pytest.raises(ValueError, match="Unsupported"):
        benchmark.deployment_forward(V15Model(), points,
                                     objective_version="unknown", mse_tolerance=1e-5)


def test_numerical_timing_uses_same_complete_repeat_count_and_median(monkeypatch):
    elapsed = iter([9.0, 3.0, 6.0])
    fit = SimpleNamespace()

    def run(method, points, **kwargs):
        return SimpleNamespace(
            fit=fit,
            parameters=torch.linspace(0, 1, points.shape[0]),
            elapsed_ms=next(elapsed),
            diagnostics={"method": method},
        )

    monkeypatch.setattr(benchmark, "run_published_baseline", run)
    monkeypatch.setattr(benchmark, "published_baseline_kwargs", lambda *a, **k: {})
    args = SimpleNamespace(end_to_end_repeats=3)
    returned = benchmark.measure_numerical_baseline(
        "uniform_gradient_pruning", torch.zeros(8, 2), args, degree=3,
    )
    assert returned[0] is fit
    assert returned[2] == 6.0
    assert returned[3] is None
    assert returned[4]["repeated_complete_timing"]["samples_ms"] == [9.0, 3.0, 6.0]


def test_v16_wrapper_uses_explicit_objective_and_dedicated_paths(monkeypatch):
    import benchmark_v16_datasets as wrapper

    called = {}

    def capture(argv, **kwargs):
        called.update(kwargs)
        called["argv"] = argv

    monkeypatch.setattr(wrapper, "shared_main", capture)
    wrapper.main(["--skip-real"])
    assert called["argv"] == ["--skip-real", "--max-knot-count", "56"]
    assert called["expected_objective"] == benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION
    assert (
        called["default_checkpoint"].name
        == "candidate_selection_v16_mse1e-4_k56_ordered_highk.pt"
    )
    assert called["default_output_dir"].name == "v16_mse1e-4_k56_ordered_highk"

    called.clear()
    wrapper.main(["--max-knot-count=12"])
    assert called["argv"] == ["--max-knot-count=12"]


def test_v16_report_labels_cannot_claim_v15(tmp_path):
    metadata = {
        "checkpoint": "v16.pt", "epoch": 1, "mse_tolerance": 1e-5,
        "objective_version": benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        "hardware": {"device": "cpu"}, "num_points": 24,
    }
    row = {"dataset": "Synthetic", "method": "ours", "status": "ok",
           "mse": 1e-6, "reference_mse": None, "fit_pass": True,
           "reference_pass": None, "final_k": 2, "total_ms": 3, "network_ms": 1}
    benchmark.write_reports(tmp_path, metadata, [row])
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "# v16 " in report
    assert "Ours v16 learned" in report
    assert "Ours v15" not in report


def test_diagnostic_benchmark_report_has_unmissable_warning(tmp_path):
    metadata = {
        "checkpoint": "v16.pt", "epoch": 1, "mse_tolerance": 2.5e-5,
        "objective_version": benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        "hardware": {"device": "cpu"}, "num_points": 24,
        "diagnostic_not_final": True,
        "checkpoint_qualification": {"reasons": ["configured target below 97%"]},
    }
    row = {"dataset": "Synthetic", "method": "ours", "status": "ok",
           "mse": 1e-6, "reference_mse": None, "fit_pass": True,
           "reference_pass": None, "final_k": 2, "total_ms": 3, "network_ms": 1}
    benchmark.write_reports(tmp_path, metadata, [row])
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "DIAGNOSTIC NOT FINAL" in report
    assert "configured target below 97%" in report


@pytest.mark.parametrize("v16, expected", [(False, (28, 40)), (True, (64, 64))])
def test_comparison_caps_keep_v15_defaults_and_match_v16_checkpoint(v16, expected):
    args = SimpleNamespace(
        max_internal_knots=None, paper_initial_knots=None,
        liang_dense_knots=None,
    )
    checkpoint = {
        "objective_version": (benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION
                              if v16 else benchmark.V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION),
        "model_config": {"max_internal_knots": 64},
    }
    caps = benchmark.resolve_comparison_capacities(args, checkpoint)
    assert (args.max_internal_knots, args.paper_initial_knots) == expected
    assert caps["network_candidates"] == 64
    assert caps["equal_initial_capacity"] is v16


def test_explicit_numerical_capacities_are_retained_and_disclosed():
    args = SimpleNamespace(
        max_internal_knots=28, paper_initial_knots=40,
        liang_dense_knots=None,
    )
    checkpoint = {"objective_version": benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
                  "model_config": {"max_internal_knots": 64}}
    caps = benchmark.resolve_comparison_capacities(args, checkpoint)
    assert caps == {
        "network_candidates": 64,
        "greedy_initial_and_yeh_max": 28,
        "kang_dense_initial": 40,
        "liang_dense_initial": 40,
        "equal_initial_capacity": False,
        "degree": 3,
        "clamped_endpoint_entries": 8,
        "network_full_knot_vector_size_at_all_keep": 72,
        "numerical_full_knot_vector_cap": 36,
    }
    args.max_internal_knots = 0
    with pytest.raises(ValueError, match="positive"):
        benchmark.resolve_comparison_capacities(args, checkpoint)


def test_full_knot_vector_notation_maps_to_internal_capacity():
    args = SimpleNamespace(
        max_internal_knots=None,
        full_knot_vector_size=64,
        paper_initial_knots=None,
        liang_dense_knots=None,
    )
    checkpoint = {
        "objective_version": benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        "model_config": {"max_internal_knots": 56, "degree": 3},
    }
    caps = benchmark.resolve_comparison_capacities(args, checkpoint)
    assert args.max_internal_knots == 56
    assert args.paper_initial_knots == 56
    assert caps["network_full_knot_vector_size_at_all_keep"] == 64
    assert caps["numerical_full_knot_vector_cap"] == 64


def test_v16_seed_reservations_include_later_configured_and_resumed_epochs():
    checkpoint = {
        "objective_version": benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        "epoch": 2, "history": [{"epoch": 1}, {"epoch": 2}],
        "training_config": {"train_seed": 42, "train_size": 4000, "epochs": 5,
                            "train_seed_stride": 10_000_000, "resample_train_each_epoch": True,
                            "val_seed": 1_000_000, "val_size": 1000},
    }
    ranges = benchmark.known_synthetic_seed_ranges(checkpoint)
    assert len(ranges) == 6
    assert ranges[4] == {"split": "train", "epoch": 5, "start": 40_000_042, "stop": 40_004_042}
    assert ranges[-1] == {"split": "val", "epoch": 1, "start": 1_000_000, "stop": 1_001_000}
    checkpoint["history"].append({"epoch": 7})
    assert len(benchmark.known_synthetic_seed_ranges(checkpoint)) == 8
    checkpoint["training_config"]["resample_train_each_epoch"] = False
    assert len(benchmark.known_synthetic_seed_ranges(checkpoint)) == 2
    checkpoint["objective_version"] = benchmark.V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION
    checkpoint["training_config"]["resample_train_each_epoch"] = True
    assert len(benchmark.known_synthetic_seed_ranges(checkpoint)) == 2


def test_v16_rejects_test_seed_colliding_with_later_epoch(monkeypatch):
    checkpoint = {
        "objective_version": benchmark.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        "training_config": {"train_seed": 42, "train_size": 4000, "epochs": 5,
                            "val_seed": 1_000_000, "val_size": 1000},
    }
    monkeypatch.setattr(benchmark, "_dataset_config_from_checkpoint", lambda *a: {})
    monkeypatch.setattr(benchmark, "_select_indices", lambda *a, **k: [42])
    args = SimpleNamespace(skip_synthetic=False, seed=20_000_000, scan_size=100,
                           selection_seed=1, min_knot_count=4, max_knot_count=20,
                           samples_per_knot_count=1)
    with pytest.raises(ValueError, match="train epoch 3"):
        benchmark.prepare_cases(args, checkpoint, {})
