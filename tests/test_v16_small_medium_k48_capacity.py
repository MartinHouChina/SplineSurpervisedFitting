"""K=4..20 sources with 48 internal candidates and an isolated study contract."""
from __future__ import annotations

# ruff: noqa: E402
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_v16
from benchmark_v15_datasets import resolve_comparison_capacities
from spline_fitting.checkpointing import (
    V16_CERTIFIED_SYNTHETIC_CONTRACT,
    V16_FEASIBLE_TEACHER_OBJECTIVE_VERSION,
    assess_v16_checkpoint,
)
from spline_fitting.data.v16_mixed import ValidationCurves
from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork
from spline_fitting.training.v16_feasible_teacher import (
    _ordered_anchor_slots,
    build_or_load_v16_feasible_teacher_cache,
)


@pytest.fixture(autouse=True)
def single_threaded():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def small_k48_profile():
    return [
        "--study-scope", "small_medium_k48",
        "--candidate-knots", "48", "--min-control-points", "8",
        "--max-control-points", "24", "--proposal-high-k-fraction", "0",
        "--joint-high-k-fraction", "0", "--synthetic-high-k-val-size", "0",
        "--mse-tolerance", "5e-5", "--num-points", "64",
        "--hidden-dim", "16", "--encoder-layers", "1", "--selector-layers", "1",
        "--train-size", "4", "--val-size", "2", "--batch-size", "2",
        "--synthetic-boundary-val-size", "1", "--minimality-audit-points", "64",
        "--epochs", "2", "--proposal-epochs", "1", "--torch-num-threads", "1",
        "--device", "cpu", "--log-every-batches", "100",
    ]


def small_model(candidate_count=48):
    return V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, selector_layers=1,
        max_internal_knots=candidate_count, one_shot_selection_policy="mass_topk",
        one_shot_adaptive_threshold=True, min_selected_knots=4,
        mse_tolerance=5e-5, keep_state_interaction=True,
        proposal_global_warp_limit=4.0,
    ).eval()


def test_k48_scope_contract_preserves_k24_and_legacy_defaults():
    parser = train_v16.parser()
    legacy = parser.parse_args([])
    train_v16.validate_args(legacy)
    assert legacy.study_scope == "legacy"
    assert train_v16.synthetic_data_contract(legacy) == V16_CERTIFIED_SYNTHETIC_CONTRACT
    args = parser.parse_args(small_k48_profile())
    train_v16.validate_args(args)
    assert train_v16.synthetic_data_contract(args) == (
        "certified_source_subset_threshold_minimal_k4_20_kc48_span001_v1"
    ) == train_v16.V16_SMALL_MEDIUM_K48_SYNTHETIC_CONTRACT
    assert args.proposal_high_k_fraction == args.joint_high_k_fraction == 0
    assert args.candidate_knots == 48
    full = small_k48_profile()
    start = full.index("--candidate-knots")
    full[start:start + 2] = ["--full-knot-vector-size", "56"]
    args = parser.parse_args(full)
    train_v16.validate_args(args)
    assert args.candidate_knots == 48
    k24 = parser.parse_args(small_k48_profile() + [
        "--study-scope", "small_medium_k24", "--candidate-knots", "24",
    ])
    train_v16.validate_args(k24)
    assert train_v16.synthetic_data_contract(k24) == train_v16.V16_SMALL_MEDIUM_SYNTHETIC_CONTRACT
    assert train_v16.synthetic_data_contract(k24) != train_v16.synthetic_data_contract(args)


@pytest.mark.parametrize("override,match", [
    (["--max-control-points", "25"], "source control points"),
    (["--min-control-points", "9"], "source control points"),
    (["--candidate-knots", "24"], "candidate-knots 48"),
    (["--candidate-knots", "49"], "candidate-knots 48"),
    (["--proposal-high-k-fraction", "0.5"], "proposal-high-k-fraction 0"),
    (["--joint-high-k-fraction", "0.5"], "joint-high-k-fraction 0"),
    (["--synthetic-high-k-val-size", "1"], "synthetic-high-k-val-size 0"),
    (["--knot-min-span", "0.02"], "knot-min-span 0.01"),
    (["--no-certified-minimal-source"], "certified minimal sources"),
])
def test_k48_scope_rejects_inconsistent_labels(override, match):
    args = train_v16.parser().parse_args(small_k48_profile() + override)
    with pytest.raises(ValueError, match=match):
        train_v16.validate_args(args)


def test_k48_forward_variable_keep_decode_refit_and_offline_teacher(tmp_path):
    args = train_v16.parser().parse_args(small_k48_profile())
    train_v16.validate_args(args)
    dataset = ValidationCurves(
        train_v16.synthetic_dataset_config(args), size=2,
        synthetic_boundary_samples=1,
    )
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["target_internal_knots"].shape == (2, 20)
    assert batch["target_single_deletion_mse"].shape == (2, 20)
    assert batch["target_internal_knot_count"][0] == 20
    assert bool(((batch["target_internal_knot_count"] >= 4)
                 & (batch["target_internal_knot_count"] <= 20)).all())
    model = small_model()
    with torch.no_grad():
        context = model.encode_candidates(batch["points"])
        output = model(batch["points"])
    assert context["proposal_internal_knots"].shape == (2, 48)
    assert bool((context["proposal_internal_knots"].diff() > 0).all())
    assert bool((output["params"].diff() > 0).all())
    assert output["learned_keep_mask"].shape == (2, 48)
    assert model.candidate_head.interval_queries.shape[0] == 49

    # Exercise the actual ragged subset decoder and ordinary cubic refit at
    # source-minimum, source-maximum and all-keep capacities, not just metadata.
    for counts in ([4, 20], [20, 48]):
        mask = torch.zeros(2, 48, dtype=torch.bool)
        for row, count in enumerate(counts):
            indices = torch.linspace(0, 47, count).round().long()
            mask[row, indices] = True
        with torch.no_grad():
            decoded = model.decode_subset(context, mask)
        assert torch.equal(decoded["learned_keep_mask"], mask)
        assert decoded["predicted_knot_count"].tolist() == counts
        for row, count in enumerate(counts):
            knots = decoded["internal_knots"][row, mask[row]]
            assert knots.numel() == count
            assert bool((knots.diff() > 0).all())
            fit = refit_bspline_control_points(
                decoded["params"][row], batch["points"][row], knots,
                degree=3, smoothness_weight=0.0,
            )
            assert fit.knot_vector.shape == (count + 8,)
            assert fit.control_points.shape == (count + 4, 2)
            assert torch.equal(fit.knot_vector[:4], torch.zeros(4))
            assert torch.equal(fit.knot_vector[-4:], torch.ones(4))
            assert bool((fit.knot_vector.diff() >= 0).all())
            assert torch.isfinite(fit.control_points).all()
            assert torch.isfinite(fit.fit_mse)
    assert fit.knot_vector.numel() == 56
    assert fit.control_points.shape[0] == 52

    candidates = torch.linspace(0.02, 0.98, 48)
    slots = list(range(2, 42, 2))
    assert _ordered_anchor_slots(candidates, candidates[slots], max_distance=1e-6) == slots
    teacher_path = tmp_path / "teacher48.pt"
    teacher_options = dict(
        mse_tolerance=5e-5, device="cpu", batch_size=2,
        teacher_strategy="synthetic_anchor_first", anchor_match_tolerance=0.05,
    )
    cache = build_or_load_v16_feasible_teacher_cache(
        model, dataset, teacher_path, **teacher_options,
    )
    assert cache.candidate_count == 48
    assert cache.batch.config.error_tolerance ** 2 == pytest.approx(5e-5)
    assert cache.labels["teacher_retained_mask"].shape == (2, 48)
    assert bool((cache.labels["teacher_count"] <= 48).all())
    assert torch.equal(cache.labels["teacher_count"], cache.labels["teacher_retained_mask"].sum(-1))
    assert torch.isfinite(cache.labels["teacher_fit_mse"]).all()
    assert torch.equal(
        cache.labels["teacher_threshold_satisfied"], cache.labels["teacher_fit_mse"] <= 5e-5,
    )
    loaded = build_or_load_v16_feasible_teacher_cache(
        model, dataset, teacher_path, **teacher_options,
    )
    assert loaded.loaded
    assert torch.equal(loaded.labels["teacher_retained_mask"], cache.labels["teacher_retained_mask"])
    with pytest.raises(ValueError, match="does not match frozen Proposal"):
        build_or_load_v16_feasible_teacher_cache(
            small_model(24), dataset, teacher_path, **teacher_options,
        )
    with pytest.raises(ValueError, match="does not match frozen Proposal"):
        build_or_load_v16_feasible_teacher_cache(
            model, dataset, teacher_path, **{**teacher_options, "mse_tolerance": 1e-4},
        )


def test_k48_comparison_uses_56_entry_full_knot_vectors():
    args = SimpleNamespace(
        max_internal_knots=None, full_knot_vector_size=56,
        paper_initial_knots=48, liang_dense_knots=48,
    )
    report = resolve_comparison_capacities(args, {
        "objective_version": V16_FEASIBLE_TEACHER_OBJECTIVE_VERSION,
        "model_config": {"max_internal_knots": 48, "degree": 3},
        "dataset_config": {"max_control_points": 24},
    })
    assert report["source_max_internal_knots"] == 20
    assert report["network_candidates"] == 48
    assert report["network_full_knot_vector_size_at_all_keep"] == 56
    assert report["numerical_full_knot_vector_cap"] == 56
    assert report["equal_initial_capacity"] is True
    assert report["formal_overcomplete_candidate_contract"] is False


def test_k48_cpu_training_gradients_teacher_and_resume_contract(tmp_path, capsys):
    output = tmp_path / "small48.pt"
    command = small_k48_profile() + [
        "--output", str(output), "--keep-state-interaction",
        "--proposal-global-warp-limit", "4", "--joint-supervision", "offline_feasible_teacher",
        "--feasible-teacher-cache-dir", str(tmp_path / "teacher48"),
        "--feasible-teacher-strategy", "synthetic_anchor_first",
        "--synthetic-count-role", "reference_only", "--no-resample-train-each-epoch",
        "--proposal-joint-lr", "0", "--parameter-joint-lr", "0",
        "--keep-boundary-ranking-weight", "0.2", "--initial-keep-fraction", "0.5",
    ]
    assert train_v16.main(command) == 0
    last_path = tmp_path / "small48.last.pt"
    checkpoint = torch.load(last_path, weights_only=True)
    assert checkpoint["study_scope"] == "small_medium_k48"
    assert checkpoint["synthetic_data_contract"] == train_v16.V16_SMALL_MEDIUM_K48_SYNTHETIC_CONTRACT
    assert checkpoint["validation_metrics"]["synthetic_boundary_knot_count"] == 20
    metrics = checkpoint["history"][-1]["train"]
    assert metrics["gradient_norm_selector"] > 0
    assert metrics["gradient_norm_decoder"] > 0
    assert checkpoint["model_config"]["max_internal_knots"] == 48
    assert checkpoint["training_config"]["mse_tolerance"] == 5e-5
    assert checkpoint["deployment_config"]["mse_tolerance"] == 5e-5
    assert checkpoint["deployment_config"]["error_tolerance"] ** 2 == pytest.approx(5e-5)
    assert not assess_v16_checkpoint(checkpoint)["formal_reporting_eligible"]
    command[command.index("--epochs") + 1] = "3"
    command += ["--resume", str(last_path)]
    assert train_v16.main(command) == 0
    resumed = torch.load(last_path, weights_only=True)
    assert resumed["study_scope"] == "small_medium_k48"
    assert resumed["synthetic_data_contract"] == checkpoint["synthetic_data_contract"]
    assert resumed["offline_feasible_teacher"]["loaded_from_cache"] is True
    with pytest.raises(SystemExit):
        train_v16.main(command + ["--epochs", "4", "--mse-tolerance", "1e-4"])
    assert "mismatch" in capsys.readouterr().err
    for scope, capacity in [("small_medium_k24", "24"), ("legacy", "48")]:
        with pytest.raises(SystemExit):
            train_v16.main(command + [
                "--epochs", "4", "--study-scope", scope, "--candidate-knots", capacity,
            ])
        assert "different synthetic source-range" in capsys.readouterr().err
