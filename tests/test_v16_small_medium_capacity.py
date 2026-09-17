"""The independent K=4..20 study uses 24 internal candidates, not 24 full knots."""
from __future__ import annotations

# ruff: noqa: E402
from copy import deepcopy
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


def small_profile():
    return [
        "--study-scope", "small_medium_k24",
        "--candidate-knots", "24", "--min-control-points", "8",
        "--max-control-points", "24", "--proposal-high-k-fraction", "0",
        "--joint-high-k-fraction", "0", "--synthetic-high-k-val-size", "0",
        "--mse-tolerance", "1e-4", "--num-points", "64",
        "--hidden-dim", "16", "--encoder-layers", "1", "--selector-layers", "1",
        "--train-size", "4", "--val-size", "2", "--batch-size", "2",
        "--synthetic-boundary-val-size", "1", "--minimality-audit-points", "64",
        "--epochs", "2", "--proposal-epochs", "1", "--torch-num-threads", "1",
        "--device", "cpu", "--log-every-batches", "100",
    ]


def test_small_scope_contract_is_explicit_and_legacy_default_is_unchanged():
    parser = train_v16.parser()
    legacy = parser.parse_args([])
    train_v16.validate_args(legacy)
    assert legacy.study_scope == "legacy"
    assert train_v16.synthetic_data_contract(legacy) == V16_CERTIFIED_SYNTHETIC_CONTRACT
    args = parser.parse_args(small_profile())
    train_v16.validate_args(args)
    assert train_v16.synthetic_data_contract(args) == train_v16.V16_SMALL_MEDIUM_SYNTHETIC_CONTRACT
    assert args.proposal_high_k_fraction == args.joint_high_k_fraction == 0
    assert args.candidate_knots == 24
    full = small_profile()
    start = full.index("--candidate-knots")
    full[start:start + 2] = ["--full-knot-vector-size", "32"]
    args = parser.parse_args(full)
    train_v16.validate_args(args)
    assert args.candidate_knots == 24


@pytest.mark.parametrize("override,match", [
    (["--max-control-points", "25"], "source control points"),
    (["--min-control-points", "9"], "source control points"),
    (["--candidate-knots", "25"], "source control points"),
    (["--proposal-high-k-fraction", "0.5"], "proposal-high-k-fraction 0"),
    (["--joint-high-k-fraction", "0.5"], "joint-high-k-fraction 0"),
    (["--synthetic-high-k-val-size", "1"], "synthetic-high-k-val-size 0"),
    (["--knot-min-span", "0.02"], "knot-min-span 0.01"),
    (["--no-certified-minimal-source"], "certified minimal sources"),
])
def test_small_scope_rejects_inconsistent_labels(override, match):
    args = train_v16.parser().parse_args(small_profile() + override)
    with pytest.raises(ValueError, match=match):
        train_v16.validate_args(args)


@pytest.mark.parametrize("mse_tolerance", [1e-4, 5e-5])
def test_source_padding_twenty_and_candidate_twentyfour_forward_teacher(tmp_path, mse_tolerance):
    args = train_v16.parser().parse_args(small_profile() + ["--mse-tolerance", str(mse_tolerance)])
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
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, selector_layers=1,
        max_internal_knots=24, one_shot_selection_policy="mass_topk",
        one_shot_adaptive_threshold=True, min_selected_knots=4,
        mse_tolerance=mse_tolerance, keep_state_interaction=True,
        proposal_global_warp_limit=4.0,
    ).eval()
    with torch.no_grad():
        context = model.encode_candidates(batch["points"])
        output = model(batch["points"])
    assert context["proposal_internal_knots"].shape == (2, 24)
    assert bool((context["proposal_internal_knots"].diff() > 0).all())
    assert bool((output["params"].diff() > 0).all())
    assert output["learned_keep_mask"].shape == (2, 24)
    for row in range(2):
        chosen = output["internal_knots"][row][output["learned_keep_mask"][row]]
        assert bool((chosen.diff() > 0).all())
    candidates = torch.linspace(0.02, 0.98, 24)
    slots = list(range(2, 22))
    assert _ordered_anchor_slots(candidates, candidates[slots], max_distance=1e-6) == slots
    cache = build_or_load_v16_feasible_teacher_cache(
        model, dataset, tmp_path / "teacher.pt", mse_tolerance=mse_tolerance,
        device="cpu", batch_size=2, teacher_strategy="synthetic_anchor_first",
        anchor_match_tolerance=0.05,
    )
    assert cache.candidate_count == 24
    assert cache.batch.config.error_tolerance ** 2 == pytest.approx(mse_tolerance)
    assert cache.labels["teacher_retained_mask"].shape == (2, 24)
    assert bool((cache.labels["teacher_count"] <= 24).all())
    assert torch.isfinite(cache.labels["teacher_fit_mse"]).all()
    assert torch.equal(
        cache.labels["teacher_threshold_satisfied"], cache.labels["teacher_fit_mse"] <= mse_tolerance,
    )
    with pytest.raises(ValueError, match="does not match frozen Proposal"):
        build_or_load_v16_feasible_teacher_cache(
            model, dataset, tmp_path / "teacher.pt", mse_tolerance=mse_tolerance / 2,
            device="cpu", batch_size=2, teacher_strategy="synthetic_anchor_first",
            anchor_match_tolerance=0.05,
        )


def test_small_capacity_comparison_uses_32_entry_full_knot_vectors():
    args = SimpleNamespace(
        max_internal_knots=None, full_knot_vector_size=32,
        paper_initial_knots=24, liang_dense_knots=24,
    )
    report = resolve_comparison_capacities(args, {
        "objective_version": V16_FEASIBLE_TEACHER_OBJECTIVE_VERSION,
        "model_config": {"max_internal_knots": 24, "degree": 3},
        "dataset_config": {"max_control_points": 24},
    })
    assert report["source_max_internal_knots"] == 20
    assert report["network_candidates"] == 24
    assert report["network_full_knot_vector_size_at_all_keep"] == 32
    assert report["numerical_full_knot_vector_cap"] == 32
    assert report["equal_initial_capacity"] is True
    assert report["formal_overcomplete_candidate_contract"] is False


@pytest.mark.parametrize("mse_tolerance", [1e-4, 5e-5])
def test_small_scope_real_joint_gradients_contract_and_resume(tmp_path, capsys, mse_tolerance):
    output = tmp_path / "small.pt"
    command = small_profile() + [
        "--mse-tolerance", str(mse_tolerance),
        "--output", str(output), "--keep-state-interaction",
        "--proposal-global-warp-limit", "4", "--joint-supervision", "offline_feasible_teacher",
        "--feasible-teacher-cache-dir", str(tmp_path / "teacher"),
        "--synthetic-count-role", "reference_only", "--no-resample-train-each-epoch",
        "--proposal-joint-lr", "0", "--parameter-joint-lr", "0",
        "--keep-boundary-ranking-weight", "0.2", "--initial-keep-fraction", "0.5",
    ]
    assert train_v16.main(command) == 0
    last_path = tmp_path / "small.last.pt"
    checkpoint = torch.load(last_path, weights_only=True)
    assert checkpoint["study_scope"] == "small_medium_k24"
    assert checkpoint["synthetic_data_contract"] == train_v16.V16_SMALL_MEDIUM_SYNTHETIC_CONTRACT
    assert checkpoint["validation_metrics"]["synthetic_boundary_knot_count"] == 20
    metrics = checkpoint["history"][-1]["train"]
    assert metrics["gradient_norm_selector"] > 0
    assert metrics["gradient_norm_decoder"] > 0
    assert checkpoint["model_config"]["max_internal_knots"] == 24
    assert checkpoint["training_config"]["mse_tolerance"] == mse_tolerance
    assert checkpoint["deployment_config"]["mse_tolerance"] == mse_tolerance
    assert checkpoint["deployment_config"]["error_tolerance"] ** 2 == pytest.approx(mse_tolerance)
    assert not assess_v16_checkpoint(checkpoint)["formal_reporting_eligible"]
    command[command.index("--epochs") + 1] = "3"
    command += ["--resume", str(last_path)]
    assert train_v16.main(command) == 0
    resumed = torch.load(last_path, weights_only=True)
    assert resumed["study_scope"] == "small_medium_k24"
    assert resumed["synthetic_data_contract"] == checkpoint["synthetic_data_contract"]
    assert resumed["offline_feasible_teacher"]["loaded_from_cache"] is True
    with pytest.raises(SystemExit):
        train_v16.main(command + ["--epochs", "4", "--mse-tolerance", str(mse_tolerance / 2)])
    assert "mismatch" in capsys.readouterr().err
    command[command.index("--epochs") + 1] = "4"
    command[command.index("--study-scope") + 1] = "legacy"
    with pytest.raises(SystemExit):
        train_v16.main(command)
    assert "different synthetic source-range" in capsys.readouterr().err


def test_small_scope_full_initialization_uses_independent_contract(tmp_path):
    source = tmp_path / "source.pt"
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, selector_layers=1,
        max_internal_knots=24, one_shot_adaptive_threshold=True,
        one_shot_selection_policy="mass_topk", min_selected_knots=4,
    )
    torch.save({
        "objective_version": V16_FEASIBLE_TEACHER_OBJECTIVE_VERSION,
        "stage": "joint", "epoch": 5, "model_config": model.get_config(),
        "model_state_dict": deepcopy(model.state_dict()),
        "training_config": {"real_fraction": 0.0},
    }, source)
    command = small_profile() + ["--output", str(tmp_path / "full.pt")]
    command[command.index("--proposal-epochs") + 1] = "0"
    command[command.index("--epochs") + 1] = "1"
    command += ["--init-full-checkpoint", str(source)]
    assert train_v16.main(command) == 0
    for name in ("full.initial.pt", "full.proposal.pt", "full.last.pt"):
        checkpoint = torch.load(tmp_path / name, weights_only=True)
        assert checkpoint["study_scope"] == "small_medium_k24"
        assert checkpoint["synthetic_data_contract"] == train_v16.V16_SMALL_MEDIUM_SYNTHETIC_CONTRACT


def test_legacy_resume_without_scope_stays_legacy(tmp_path, capsys):
    command = small_profile() + ["--output", str(tmp_path / "legacy.pt")]
    flag = command.index("--study-scope")
    del command[flag:flag + 2]
    assert train_v16.main(command) == 0
    last_path = tmp_path / "legacy.last.pt"
    checkpoint = torch.load(last_path, weights_only=True)
    assert checkpoint["synthetic_data_contract"] == V16_CERTIFIED_SYNTHETIC_CONTRACT
    del checkpoint["study_scope"]
    del checkpoint["training_config"]["study_scope"]
    torch.save(checkpoint, last_path)
    command[command.index("--epochs") + 1] = "3"
    command += ["--resume", str(last_path)]
    assert train_v16.main(command) == 0
    resumed = torch.load(last_path, weights_only=True)
    assert resumed["study_scope"] == "legacy"
    assert resumed["training_config"]["study_scope"] == "legacy"
    command[command.index("--epochs") + 1] = "4"
    command += ["--study-scope", "small_medium_k24"]
    with pytest.raises(SystemExit):
        train_v16.main(command)
    assert "different synthetic source-range" in capsys.readouterr().err
