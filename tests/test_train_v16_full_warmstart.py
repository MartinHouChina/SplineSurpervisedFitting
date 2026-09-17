from __future__ import annotations

# ruff: noqa: E402

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_v16
from spline_fitting.checkpointing import V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork


def tiny_model(**overrides):
    return V16CandidateSelectionNetwork(**{
        "hidden_dim": 16, "encoder_layers": 1, "max_internal_knots": 8,
        "attention_heads": 4, "selector_layers": 1,
        "one_shot_selection_policy": "mass_topk", "one_shot_adaptive_threshold": True,
        "one_shot_safety_knots": 2, "one_shot_safety_sigma": 0.25,
        "min_selected_knots": 4, "mse_tolerance": 0.001,
        "relocation_blend": 0.03, **overrides,
    })


def source_payload(model, *, real_fraction=0.0):
    return {
        "objective_version": V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        "model_config": model.get_config(),
        "model_state_dict": deepcopy(model.state_dict()),
        "training_config": {"real_fraction": real_fraction},
        "epoch": 99, "stage": "joint",
        "history": [{"epoch": 99}],
        "optimizer_state_dict": {"must_not_restore": True},
    }


def tiny_command(output, source):
    return [
        "--epochs", "1", "--proposal-epochs", "0",
        "--init-full-checkpoint", str(source),
        "--train-size", "4", "--val-size", "2", "--batch-size", "2",
        "--num-points", "24", "--candidate-knots", "8", "--hidden-dim", "16",
        "--encoder-layers", "1", "--selector-layers", "1", "--attention-heads", "4",
        "--min-control-points", "8", "--max-control-points", "12",
        "--policy-samples", "2", "--counterfactual-edits", "1",
        "--minimality-audit-points", "32", "--mse-tolerance", "0.001",
        "--torch-num-threads", "1", "--device", "cpu",
        "--log-every-batches", "100", "--output", str(output),
    ]


def test_full_transfer_preserves_selector_decoder_and_predictions():
    torch.set_num_threads(1)
    source = tiny_model().eval()
    target = tiny_model().eval()
    with torch.no_grad():
        source.keep_head.weight.add_(0.1)
        source.relocation_update.bias.add_(0.2)
    report = train_v16.transfer_full_model_weights(target, source_payload(source))
    assert report["copied_tensor_count"] == len(source.state_dict())
    assert report["zero_initialized_extensions"] == []
    for name, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[name], value), name
    points = torch.randn(2, 24, 2)
    with torch.no_grad():
        before, after = source(points), target(points)
    for key in ("params", "internal_knots", "keep_probabilities", "learned_keep_mask"):
        assert torch.equal(before[key], after[key]), key


@pytest.mark.parametrize("stage", ["proposal", None])
def test_full_transfer_requires_joint_source(stage):
    payload = source_payload(tiny_model())
    payload["stage"] = stage
    with pytest.raises(ValueError, match="requires a Joint checkpoint"):
        train_v16.transfer_full_model_weights(tiny_model(), payload)


def test_full_transfer_audits_ignored_constructor_initializers():
    source = tiny_model(initial_keep_fraction=0.9, relocation_blend=0.25)
    target = tiny_model(initial_keep_fraction=0.4, relocation_blend=0.01)
    report = train_v16.transfer_full_model_weights(target, source_payload(source))
    assert report["configuration_changes"] == {}
    assert set(report["ignored_initialization_settings"]) == {
        "initial_keep_fraction", "relocation_blend",
    }
    assert report["ignored_initialization_settings"]["relocation_blend"]["requested"] == 0.01
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value), key


@pytest.mark.parametrize("revision", [None, "future_or_incompatible_revision"])
def test_offline_joint_resume_rejects_different_loss_semantics(revision):
    payload = {"stage": "joint", "loss_semantics_revision": revision}
    with pytest.raises(ValueError, match="--init-full-checkpoint"):
        train_v16.resume_loss_semantics_migration(
            payload, joint_supervision="offline_feasible_teacher",
        )
    # The new guard does not reinterpret other objectives.
    assert train_v16.resume_loss_semantics_migration(
        payload, joint_supervision="synthetic_ground_truth",
    ) is None


@pytest.mark.parametrize("change,match", [
    ({"max_internal_knots": 9}, "architecture/capacity"),
    ({"attention_heads": 2}, "architecture/capacity"),
    ({"subset_parameter_residual_limit": 0.5}, "architecture/capacity"),
    ({"one_shot_adaptive_threshold": False}, "architecture/capacity"),
])
def test_full_transfer_rejects_architecture_and_capacity_changes(change, match):
    with pytest.raises(ValueError, match=match):
        train_v16.transfer_full_model_weights(tiny_model(**change), source_payload(tiny_model()))


@pytest.mark.parametrize("name", ["keep_head.weight", "relocation_blend_logit", "encoder.local_stack.0.weight"])
def test_full_transfer_rejects_missing_core_tensors(name):
    model = tiny_model()
    payload = source_payload(model)
    # Use a real encoder tensor without depending on its implementation layout.
    if name.startswith("encoder."):
        name = next(key for key in payload["model_state_dict"] if key.startswith("encoder."))
    del payload["model_state_dict"][name]
    with pytest.raises(ValueError, match="requires every model tensor"):
        train_v16.transfer_full_model_weights(tiny_model(), payload)


def test_full_transfer_only_allows_explicit_zero_extensions():
    source = tiny_model().eval()
    payload = source_payload(source)
    # A genuinely historical config lacks both extension keys.
    payload["model_config"].pop("keep_state_interaction")
    payload["model_config"].pop("proposal_global_warp_limit")
    target = tiny_model(keep_state_interaction=True, proposal_global_warp_limit=4.0).eval()
    report = train_v16.transfer_full_model_weights(target, payload)
    assert set(report["zero_initialized_extensions"]) == {
        "keep_state_embedding.weight", "candidate_head.interval_warp_score.weight",
        "candidate_head.interval_warp_score.bias",
    }
    points = torch.randn(2, 24, 2)
    with torch.no_grad():
        before, after = source(points), target(points)
    for key in ("params", "internal_knots", "keep_probabilities", "learned_keep_mask"):
        torch.testing.assert_close(before[key], after[key], rtol=1e-5, atol=1e-6)
    with torch.no_grad():
        target.keep_state_embedding.weight.fill_(1)
    with pytest.raises(ValueError, match="must start at zero"):
        train_v16.transfer_full_model_weights(target, payload)
    with pytest.raises(ValueError, match="cannot discard"):
        train_v16.transfer_full_model_weights(source, source_payload(target))


def test_initialization_preserves_mixed_or_unknown_ancestry(tmp_path):
    source = source_payload(tiny_model(), real_fraction=0.35)
    provenance = train_v16.initialization_provenance(
        source, path=tmp_path / "mixed.pt", mode="full_model", real_fraction=0,
    )
    assert provenance["source_training_real_fraction"] == 0.35
    assert provenance["synthetic_only_model_lineage"] is False
    source["training_config"]["real_fraction"] = 0.0
    source["initialization_provenance"] = provenance
    descendant = train_v16.initialization_provenance(
        source, path=tmp_path / "synthetic_finetuned.pt", mode="full_model", real_fraction=0,
    )
    assert descendant["synthetic_only_model_lineage"] is False
    del source["initialization_provenance"]
    source["training_config"]["init_checkpoint"] = "untraced.pt"
    assert not train_v16.initialization_provenance(
        source, path=tmp_path / "unknown.pt", mode="full_model", real_fraction=0,
    )["synthetic_only_model_lineage"]


def test_zero_proposal_and_initialization_flags_are_checked():
    parser = train_v16.parser()
    with pytest.raises(ValueError, match="requires --init-full-checkpoint"):
        train_v16.validate_args(parser.parse_args(["--proposal-epochs", "0"]))
    for conflicting in ("--resume", "--init-checkpoint"):
        args = parser.parse_args(["--init-full-checkpoint", "full.pt", conflicting, "other.pt"])
        with pytest.raises(ValueError, match="mutually exclusive"):
            train_v16.validate_args(args)
    args = parser.parse_args(["--proposal-epochs", "0", "--init-full-checkpoint", "full.pt"])
    train_v16.validate_args(args)
    args.train_size, args.batch_size, args.epochs = 600, 128, 12
    assert train_v16.training_update_budget(args) == {
        "steps_per_epoch": 5, "remaining_proposal_updates": 0,
        "remaining_joint_updates": 60, "remaining_total_updates": 60,
    }


def test_full_warmstart_skips_proposal_preserves_baseline_and_resumes(tmp_path):
    source = tmp_path / "source.pt"
    original = source_payload(tiny_model(), real_fraction=0.35)
    torch.save(original, source)
    output = tmp_path / "finetune.pt"
    command = tiny_command(output, source) + ["--keep-state-interaction"]
    assert train_v16.main(command) == 0
    initial = torch.load(tmp_path / "finetune.initial.pt", weights_only=True)
    last_path = tmp_path / "finetune.last.pt"
    last = torch.load(last_path, weights_only=True)
    assert initial["epoch"] == 0
    assert initial["history"] == []
    for key, value in original["model_state_dict"].items():
        assert torch.equal(initial["model_state_dict"][key], value), key
    assert [entry["epoch"] for entry in last["history"]] == [1]
    assert last["history"][0]["stage"] == "joint"
    assert last["proposal_initializer_epoch"] == 0
    assert last["initial_validation"] == initial["validation_metrics"]
    assert last["cumulative_optimizer_updates"] == 2
    assert last["cumulative_joint_optimizer_updates"] == 2
    assert last["initialization_provenance"]["source_training_real_fraction"] == 0.35
    assert not last["initialization_provenance"]["synthetic_only_model_lineage"]
    assert json.loads((tmp_path / "finetune.initial.validation.json").read_text())["validation_metrics"] == initial["validation_metrics"]
    assert (tmp_path / "finetune.proposal.pt").is_file()
    assert (tmp_path / "finetune.proposal.final.pt").is_file()
    flag = command.index("--init-full-checkpoint")
    del command[flag:flag + 2]
    command[command.index("--epochs") + 1] = "2"
    command += ["--resume", str(last_path)]
    assert train_v16.main(command) == 0
    resumed = torch.load(last_path, weights_only=True)
    assert [entry["epoch"] for entry in resumed["history"]] == [1, 2]
    assert resumed["cumulative_joint_optimizer_updates"] == 4
    assert resumed["initialization_provenance"] == last["initialization_provenance"]
    assert resumed["training_config"]["init_full_checkpoint"] == str(source)


def test_full_warmstart_zero_proposal_supports_offline_teacher_cache(tmp_path, capsys):
    source = tmp_path / "source.pt"
    torch.save(source_payload(tiny_model()), source)
    output = tmp_path / "cached.pt"
    command = tiny_command(output, source) + [
        "--joint-supervision", "offline_feasible_teacher",
        "--feasible-teacher-cache-dir", str(tmp_path / "cache"),
        "--synthetic-count-role", "reference_only", "--no-resample-train-each-epoch",
        "--proposal-joint-lr", "0", "--parameter-joint-lr", "0",
        "--keep-state-interaction", "--keep-boundary-ranking-weight", "0.2",
    ]
    assert train_v16.main(command) == 0
    assert (tmp_path / "cache/train.pt").is_file()
    last = torch.load(tmp_path / "cached.last.pt", weights_only=True)
    assert last["offline_feasible_teacher"]["sample_count"] == 4
    assert last["proposal_initializer_epoch"] == 0
    assert last["loss_config"]["weights"]["keep_boundary_ranking_weight"] == 0.2
    assert "teacher_count_topk_recall" in last["train_metrics"]
    assert last["loss_semantics_revision"] == train_v16.V16_OFFLINE_LOSS_SEMANTICS_REVISION
    initial = torch.load(tmp_path / "cached.initial.pt", weights_only=True)
    assert initial["loss_semantics_revision"] == last["loss_semantics_revision"]
    flag = command.index("--init-full-checkpoint")
    del command[flag:flag + 2]
    command[command.index("--epochs") + 1] = "2"
    last_path = tmp_path / "cached.last.pt"
    command += ["--resume", str(last_path)]
    assert train_v16.main(command) == 0
    resumed = torch.load(last_path, weights_only=True)
    assert resumed["epoch"] == 2
    assert resumed["offline_feasible_teacher"]["loaded_from_cache"] is True
    # Old Joint weights remain eligible for a fresh optimizer warm start, but
    # old Adam moments must not silently resume the revised geometry objective.
    del resumed["loss_semantics_revision"]
    train_v16.transfer_full_model_weights(
        tiny_model(keep_state_interaction=True), resumed,
    )
    torch.save(resumed, last_path)
    command[command.index("--epochs") + 1] = "3"
    with pytest.raises(SystemExit):
        train_v16.main(command)
    assert "incompatible loss semantics" in capsys.readouterr().err
    assert torch.load(last_path, weights_only=True)["epoch"] == 2


def test_full_warmstart_proposal_adaptation_retains_initial_best(tmp_path, monkeypatch):
    source = tmp_path / "source.pt"
    torch.save(source_payload(tiny_model()), source)
    output = tmp_path / "adapted.pt"
    command = tiny_command(output, source) + [
        "--keep-state-interaction", "--proposal-global-warp-limit", "4",
    ]
    command[command.index("--epochs") + 1] = "2"
    command[command.index("--proposal-epochs") + 1] = "1"
    monkeypatch.setattr(train_v16, "proposal_checkpoint_rank", lambda _metrics: (0.0,))
    assert train_v16.main(command) == 0
    best = torch.load(tmp_path / "adapted.proposal.pt", weights_only=True)
    final = torch.load(tmp_path / "adapted.proposal.final.pt", weights_only=True)
    last = torch.load(tmp_path / "adapted.last.pt", weights_only=True)
    assert best["epoch"] == 0
    assert final["epoch"] == 1
    assert last["proposal_initializer_epoch"] == 0
    assert last["cumulative_optimizer_updates"] == 4
    assert last["cumulative_joint_optimizer_updates"] == 2
    assert [entry["stage"] for entry in last["history"]] == ["proposal", "joint"]
    assert best["model_config"]["proposal_global_warp_limit"] == 4


def test_zero_proposal_initialization_is_resumable_after_teacher_failure(tmp_path, monkeypatch):
    source = tmp_path / "source.pt"
    torch.save(source_payload(tiny_model()), source)
    output = tmp_path / "interrupted.pt"
    command = tiny_command(output, source) + [
        "--joint-supervision", "offline_feasible_teacher",
        "--feasible-teacher-cache-dir", str(tmp_path / "cache"),
        "--synthetic-count-role", "reference_only", "--no-resample-train-each-epoch",
        "--proposal-joint-lr", "0", "--parameter-joint-lr", "0",
    ]
    original_builder = train_v16.build_or_load_v16_feasible_teacher_cache

    def interrupted_builder(*_args, **_kwargs):
        raise RuntimeError("interrupted offline cache")

    monkeypatch.setattr(train_v16, "build_or_load_v16_feasible_teacher_cache", interrupted_builder)
    with pytest.raises(RuntimeError, match="interrupted offline cache"):
        train_v16.main(command)
    last_path = tmp_path / "interrupted.last.pt"
    interrupted = torch.load(last_path, weights_only=True)
    assert interrupted["epoch"] == 0
    assert interrupted["stage"] == "proposal"
    # Older Proposal checkpoints have not accumulated Joint Adam moments and
    # may safely adopt the new objective without rebuilding teacher labels.
    del interrupted["loss_semantics_revision"]
    torch.save(interrupted, last_path)
    monkeypatch.setattr(train_v16, "build_or_load_v16_feasible_teacher_cache", original_builder)
    flag = command.index("--init-full-checkpoint")
    del command[flag:flag + 2]
    command += ["--resume", str(last_path)]
    assert train_v16.main(command) == 0
    resumed = torch.load(last_path, weights_only=True)
    assert resumed["epoch"] == 1
    assert resumed["loss_semantics_revision"] == train_v16.V16_OFFLINE_LOSS_SEMANTICS_REVISION
    assert resumed["loss_semantics_migration"]["source_revision"] is None
    assert resumed["loss_semantics_migration"]["source_stage"] == "proposal"
