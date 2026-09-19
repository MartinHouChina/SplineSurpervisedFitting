"""Explicit shared K32 experiments; old K64 workflows stay reproducible."""
from copy import deepcopy

import pytest

from test_run_v16_1070_overnight_linux import (
    RUN_NAME, _bash_path, _commands, _parsed, _run, preflight, resume_bundle,
)


def test_k32_capacity_propagates_to_training_every_method_and_preflight(tmp_path):
    source = tmp_path / "stable64.pt"
    source.write_bytes(b"dry-run fixture")
    result = _run(tmp_path, "--stable-selection", "--candidate-knots", "32",
                  "--warm-start-checkpoint", _bash_path(source),
                  "--resize-candidate-warm-start", "--epochs", "32", "--proposal-epochs", "12")
    commands = _commands(result)
    train = _parsed(commands["train_fresh"])
    assert train.candidate_knots == 32 and train.resize_candidate_warm_start
    assert (train.epochs, train.proposal_epochs) == (32, 12)
    assert (train.min_control_points, train.max_control_points) == (8, 28)
    assert train.mse_tolerance == pytest.approx(5e-5)
    assert train.teacher_greedy_trajectory_checks == 4
    for stage in ("benchmark_six_methods", "plot_ours_cases", "plot_six_method_real_cases"):
        args = _parsed(commands[stage])
        assert (args.max_internal_knots, args.paper_initial_knots, args.liang_dense_knots) == (32, 32, 32)
        assert {m.split("=", 1)[0] for m in args.manifest} == {
            "UJI", "NaturalEarth", "USGS", "IndustrialOffset"}
    precheck = commands["check_data_and_provenance"]
    assert precheck[precheck.index("--candidate-knots")+1] == "32"
    assert "--resize-candidate-warm-start" in precheck
    assert "full cubic vector <= 40" in result.stdout
    assert list(tmp_path.iterdir()) == [source]


def test_k32_evaluation_uses_checkpoint_without_silently_resizing(tmp_path):
    source = tmp_path / "trained32.pt"
    source.write_bytes(b"dry-run fixture")
    commands = _commands(_run(tmp_path, "--candidate-knots", "32",
                              "--checkpoint", _bash_path(source)))
    assert "train_fresh" not in commands and "train_resume" not in commands
    assert "--resize-candidate-warm-start" not in commands["check_data_and_provenance"]
    assert _parsed(commands["benchmark_six_methods"]).max_internal_knots == 32


@pytest.mark.parametrize("value", ["0", "23", "189", "1.5", "bad", "-32"])
def test_invalid_capacity_fails_without_side_effects(tmp_path, value):
    result = _run(tmp_path, "--candidate-knots", value)
    assert result.returncode != 0 and "candidate-knots" in result.stderr
    assert not list(tmp_path.iterdir())


def test_resize_flag_alone_cannot_relabel_an_old_checkpoint(tmp_path):
    result = _run(tmp_path, "--resize-candidate-warm-start", "--candidate-knots", "32")
    assert result.returncode != 0 and "requires a new" in result.stderr
    assert not list(tmp_path.iterdir())


def test_preflight_capacity_matching_and_explicit_downsize(tmp_path, preflight, monkeypatch):
    import torch
    import spline_fitting.checkpointing as checkpointing

    path = tmp_path / "source64.pt"
    payload = {"objective_version": checkpointing.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
               "model_config": {"max_internal_knots": 64}, "training_config": {"mse_tolerance": 5e-5}}
    torch.save(payload, path)
    monkeypatch.setattr(checkpointing, "build_model_from_checkpoint",
                        lambda p: (None, p["model_config"], None))
    with pytest.raises(ValueError, match="32-internal-candidate"):
        preflight.checkpoint_record(path, deployment=True, tolerance=5e-5, candidate_knots=32)
    record = preflight.checkpoint_record(path, deployment=True, tolerance=5e-5,
                                        candidate_knots=32, resize_candidates=True)
    assert record["source_candidate_knots"] == 64
    assert record["requested_candidate_knots"] == 32
    assert record["resize_candidate_warm_start"]
    for cap in (64, 96):
        with pytest.raises(ValueError, match="strictly smaller"):
            preflight.checkpoint_record(path, deployment=True, tolerance=5e-5,
                                        candidate_knots=cap, resize_candidates=True)
    payload["model_config"]["max_internal_knots"] = 32
    torch.save(payload, path)
    assert preflight.checkpoint_record(path, deployment=True, tolerance=5e-5,
                                      candidate_knots=32)["source_candidate_knots"] == 32


def test_k32_same_run_resume_validates_all_companion_capacities(preflight, resume_bundle):
    import torch
    from spline_fitting.models.v16_network import V16CandidateSelectionNetwork

    model = V16CandidateSelectionNetwork(hidden_dim=16, encoder_layers=1, selector_layers=1,
                                        max_internal_knots=32, mse_tolerance=5e-5)
    payload = deepcopy(resume_bundle["payload"])
    payload["model_config"] = model.get_config()
    payload["model_state_dict"] = model.state_dict()
    payload["training_config"]["candidate_knots"] = 32
    payload["training_config"]["resize_candidate_warm_start"] = True
    for name in ("last", "output", "proposal"):
        torch.save({**payload, **({"stage": "proposal", "epoch": 4} if name == "proposal" else {})},
                   resume_bundle[name])
    arguments = list(resume_bundle["arguments"])
    arguments[arguments.index("--candidate-knots")+1] = "32"
    assert preflight.resume_status(resume_bundle["last"], arguments) == "completed"
    arguments[arguments.index("--candidate-knots")+1] = "64"
    with pytest.raises(ValueError, match="capacity mismatch"):
        preflight.resume_status(resume_bundle["last"], arguments)


def test_k32_resume_dry_run_does_not_repeat_capacity_migration(tmp_path):
    last = tmp_path / "results with spaces/checkpoints" / f"{RUN_NAME}.last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"resume dry-run fixture")
    commands = _commands(_run(tmp_path, "--stable-selection", "--candidate-knots", "32", "--resume-run"))
    train = _parsed(commands["train_resume"])
    assert train.candidate_knots == 32 and not train.resize_candidate_warm_start
    assert train.warm_start_checkpoint is None
