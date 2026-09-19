"""Explicit candidate downsizing is a fresh, auditable full-model warm start."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_v16  # noqa: E402
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork  # noqa: E402

QUERY = "candidate_head.interval_queries"
ANCHORS = "candidate_head.interval_query_anchors"


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_model(capacity=64, **changes):
    options = dict(point_dim=2, hidden_dim=16, encoder_layers=1,
                   max_internal_knots=capacity, attention_heads=4, selector_layers=1,
                   one_shot_selection_policy="mass_topk", one_shot_adaptive_threshold=True,
                   one_shot_coverage_bins=4, min_selected_knots=4)
    options.update(changes)
    return V16CandidateSelectionNetwork(**options)


def native_payload(model):
    return dict(
        objective_version=train_v16.V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        architecture_revision=train_v16.V16_ADAPTIVE_SELECTION_REVISION,
        simplification_contract=train_v16.V16_SIMPLIFICATION_CONTRACT,
        model_config=model.get_config(), model_state_dict=model.state_dict(),
    )


def curves():
    t = torch.linspace(0, 1, 96)
    return torch.stack((torch.stack((t, 0.2 * torch.sin(t * 8)), -1),
                        torch.stack((t, 0.1 * torch.cos(t * 5)), -1)))


def assert_valid_deployment(model):
    points = curves()
    model.eval()
    with torch.inference_mode():
        result = model(points)
        assert result["learned_keep_mask"].shape == (2, 32)
        assert (result["predicted_knot_count"] <= 32).all()
        for value in result.values():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all()
        for index in range(len(points)):
            selected = result["internal_knots"][index][result["learned_keep_mask"][index]]
            fit = train_v16.refit_bspline_control_points(
                result["params"][index], points[index], selected, degree=3)
            assert torch.isfinite(fit.fit_mse)
            assert torch.isfinite(fit.reconstructed_points).all()
    return result


@pytest.mark.parametrize("trust_enabled", [False, True])
def test_resize_64_to_32_copies_every_other_tensor_and_roundtrips(tmp_path, trust_enabled):
    source = small_model(parameter_trust_enabled=trust_enabled)
    target = small_model(32, parameter_trust_enabled=trust_enabled)
    original = deepcopy(source.state_dict())
    metadata = {}
    copied = train_v16.transfer_all_weights(
        target, native_payload(source), resize_candidate_warm_start=True,
        transfer_metadata=metadata)
    assert set(copied) == set(original) - {QUERY, ANCHORS}
    assert set(metadata["exact_copied_tensor_names"]) == set(copied)
    for name in copied:
        assert torch.equal(original[name], target.state_dict()[name]), name
    # 65 -> 33 aligns every second rank exactly, preserving both endpoints.
    assert torch.equal(target.state_dict()[QUERY], original[QUERY][::2])
    expected_anchors = (torch.arange(33) + 0.5) / 33
    assert torch.equal(target.state_dict()[ANCHORS], expected_anchors)
    assert metadata["source_candidate_knots"] == 64
    assert metadata["target_candidate_knots"] == 32
    assert metadata["resized_tensor_names"] == [QUERY]
    assert metadata["regenerated_buffer_names"] == [ANCHORS]
    assert metadata["method"] == "linear_interval_rank_align_corners"
    assert metadata["source_shapes"][QUERY] == [65, 16]
    assert metadata["target_shapes"][QUERY] == [33, 16]
    for name, value in source.state_dict().items():
        assert torch.equal(value, original[name]), name
    before = assert_valid_deployment(target)
    path = tmp_path / "native32.pt"
    payload = native_payload(target)
    payload["initializer_provenance"] = {"capacity_transfer": metadata}
    torch.save(payload, path)
    restored_payload = torch.load(path, weights_only=True)
    restored, config, _ = train_v16.build_model_from_checkpoint(restored_payload)
    assert config["max_internal_knots"] == 32
    assert restored_payload["initializer_provenance"]["capacity_transfer"] == metadata
    after = assert_valid_deployment(restored)
    for name in before:
        if torch.is_tensor(before[name]):
            assert torch.equal(before[name], after[name]), name


def test_nonaligned_capacity_uses_linear_rank_interpolation():
    source, target = small_model(64), small_model(30)
    with torch.no_grad():
        source.candidate_head.interval_queries.copy_(
            torch.arange(65, dtype=torch.float32).unsqueeze(1).expand(65, 16))
    train_v16.transfer_all_weights(target, native_payload(source),
                                  resize_candidate_warm_start=True)
    expected = torch.linspace(0, 64, 31).unsqueeze(1).expand(31, 16)
    torch.testing.assert_close(target.state_dict()[QUERY], expected)


def test_legacy_same_capacity_is_exact_and_capacity_change_requires_opt_in():
    source, same = small_model(), small_model()
    metadata = {}
    copied = train_v16.transfer_all_weights(same, native_payload(source),
                                           transfer_metadata=metadata)
    assert len(copied) == len(source.state_dict())
    assert metadata == {}
    for name, value in source.state_dict().items():
        assert torch.equal(same.state_dict()[name], value)
    with pytest.raises(ValueError, match="configuration mismatch"):
        train_v16.transfer_all_weights(small_model(32), native_payload(source))


@pytest.mark.parametrize("capacity", [64, 96])
def test_resize_rejects_equal_or_larger_capacity(capacity):
    with pytest.raises(ValueError, match="strict candidate-capacity downsize"):
        train_v16.transfer_all_weights(small_model(capacity), native_payload(small_model()),
                                      resize_candidate_warm_start=True)


@pytest.mark.parametrize("change", ["hidden", "policy", "tolerance", "trust_remove",
                                    "tensor", "extra", "missing", "anchors", "query_nan",
                                    "unknown_config", "contract"])
def test_resize_rejects_other_config_or_tensor_corruption_without_mutation(change):
    source, target = small_model(), small_model(32)
    payload = native_payload(source)
    if change == "hidden":
        target = small_model(32, hidden_dim=32)
    elif change == "policy":
        target = small_model(32, one_shot_selection_policy="threshold")
    elif change == "tolerance":
        target = small_model(32, mse_tolerance=5e-5)
    elif change == "trust_remove":
        payload = native_payload(small_model(parameter_trust_enabled=True))
    elif change == "tensor":
        payload["model_state_dict"]["keep_head.weight"] = torch.zeros(1, 17)
    elif change == "extra":
        payload["model_state_dict"]["unknown_candidate_tensor"] = torch.zeros(65)
    elif change == "missing":
        payload["model_state_dict"].pop("keep_head.weight")
    elif change == "anchors":
        payload["model_state_dict"][ANCHORS] = torch.zeros(65)
    elif change == "query_nan":
        payload["model_state_dict"][QUERY] = torch.full((65, 16), float("nan"))
    elif change == "unknown_config":
        payload["model_config"]["unknown_capacity_axis"] = 64
    elif change == "contract":
        payload["architecture_revision"] = "other"
    before = deepcopy(target.state_dict())
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        train_v16.transfer_all_weights(target, payload, resize_candidate_warm_start=True)
    for name, value in before.items():
        assert torch.equal(target.state_dict()[name], value), name


@pytest.mark.parametrize("other", [[], ["--resume", "x.pt"], ["--init-checkpoint", "x.pt"]])
def test_cli_resize_requires_full_warm_start_not_resume_or_proposal_transfer(other):
    args = train_v16.parser().parse_args(["--resize-candidate-warm-start", *other])
    with pytest.raises(ValueError, match="requires --warm-start-checkpoint"):
        train_v16.validate_args(args)


def test_resize_flag_is_omitted_by_default_and_ignored_only_as_initialization_metadata():
    args = train_v16.parser().parse_args([])
    train_v16.validate_args(args)
    original = train_v16.serial_args(args)
    assert "resize_candidate_warm_start" not in original
    previous = {**original, "resize_candidate_warm_start": True}
    assert train_v16.training_config_changes(original, previous, set()) == []
    assert train_v16.training_config_changes(previous, original, set()) == []
    previous["candidate_knots"] = 32
    assert train_v16.training_config_changes(original, previous, set()) == ["candidate_knots"]
    args = train_v16.parser().parse_args([
        "--warm-start-checkpoint", "source.pt", "--resize-candidate-warm-start"])
    train_v16.validate_args(args)
    assert train_v16.serial_args(args)["resize_candidate_warm_start"] is True


def training_command(output):
    return ["--epochs", "2", "--proposal-epochs", "1", "--train-size", "4",
            "--val-size", "2", "--batch-size", "2", "--num-points", "72",
            "--candidate-knots", "32", "--hidden-dim", "16", "--encoder-layers", "1",
            "--selector-layers", "1", "--attention-heads", "4", "--min-control-points", "8",
            "--max-control-points", "12", "--policy-samples", "2", "--counterfactual-edits", "1",
            "--teacher-prefix-search-steps", "2", "--no-certified-minimal-source",
            "--mse-tolerance", "0.001", "--proposal-pass-target", "0",
            "--torch-num-threads", "1", "--device", "cpu", "--log-every-batches", "100",
            "--output", str(output)]


def test_resize_training_starts_fresh_then_resumes_without_reapplying_resize(tmp_path, capsys):
    source_path, output = tmp_path / "source64.pt", tmp_path / "target32.pt"
    source_payload = native_payload(small_model(
        mse_tolerance=0.001, relocation_blend=train_v16.parser().parse_args([]).relocation_blend))
    # Deliberately unusable resume state must never be loaded for a warm start.
    source_payload.update(epoch=99, stage="joint", optimizer_state_dict={"old": True},
                          history=[{"epoch": 99}], complexity_scale=999,
                          next_selection_safety_scale=-1)
    torch.save(source_payload, source_path)
    source_bytes = source_path.read_bytes()
    command = training_command(output)
    assert train_v16.main(command + ["--warm-start-checkpoint", str(source_path),
                                    "--resize-candidate-warm-start"]) == 0
    last = tmp_path / "target32.last.pt"
    payload = torch.load(last, weights_only=True)
    assert [row["epoch"] for row in payload["history"]] == [1, 2]
    assert payload["model_config"]["max_internal_knots"] == 32
    assert payload["training_config"]["resize_candidate_warm_start"] is True
    record = payload["initializer_provenance"]
    assert record["mode"] == "full_model"
    assert record["epoch"] == 99
    assert record["initialized_tensor_names"] == []
    assert record["sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert record["capacity_transfer"]["optimizer_state"] == "fresh"
    assert record["capacity_transfer"]["history"] == "fresh"
    assert all(float(state["step"]) <= 4 for state in payload["optimizer_state_dict"]["state"].values())
    assert source_path.read_bytes() == source_bytes
    restored, _, _ = train_v16.build_model_from_checkpoint(payload)
    assert_valid_deployment(restored)
    command[command.index("--epochs") + 1] = "3"
    assert train_v16.main(command + ["--resume", str(last)]) == 0
    resumed = torch.load(last, weights_only=True)
    assert [row["epoch"] for row in resumed["history"]] == [1, 2, 3]
    assert resumed["initializer_provenance"] == record
    assert "resize_candidate_warm_start" not in resumed["training_config"]
    command[command.index("--epochs") + 1] = "4"
    command[command.index("--candidate-knots") + 1] = "16"
    with pytest.raises(SystemExit):
        train_v16.main(command + ["--resume", str(last)])
    assert "candidate_knots" in capsys.readouterr().err


def test_local_compact_checkpoint_can_downsize_and_fit_without_mutation():
    """Optional local integration: distribution tests do not require private weights."""
    path = ROOT / "outputs/checkpoints/overnight_compact_3090_r1.pt"
    if not path.exists():
        pytest.skip("local compact checkpoint is not distributed with the code")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["model_config"]["max_internal_knots"] == 64
    target = V16CandidateSelectionNetwork(**{**payload["model_config"], "max_internal_knots": 32})
    metadata = {}
    train_v16.transfer_all_weights(target, payload, resize_candidate_warm_start=True,
                                  transfer_metadata=metadata)
    assert metadata["source_candidate_knots"] == 64
    assert_valid_deployment(target)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
