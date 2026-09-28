"""Runner safety and tiny random-checkpoint smoke tests, not paper evidence."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_v16_short_ablation as entry
from spline_fitting.checkpointing import V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def options(output, *extra):
    return entry.parse_args([
        "--output-dir", str(output), "--device", "cpu", "--epochs", "1",
        "--train-size", "2", "--val-size", "2", "--test-size", "2",
        "--batch-size", "2", "--num-points", "16", "--skip-real-data",
        "--min-source-knots", "0", "--max-source-knots", "1",
        "--max-internal-knots", "4", "--torch-num-threads", "1", *extra,
    ])


@pytest.fixture
def tiny_checkpoint(tmp_path):
    torch.manual_seed(711)
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=4,
        attention_heads=2, selector_layers=1, min_selected_knots=1,
        one_shot_selection_policy="mass_topk", one_shot_adaptive_threshold=True,
    )
    payload = dict(
        objective_version=V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        model_config=model.get_config(), model_state_dict=model.state_dict(),
        epoch=17, role="random_tiny_test_fixture_not_trained",
        formal_qualification=True,  # Ancestor claims must not be inherited.
    )
    path = tmp_path / "random_tiny.pt"
    torch.save(payload, path)
    return path, payload, model


@pytest.mark.parametrize("arguments", [
    ["--epochs", "0"], ["--train-size", "0"], ["--batch-size", "0"],
    ["--teacher-max-deletions", "-1"], ["--learning-rate", "nan"],
    ["--mse-tolerance", "0"], ["--max-squared-error-tolerance", "inf"],
    ["--count-weight", "-1"], ["--min-source-knots", "2"],
    ["--num-points", "7"], ["--seed", "-1"],
    ["--modes", "fixed", "fixed"], ["--max-internal-knots", "33", "--num-points", "64"],
])
def test_invalid_arguments_rejected_before_run(tmp_path, arguments):
    with pytest.raises(SystemExit):
        options(tmp_path / "unused", *arguments)
    assert not (tmp_path / "unused").exists()


def test_panels_use_disjoint_seeds_groups_and_points_and_only_synthetic_train(tmp_path):
    panels = entry.build_panels(options(tmp_path / "run"))
    assert {split: len(rows) for split, rows in panels.items()} == {"train": 2, "val": 2, "test": 2}
    assert all(row["dataset"] == "Synthetic" for rows in panels.values() for row in rows)
    ids = [row["sample_id"] for rows in panels.values() for row in rows]
    hashes = [entry.tensor_hash(row["points"]) for rows in panels.values() for row in rows]
    assert len(set(ids)) == len(set(hashes)) == 6
    metadata = entry.panel_metadata(panels)
    assert "points" not in metadata["train"][0]
    assert metadata["train"][0]["points_sha256"] == hashes[0]


@pytest.mark.parametrize("kind", ["sample_id", "group_id", "points", "real_train", "split"])
def test_panel_leakage_and_real_training_are_rejected(tmp_path, kind):
    panels = entry.build_panels(options(tmp_path / "run"))
    if kind == "real_train":
        panels["train"][0]["dataset"] = "UJI"
    elif kind == "split":
        panels["val"][0]["split"] = "test"
    else:
        panels["test"][0][kind] = deepcopy(panels["train"][0][kind])
    with pytest.raises(ValueError, match="leakage|real-data|split mismatch"):
        entry.validate_panels(panels)


def _row(dataset, *, passed, failed=False, count=2, mse=1e-5):
    return dict(dataset=dataset, status="failed" if failed else "ok",
                joint_pass=passed, fit_pass=passed, k=count, mse=mse,
                max_squared_error=mse * 2, total_ms=2.0, network_ms=1.0)


def test_summary_keeps_failures_in_denominator_and_uses_source_macro():
    rows = [_row("A", passed=True), _row("A", passed=False, failed=True),
            _row("A", passed=False), _row("B", passed=True)]
    report = entry.summary(rows)
    assert report["overall"]["n"] == 4 and report["overall"]["failures"] == 1
    assert report["overall"]["dual_pass_rate"] == 0.5
    assert report["per_source"]["A"]["dual_pass_rate"] == pytest.approx(1 / 3)
    assert report["macro_dual_pass"] == pytest.approx(2 / 3)
    assert report["worst_source_dual_pass"] == pytest.approx(1 / 3)
    failed = entry.summary([_row("A", passed=False, failed=True)])
    assert failed["overall"]["mean_k"] is None
    assert failed["overall"]["dual_pass_rate"] == 0.0


def test_validation_rank_prioritizes_pass_over_count():
    reliable = {"summary": entry.summary([_row("A", passed=True, count=4)])}
    sparse_failure = {"summary": entry.summary([_row("A", passed=False, count=0)])}
    assert entry.validation_rank(reliable) < entry.validation_rank(sparse_failure)


def test_evaluation_warmup_and_curve_failure_do_not_remove_cases(tmp_path, tiny_checkpoint, monkeypatch):
    _, _, model = tiny_checkpoint
    a = options(tmp_path / "run", "--mse-tolerance", "1", "--max-squared-error-tolerance", "1")
    cases = entry.build_panels(a)["test"]
    original_forward = model.forward_deployment
    calls = 0
    def failing_first_curve(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls <= 2:  # Warm-up and first measured curve fail numerically.
            raise RuntimeError("synthetic numerical-failure fixture")
        return original_forward(*args, **kwargs)
    monkeypatch.setattr(model, "forward_deployment", failing_first_curve)
    report = entry.evaluate(model, cases, a)
    assert calls == 3 and len(report["rows"]) == 2
    assert report["rows"][0]["status"] == "failed"
    assert report["rows"][1]["status"] == "ok"
    assert report["summary"]["overall"]["n"] == 2
    assert report["summary"]["overall"]["failures"] == 1
    assert report["summary"]["overall"]["dual_pass_rate"] == 0.5


def test_teacher_cache_reused_only_for_identical_protocol_and_ordered_points(tmp_path, tiny_checkpoint, monkeypatch):
    _, _, model = tiny_checkpoint
    a = options(tmp_path / "run", "--mse-tolerance", "1", "--max-squared-error-tolerance", "1")
    cases = entry.build_panels(a)["train"]
    path = tmp_path / "teacher.pt"
    labels = entry.build_cache(model, cases, path, a, "protocol-a")
    assert len(labels) == 2 and all(row["feasible"] for row in labels)
    before = entry.sha256(path)
    monkeypatch.setattr(entry, "build_fixed_teacher", lambda *args, **kwargs: pytest.fail("cached teacher regenerated"))
    reloaded = entry.build_cache(model, cases, path, a, "protocol-a")
    assert entry.sha256(path) == before
    assert torch.equal(reloaded[0]["mask"], labels[0]["mask"])
    for changed_cases, protocol in ((cases, "protocol-b"), (list(reversed(cases)), "protocol-a")):
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            entry.build_cache(model, changed_cases, path, a, protocol)
    assert entry.sha256(path) == before


def test_existing_output_refused_without_modifying_contents(tmp_path, tiny_checkpoint):
    path, _, _ = tiny_checkpoint
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "user_data.json"
    entry.atomic_json(sentinel, {"preserve": True})
    before = sentinel.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        entry.main(["--checkpoint", str(path), "--output-dir", str(output)])
    assert sentinel.read_bytes() == before and list(output.iterdir()) == [sentinel]


def test_arm_selects_epoch_on_validation_never_test_and_logs_loss_once(tmp_path, tiny_checkpoint, monkeypatch):
    path, original, model = tiny_checkpoint
    output = tmp_path / "run"
    output.mkdir()
    a = options(output, "--checkpoint", str(path), "--mse-tolerance", "1", "--max-squared-error-tolerance", "1")
    panels = entry.build_panels(a)
    labels = entry.build_fixed_teacher(model, torch.stack([row["points"] for row in panels["train"]]),
                                       mse_tolerance=1, max_squared_error_tolerance=1)
    teachers = {"train": labels, "val": labels}
    calls = []
    def fake_evaluate(current_model, cases, args, *, geometry=False, label="validation"):
        split = cases[0]["split"]
        calls.append((split, label))
        # Epoch 0 passes validation; training epoch 1 does not. Test always
        # passes and must never be eligible to change the selected epoch.
        passed = "epoch0" in label or split == "test"
        return {"rows": [], "summary": entry.summary([_row("Synthetic", passed=passed)])}
    def fake_loss(current_model, points, target, mode, **kwargs):
        first = next(p for p in current_model.parameters() if p.requires_grad)
        loss = first.square().sum() * 0 + 2.0
        return loss, {"loss": loss.detach()}
    monkeypatch.setattr(entry, "evaluate", fake_evaluate)
    monkeypatch.setattr(entry, "ablation_loss", fake_loss)
    monkeypatch.setattr(entry, "teacher_replay", lambda *args: {})
    result = entry.run_arm("fixed", original, panels, teachers, a,
                           {"checkpoint_sha256": entry.sha256(path), "arguments": {}})
    assert result["best"]["epoch"] == 0 and result["last"]["epoch"] == 1
    assert [split for split, _ in calls] == ["val", "val", "test", "test"]
    history = json.loads((output / "fixed/history.json").read_text(encoding="utf-8"))
    assert history[1]["train"]["loss"] == pytest.approx(2.0)
    assert not history[1]["selected"]


def test_tiny_cli_all_arms_resume_without_extra_optimization(tmp_path, tiny_checkpoint, monkeypatch):
    path, _, _ = tiny_checkpoint
    output = tmp_path / "tiny_all_arms"
    argv = [
        "--checkpoint", str(path), "--output-dir", str(output), "--device", "cpu",
        "--epochs", "1", "--train-size", "2", "--val-size", "2", "--test-size", "2",
        "--batch-size", "2", "--num-points", "16", "--skip-real-data",
        "--min-source-knots", "0", "--max-source-knots", "1",
        "--max-internal-knots", "4", "--torch-num-threads", "1",
        # Permissive thresholds are solely to exercise optimizer paths with a
        # RANDOM tiny checkpoint. They are not scientific success thresholds.
        "--mse-tolerance", "1", "--max-squared-error-tolerance", "1",
        "--modes", "fixed", "legacy", "geometry",
    ]
    monkeypatch.setattr(entry, "default_manifests", lambda *args: pytest.fail("synthetic smoke accessed real data"))
    assert entry.main(argv) == 0
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert "no test-based selection" in protocol["selection"]
    assert "not formal paper evidence" in protocol["limitation"]
    assert not (output / "teacher_test.pt").exists()
    assert (output / "ablation_comparison.png").is_file()
    assert (output / "summary.csv").is_file()
    result = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert [arm["mode"] for arm in result["arms"]] == ["fixed", "legacy", "geometry"]
    immutable_paths = [output / "teacher_train.pt", output / "teacher_val.pt"]
    for mode in ("fixed", "legacy", "geometry"):
        folder = output / mode
        last = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
        assert last["epoch"] == 1 and last["history"][-1]["train"]["optimized_batches"] == 1
        assert last["formal_qualification"] is False
        assert last["role"] == "diagnostic_short_finetune_not_formal_paper_model"
        assert last["parent_checkpoint_sha256"] == entry.sha256(path)
        assert last["objective_version"] == V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION
        best_report = json.loads((folder / "test_best.json").read_text(encoding="utf-8"))
        assert best_report["summary"]["overall"]["n"] == 2
        assert all(row["split"] == "test" for row in best_report["rows"])
        immutable_paths.extend([folder / "best.pt", folder / "last.pt", folder / "history.json"])
    hashes = {str(p): entry.sha256(p) for p in immutable_paths}
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *args, **kwargs: pytest.fail("completed resume optimized again"))
    monkeypatch.setattr(entry, "build_fixed_teacher", lambda *args, **kwargs: pytest.fail("completed resume rebuilt teacher"))
    assert entry.main([*argv, "--resume"]) == 0
    assert {str(p): entry.sha256(p) for p in immutable_paths} == hashes
