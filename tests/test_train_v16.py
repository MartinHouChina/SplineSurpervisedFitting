from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_v16
from spline_fitting.checkpointing import V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION
from spline_fitting.data.v16_mixed import (
    MixedTrainingCurves,
    ValidationCurves,
    _curve_record,
    grouped_indices,
    load_real_sources,
)
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork


def test_default_profile_is_engineering_90_and_reduced_cost():
    args = train_v16.parser().parse_args([])
    train_v16.validate_args(args)
    assert args.proposal_pass_target == pytest.approx(0.90)
    assert args.deployment_pass_target == pytest.approx(0.90)
    assert (args.epochs, args.proposal_epochs) == (60, 20)
    assert (args.train_size, args.val_size, args.real_val_size) == (2400, 500, 100)
    assert args.synthetic_boundary_val_size == 32
    assert args.batch_size == 16
    assert args.proposal_high_k_fraction == pytest.approx(0.5)
    assert args.proposal_high_k_min_knots == 40
    assert (args.min_control_points, args.max_control_points) == (8, 60)
    assert args.candidate_knots == 56
    assert args.knot_min_span == pytest.approx(0.01)
    assert args.one_shot_selection_policy == "mass_topk"
    assert args.one_shot_safety_sigma == pytest.approx(0.25)
    assert args.one_shot_safety_knots == 2
    assert args.final_safety_sigma == pytest.approx(0.05)
    assert args.final_safety_knots == 0
    assert args.safety_anneal_epochs == 10
    assert args.teacher_prefix_search_steps == 7
    assert args.teacher_low_count_sweep == 16
    assert args.synthetic_count_role == "upper_bound"
    assert args.initial_keep_fraction == pytest.approx(30 / 56)
    assert args.synthetic_geometry_oracle_teacher is False
    assert args.oracle_teacher_extra_knots == 2
    assert args.supervised_count_weight == pytest.approx(1.0)
    assert args.supervised_over_count_weight == pytest.approx(1.0)
    assert args.complexity_max_scale == pytest.approx(4.0)
    assert args.true_parameter_weight == pytest.approx(0.1)
    assert args.proposal_knot_coverage_weight == pytest.approx(1.0)
    assert args.proposal_knot_assignment_weight == pytest.approx(1.0)
    assert args.selected_knot_position_weight == pytest.approx(1.0)
    assert args.knot_position_beta == pytest.approx(0.01)
    assert args.certified_minimal_source is True
    assert args.minimality_margin == pytest.approx(0.2)
    assert args.minimality_audit_points == 512
    config = train_v16.synthetic_dataset_config(args)
    assert config["min_control_points"] == 8
    assert config["max_control_points"] == 60
    assert config["knot_min_span"] == pytest.approx(0.01)
    assert config["certified_minimal_source"] is True
    assert config["canonical_knot_tolerance"] == pytest.approx(
        args.mse_tolerance ** 0.5
    )


def test_uncertified_synthetic_ablation_disables_canonical_certificate():
    args = train_v16.parser().parse_args(["--no-certified-minimal-source"])
    train_v16.validate_args(args)
    config = train_v16.synthetic_dataset_config(args)
    assert config["certified_minimal_source"] is False
    assert config["canonical_knot_tolerance"] == 0.0


def test_certified_count_labels_require_enough_candidate_capacity():
    args = train_v16.parser().parse_args([
        "--candidate-knots", "15", "--max-control-points", "24",
    ])
    with pytest.raises(ValueError, match="maximum synthetic internal-knot count"):
        train_v16.validate_args(args)


def test_synthetic_knot_span_must_support_the_maximum_source_count():
    args = train_v16.parser().parse_args(["--knot-min-span", "0.02"])
    with pytest.raises(ValueError, match="knot-min-span.*max-control-points"):
        train_v16.validate_args(args)


def test_certified_count_labels_require_reachable_minimum_count():
    args = train_v16.parser().parse_args([
        "--min-control-points", "7", "--min-selected-knots", "4",
    ])
    with pytest.raises(ValueError, match="minimum synthetic internal-knot count"):
        train_v16.validate_args(args)


@pytest.mark.parametrize(
    "arguments,reason",
    [
        (["--proposal-high-k-fraction", "1.1"], "must lie in"),
        (["--proposal-high-k-min-knots", "57"], "inside the synthetic"),
        (
            ["--proposal-high-k-min-knots", "4"],
            "requires a nonempty low-K range",
        ),
    ],
)
def test_proposal_high_k_sampling_arguments_are_strict(arguments, reason):
    args = train_v16.parser().parse_args(arguments)
    with pytest.raises(ValueError, match=reason):
        train_v16.validate_args(args)


def test_full_knot_vector_size_64_means_56_internal_candidates():
    args = train_v16.parser().parse_args(["--full-knot-vector-size", "64"])
    train_v16.validate_args(args)
    assert args.candidate_knots == 56
    both = train_v16.parser().parse_args([
        "--candidate-knots", "64", "--full-knot-vector-size", "72",
    ])
    with pytest.raises(ValueError, match="either"):
        train_v16.validate_args(both)


def test_proposal_transfer_includes_parameter_head_but_not_selector():
    source = V16CandidateSelectionNetwork(
        point_dim=2, hidden_dim=16, encoder_layers=1,
        max_internal_knots=4, attention_heads=4, selector_layers=1,
    )
    target = V16CandidateSelectionNetwork(
        point_dim=2, hidden_dim=16, encoder_layers=1,
        max_internal_knots=8, attention_heads=4, selector_layers=1,
    )
    with torch.no_grad():
        for name, value in source.named_parameters():
            if name.startswith("parameter_head."):
                value.fill_(0.125)
    selector_before = target.keep_head.weight.detach().clone()
    transfer = train_v16.transfer_proposal_weights(
        target, {"model_state_dict": source.state_dict()},
    )
    transferred = transfer.copied + transfer.resized
    assert any(name.startswith("encoder.") for name in transferred)
    assert any(name.startswith("parameter_head.") for name in transferred)
    assert any(name.startswith("candidate_head.") for name in transferred)
    assert not any(name.startswith("keep_head.") for name in transferred)
    assert transfer.resized == ("candidate_head.interval_queries",)
    assert transfer.retained_target == (
        "candidate_head.interval_query_anchors",
    )
    assert all(
        torch.allclose(value, torch.full_like(value, 0.125))
        for name, value in target.named_parameters()
        if name.startswith("parameter_head.")
    )
    torch.testing.assert_close(target.keep_head.weight, selector_before)


def test_k64_to_k56_transfer_interpolates_only_interval_query_ranks():
    source = V16CandidateSelectionNetwork(
        point_dim=2, hidden_dim=16, encoder_layers=1,
        max_internal_knots=64, attention_heads=4, selector_layers=1,
    )
    target = V16CandidateSelectionNetwork(
        point_dim=2, hidden_dim=16, encoder_layers=1,
        max_internal_knots=56, attention_heads=4, selector_layers=1,
    )
    with torch.no_grad():
        ranks = (
            (torch.arange(65, dtype=torch.float32) + 0.5) / 65
        ).unsqueeze(-1)
        source.candidate_head.interval_queries.copy_(
            ranks.expand(-1, source.hidden_dim)
        )
    target_anchors = target.candidate_head.interval_query_anchors.clone()
    target_selector = target.keep_head.weight.detach().clone()

    transfer = train_v16.transfer_proposal_weights(
        target, {"model_state_dict": source.state_dict()},
    )

    assert transfer.resized == ("candidate_head.interval_queries",)
    assert transfer.retained_target == (
        "candidate_head.interval_query_anchors",
    )
    expected = (
        ((torch.arange(57, dtype=torch.float32) + 0.5) / 57)
        .unsqueeze(-1)
        .expand(-1, 16)
    )
    torch.testing.assert_close(
        target.candidate_head.interval_queries.detach(), expected,
    )
    torch.testing.assert_close(
        target.candidate_head.interval_query_anchors, target_anchors,
    )
    torch.testing.assert_close(target.keep_head.weight, target_selector)


def test_proposal_transfer_never_overwrites_deterministic_target_anchors():
    source = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=56,
        attention_heads=4, selector_layers=1,
    )
    target = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=56,
        attention_heads=4, selector_layers=1,
    )
    with torch.no_grad():
        source.candidate_head.interval_query_anchors.fill_(0.123)
    expected = target.candidate_head.interval_query_anchors.clone()
    transfer = train_v16.transfer_proposal_weights(
        target, {"model_state_dict": source.state_dict()},
    )
    assert transfer.retained_target == (
        "candidate_head.interval_query_anchors",
    )
    torch.testing.assert_close(
        target.candidate_head.interval_query_anchors, expected,
    )


def test_proposal_transfer_rejects_wrong_objective_or_partial_contract():
    source = V16CandidateSelectionNetwork(
        point_dim=2, hidden_dim=16, encoder_layers=1,
        max_internal_knots=64, attention_heads=4, selector_layers=1,
    )
    target = V16CandidateSelectionNetwork(
        point_dim=2, hidden_dim=16, encoder_layers=1,
        max_internal_knots=56, attention_heads=4, selector_layers=1,
    )
    checkpoint = {
        "objective_version": "unrelated_objective",
        "model_config": source.get_config(),
        "model_state_dict": source.state_dict(),
    }
    with pytest.raises(ValueError, match="objective is not compatible"):
        train_v16.transfer_proposal_weights(target, checkpoint)

    checkpoint["objective_version"] = V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION
    checkpoint["model_config"] = {
        **source.get_config(),
        "point_dim": 3,
    }
    with pytest.raises(ValueError, match="incompatible proposal contract.*point_dim"):
        train_v16.transfer_proposal_weights(target, checkpoint)

    checkpoint["model_config"] = source.get_config()
    incomplete = dict(source.state_dict())
    incomplete.pop("candidate_head.interval_queries")
    checkpoint["model_state_dict"] = incomplete
    with pytest.raises(ValueError, match="missing required proposal tensors"):
        train_v16.transfer_proposal_weights(target, checkpoint)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def dataset_config():
    return dict(num_points=24, point_dim=2, min_control_points=8,
                max_control_points=12, noise_std=0.001, normalize=True,
                canonical_knot_tolerance=0.0)


def write_manifest(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


@pytest.fixture
def real_manifest(tmp_path):
    specifications = [("train", "writer_train_a"), ("train", "writer_train_b"),
                      ("val", "writer_val_a"), ("val", "writer_val_a"),
                      ("val", "writer_val_b"), ("val", "writer_val_b"),
                      ("test", "writer_test_a"), ("test", "writer_test_b")]
    records = []
    for index, (split, group) in enumerate(specifications):
        t = np.linspace(0, 1, 29)
        points = np.stack([t, np.sin((index + 1) * t)], -1)
        point_path = tmp_path / f"points_{index}.npy"
        np.save(point_path, points)
        records.append(dict(sample_id=f"curve_{index}", source_dataset="TestReal",
                            group_id=group, split=split, points_path=point_path.name,
                            num_points=len(points), point_dim=2, has_knot_labels=False,
                            metadata={}))
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, records)
    return path, records


def test_real_sources_include_only_requested_train_and_val_splits(real_manifest):
    path, _ = real_manifest
    sources, provenance = load_real_sources([path], num_points=24, point_dim=2)
    label, train, val = sources[0]
    assert label == "TestReal"
    assert len(train) == 2 and len(val) == 4
    assert {row["split"] for row in train.records} == {"train"}
    assert {row["split"] for row in val.records} == {"val"}
    assert {row["group_id"] for row in train.records}.isdisjoint(
        {row["group_id"] for row in val.records}
    )
    assert train[0]["points"].shape == val[0]["points"].shape == (24, 2)
    assert provenance[0]["test_size"] == 2
    assert len(provenance[0]["manifest_sha256"]) == 64


@pytest.mark.parametrize("kind", ["group", "file"])
def test_real_sources_reject_cross_split_leakage(real_manifest, kind):
    path, records = real_manifest
    if kind == "group":
        records[2]["group_id"] = records[0]["group_id"]
    else:
        records[2]["points_path"] = records[0]["points_path"]
    write_manifest(path, records)
    with pytest.raises(ValueError, match="leakage"):
        load_real_sources([path], num_points=24, point_dim=2)


def test_real_sources_reject_duplicate_manifest(real_manifest):
    path, _ = real_manifest
    with pytest.raises(ValueError, match="duplicate real manifest"):
        load_real_sources([path, path.resolve()], num_points=24, point_dim=2)


def test_real_sources_reject_dimension_mismatch(real_manifest):
    path, _ = real_manifest
    with pytest.raises(ValueError, match="dimension mismatch"):
        load_real_sources([path], num_points=24, point_dim=3)


def test_actual_file_dimension_is_checked_when_loaded(real_manifest):
    path, records = real_manifest
    np.save(path.parent / records[0]["points_path"], np.ones((29, 3)))
    sources, _ = load_real_sources([path], num_points=24, point_dim=2)
    with pytest.raises(ValueError, match="does not match"):
        sources[0][1][0]


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_fixed_epoch_mixture_is_deterministic_and_respects_source_fraction(
    real_manifest, dataset_config, fraction,
):
    path, _ = real_manifest
    sources, _ = load_real_sources([path], num_points=24, point_dim=2)
    options = dict(size=6, seed=42, epoch=2, real_fraction=fraction)
    first = MixedTrainingCurves(dataset_config, sources, **options)
    second = MixedTrainingCurves(dataset_config, sources, **options)
    expected_label = "TestReal" if fraction == 1 else "Synthetic"
    for index in range(len(first)):
        a, b = first[index], second[index]
        assert a["source"] == b["source"] == expected_label
        torch.testing.assert_close(a["points"], b["points"], rtol=0, atol=0)
        assert a["target_internal_knot_count"].dtype == torch.long
        assert a["target_internal_knot_count_valid"].dtype == torch.bool
        assert int(a["target_internal_knot_count"]) == 0
        assert not bool(a["target_internal_knot_count_valid"])
        assert not bool(a["target_geometry_valid"])
        assert torch.count_nonzero(a["target_params"]) == 0
        assert torch.count_nonzero(a["target_internal_knots"]) == 0
        assert not a["target_internal_knot_mask"].any()
        if fraction == 1:
            train_points = [sources[0][1][i]["points"] for i in range(len(sources[0][1]))]
            assert any(torch.equal(a["points"], value) for value in train_points)
        else:
            torch.testing.assert_close(a["points"], first.synthetic[index]["points"], rtol=0, atol=0)


def test_proposal_high_k_strata_are_exact_and_reproducible(dataset_config):
    options = dict(
        size=20,
        seed=42,
        epoch=3,
        real_fraction=0.0,
        synthetic_high_k_fraction=0.5,
        synthetic_high_k_min_knots=7,
    )
    first = MixedTrainingCurves(dataset_config, **options)
    repeated = MixedTrainingCurves(dataset_config, **options)
    assert first.synthetic_draw_count == 20
    assert len(first.synthetic_high_k_indices) == 10
    assert first.synthetic_high_k_indices == repeated.synthetic_high_k_indices
    assert first.synthetic_low is not None
    assert first.synthetic_low.min_control_points == 8
    assert first.synthetic_low.max_control_points == 10
    assert first.synthetic_high is not None
    assert first.synthetic_high.min_control_points == 11
    assert first.synthetic_high.max_control_points == 12
    for index in range(len(first)):
        torch.testing.assert_close(
            first[index]["points"],
            repeated[index]["points"],
            rtol=0,
            atol=0,
        )

    joint_distribution = MixedTrainingCurves(
        dataset_config,
        size=20,
        seed=42,
        epoch=3,
        real_fraction=0.0,
        synthetic_high_k_fraction=0.0,
        synthetic_high_k_min_knots=7,
    )
    assert not joint_distribution.synthetic_high_k_indices
    assert joint_distribution.synthetic_low is None
    assert joint_distribution.synthetic_high is None
    assert joint_distribution.synthetic.min_control_points == 8
    assert joint_distribution.synthetic.max_control_points == 12


def test_narrow_low_k_targets_are_padded_to_global_capacity():
    points = torch.randn(12, 2)
    sample = {
        "source_minimality_certified": True,
        "source_internal_knot_count": 2,
        "true_params": torch.linspace(0.0, 1.0, 12),
        "true_internal_knots": torch.tensor([0.25, 0.75]),
        "true_internal_knot_mask": torch.tensor([True, True]),
    }
    record = _curve_record(
        points,
        "Synthetic",
        max_internal_knots=8,
        synthetic_sample=sample,
        certified=True,
    )
    assert record["target_internal_knots"].shape == (8,)
    assert record["target_internal_knot_mask"].shape == (8,)
    torch.testing.assert_close(
        record["target_internal_knots"][:2], torch.tensor([0.25, 0.75])
    )
    assert int(record["target_internal_knot_mask"].sum()) == 2
    assert torch.count_nonzero(record["target_internal_knots"][2:]) == 0


def test_certified_synthetic_minimal_count_targets_collate_with_real_curves(
    real_manifest, dataset_config,
):
    path, _ = real_manifest
    sources, _ = load_real_sources([path], num_points=24, point_dim=2)
    certified_config = dict(dataset_config)
    certified_config.update(
        min_control_points=8,
        max_control_points=8,
        canonical_knot_tolerance=5e-3,
        certified_minimal_source=True,
        minimality_margin=0.2,
        minimality_audit_points=64,
    )

    training = MixedTrainingCurves(
        certified_config,
        size=2,
        seed=9182,
        real_fraction=0.0,
    )
    training_sample = training[0]
    assert set(training_sample) == {
        "points",
        "source",
        "target_internal_knot_count",
        "target_internal_knot_count_valid",
        "target_params",
        "target_internal_knots",
        "target_internal_knot_mask",
        "target_geometry_valid",
    }
    assert training_sample["target_internal_knot_count"].dtype == torch.long
    assert training_sample["target_internal_knot_count_valid"].dtype == torch.bool
    assert int(training_sample["target_internal_knot_count"]) == 4
    assert bool(training_sample["target_internal_knot_count_valid"])
    assert bool(training_sample["target_geometry_valid"])
    assert training_sample["target_params"].shape == (24,)
    assert torch.all(training_sample["target_params"].diff() > 0)
    assert training_sample["target_params"][0] == 0
    assert training_sample["target_params"][-1] == 1
    assert training_sample["target_internal_knots"].shape == (4,)
    assert training_sample["target_internal_knot_mask"].shape == (4,)
    assert int(training_sample["target_internal_knot_mask"].sum()) == 4
    assert torch.all(training_sample["target_internal_knots"].diff() > 0)

    validation = ValidationCurves(
        certified_config,
        sources,
        size=1,
        seed=9182,
        real_per_source=2,
    )
    batch = next(iter(DataLoader(validation, batch_size=3, shuffle=False)))
    assert batch["source"] == ["Synthetic", "TestReal", "TestReal"]
    torch.testing.assert_close(
        batch["target_internal_knot_count"],
        torch.tensor([4, 0, 0], dtype=torch.long),
    )
    torch.testing.assert_close(
        batch["target_internal_knot_count_valid"],
        torch.tensor([True, False, False]),
    )
    torch.testing.assert_close(
        batch["target_geometry_valid"],
        torch.tensor([True, False, False]),
    )
    assert batch["target_params"].shape == (3, 24)
    assert batch["target_internal_knots"].shape == (3, 4)
    assert batch["target_internal_knot_mask"].shape == (3, 4)
    assert torch.count_nonzero(batch["target_params"][1:]) == 0
    assert torch.count_nonzero(batch["target_internal_knots"][1:]) == 0
    assert not batch["target_internal_knot_mask"][1:].any()


def test_validation_is_group_balanced_and_never_uses_test(real_manifest, dataset_config):
    path, records = real_manifest
    sources, _ = load_real_sources([path], num_points=24, point_dim=2)
    validation = ValidationCurves(dataset_config, sources, size=2, seed=1234, real_per_source=2)
    repeated = ValidationCurves(dataset_config, sources, size=2, seed=1234, real_per_source=2)
    selected = validation.selected_real_ids["TestReal"]
    assert repeated.selected_real_ids == validation.selected_real_ids
    lookup = {record["sample_id"]: record for record in records}
    assert len(validation) == 4 and len(selected) == 2
    assert {lookup[sample_id]["split"] for sample_id in selected} == {"val"}
    assert len({lookup[sample_id]["group_id"] for sample_id in selected}) == 2
    assert all(validation[index]["source"] == "Synthetic" for index in range(2))
    assert all(validation[index]["source"] == "TestReal" for index in range(2, 4))


def test_validation_reserves_deterministic_maximum_complexity_slots(
    dataset_config,
):
    validation = ValidationCurves(
        dataset_config,
        size=5,
        seed=1234,
        synthetic_boundary_samples=3,
    )
    repeated = ValidationCurves(
        dataset_config,
        size=5,
        seed=1234,
        synthetic_boundary_samples=3,
    )
    assert len(validation) == 5
    assert validation.synthetic_boundary is not None
    assert validation.synthetic_boundary.min_control_points == 12
    assert validation.synthetic_boundary.max_control_points == 12
    assert all(
        validation.entries[index][1] is validation.synthetic_boundary
        for index in range(3)
    )
    assert all(
        validation.entries[index][1] is validation.synthetic
        for index in range(3, 5)
    )
    for index in range(5):
        torch.testing.assert_close(
            validation[index]["points"], repeated[index]["points"],
            rtol=0,
            atol=0,
        )

    with pytest.raises(ValueError, match="non-negative integer"):
        ValidationCurves(
            dataset_config,
            size=2,
            synthetic_boundary_samples=True,
        )


def test_group_sampler_caps_without_duplicates_and_balances_prefix():
    records = [dict(group_id="large") for _ in range(8)] + [dict(group_id="small")]
    selected = grouped_indices(records, 100, seed=42)
    assert len(selected) == len(set(selected)) == len(records)
    assert {records[index]["group_id"] for index in selected[:2]} == {"large", "small"}


def test_checkpoint_ranking_enforces_worst_source_feasibility_before_minimum_k():
    def metrics(worst, count, mse):
        return dict(worst_deployment_pass_rate=worst, keep_count=count, deployment_mse=mse)

    target = 0.97
    feasible = train_v16.checkpoint_rank(metrics(0.97, 20, 2e-5), target)
    fewer_but_infeasible = train_v16.checkpoint_rank(metrics(0.96, 2, 1e-6), target)
    assert feasible > fewer_but_infeasible
    assert train_v16.checkpoint_rank(metrics(0.98, 10, 2.4e-5), target) > feasible
    assert train_v16.checkpoint_rank(metrics(0.98, 10, 1e-5), target) > train_v16.checkpoint_rank(
        metrics(1.0, 10, 2e-5), target
    )
    assert train_v16.checkpoint_rank(metrics(0.9, 25, 5e-5), target) > train_v16.checkpoint_rank(
        metrics(0.8, 1, 1e-6), target
    )


def test_validation_summary_reports_certified_count_and_position_accuracy():
    rows = [
        dict(
            source="Synthetic", mse=1e-5, dense_mse=1e-6, k=11,
            target_k=12, probability_mass=10.5, adaptive_threshold=1.0,
            parameter_rmse=0.01, knot_match_predicted=11,
            knot_match_true=12, knot_match_matched=10,
            knot_match_error_sum=0.04,
        ),
        dict(
            source="Synthetic", mse=2e-5, dense_mse=2e-6, k=14,
            target_k=12, probability_mass=13.2, adaptive_threshold=1.2,
            parameter_rmse=0.02, knot_match_predicted=14,
            knot_match_true=12, knot_match_matched=11,
            knot_match_error_sum=0.055,
        ),
    ]
    result = train_v16.summarize(rows, 2.5e-5)
    synthetic = result["by_source"]["Synthetic"]
    assert synthetic["target_count_mean"] == pytest.approx(12)
    assert synthetic["count_mae"] == pytest.approx(1.5)
    assert synthetic["count_bias"] == pytest.approx(0.5)
    assert synthetic["knot_match_precision"] == pytest.approx(21 / 25)
    assert synthetic["knot_match_recall"] == pytest.approx(21 / 24)
    assert synthetic["knot_matched_mae"] == pytest.approx(0.095 / 21)
    assert result["synthetic_count_mae"] == pytest.approx(1.5)


def test_validation_summary_audits_capacity_boundary_separately():
    rows = [
        dict(
            source="Synthetic", mse=mse, dense_mse=dense_mse, k=keep,
            target_k=target, probability_mass=float(keep),
            adaptive_threshold=0.5,
        )
        for mse, dense_mse, keep, target in (
            (5e-5, 1e-5, 52, 56),
            (2e-4, 2e-4, 50, 56),
            (1e-5, 1e-5, 4, 4),
        )
    ]
    result = train_v16.summarize(
        rows,
        1e-4,
        synthetic_boundary_knot_count=56,
    )
    assert result["synthetic_boundary_knot_count"] == 56
    assert result["synthetic_boundary_sample_count"] == 2
    assert result["synthetic_boundary_dense_pass_rate"] == pytest.approx(0.5)
    assert result["synthetic_boundary_deployment_pass_rate"] == pytest.approx(0.5)
    assert result["qualification_dense_pass_rate"] == pytest.approx(0.5)
    assert result["qualification_deployment_pass_rate"] == pytest.approx(0.5)


def test_mature_checkpoint_ranking_prefers_certified_count_and_knot_accuracy():
    common = dict(
        worst_deployment_pass_rate=0.95,
        deployment_mse=1e-5,
        synthetic_knot_match_f1=0.8,
        synthetic_knot_matched_mae=0.004,
    )
    close = dict(common, keep_count=14, synthetic_count_mae=0.5)
    too_small = dict(common, keep_count=10, synthetic_count_mae=2.0)
    assert train_v16.checkpoint_rank(close, 0.9, 0.02) > train_v16.checkpoint_rank(
        too_small, 0.9, 0.02
    )

    better_position = dict(close, synthetic_knot_match_f1=0.9)
    assert train_v16.checkpoint_rank(
        better_position, 0.9, 0.02
    ) > train_v16.checkpoint_rank(close, 0.9, 0.02)

    assert train_v16.checkpoint_rank(
        close, 0.9, 0.02, simplification_ready=True
    ) > train_v16.checkpoint_rank(
        close, 0.9, 0.02, simplification_ready=False
    )

    slightly_better_count_bad_geometry = dict(
        close, synthetic_count_mae=0.49, synthetic_knot_match_f1=0.6,
    )
    assert train_v16.checkpoint_rank(
        better_position, 0.9, 0.02,
    ) > train_v16.checkpoint_rank(
        slightly_better_count_bad_geometry, 0.9, 0.02,
    )

    missing_labels = {
        "worst_deployment_pass_rate": 0.95,
        "deployment_mse": 1e-6,
        "deployment_mse_p95": 2e-6,
        "keep_count": 8,
    }
    assert train_v16.checkpoint_rank(
        close, 0.9, 0.02,
    ) > train_v16.checkpoint_rank(
        missing_labels, 0.9, 0.02,
    )

    reportable = dict(
        close,
        worst_deployment_pass_rate=0.91,
        worst_dense_pass_rate=0.95,
        synthetic_count_mae=1.0,
        synthetic_knot_match_f1=0.70,
        synthetic_knot_matched_mae=0.004,
    )
    safe_but_bad_geometry = dict(
        reportable,
        worst_deployment_pass_rate=0.95,
        synthetic_count_mae=0.5,
        synthetic_knot_match_f1=0.40,
    )
    assert train_v16.checkpoint_rank(
        reportable, 0.9, 0.02,
    ) > train_v16.checkpoint_rank(
        safe_but_bad_geometry, 0.9, 0.02,
    )


def test_selection_safety_anneals_conservatively_and_requires_final_state():
    args = train_v16.parser().parse_args([])
    train_v16.validate_args(args)
    assert train_v16.selection_safety(args, 1.0) == pytest.approx((0.25, 2))
    assert train_v16.selection_safety(args, 0.5) == pytest.approx((0.15, 1))
    assert train_v16.selection_safety(args, 0.0) == pytest.approx((0.05, 0))
    assert not train_v16.simplification_is_ready(
        args, epoch=30, stage="joint",
        applied_safety_scale=0.1, applied_complexity_scale=1.0,
    )
    assert train_v16.simplification_is_ready(
        args, epoch=30, stage="joint",
        applied_safety_scale=0.0, applied_complexity_scale=1.0,
    )


def test_pass_feedback_controller_progresses_in_margin_band_and_rolls_back():
    args = train_v16.parser().parse_args([])
    train_v16.validate_args(args)
    assert train_v16.update_simplification_controller(
        args, pass_rate=0.93, complexity_scale=0.0, safety_scale=1.0,
    ) == pytest.approx((0.1, 0.9))
    # The old controller froze forever in this 90--92% band.
    assert train_v16.update_simplification_controller(
        args, pass_rate=0.91, complexity_scale=0.0, safety_scale=1.0,
    ) == pytest.approx((0.05, 0.95))
    assert train_v16.update_simplification_controller(
        args, pass_rate=0.89, complexity_scale=1.0, safety_scale=0.5,
    ) == pytest.approx((0.8, 0.7))


def training_command(output, *, tolerance="0.001", proposal_target="0"):
    return ["--epochs", "2", "--proposal-epochs", "1", "--train-size", "4",
            "--val-size", "2", "--batch-size", "2", "--num-points", "24",
            "--candidate-knots", "8", "--hidden-dim", "16", "--encoder-layers", "1",
            "--selector-layers", "1", "--attention-heads", "4", "--min-control-points", "8",
             "--max-control-points", "12", "--policy-samples", "2", "--counterfactual-edits", "1",
            "--no-certified-minimal-source",
            "--mse-tolerance", tolerance, "--proposal-pass-target", proposal_target,
            "--torch-num-threads", "1", "--device", "cpu", "--log-every-batches", "100",
            "--output", str(output)]


def test_two_stage_tiny_training_and_resume_preserves_history_and_optimizer(tmp_path):
    output = tmp_path / "tiny_v16.pt"
    command = training_command(output)
    assert train_v16.main(command) == 0
    last_path = tmp_path / "tiny_v16.last.pt"
    original = torch.load(last_path, map_location="cpu", weights_only=True)
    assert original["objective_version"] == V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION
    assert original["epoch"] == 2 and original["stage"] == "joint"
    assert original["qualification"]["schema_version"] == 4
    assert original["qualification"]["required_reporting_pass_rate"] == 0.90
    assert original["qualification"]["configured_proposal_pass_target"] == 0.0
    assert not original["qualification"]["formal_reporting_eligible"]
    assert (
        original["current_deployment_pass_constraint_satisfied"]
        is original["best_deployment_pass_constraint_satisfied"]
    )
    assert [entry["stage"] for entry in original["history"]] == ["proposal", "joint"]
    assert "proposal_knot_assignment_mae" in original["history"][0]["train"]
    assert original["loss_config"]["weights"][
        "proposal_knot_assignment_weight"
    ] == pytest.approx(1.0)
    assert original["proposal_ready"]
    assert output.is_file() and (tmp_path / "tiny_v16.proposal.pt").is_file()
    step_before = max(float(state["step"]) for state in original["optimizer_state_dict"]["state"].values())
    resumed_command = deepcopy(command)
    resumed_command[resumed_command.index("--epochs") + 1] = "3"
    resumed_command.extend([
        "--initial-keep-fraction", "0.33",
        "--resume", str(last_path),
    ])
    assert train_v16.main(resumed_command) == 0
    resumed = torch.load(last_path, map_location="cpu", weights_only=True)
    assert resumed["epoch"] == 3 and resumed["stage"] == "joint"
    assert resumed["history"][:2] == original["history"]
    assert [entry["epoch"] for entry in resumed["history"]] == [1, 2, 3]
    step_after = max(float(state["step"]) for state in resumed["optimizer_state_dict"]["state"].values())
    assert step_after == step_before + 2
    assert json.loads((tmp_path / "tiny_v16.history.json").read_text()) == resumed["history"]
    assert resumed["training_config"]["epochs"] == 3
    assert resumed["training_config"]["initial_keep_fraction"] == pytest.approx(
        original["training_config"]["initial_keep_fraction"]
    )
    assert resumed["training_config"]["proposal_high_k_fraction"] == pytest.approx(
        0.5
    )
    assert resumed["training_config"]["proposal_high_k_min_knots"] == 8
    best = torch.load(output, map_location="cpu", weights_only=True)
    assert best["epoch"] in (2, 3)
    assert best["deployment_config"]["network_forwards"] == 1
    assert best["deployment_config"]["final_refits"] == 1


def test_failed_proposal_gate_retains_diagnostics_but_creates_no_final_checkpoint(tmp_path, capsys):
    output = tmp_path / "infeasible.pt"
    command = training_command(output, tolerance="1e-12", proposal_target="1")
    assert train_v16.main(command) == 2
    assert not output.exists()
    assert (tmp_path / "infeasible.proposal.pt").is_file()
    last = torch.load(tmp_path / "infeasible.last.pt", map_location="cpu", weights_only=True)
    assert last["epoch"] == 1 and last["stage"] == "proposal"
    assert last["checkpoint_quality"] == "target_not_met"
    assert not last["best_deployment_pass_constraint_satisfied"]
    assert last["validation_metrics"]["worst_dense_pass_rate"] < 1
    assert "STOP: dense proposal did not reach" in capsys.readouterr().out


def test_failed_proposal_can_resume_with_more_warmup_and_preserved_loss_weights(tmp_path):
    output = tmp_path / "extended.pt"
    command = training_command(output, tolerance="1e-12", proposal_target="1")
    assert train_v16.main(command) == 2
    last_path = tmp_path / "extended.last.pt"
    payload = torch.load(last_path, map_location="cpu", weights_only=True)
    payload["loss_config"]["weights"]["policy_weight"] = 0.123
    torch.save(payload, last_path)
    command[command.index("--epochs") + 1] = "3"
    command[command.index("--proposal-epochs") + 1] = "2"
    command.extend(["--resume", str(last_path)])
    assert train_v16.main(command) == 2
    resumed = torch.load(last_path, map_location="cpu", weights_only=True)
    assert resumed["epoch"] == 2 and resumed["stage"] == "proposal"
    assert len(resumed["history"]) == 2
    assert resumed["loss_config"]["weights"]["policy_weight"] == 0.123
    assert not output.exists()


def test_resume_rejects_changed_output_directory(tmp_path):
    output = tmp_path / "original.pt"
    command = training_command(output)
    assert train_v16.main(command) == 0
    command[command.index("--epochs") + 1] = "3"
    command[command.index("--output") + 1] = str(tmp_path / "other.pt")
    command.extend(["--resume", str(tmp_path / "original.last.pt")])
    with pytest.raises(SystemExit) as error:
        train_v16.main(command)
    assert error.value.code == 2
    assert not (tmp_path / "other.pt").exists()


def test_resume_rejects_changed_proposal_high_k_sampling(tmp_path, capsys):
    output = tmp_path / "sampling.pt"
    command = training_command(output)
    assert train_v16.main(command) == 0
    command[command.index("--epochs") + 1] = "3"
    command.extend(
        [
            "--proposal-high-k-fraction",
            "0.25",
            "--resume",
            str(tmp_path / "sampling.last.pt"),
        ]
    )
    with pytest.raises(SystemExit) as error:
        train_v16.main(command)
    assert error.value.code == 2
    assert "proposal_high_k_fraction" in capsys.readouterr().err


def test_resume_rejects_an_older_simplification_revision(tmp_path):
    output = tmp_path / "old_revision.pt"
    command = training_command(output)
    assert train_v16.main(command) == 0
    last_path = tmp_path / "old_revision.last.pt"
    payload = torch.load(last_path, map_location="cpu", weights_only=True)
    payload["architecture_revision"] = "v16_adaptive_beta_mass_topk"
    torch.save(payload, last_path)
    command[command.index("--epochs") + 1] = "3"
    command.extend(["--resume", str(last_path)])
    with pytest.raises(SystemExit) as error:
        train_v16.main(command)
    assert error.value.code == 2
