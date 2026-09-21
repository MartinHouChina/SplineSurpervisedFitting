"""Explicit specialist scopes, independent test ranges, and synthetic labels."""
import pytest
import torch

from test_run_v16_1070_overnight_linux import _bash_path, _commands, _parsed, _run
from spline_fitting.data.v16_mixed import MixedTrainingCurves, _CompactShapeSyntheticDataset


def test_low_capacity_specialist_keeps_all_test_sources_and_common_test_range(tmp_path):
    checkpoint = tmp_path / "k32.pt"
    checkpoint.write_bytes(b"dry run only")
    commands = _commands(_run(tmp_path, "--anchored-selection", "--warm-start-checkpoint", _bash_path(checkpoint),
        "--candidate-knots", "16", "--source-max-knots", "16", "--resize-candidate-warm-start",
        "--validation-source", "UJI", "--synthetic-shape-domain", "handwriting",
        "--synthetic-shape-fraction", ".5", "--synthetic-simple-fraction", ".25",
        "--candidate-refinement-layers", "2", "--selection-refinement-layers", "2",
        "--decoder-refinement-layers", "2", "--max-point-error-weight", ".05",
        "--max-point-error-tolerance", "5e-4"))
    train = _parsed(commands["train_fresh"])
    assert train.real_fraction == 0 and len(train.real_manifest) == 1
    assert train.real_manifest[0].name == "uji_pen_v2.jsonl"
    assert train.max_control_points == 20 and train.candidate_knots == 16
    assert train.synthetic_shape_domain == "handwriting"
    assert train.synthetic_shape_fraction == .5 and train.synthetic_simple_fraction == .25
    assert train.proposal_refinement_layers == train.selection_refinement_layers == train.survivor_refinement_layers == 2
    assert train.max_point_error_tolerance == 5e-4 and train.max_point_error_weight == .05
    assert commands["train_fresh"].count("--synthetic-shape-fraction") == 1
    benchmark = _parsed(commands["benchmark_six_methods"])
    assert benchmark.max_internal_knots == 16
    assert benchmark.max_knot_count == benchmark.synthetic_source_max_knots == 24
    for name in ("benchmark_six_methods", "plot_ours_cases", "plot_six_method_real_cases"):
        assert len(_parsed(commands[name]).manifest) == 4
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_real_training_requires_explicit_scope_and_keeps_tests_held_out(tmp_path):
    checkpoint = tmp_path / "k32.pt"
    checkpoint.write_bytes(b"dry run only")
    commands = _commands(_run(tmp_path, "--anchored-selection", "--warm-start-checkpoint", _bash_path(checkpoint),
        "--training-source", "USGS", "--training-real-fraction", ".4"))
    train = _parsed(commands["train_fresh"])
    assert train.real_fraction == .4 and len(train.real_manifest) == 1
    assert "usgs_contours" in train.real_manifest[0].as_posix()
    assert "--training-source" in commands["check_data_and_provenance"]
    assert len(_parsed(commands["benchmark_six_methods"]).manifest) == 4


@pytest.mark.parametrize("options", [
    ("--candidate-knots", "16"),
    ("--source-max-knots", "33", "--candidate-knots", "32"),
    ("--training-source", "test"),
    ("--training-source", "UJI", "--validation-source", "USGS"),
])
def test_unsafe_domain_options_fail_before_writes(tmp_path, options):
    result = _run(tmp_path, *options)
    assert result.returncode != 0 and not list(tmp_path.iterdir())


@pytest.mark.parametrize("domain", ["handwriting", "terrain", "industrial"])
def test_shape_domain_is_deterministic_and_has_no_fabricated_knot_labels(domain):
    config = dict(num_points=24, point_dim=2, min_control_points=8, max_control_points=20)
    data = MixedTrainingCurves(config, size=3, seed=24, real_fraction=0,
                              synthetic_shape_fraction=1, synthetic_shape_domain=domain)
    for index in range(3):
        sample = data[index]
        assert sample["synthetic_family"] == "shape_" + domain
        assert not sample["target_geometry_valid"]
        assert not sample["target_internal_knot_count_valid"]
        torch.testing.assert_close(sample["points"], data[index]["points"], rtol=0, atol=0)


def test_mixed_domain_preserves_historical_shapes_exactly():
    config = dict(num_points=24, point_dim=2)
    implicit = _CompactShapeSyntheticDataset(config, size=3, seed=99)
    explicit = _CompactShapeSyntheticDataset(config, size=3, seed=99, domain="mixed")
    for index in range(3):
        assert implicit[index]["synthetic_family"] == explicit[index]["synthetic_family"]
        torch.testing.assert_close(implicit[index]["points"], explicit[index]["points"], rtol=0, atol=0)


def test_specialist_benchmark_generator_override_does_not_relabel_training(monkeypatch):
    import argparse
    import benchmark_v15_datasets as benchmark

    captured = {}
    original = dict(num_points=192, min_control_points=8, max_control_points=20)
    monkeypatch.setattr(benchmark, "_dataset_config_from_checkpoint", lambda *_: original)
    def stop(config, **kwargs):
        captured.update(config)
        raise RuntimeError("selected generator")
    monkeypatch.setattr(benchmark, "_select_indices", stop)
    args = argparse.Namespace(synthetic_source_max_knots=24, skip_synthetic=False,
        seed=20_000, scan_size=50, selection_seed=42, min_knot_count=4, max_knot_count=24,
        samples_per_knot_count=1)
    with pytest.raises(RuntimeError, match="selected generator"):
        benchmark.prepare_cases(args, {"training_config": {}}, {})
    assert captured["max_control_points"] == 28 and original["max_control_points"] == 20
