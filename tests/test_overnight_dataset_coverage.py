"""All default external sources are required and shared, without changing training."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import benchmark_v15_datasets as benchmark
import overnight_datasets as catalog
import overnight_linux_preflight as preflight
import visualize_v16_real_deployments as visualizer
import visualize_v16_ours_cases as ours
from test_run_v16_1070_overnight_linux import _bash_path, _commands, _parsed, _run


SOURCES = {"UJI", "NaturalEarth", "USGS", "IndustrialOffset"}


def write_source(path, name):
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index, split in enumerate(("train", "val", "test")):
        points = np.stack((np.linspace(0, 1, 32), np.linspace(0, 1, 32) ** (2 + index)), axis=-1)
        filename = f"{name}_{split}.npy"
        np.save(path.parent / filename, points)
        metadata = ({"benchmark_kind": "cad_driven_semi_synthetic_offset_curve",
                     "source_geometry_sha256": f"parent-{split}",
                     "profile_family": "test_family", "profile_variant": index}
                    if name == "IndustrialOffset" else {})
        records.append(dict(sample_id=f"{name}_{split}", source_dataset=name,
                            group_id=f"{name}_{split}_group", split=split,
                            points_path=filename, num_points=32, point_dim=2,
                            has_knot_labels=False, metadata=metadata))
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return records


@pytest.fixture
def external_data(tmp_path):
    root = tmp_path / "external data"
    for name, path in catalog.default_manifests(root).items():
        write_source(path, name)
    return root


def test_catalog_has_four_independent_defaults_not_archived_usgs_snapshots():
    paths = catalog.default_manifests(Path("dataset"))
    assert set(paths) == SOURCES
    assert len(set(paths.values())) == 4
    assert "large_scale_initial" not in str(paths)


@pytest.mark.parametrize("values", [
    ["Synthetic=foo.jsonl"], ["UJI=other.jsonl"], ["bad"], ["Empty="],
    ["Other=data/splits/uji_pen_v2.jsonl"],
])
def test_catalog_rejects_duplicate_reserved_or_invalid_extra_sources(values):
    with pytest.raises(ValueError):
        catalog.parse_manifests(values, defaults=catalog.default_manifests(Path("data")))


def test_benchmark_and_both_visualizers_share_all_external_case_selection(external_data, monkeypatch):
    monkeypatch.setattr(benchmark, "_dataset_config_from_checkpoint",
                        lambda *args: {"num_points": 16, "point_dim": 2})
    arguments = ["--data-root", str(external_data), "--real-samples-per-dataset", "1"]
    benchmark_args = benchmark.parser().parse_args([
        "--checkpoint", "not-loaded.pt", "--output-dir", "not-written", "--skip-synthetic", *arguments])
    cases, provenance = benchmark.prepare_cases(benchmark_args, {}, {})
    assert {case["dataset"] for case in cases} == SOURCES
    assert all(case["sample_id"].endswith("_test") for case in cases)
    for cli in (arguments, ours.with_ours_defaults(arguments)):
        args = visualizer.parser().parse_args(cli)
        args.skip_synthetic, args.skip_real = True, False
        other_cases, other_provenance = visualizer.prepare_cases(args, {}, {})
        assert [(case["dataset"], case["sample_id"]) for case in other_cases] == [
            (case["dataset"], case["sample_id"]) for case in cases]
        assert other_provenance == provenance
    offset = next(case for case in cases if case["dataset"] == "IndustrialOffset")
    assert offset["source_kind"] == "procedural_cad_offset"
    assert "not measured" in offset["dataset_label"]
    assert offset["canonical_k"] is None and offset["source_k"] is None
    offset_provenance = next(item for item in provenance if item["dataset"] == "IndustrialOffset")
    assert not offset_provenance["present_in_checkpoint_validation_manifests"]


def test_preflight_requires_and_records_industrial_without_adding_validation(external_data, monkeypatch):
    import spline_fitting.data.v16_mixed as mixed

    validated = []
    monkeypatch.setattr(mixed, "load_real_sources", lambda paths, **kwargs: (validated.extend(paths), []))
    record = preflight.main(["--device", "cpu", "--data-root", str(external_data), "--validate-real-splits"])
    assert set(record["datasets"]) == SOURCES
    assert set(validated) == {catalog.default_manifests(external_data)[name]
                              for name in catalog.VALIDATION_SOURCE_NAMES}
    industrial = record["datasets"]["IndustrialOffset"]
    assert industrial["source_kind"] == "procedural_cad_offset"
    assert not industrial["used_for_validation_by_this_runner"]
    assert industrial["test_curves"] == 1


def test_missing_industrial_is_not_silently_skipped(external_data, monkeypatch):
    path = catalog.default_manifests(external_data)["IndustrialOffset"]
    path.unlink()
    with pytest.raises(FileNotFoundError, match="industrial_offsets"):
        preflight.main(["--device", "cpu", "--data-root", str(external_data)])
    monkeypatch.setattr(benchmark, "_dataset_config_from_checkpoint", lambda *a: {"num_points": 16, "point_dim": 2})
    args = SimpleNamespace(skip_synthetic=True, skip_real=False, data_root=external_data,
                           manifest=[], real_samples_per_dataset=1, selection_seed=42)
    with pytest.raises(FileNotFoundError, match="industrial_offsets"):
        benchmark.prepare_cases(args, {}, {})


def test_missing_reference_point_file_fails_preflight(external_data):
    path = catalog.default_manifests(external_data)["IndustrialOffset"]
    (path.parent / "IndustrialOffset_test.npy").unlink()
    with pytest.raises(FileNotFoundError, match="IndustrialOffset.*missing point file"):
        preflight.main(["--device", "cpu", "--data-root", str(external_data)])


def test_preflight_reports_source_and_path_progress(external_data, capsys):
    path = catalog.default_manifests(external_data)["UJI"]
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records.extend({**records[0], "sample_id": f"progress_fixture_{index}"} for index in range(1997))
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    preflight.main(["--device", "cpu", "--data-root", str(external_data)])
    output = capsys.readouterr().out
    assert "UJI: reading required manifest" in output
    assert "UJI: checked 2000/2000 point-file paths" in output
    assert "IndustrialOffset: reading required manifest" in output


def test_offset_parent_cannot_cross_split_even_with_different_group_ids(tmp_path):
    records = write_source(tmp_path / "manifest.jsonl", "IndustrialOffset")
    records[2]["metadata"]["source_geometry_sha256"] = records[0]["metadata"]["source_geometry_sha256"]
    with pytest.raises(ValueError, match="parent-profile leakage"):
        catalog.validate_source_records("IndustrialOffset", records)


def test_duplicate_source_snapshot_rejected_even_with_different_paths(tmp_path):
    records = write_source(tmp_path / "manifest.jsonl", "USGS")
    seen = set()
    catalog.validate_source_records("USGS", records, seen_sources=seen)
    with pytest.raises(ValueError, match="Duplicate source_dataset"):
        catalog.validate_source_records("USGS_old_snapshot", records, seen_sources=seen)


def test_runner_extra_source_reaches_all_evaluation_stages_but_not_training(tmp_path):
    extra = tmp_path / "extra manifest.jsonl"
    commands = _commands(_run(tmp_path, "--reliable-selection", "--manifest", f"Extra={_bash_path(extra)}"))
    train = _parsed(commands["train_fresh"])
    assert len(train.real_manifest) == 3 and train.real_fraction == 0
    for stage in ("benchmark_six_methods", "plot_ours_cases", "plot_six_method_real_cases"):
        assert {value.split("=", 1)[0] for value in _parsed(commands[stage]).manifest} == SOURCES | {"Extra"}
    assert f"Extra={_bash_path(extra)}" in commands["check_data_and_provenance"]
    assert not list(tmp_path.iterdir())


def test_runner_preparation_includes_local_offset_generation_without_overwrite(tmp_path):
    commands = _commands(_run(tmp_path, "--prepare-real-data", "--data-root", _bash_path(tmp_path / "absent")))
    command = commands["prepare_industrial_offsets"]
    assert "scripts/prepare_industrial_offsets.py" in command
    assert "--download" not in command and "--overwrite" not in command
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["UJI", "NaturalEarth", "USGS", "IndustrialOffset", "Synthetic"])
def test_runner_cannot_replace_required_source_with_extra_manifest(tmp_path, name):
    result = _run(tmp_path, "--manifest", f"{name}=other.jsonl")
    assert result.returncode != 0 and "reserved dataset" in result.stderr
    assert not list(tmp_path.iterdir())
