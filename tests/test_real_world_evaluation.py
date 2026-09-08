from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import V13_SET_RELOCATION_OBJECTIVE_VERSION  # noqa: E402
from spline_fitting.data import write_curve_manifest  # noqa: E402
from spline_fitting.models import SplineFittingNetwork  # noqa: E402


def _checkpoint(path: Path) -> None:
    config = {
        "point_dim": 2,
        "degree": 3,
        "hidden_dim": 16,
        "encoder_layers": 1,
        "max_internal_knots": 5,
        "min_parameter_gap": 1e-4,
        "min_knot_gap": 1e-3,
        "gap_parameterization": "strict",
        "lambda_poly": 1e-6,
        "lambda_knot": 1e-5,
        "structure_mode": "candidate_pruning_one_shot",
        "structure_attention_heads": 4,
        "geometry_feature_mode": "chord_derivatives",
        "pruning_residual_bandwidth": 0.05,
        "pruning_initial_keep_probability": 0.55,
        "one_shot_fixed_proposal_geometry": True,
        "one_shot_selection_policy": "threshold",
        "one_shot_safety_sigma": 0.0,
        "one_shot_selector_layers": 1,
        "one_shot_coverage_bins": 0,
        "candidate_local_attention_bandwidth": 0.08,
        "one_shot_joint_position_refinement": True,
        "one_shot_survivor_relocation": True,
        "one_shot_max_position_shift": 0.15,
        "stable_pilot_descriptors": True,
        "compute_first_derivative": False,
    }
    model = SplineFittingNetwork(**config)
    torch.save(
        {
            "objective_version": V13_SET_RELOCATION_OBJECTIVE_VERSION,
            "model_config": config,
            "model_state_dict": model.state_dict(),
            "dataset_config": {
                "num_points": 24,
                "point_dim": 2,
                "canonical_knot_tolerance": 0.005,
            },
            "deployment_config": {"error_tolerance": 0.005},
        },
        path,
    )


def test_real_world_evaluator_writes_unlabeled_metrics(tmp_path: Path) -> None:
    records = []
    for index in range(2):
        x = np.linspace(0.0, 1.0, 31, dtype=np.float32)
        points = np.stack([x, 0.2 * np.sin((index + 1) * np.pi * x)], axis=-1)
        points_path = tmp_path / f"curve_{index}.npy"
        np.save(points_path, points)
        records.append(
            {
                "sample_id": f"fixture-{index}",
                "source_dataset": "fixture_real",
                "group_id": f"region-{index}",
                "split": "test",
                "points_path": points_path.name,
                "num_points": 31,
                "point_dim": 2,
                "has_knot_labels": False,
                "metadata": {},
            }
        )
    manifest = write_curve_manifest(records, tmp_path / "manifest.jsonl")
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint)
    output = tmp_path / "evaluation.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_real_world.py",
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--max-samples",
            "2",
            "--batch-size",
            "2",
            "--timing-warmup",
            "0",
            "--timing-repeats",
            "1",
            "--geometry-samples",
            "16",
            "--device",
            "cpu",
            "--json-output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["overall"]["sample_count"] == 2
    assert report["knot_ground_truth_available"] is False
    assert report["timing"]["scope"].startswith("network forward_deployment only")
    assert report["overall"]["normalized_reference_mse_mean"] >= 0.0
    assert len(report["samples"]) == 2
    assert output.with_suffix(".csv").is_file()


def test_real_world_certified_mode_keeps_learned_result_and_audits_repair(
    tmp_path: Path,
) -> None:
    x = np.linspace(0.0, 1.0, 17, dtype=np.float32)
    points = np.stack([x, 0.35 * np.sin(5.0 * np.pi * x)], axis=-1)
    points_path = tmp_path / "curve.npy"
    np.save(points_path, points)
    manifest = write_curve_manifest(
        [
            {
                "sample_id": "certified-fixture",
                "source_dataset": "fixture_real",
                "group_id": "region-cert",
                "split": "test",
                "points_path": points_path.name,
                "num_points": 17,
                "point_dim": 2,
                "has_knot_labels": False,
                "metadata": {},
            }
        ],
        tmp_path / "manifest.jsonl",
    )
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint)
    output = tmp_path / "certified.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_real_world.py",
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--max-samples",
            "1",
            "--batch-size",
            "1",
            "--timing-warmup",
            "0",
            "--timing-repeats",
            "1",
            "--geometry-samples",
            "0",
            "--device",
            "cpu",
            "--deployment-mode",
            "certified",
            "--certified-mse-target",
            "1e-10",
            "--certified-position-sweeps",
            "0",
            "--json-output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    row = report["samples"][0]
    assert report["schema_version"] == "real_world_reference_certified_deployment_v3"
    assert report["deployment_mode"] == "certified"
    assert report["knot_ground_truth_available"] is False
    assert report["overall"]["reference_certificate_valid_fraction"] == 1.0
    assert row["learned_normalized_reference_mse"] >= 0.0
    assert row["normalized_reference_mse"] <= 1e-10
    assert row["reference_certificate_valid"] is True
    assert row["continuous_curve_guaranteed"] is False
    assert row["globally_minimal_knot_count_guaranteed"] is False
    assert report["timing"]["network_amortized_time_ms_per_curve"] >= 0.0
    assert report["timing"]["certified_postprocess_mean_time_ms_per_curve"] >= 0.0
