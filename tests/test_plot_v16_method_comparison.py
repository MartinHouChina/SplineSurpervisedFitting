from __future__ import annotations

import copy
import struct
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import plot_v16_method_comparison as plotting  # noqa: E402


@pytest.fixture
def report() -> dict:
    summary = []
    measurements = []
    for dataset in ("Synthetic", "UJI"):
        for index, method in enumerate(plotting.PUBLISHED_METHODS):
            summary.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n": 1,
                    "failed": 0,
                    "mse_mean": 8e-6 * (index + 1),
                    "fit_pass_rate": 1.0 if index < 3 else 0.0,
                    "reference_mse_mean": (
                        9e-6 * (index + 1) if dataset == "UJI" else None
                    ),
                    "reference_pass_rate": (
                        1.0 if dataset == "UJI" and index < 2 else (
                            0.0 if dataset == "UJI" else None
                        )
                    ),
                    "final_k_mean": float(10 + index),
                    "canonical_k_mean": 8.5 if dataset == "Synthetic" else None,
                    "total_ms_mean": float(2 + 4 * index),
                    "network_ms_mean": 1.25 if method == "ours" else None,
                }
            )
            measurements.append(
                {
                    "dataset": dataset,
                    "sample_id": f"{dataset.lower()}-0",
                    "method": method,
                    "status": "ok",
                }
            )
    metadata = {
        "objective_version": "candidate_selection_counterfactual_bspline_v16",
        "checkpoint": "outputs/checkpoints/formal-v16.pt",
        "checkpoint_sha256": "0" * 64,
        "diagnostic_not_final": False,
        "mse_tolerance": 2.5e-5,
        "num_points": 192,
        "hardware": {"device": "cuda", "gpu": "test GPU", "threads": 1},
        "datasets": [],
    }
    metadata["fingerprint"] = plotting._metadata_fingerprint(metadata)
    return {
        "metadata": metadata,
        "summary": summary,
        "measurements": measurements,
    }


def test_formal_validation_rejects_relabelled_partial_or_tampered_reports(report) -> None:
    plotting.validate_v16_benchmark(report)

    v15 = copy.deepcopy(report)
    v15["metadata"]["objective_version"] = (
        "candidate_pruning_deployment_aligned_feedback_v15"
    )
    v15["metadata"]["fingerprint"] = plotting._metadata_fingerprint(v15["metadata"])
    with pytest.raises(ValueError, match="native v16"):
        plotting.validate_v16_benchmark(v15)

    partial = copy.deepcopy(report)
    missing_method = plotting.PUBLISHED_METHODS[-1]
    partial["summary"] = [
        row for row in partial["summary"] if row["method"] != missing_method
    ]
    with pytest.raises(ValueError, match="incomplete"):
        plotting.validate_v16_benchmark(partial)

    tampered = copy.deepcopy(report)
    tampered["metadata"]["mse_tolerance"] = 1e-3
    with pytest.raises(ValueError, match="fingerprint"):
        plotting.validate_v16_benchmark(tampered)


def test_unqualified_report_requires_explicit_diagnostic_opt_in(report) -> None:
    report["metadata"]["diagnostic_not_final"] = True
    report["metadata"]["fingerprint"] = plotting._metadata_fingerprint(
        report["metadata"]
    )
    with pytest.raises(ValueError, match="unqualified"):
        plotting.validate_v16_benchmark(report)
    plotting.validate_v16_benchmark(
        report,
        allow_unqualified_diagnostic=True,
    )


def test_four_metric_renderer_writes_input_and_reference_pngs_without_mutation(
    tmp_path: Path,
    report: dict,
) -> None:
    before = copy.deepcopy(report)
    outputs = [
        plotting.render_comparison(report, tmp_path, dpi=40, reference=False),
        plotting.render_comparison(report, tmp_path, dpi=40, reference=True),
    ]

    assert [path.name for path in outputs] == [
        "v16_published_methods_input.png",
        "v16_published_methods_reference.png",
    ]
    for path in outputs:
        data = path.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", data[16:24])
        assert width >= 500
        assert height >= 300
    assert report == before


def test_only_synthetic_canonical_k_is_used_as_reference(report: dict) -> None:
    index = {
        (row["dataset"], row["method"]): row for row in report["summary"]
    }
    assert plotting._synthetic_canonical_k(
        index,
        plotting.PUBLISHED_METHODS,
    ) == pytest.approx(8.5)

    index[("Synthetic", plotting.PUBLISHED_METHODS[-1])]["canonical_k_mean"] = 9.0
    with pytest.raises(ValueError, match="inconsistent"):
        plotting._synthetic_canonical_k(index, plotting.PUBLISHED_METHODS)


def test_reference_renderer_skips_report_without_reference_rows(
    tmp_path: Path,
    report: dict,
) -> None:
    report["summary"] = [
        row for row in report["summary"] if row["dataset"] == "Synthetic"
    ]

    assert plotting.render_comparison(report, tmp_path, reference=True) is None
    assert not (tmp_path / "v16_published_methods_reference.png").exists()
