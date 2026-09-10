"""Paired synthetic / real-world evaluation with explicit timing boundaries."""
from __future__ import annotations

# ruff: noqa: E402

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from spline_fitting.checkpointing import (
    V16_FORMAL_PASS_RATE,
    assess_v16_checkpoint,
    build_model_from_checkpoint,
    V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
    V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
    V16_OBJECTIVE_VERSIONS,
    V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION,
)
from spline_fitting.data.point_cloud_io import interpolate_parameters_by_chord
from spline_fitting.data.real_world import RealWorldCurveDataset, read_curve_manifest
from spline_fitting.data.synthetic import SyntheticCubicBSplineDataset
from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
from spline_fitting.evaluation.gradient_knot_pruning import chord_length_parameters
from spline_fitting.evaluation.published_baselines import (
    COMPARISON_BASELINE_METHODS,
    run_published_baseline,
)
from spline_fitting.evaluation.timing import measure_synchronized_wall_time
from compare_knot_methods import _select_indices
from visualize_batch_comparison import _dataset_config_from_checkpoint

METHODS = ("ours", *COMPARISON_BASELINE_METHODS)
PUBLISHED_METHODS = (
    "ours",
    "park_dominant_point_2007_adaptation",
    "liang_feature_iki_2017_adaptation",
    "dung_direct_knot_2017_adaptation",
    "kang_sparse_2015_adaptation",
    "luo_linf_de_2022_adaptation",
)
LABELS = {
    "ours": "Ours v15 learned",
    "park_dominant_point_2007_adaptation": "Park & Lee 2007 (DOM adaptation)",
    "liang_feature_iki_2017_adaptation": "Liang et al. 2017 (feature-IKI adaptation)",
    "dung_direct_knot_2017_adaptation": "Dung & Tjahjowidodo 2017 (serial adaptation)",
    "kang_sparse_2015_adaptation": "Kang 2015 (ADMM adaptation)",
    "luo_linf_de_2022_adaptation": "Luo et al. 2022 (l-infinity,1 + DE adaptation)",
    "yeh_feature_cdf_2020": "Yeh 2020 (feature-CDF + K scan)",
    "uniform_gradient_pruning": "Uniform Kmax greedy + gradient",
}
OBJECTIVE_LABELS = {
    V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION: "v15",
    V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION: "v16-legacy",
    V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION: "v16",
}
DEFAULT_MANIFESTS = {
    "UJI": ROOT / "data/splits/uji_pen_v2.jsonl",
    "NaturalEarth": ROOT / "data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl",
    "USGS": ROOT / "data/processed/usgs_contours/large_scale/manifest.jsonl",
}


def parser(*, default_checkpoint: Path | None = None,
           default_output_dir: Path | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=default_checkpoint is None,
                   default=default_checkpoint)
    p.add_argument("--output-dir", type=Path, required=default_output_dir is None,
                   default=default_output_dir)
    p.add_argument("--samples-per-knot-count", type=int, default=1)
    p.add_argument("--min-knot-count", type=int, default=4)
    p.add_argument("--max-knot-count", type=int, default=20)
    p.add_argument("--scan-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--selection-seed", type=int, default=20260908)
    p.add_argument("--real-samples-per-dataset", type=int, default=10)
    p.add_argument(
        "--method-set",
        choices=("published", "all"),
        default="all",
        help=(
            "'published' evaluates Ours, Park, Liang, Dung, Kang and Luo; "
            "'all' additionally evaluates Yeh and uniform greedy"
        ),
    )
    p.add_argument("--manifest", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--skip-synthetic", action="store_true")
    p.add_argument("--skip-real", action="store_true")
    p.add_argument("--mse-tolerance", type=float, default=2.5e-5)
    p.add_argument("--max-internal-knots", type=int, default=None,
                   help="Numerical baseline cap; default 28 for v15, checkpoint candidate capacity for v16")
    p.add_argument(
        "--full-knot-vector-size", type=int, default=None,
        help=("Set the numerical cap using full open-clamped knot-vector notation. "
              "For cubic curves, 64 total entries means 56 internal knots. "
              "Mutually exclusive with --max-internal-knots"),
    )
    p.add_argument(
        "--allow-unequal-capacity", action="store_true",
        help=("Explicit diagnostic ablation only: allow numerical initial caps "
              "to differ from the learned checkpoint's internal Kc"),
    )
    p.add_argument("--gradient-steps", type=int, default=12)
    p.add_argument("--paper-initial-knots", type=int, default=None,
                   help="Kang dense initial cap; default 40 for v15, checkpoint candidate capacity for v16")
    p.add_argument("--paper-admm-iterations", type=int, default=400)
    p.add_argument("--paper-lambda-bisections", type=int, default=8)
    p.add_argument("--paper-relocation-iterations", type=int, default=8)
    p.add_argument("--park-shape-weight", type=float, default=0.8)
    p.add_argument("--liang-dense-knots", type=int, default=None,
                   help="Dense preliminary knots; default follows --paper-initial-knots")
    p.add_argument("--liang-initial-knots", type=int, default=4)
    p.add_argument("--liang-curvature-weight", type=float, default=0.5)
    p.add_argument("--liang-feature-samples", type=int, default=1025)
    p.add_argument("--dung-max-error", type=float, default=None,
                   help="Native maximum Euclidean segmentation error; default sqrt(MSE tolerance)")
    p.add_argument("--dung-scan-intervals", type=int, default=10)
    p.add_argument("--dung-optimization-iterations", type=int, default=10)
    p.add_argument("--luo-eta", type=float, default=0.5)
    p.add_argument("--luo-de-population", type=int, default=10)
    p.add_argument("--luo-de-iterations", type=int, default=50)
    p.add_argument("--luo-seed", type=int, default=2022)
    p.add_argument("--network-warmups", type=int, default=3)
    p.add_argument("--network-repeats", type=int, default=10)
    p.add_argument("--end-to-end-repeats", type=int, default=3)
    p.add_argument("--torch-num-threads", type=int, default=4)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument(
        "--allow-unqualified-diagnostic", action="store_true",
        help=("Permit a proposal-stage or target-not-met v16 checkpoint for visibly "
              "marked troubleshooting only"),
    )
    p.add_argument("--resume", action="store_true", help="Reuse completed rows only when the experiment fingerprint matches.")
    return p


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def recover_journal(path: Path) -> list[dict]:
    """Recover only a torn final write; preserve its bytes for audit."""
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows, offset = [], 0
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Corrupt completed journal row {i+1}: {path}")
            path.with_suffix(".interrupted_tail.bin").write_bytes(line)
            with path.open("r+b") as handle:
                handle.truncate(offset)
            break
        if not isinstance(row, dict):
            raise ValueError(f"Journal row {i+1} is not an object")
        rows.append(row)
        offset += len(line)
    if rows and raw and not raw.endswith(b"\n") and offset == len(raw):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return rows


def balanced_indices(records: list[dict], count: int, seed: int) -> list[int]:
    """Round-robin randomized groups before taking additional curves per group."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        groups[str(record["group_id"])].append(i)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for group in groups.values():
        rng.shuffle(group)
    selected: list[int] = []
    while len(selected) < min(count, len(records)):
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                if len(selected) == min(count, len(records)):
                    return selected
    return selected


def resolve_comparison_capacities(args, checkpoint: dict) -> dict:
    """Retain v15 defaults and pair v16 numerical budgets with its actual model."""
    capacity = int(checkpoint["model_config"]["max_internal_knots"])
    degree = int(checkpoint["model_config"].get("degree", 3))
    endpoint_entries = 2 * (degree + 1)
    v16 = checkpoint.get("objective_version") in V16_OBJECTIVE_VERSIONS
    requested_full = getattr(args, "full_knot_vector_size", None)
    if requested_full is not None:
        if args.max_internal_knots is not None:
            raise ValueError(
                "use either --max-internal-knots or --full-knot-vector-size, not both"
            )
        args.max_internal_knots = requested_full - endpoint_entries
    if args.max_internal_knots is None:
        args.max_internal_knots = capacity if v16 else 28
    if args.paper_initial_knots is None:
        args.paper_initial_knots = capacity if v16 else 40
    liang_capacity = (
        args.paper_initial_knots
        if getattr(args, "liang_dense_knots", None) is None
        else args.liang_dense_knots
    )
    if min(
        capacity, args.max_internal_knots, args.paper_initial_knots,
        liang_capacity,
    ) < 1:
        raise ValueError("all candidate and numerical knot capacities must be positive")
    return {
        "network_candidates": capacity,
        "greedy_initial_and_yeh_max": args.max_internal_knots,
        "kang_dense_initial": args.paper_initial_knots,
        "liang_dense_initial": liang_capacity,
        "equal_initial_capacity": (
            capacity == args.max_internal_knots
            == args.paper_initial_knots == liang_capacity
        ),
        "degree": degree,
        "clamped_endpoint_entries": endpoint_entries,
        "network_full_knot_vector_size_at_all_keep": capacity + endpoint_entries,
        "numerical_full_knot_vector_cap": args.max_internal_knots + endpoint_entries,
    }


def published_baseline_kwargs(args, *, degree: int, warmup: bool = False) -> dict:
    """Build one disclosed option set for every numerical paper adaptation."""
    values = {
        "mse_tolerance": args.mse_tolerance,
        "max_internal_knots": 3 if warmup else args.max_internal_knots,
        "degree": degree,
        "gradient_steps": 1 if warmup else args.gradient_steps,
        "paper_initial_knots": 6 if warmup else args.paper_initial_knots,
        "paper_admm_iterations": 10 if warmup else args.paper_admm_iterations,
        "paper_lambda_bisections": 1 if warmup else args.paper_lambda_bisections,
        "paper_relocation_iterations": 1 if warmup else args.paper_relocation_iterations,
        "park_shape_weight": args.park_shape_weight,
        "liang_dense_knots": (
            6 if warmup else (
                args.paper_initial_knots
                if args.liang_dense_knots is None
                else args.liang_dense_knots
            )
        ),
        "liang_initial_knots": 2 if warmup else args.liang_initial_knots,
        "liang_curvature_weight": args.liang_curvature_weight,
        "liang_feature_samples": 65 if warmup else args.liang_feature_samples,
        "dung_max_error": args.dung_max_error,
        "dung_scan_intervals": 2 if warmup else args.dung_scan_intervals,
        "dung_optimization_iterations": (
            1 if warmup else args.dung_optimization_iterations
        ),
        "luo_eta": args.luo_eta,
        "luo_de_population": 5 if warmup else args.luo_de_population,
        "luo_de_iterations": 1 if warmup else args.luo_de_iterations,
        "luo_seed": args.luo_seed,
    }
    return values


def known_synthetic_seed_ranges(checkpoint: dict) -> list[dict]:
    """Reserve every v16 epoch's synthetic seed range, not only epoch zero.

    The full configured run is reserved conservatively even when the selected
    best checkpoint predates its last epoch. Historical v15 semantics remain
    the original fixed train/validation intervals.
    """
    config = checkpoint.get("training_config", {})
    ranges = []
    v16 = checkpoint.get("objective_version") in V16_OBJECTIVE_VERSIONS
    for split in ("train", "val"):
        base, size = config.get(f"{split}_seed"), config.get(f"{split}_size")
        if base is None or size is None:
            continue
        base, size = int(base), int(size)
        epochs, stride = 1, 0
        if split == "train" and v16 and config.get("resample_train_each_epoch", True):
            history_epochs = [int(row.get("epoch", 0)) for row in checkpoint.get("history", [])]
            epochs = max(1, int(config.get("epochs", 1)), int(checkpoint.get("epoch", 0)), *history_epochs)
            stride = int(config.get("train_seed_stride", 10_000_000))
            if stride < 1:
                raise ValueError("v16 train_seed_stride must be positive")
        if base < 0 or size < 1:
            raise ValueError("checkpoint synthetic seed ranges are invalid")
        for epoch in range(epochs):
            ranges.append({"split": split, "epoch": epoch + 1,
                           "start": base + epoch * stride,
                           "stop": base + epoch * stride + size})
    return ranges


def prepare_cases(args, checkpoint: dict, model_config: dict) -> tuple[list[dict], list[dict]]:
    cases, provenance = [], []
    config = _dataset_config_from_checkpoint(checkpoint, model_config)
    if not args.skip_synthetic:
        train_config = checkpoint.get("training_config", {})
        if args.seed in (train_config.get("train_seed"), train_config.get("val_seed")):
            raise ValueError("Synthetic test seed overlaps the checkpoint train/validation seed")
        indices = _select_indices(
            config, scan_size=args.scan_size, dataset_seed=args.seed,
            selection_seed=args.selection_seed, minimum_count=args.min_knot_count,
            maximum_count=args.max_knot_count, samples_per_count=args.samples_per_knot_count,
        )
        excluded_ranges = known_synthetic_seed_ranges(checkpoint)
        for interval in excluded_ranges:
            if any(interval["start"] <= args.seed + i < interval["stop"] for i in indices):
                raise ValueError(
                    f"Selected synthetic per-sample seeds overlap {interval['split']} "
                    f"epoch {interval['epoch']}"
                )
        dataset = SyntheticCubicBSplineDataset(
            size=max(indices) + 1, seed=args.seed, cache_samples=False, **config,
        )
        for i in indices:
            sample = dataset[i]
            if config.get("certified_minimal_source"):
                if sample.get("source_minimality_certified") is not True:
                    raise RuntimeError(
                        f"Synthetic sample {i} violated its saved minimality contract"
                    )
                source_k = int(sample["source_internal_knot_count"])
                canonical_k = int(sample["true_internal_knot_mask"].sum())
                if source_k != canonical_k:
                    raise RuntimeError(
                        f"Synthetic sample {i} changed K after minimality certification"
                    )
            else:
                source_k = int(sample["source_internal_knot_count"])
                canonical_k = int(sample["true_internal_knot_mask"].sum())
            cases.append({
                "dataset": "Synthetic", "sample_id": f"seed{args.seed}_sample{i}",
                "group_id": f"independent_{i}", "points": sample["points"],
                "reference": None,
                "source_k": source_k,
                "canonical_k": canonical_k,
                "reference_grid": None,
            })
        provenance.append({"dataset": "Synthetic", "config": config, "seed": args.seed,
                           "selected_indices": indices, "selected_count": len(indices),
                           "excluded_training_seed_ranges": excluded_ranges})
    if args.skip_real:
        return cases, provenance
    manifests = dict(DEFAULT_MANIFESTS)
    if args.manifest:
        manifests = {}
        for value in args.manifest:
            name, sep, path = value.partition("=")
            if not sep or not name:
                raise ValueError("--manifest must be NAME=PATH")
            if name in manifests or name == "Synthetic":
                raise ValueError(f"Duplicate or reserved dataset name: {name}")
            manifests[name] = Path(path)
    for name, path in manifests.items():
        records = read_curve_manifest(path)
        group_splits: dict[str, set[str]] = defaultdict(set)
        for record in records:
            group_splits[record["group_id"]].add(record["split"])
        if any(len(splits) > 1 for splits in group_splits.values()):
            raise ValueError(f"Group leakage between splits in {path}")
        dataset = RealWorldCurveDataset(path, split="test", num_points=config["num_points"])
        if not len(dataset):
            raise ValueError(f"No test curves in {path}")
        indices = balanced_indices(dataset.records, args.real_samples_per_dataset, args.selection_seed)
        for i in indices:
            sample = dataset[i]
            if sample["points"].shape[-1] != config["point_dim"]:
                raise ValueError(f"Point dimension disagrees with checkpoint: {path}")
            raw = dataset.load_reference_points(i).double()
            reference = (raw - sample["center"].double()) / sample["scale"].double()
            # The loader samples the original polyline at uniform ORIGINAL
            # arc-length positions. These positions, rather than remeasured
            # chords across a coarse grid's corners, align the dense reference.
            reference_grid = chord_length_parameters(raw)
            cases.append({
                "dataset": name, "sample_id": sample["curve_id"],
                "group_id": sample["group_id"], "points": sample["points"],
                "reference": reference, "reference_grid": reference_grid,
                "source_k": None, "canonical_k": None,
            })
        provenance.append({
            "dataset": name, "manifest": str(path.resolve()), "manifest_sha256": sha256_file(path),
            "split": "test", "available_test_curves": len(dataset),
            "available_test_groups": len({r["group_id"] for r in dataset.records}),
            "selected_count": len(indices), "selected_indices": indices,
            "selected_groups": len({dataset.records[i]["group_id"] for i in indices}),
            "has_knot_labels": False, "sampling": "seeded_group_round_robin",
        })
    return cases, provenance


def reference_mse(fit, parameters: torch.Tensor, case: dict) -> float | None:
    if case["reference"] is None:
        return None
    source_grid = torch.linspace(0, 1, parameters.numel(), dtype=torch.float64)
    reference_params = interpolate_parameters_by_chord(
        source_grid, parameters.double(), case["reference_grid"],
    )
    return float((fit.evaluate(reference_params) - case["reference"]).square().sum(-1).mean())


def benchmark_version(objective_version: str, *, expected_objective: str) -> str:
    """Reject mislabeled/unknown checkpoints instead of changing architecture."""
    if expected_objective not in OBJECTIVE_LABELS:
        raise ValueError(f"Unsupported benchmark objective: {expected_objective}")
    if objective_version != expected_objective:
        raise ValueError(
            f"This {OBJECTIVE_LABELS[expected_objective]} experiment requires "
            f"objective_version={expected_objective!r}; got {objective_version!r}"
        )
    return OBJECTIVE_LABELS[objective_version]


def validate_checkpoint_for_benchmark(
    checkpoint: dict, *, mse_tolerance: float,
    allow_unqualified_diagnostic: bool,
) -> tuple[dict | None, bool]:
    """Return the v16 qualification audit and diagnostic-watermark state."""
    if checkpoint.get("objective_version") != V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION:
        return None, False
    qualification = assess_v16_checkpoint(
        checkpoint,
        required_pass_rate=V16_FORMAL_PASS_RATE,
        required_mse_tolerance=mse_tolerance,
    )
    diagnostic = not qualification["formal_reporting_eligible"]
    if diagnostic and not allow_unqualified_diagnostic:
        raise ValueError(
            "v16 checkpoint is not eligible for formal reporting: "
            + "; ".join(qualification["reasons"])
            + ". Use --allow-unqualified-diagnostic for marked troubleshooting."
        )
    return qualification, diagnostic


def deployment_forward(model, points, *, objective_version: str,
                       mse_tolerance: float):
    if objective_version in V16_OBJECTIVE_VERSIONS:
        # v16 conditions the actual subset decision on this MSE budget. Use the
        # reported threshold, not the potentially different checkpoint default.
        return model.forward_deployment(points, mse_tolerance=mse_tolerance)
    if objective_version == V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION:
        # v15 never received a tolerance condition. Preserve exact semantics.
        return model.forward_deployment(points)
    raise ValueError(f"Unsupported deployment objective: {objective_version}")


def measure_ours(model, points, device, args, *,
                 objective_version=V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION):
    device_points = points.unsqueeze(0).to(device)

    def forward():
        with torch.inference_mode():
            return deployment_forward(model, device_points,
                                      objective_version=objective_version,
                                      mse_tolerance=args.mse_tolerance)

    network = measure_synchronized_wall_time(
        forward, synchronization_device=device, warmup_repeats=args.network_warmups,
        timing_repeats=args.network_repeats,
    )

    def deploy():
        # Time from normalized host points, including H2D, network, D2H and
        # CPU float64 endpoint-constrained refit. Metric evaluation is excluded.
        with torch.inference_mode():
            output = deployment_forward(model, points.unsqueeze(0).to(device),
                                        objective_version=objective_version,
                                        mse_tolerance=args.mse_tolerance)
        params = output["params"][0].detach().cpu().double()
        mask = output["learned_keep_mask"][0].detach().cpu().bool()
        candidates = output["internal_knots"][0].detach().cpu().double()
        selected = candidates[mask].sort().values
        fit = refit_bspline_control_points(
            params, points.double(), selected, degree=model.degree,
            smoothness_weight=0.0, control_ridge=0.0, interpolate_endpoints=True,
        )
        return fit, params

    full = measure_synchronized_wall_time(
        deploy, synchronization_device=device, warmup_repeats=1,
        timing_repeats=args.end_to_end_repeats,
    )
    fit, params = full.result
    return fit, params, full.latency.p50_ms, network.latency.p50_ms, {
        "network_latency": network.latency.as_dict(),
        "end_to_end_latency": full.latency.as_dict(), "postprocessing": "none",
        "tolerance_conditioned_network": (
            objective_version in V16_OBJECTIVE_VERSIONS
        ),
        "network_mse_tolerance": (
            args.mse_tolerance
            if objective_version in V16_OBJECTIVE_VERSIONS
            else None
        ),
    }


def measure_numerical_baseline(method, points, args, *, degree):
    """Repeat a complete numerical method and report its median wall time.

    The numerical implementation owns its timer, which begins at the normalized
    CPU points and includes chord parameterization, knot search/relocation and
    the common final refit.  Reusing ``end_to_end_repeats`` gives Ours and every
    numerical control the same number of timed complete executions.
    """
    results = [
        run_published_baseline(
            method,
            points.double(),
            **published_baseline_kwargs(args, degree=degree),
        )
        for _ in range(args.end_to_end_repeats)
    ]
    elapsed = [float(result.elapsed_ms) for result in results]
    result = results[-1]
    diagnostics = dict(result.diagnostics)
    diagnostics["repeated_complete_timing"] = {
        "repeats": len(elapsed),
        "samples_ms": elapsed,
        "p50_ms": float(statistics.median(elapsed)),
        "mean_ms": float(statistics.fmean(elapsed)),
    }
    return (
        result.fit,
        result.parameters,
        float(statistics.median(elapsed)),
        None,
        diagnostics,
    )


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"])].append(row)
    result = []
    for (dataset, method), values in groups.items():
        valid = [r for r in values if r["status"] == "ok"]
        refs = [r for r in valid if r["reference_mse"] is not None]
        knot_labels = [r for r in valid if r.get("canonical_k") is not None]
        count_errors = [
            r["final_k"] - r["canonical_k"] for r in knot_labels
        ]
        def mean(field, items=valid):
            return statistics.fmean(r[field] for r in items) if items else None
        result.append({
            "dataset": dataset, "method": method, "n": len(values),
            "failed": len(values) - len(valid),
            "mse_mean": mean("mse"),
            "mse_p95": float(torch.quantile(torch.tensor([r["mse"] for r in valid], dtype=torch.float64), .95)) if valid else None,
            "fit_pass_rate": sum(r["fit_pass"] for r in values) / len(values),
            "reference_mse_mean": mean("reference_mse", refs),
            "reference_pass_rate": sum(bool(r["reference_pass"]) for r in values) / len(values) if any(r.get("has_reference", r["reference_mse"] is not None) for r in values) else None,
            "final_k_mean": mean("final_k"),
            "canonical_k_mean": (
                statistics.fmean(r["canonical_k"] for r in knot_labels)
                if knot_labels else None
            ),
            "canonical_count_mae": (
                statistics.fmean(abs(error) for error in count_errors)
                if count_errors else None
            ),
            "canonical_count_bias": (
                statistics.fmean(count_errors) if count_errors else None
            ),
            "canonical_count_exact_rate": (
                sum(error == 0 for error in count_errors) / len(count_errors)
                if count_errors else None
            ),
            "canonical_count_within_one_rate": (
                sum(abs(error) <= 1 for error in count_errors) / len(count_errors)
                if count_errors else None
            ),
            "total_ms_mean": mean("total_ms", values),
            "network_ms_mean": mean("network_ms") if method == "ours" else None,
        })
    return result


def write_reports(directory: Path, metadata: dict, rows: list[dict]):
    summary = summarize(rows)
    report = {"metadata": metadata, "summary": summary, "measurements": rows}
    (directory / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    for filename, items in (("summary.csv", summary), ("measurements.csv", rows)):
        if not items:
            continue
        with (directory / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
    version = OBJECTIVE_LABELS[metadata["objective_version"]]
    labels = {**LABELS, "ours": f"Ours {version} learned"}
    lines = [
        f"# {version} 多数据集配对测试",
        "",
        (
            f"Checkpoint: `{metadata['checkpoint']}`（"
            f"{metadata['objective_version']}，epoch {metadata['epoch']}）。"
        ),
        (
            f"统一阈值：MSE ≤ {metadata['mse_tolerance']:.3e}；"
            "MSE = mean_i ||C(t_i)-Q_i||²，不开方。"
        ),
        (
            "所有方法接收相同归一化有序点，并以 CPU float64、端点插值、"
            "无正则标准 B 样条最小二乘作为最终报告拟合。"
        ),
        (
            "参数化并非完全相同：所有方法都以弦长参数为起点；数值基线固定弦长参数，"
            f"Ours {version} 使用网络预测的弦长残差参数。若要隔离节点选择贡献，应另做 "
            "fixed-chord ablation。"
        ),
        (
            "完整耗时从归一化 CPU 输入开始，包含参数化、方法本身及最终 refit；"
            "不含数据加载、归一化和评价指标计算。Ours 的网络时间另列。"
        ),
        (
            "节点容量（分别列出，不隐含相等）："
            f"{metadata.get('knot_capacities', '见 experiment.json configuration')}。"
        ),
        f"设备：{metadata['hardware']}。",
        "",
        "| 数据集 | 方法 | n | 拟合通过率 | 最终 MSE | 平均 K | 完整耗时 ms | 网络 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    if metadata.get("diagnostic_not_final"):
        reasons = metadata.get("checkpoint_qualification", {}).get("reasons", [])
        lines[2:2] = [
            "**DIAGNOSTIC NOT FINAL — UNQUALIFIED CHECKPOINT**",
            "",
            "资格检查：" + ("；".join(reasons) if reasons else "未通过正式资格检查。"),
            "",
        ]

    def fmt(value, form):
        return format(value, form) if value is not None else "—"

    for s in summary:
        lines.append(
            f"| {s['dataset']} | {labels[s['method']]} | {s['n']} | "
            f"{s['fit_pass_rate']:.1%} | {fmt(s['mse_mean'], '.3e')} | "
            f"{fmt(s['final_k_mean'], '.2f')} | "
            f"{fmt(s['total_ms_mean'], '.2f')} | "
            f"{fmt(s['network_ms_mean'], '.2f')} |"
        )
    lines += [
        "",
        (
            f"真实曲线原始参考点测试：仅在 {metadata['num_points']} 个重采样输入点上"
            "拟合，原始参考点集不直接用于 refit。UJI 通常从较少原始点上采样，"
            "这不产生新的独立观测。通过率由参考点 MSE 单独判定。"
        ),
        "",
        "| 数据集 | 方法 | 参考点通过率 | 参考点 MSE |",
        "|---|---|---:|---:|",
    ]
    for s in summary:
        if s["reference_mse_mean"] is not None:
            lines.append(
                f"| {s['dataset']} | {labels[s['method']]} | "
                f"{s['reference_pass_rate']:.1%} | "
                f"{s['reference_mse_mean']:.3e} |"
            )
    lines += [
        "",
        (
            "复现边界：Park 保留 DOM 核心并改用公共 MSE 停止；Liang 是公开摘要所述"
            "特征积分 + IKI 的显式适配；Dung 仅复现串行、单重节点路径；Kang 是 "
            "group-L1 ADMM 适配；Luo 保留 l∞,1、局部极大值筛选和 DE，正则参数按公共 "
            "MSE 预算选择；Yeh 使用公开布点公式加递增 K 扫描。均不宣称与作者代码"
            "逐位一致。"
        ),
        (
            "失败样本计入通过率分母；无有限解时 MSE 均值只含成功样本，failed 列"
            "单独保存。小样本结果仅用于初步比较；同一 writer/tile 的相关性会降低"
            "真实数据的有效独立样本数。完整逐样本记录与配置保存在 comparison.json。"
        ),
        "",
        (
            "论文来源：[Park & Lee 2007](https://doi.org/10.1016/j.cad.2006.12.006)，"
            "[Liang et al. 2017](https://doi.org/10.1088/1361-6501/aa6a05)，"
            "[Dung & Tjahjowidodo 2017](https://doi.org/10.1371/journal.pone.0173857)，"
            "[Kang 2015](https://doi.org/10.1016/j.cad.2014.08.022)，"
            "[Luo–Kang–Yang 2022](https://doi.org/10.4208/jcm.2012-m2020-0203)，"
            "[Yeh 2020](https://doi.org/10.1016/j.cad.2020.102905)。"
        ),
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None, *, expected_objective=V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
         default_checkpoint: Path | None = None,
         default_output_dir: Path | None = None):
    p = parser(default_checkpoint=default_checkpoint,
               default_output_dir=default_output_dir)
    args = p.parse_args(argv)
    if not math.isfinite(args.mse_tolerance) or args.mse_tolerance <= 0:
        p.error("--mse-tolerance must be finite and positive")
    for name in (
        "samples_per_knot_count", "real_samples_per_dataset", "torch_num_threads",
        "network_repeats", "end_to_end_repeats", "liang_feature_samples",
        "dung_scan_intervals", "dung_optimization_iterations",
        "luo_de_population", "luo_de_iterations",
    ):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("park_shape_weight", "liang_curvature_weight", "luo_eta"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            p.error(f"--{name.replace('_', '-')} must lie in [0,1]")
    if args.luo_de_population < 5:
        p.error("--luo-de-population must be at least 5")
    if args.liang_dense_knots is not None and args.liang_dense_knots < 0:
        p.error("--liang-dense-knots must be non-negative")
    if args.liang_initial_knots < 0:
        p.error("--liang-initial-knots must be non-negative")
    if args.dung_max_error is not None and (
        not math.isfinite(args.dung_max_error) or args.dung_max_error <= 0.0
    ):
        p.error("--dung-max-error must be finite and positive")
    torch.set_num_threads(args.torch_num_threads)
    torch.manual_seed(args.selection_seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if not args.checkpoint.is_file():
        p.error(
            f"checkpoint does not exist: {args.checkpoint}. The benchmark does "
            "not train a model; run scripts/train_v16.py first, then pass its "
            "best joint .pt file (not .proposal.pt or .last.pt)"
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    objective_version = checkpoint.get("objective_version")
    try:
        version = benchmark_version(objective_version, expected_objective=expected_objective)
    except ValueError as error:
        p.error(str(error))
    try:
        qualification, diagnostic = validate_checkpoint_for_benchmark(
            checkpoint,
            mse_tolerance=args.mse_tolerance,
            allow_unqualified_diagnostic=args.allow_unqualified_diagnostic,
        )
    except ValueError as error:
        p.error(str(error))
    if diagnostic:
        print(
            "DIAGNOSTIC NOT FINAL — unqualified v16 checkpoint: "
            + "; ".join(qualification["reasons"]),
            flush=True,
        )
    model, model_config, _ = build_model_from_checkpoint(checkpoint)
    if model_config.get("structure_mode") != "candidate_pruning_one_shot":
        p.error("Checkpoint must expose one-shot candidate pruning")
    try:
        capacities = resolve_comparison_capacities(args, checkpoint)
    except ValueError as error:
        p.error(str(error))
    if (
        objective_version in V16_OBJECTIVE_VERSIONS
        and not capacities["equal_initial_capacity"]
        and not args.allow_unequal_capacity
    ):
        p.error(
            "v16 main comparisons require the same internal candidate cap for "
            "Ours and configurable numerical methods; change all caps to the "
            "checkpoint Kc or add --allow-unequal-capacity for a disclosed ablation"
        )
    model.to(device).eval()
    print("Preparing fixed paired test samples...", flush=True)
    cases, provenance = prepare_cases(args, checkpoint, model_config)
    if not cases:
        p.error("No selected datasets")
    hardware = {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "cpu": platform.processor(), "torch": str(torch.__version__), "threads": torch.get_num_threads(),
                "python": platform.python_version()}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ("resume", "output_dir")}
    code_paths = [Path(__file__), ROOT / "src/spline_fitting/evaluation/published_baselines.py",
                  ROOT / "src/spline_fitting/evaluation/gradient_knot_pruning.py",
                  ROOT / "src/spline_fitting/evaluation/sparse_knot_paper.py",
                  ROOT / "src/spline_fitting/evaluation/feature_cdf_knot_placement.py",
                  ROOT / "scripts/compare_knot_methods.py", ROOT / "scripts/visualize_batch_comparison.py"]
    if objective_version in V16_OBJECTIVE_VERSIONS:
        code_paths.append(ROOT / "scripts/benchmark_v16_datasets.py")
    code_paths = sorted(set(code_paths) | set((ROOT / "src").rglob("*.py")))
    metadata = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint),
                "objective_version": checkpoint.get("objective_version"), "epoch": checkpoint.get("epoch"),
                "checkpoint_stage": checkpoint.get("stage"),
                "checkpoint_quality": checkpoint.get("checkpoint_quality"),
                "checkpoint_qualification": qualification,
                "diagnostic_not_final": diagnostic,
                "model_version": version,
                "method_labels": {**LABELS, "ours": f"Ours {version} learned"},
                "network_tolerance_conditioned": objective_version in V16_OBJECTIVE_VERSIONS,
                "knot_capacities": capacities,
                "num_points": int(cases[0]["points"].shape[0]),
                "timing_protocol": "global numerical-backend warmup; every method uses the configured repeated complete-run median per curve; Ours additionally reports a separately warmed network-only median; dataset summary averages per-curve medians",
                "mse_tolerance": args.mse_tolerance, "configuration": config, "hardware": hardware,
                "datasets": provenance, "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in code_paths},
                "sample_content_sha256": [hashlib.sha256(c["points"].numpy().tobytes() + (c["reference"].numpy().tobytes() if c["reference"] is not None else b"")).hexdigest() for c in cases]}
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    metadata["fingerprint"] = fingerprint
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "experiment.json"
    journal = directory / "measurements.jsonl"
    rows = []
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not args.resume or previous["fingerprint"] != fingerprint:
            p.error("Output exists or experiment fingerprint changed; use a new output directory")
        if journal.exists():
            rows = recover_journal(journal)
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    methods = PUBLISHED_METHODS if args.method_set == "published" else METHODS
    done = {(r["dataset"], r["sample_id"], r["method"]) for r in rows}
    expected = {(c["dataset"], c["sample_id"], m) for c in cases for m in methods}
    if len(done) != len(rows) or not done.issubset(expected):
        p.error("Journal contains duplicate or unexpected experiment records")
    if done != expected:
        print("Warming numerical solvers (excluded from timing)...", flush=True)
        t = torch.linspace(0, 1, 32, dtype=torch.float64)
        warm_points = torch.stack((t, t.square()), dim=-1)
        for method in methods[1:]:
            run_published_baseline(
                method,
                warm_points,
                **published_baseline_kwargs(args, degree=model.degree, warmup=True),
            )
    print(f"{len(cases)} curves x {len(methods)} methods; device={device}; MSE tolerance={args.mse_tolerance:g}", flush=True)
    with journal.open("a", encoding="utf-8") as handle:
        for i, case in enumerate(cases, 1):
            for method in methods:
                if (case["dataset"], case["sample_id"], method) in done:
                    continue
                row = {"dataset": case["dataset"], "sample_id": case["sample_id"], "group_id": case["group_id"],
                       "source_k": case["source_k"], "canonical_k": case["canonical_k"], "method": method,
                       "has_reference": case["reference"] is not None,
                       "status": "ok", "mse": None, "fit_pass": False, "reference_mse": None,
                       "reference_pass": None, "final_k": None, "total_ms": None, "network_ms": None,
                       "knots": [], "diagnostics": {}, "error": None}
                started = time.perf_counter()
                try:
                    if method == "ours":
                        fit, params, total_ms, network_ms, diagnostics = measure_ours(
                            model, case["points"], device, args,
                            objective_version=objective_version,
                        )
                    else:
                        fit, params, total_ms, network_ms, diagnostics = measure_numerical_baseline(
                            method,
                            case["points"],
                            args,
                            degree=model.degree,
                        )
                    mse = float(fit.fit_mse)
                    ref_mse = reference_mse(fit, params, case)
                    if not math.isfinite(mse) or (ref_mse is not None and not math.isfinite(ref_mse)):
                        raise RuntimeError("non-finite fitted MSE")
                    row.update(mse=mse, fit_pass=mse <= args.mse_tolerance,
                               reference_mse=ref_mse, reference_pass=ref_mse <= args.mse_tolerance if ref_mse is not None else None,
                               final_k=int(fit.internal_knots.numel()), total_ms=total_ms, network_ms=network_ms,
                               knots=fit.internal_knots.detach().cpu().tolist(), diagnostics=diagnostics)
                except (RuntimeError, ValueError) as error:
                    row.update(status="failed", error=f"{type(error).__name__}: {error}", total_ms=(time.perf_counter()-started)*1000)
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                rows.append(row)
                print(f"[{i}/{len(cases)}] {case['dataset']} {case['sample_id']} {method}: K={row['final_k']} MSE={row['mse']} time={row['total_ms']:.1f} ms {row['status']}", flush=True)
            write_reports(directory, metadata, rows)
    print(f"Saved {directory / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
