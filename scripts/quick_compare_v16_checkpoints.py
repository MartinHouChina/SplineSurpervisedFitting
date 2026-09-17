"""Paired, native one-shot v16 deployment check on fixed K=4..56 curves.

This is a diagnostic, not the multi-method benchmark. Each network predicts
one mask once; each mask receives exactly one standard B-spline control-point
refit. There is no verified/add-back, hard pruning, or numerical mask repair.
"""
from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_v15_datasets import balanced_indices, known_synthetic_seed_ranges
from spline_fitting.checkpointing import V16_OBJECTIVE_VERSIONS, build_model_from_checkpoint
from spline_fitting.data.real_world import RealWorldCurveDataset, read_curve_manifest
from spline_fitting.data.synthetic import SyntheticCubicBSplineDataset
from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
from visualize_batch_comparison import _dataset_config_from_checkpoint


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-checkpoint", type=Path, required=True)
    p.add_argument("--new-checkpoint", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--mse-tolerance", type=float, default=1e-4)
    p.add_argument("--min-source-k", type=int, default=4)
    p.add_argument("--max-source-k", type=int, default=56)
    p.add_argument("--samples-per-k", type=int, default=1)
    p.add_argument("--seed", type=int, default=9_000_000_000)
    p.add_argument("--max-generate-attempts", type=int, default=8)
    p.add_argument("--manifest", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--real-samples-per-dataset", type=int, default=5)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return p


def _check_seed_not_trained(seed: int, checkpoints: tuple[dict, dict]) -> None:
    for checkpoint in checkpoints:
        for interval in known_synthetic_seed_ranges(checkpoint):
            if interval["start"] <= seed < interval["stop"]:
                raise ValueError(
                    f"test seed {seed} overlaps checkpoint {interval['split']} "
                    f"epoch {interval['epoch']}"
                )


def _synthetic_cases(args, old: dict, new: dict, old_config: dict):
    config = _dataset_config_from_checkpoint(old, old_config)
    if int(config["point_dim"]) != 2:
        raise ValueError("this synthetic stratification expects 2D cubic curves")
    config["cache_samples"] = False
    config["return_ground_truth"] = True
    for k in range(args.min_source_k, args.max_source_k + 1):
        exact_config = dict(config)
        exact_config["min_control_points"] = k + 4
        exact_config["max_control_points"] = k + 4
        # Disjoint from the default epoch-resampling seed bands. The chosen
        # per-curve seed is recorded, so both checkpoints see identical data.
        base_seed = args.seed + (k - args.min_source_k) * 100_003
        dataset = SyntheticCubicBSplineDataset(
            size=args.samples_per_k * args.max_generate_attempts,
            seed=base_seed,
            **exact_config,
        )
        selected = 0
        for index in range(len(dataset)):
            if selected >= args.samples_per_k:
                break
            curve_seed = base_seed + index
            _check_seed_not_trained(curve_seed, (old, new))
            try:
                sample = dataset[index]
            except RuntimeError as error:
                # Certification is stochastic. Try another deterministic seed,
                # but never silently replace a failed sample with a different K.
                if "minimal" not in str(error).lower() and "certif" not in str(error).lower():
                    raise
                continue
            actual_k = int(sample["source_internal_knot_count"])
            if actual_k != k:
                raise RuntimeError(f"exact-K generator returned K={actual_k}, expected {k}")
            if config.get("certified_minimal_source") and not bool(
                sample.get("source_minimality_certified", False)
            ):
                raise RuntimeError("synthetic minimality certification was lost")
            yield {
                "dataset": "Synthetic", "case_id": f"K{k}_seed{curve_seed}",
                "source_k": k, "points": sample["points"],
                "synthetic_seed": curve_seed,
            }
            selected += 1
        if selected != args.samples_per_k:
            raise RuntimeError(
                f"could certify only {selected}/{args.samples_per_k} K={k} curves; "
                "increase --max-generate-attempts"
            )


def _real_cases(args, *, num_points: int, point_dim: int):
    names = set()
    for entry in args.manifest:
        name, sep, raw_path = entry.partition("=")
        if not sep or not name or not raw_path or name == "Synthetic" or name in names:
            raise ValueError("--manifest must be unique NAME=PATH, excluding Synthetic")
        names.add(name)
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        records = read_curve_manifest(path)
        group_splits: dict[str, set[str]] = {}
        for record in records:
            group_splits.setdefault(str(record["group_id"]), set()).add(
                str(record["split"])
            )
        if any(len(splits) > 1 for splits in group_splits.values()):
            raise ValueError(f"real-data group leakage between splits in {path}")
        dataset = RealWorldCurveDataset(path, split="test", num_points=num_points)
        indices = balanced_indices(
            dataset.records, args.real_samples_per_dataset, args.seed + len(names)
        )
        for index in indices:
            sample = dataset[index]
            if sample["points"].shape[-1] != point_dim:
                raise ValueError(f"point dimension mismatch in {path}")
            yield {
                "dataset": name, "case_id": sample["curve_id"],
                "source_k": None, "points": sample["points"],
                "manifest": str(path.resolve()),
            }


def _native_refit(model, points: torch.Tensor, device: torch.device, tolerance: float):
    with torch.inference_mode():
        output = model.forward_deployment(
            points.unsqueeze(0).to(device), mse_tolerance=tolerance
        )
    params = output["params"][0].detach().cpu().double()
    proposed = output["internal_knots"][0].detach().cpu().double()
    mask = output["learned_keep_mask"][0].detach().cpu().bool()
    if proposed.ndim != 1 or mask.shape != proposed.shape:
        raise RuntimeError("invalid one-shot knot/mask shape")
    selected = proposed[mask].sort().values
    fit = refit_bspline_control_points(
        params, points.double(), selected,
        degree=model.degree, smoothness_weight=0.0, control_ridge=0.0,
        interpolate_endpoints=True,
    )
    mse = float(fit.fit_mse)
    if not math.isfinite(mse):
        raise RuntimeError("non-finite native deployment MSE")
    return {"mse": mse, "k": int(selected.numel())}


def summarize(rows: list[dict], *, tolerance: float) -> dict:
    """Summaries are per-curve means; pass is measured before any repair."""
    def group(subset):
        answer = {"n": len(subset)}
        for label in ("old", "new"):
            answer[label] = {
                "pass_rate": sum(row[label]["mse"] <= tolerance for row in subset) / len(subset)
                if subset else None,
                "mse_mean": sum(row[label]["mse"] for row in subset) / len(subset)
                if subset else None,
                "k_mean": sum(row[label]["k"] for row in subset) / len(subset)
                if subset else None,
            }
        return answer

    synthetic = [row for row in rows if row["dataset"] == "Synthetic"]
    output = {
        "synthetic_overall": group(synthetic),
        "synthetic_high_k_ge_45": group(
            [row for row in synthetic if row["source_k"] >= 45]
        ),
        "synthetic_k_56": group(
            [row for row in synthetic if row["source_k"] == 56]
        ),
        "real_by_dataset": {},
    }
    for name in sorted({row["dataset"] for row in rows} - {"Synthetic"}):
        output["real_by_dataset"][name] = group(
            [row for row in rows if row["dataset"] == name]
        )
    return output


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if not (math.isfinite(args.mse_tolerance) and args.mse_tolerance > 0):
        raise ValueError("--mse-tolerance must be positive and finite")
    if not (4 <= args.min_source_k <= args.max_source_k <= 56):
        raise ValueError("source K range must lie within 4..56")
    if args.samples_per_k < 1 or args.max_generate_attempts < 1:
        raise ValueError("synthetic sample and attempt counts must be positive")
    if args.real_samples_per_dataset < 1 and args.manifest:
        raise ValueError("--real-samples-per-dataset must be positive with manifests")
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_json}")
    checkpoints = []
    restored = []
    for path in (args.old_checkpoint, args.new_checkpoint):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("objective_version") not in V16_OBJECTIVE_VERSIONS:
            raise ValueError(f"not a v16 deployment checkpoint: {path}")
        model, model_config, _ = build_model_from_checkpoint(checkpoint)
        checkpoints.append(checkpoint)
        restored.append((model, model_config))
    old, new = checkpoints
    old_model, old_config = restored[0]
    new_model, new_config = restored[1]
    old_data = _dataset_config_from_checkpoint(old, old_config)
    new_data = _dataset_config_from_checkpoint(new, new_config)
    for key in ("num_points", "point_dim"):
        if old_data[key] != new_data[key]:
            raise ValueError(f"checkpoints disagree on {key}; paired input is invalid")
    if old_model.degree != new_model.degree:
        raise ValueError("checkpoints disagree on B-spline degree")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    old_model.to(device).eval()
    new_model.to(device).eval()
    cases = list(_synthetic_cases(args, old, new, old_config))
    cases.extend(_real_cases(
        args, num_points=int(old_data["num_points"]),
        point_dim=int(old_data["point_dim"]),
    ))
    rows = []
    for index, case in enumerate(cases, 1):
        points = case["points"]
        row = {key: value for key, value in case.items() if key != "points"}
        row["old"] = _native_refit(old_model, points, device, args.mse_tolerance)
        row["new"] = _native_refit(new_model, points, device, args.mse_tolerance)
        rows.append(row)
        print(
            f"[{index}/{len(cases)}] {row['case_id']} "
            f"old={row['old']['mse']:.3e}/K{row['old']['k']} "
            f"new={row['new']['mse']:.3e}/K{row['new']['k']}",
            flush=True,
        )
    summary = summarize(rows, tolerance=args.mse_tolerance)
    report = {
        "protocol": "native_one_shot_one_standard_bspline_refit_no_verified_or_hard_repair",
        "mse_definition": "mean squared Euclidean distance at the observed ordered points",
        "mse_tolerance": args.mse_tolerance,
        "old_checkpoint": str(args.old_checkpoint.resolve()),
        "new_checkpoint": str(args.new_checkpoint.resolve()),
        "synthetic_generator_checkpoint": str(args.old_checkpoint.resolve()),
        "synthetic_source_k_range": [args.min_source_k, args.max_source_k],
        "samples_per_k": args.samples_per_k,
        "device": str(device), "summary": summary, "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    for name in ("synthetic_overall", "synthetic_high_k_ge_45", "synthetic_k_56"):
        value = summary[name]
        def show(label: str) -> str:
            metrics = value[label]
            if not value["n"]:
                return "n/a"
            return (
                f"pass={metrics['pass_rate']:.1%} "
                f"MSE={metrics['mse_mean']:.3e} K={metrics['k_mean']:.2f}"
            )
        print(
            f"{name} n={value['n']}: "
            f"old {show('old')}; new {show('new')}"
        )
    print(f"Saved paired diagnostic: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
