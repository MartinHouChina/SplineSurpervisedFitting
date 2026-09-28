"""Screen compatible historical weights on a common validation panel, never test cases."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_geometry import file_sha256
from overnight_datasets import default_manifests
from spline_fitting.checkpointing import build_model_from_checkpoint, V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION
from spline_fitting.data.real_world import RealWorldCurveDataset
from spline_fitting.data.synthetic import SyntheticCubicBSplineDataset
from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
from visualize_batch_comparison import deployment_geometries


def network_forward(model, checkpoint, points, device, mse_tolerance):
    with torch.inference_mode():
        batch = points.unsqueeze(0).to(device=device, dtype=next(model.parameters()).dtype)
        if checkpoint.get("objective_version") == V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION:
            return model.forward_deployment(batch, mse_tolerance=mse_tolerance)
        elif callable(getattr(model, "forward_deployment", None)):
            return model.forward_deployment(batch)
        else:
            return model(batch)


def raw_network_fit(model, checkpoint, points, device, mse_tolerance):
    """Use the historical authoritative mask and coordinates, with a common LS refit."""
    output = network_forward(model, checkpoint, points, device, mse_tolerance)
    with torch.inference_mode():
        _, knots, mask = deployment_geometries(output)
        parameters = output["params"][0].detach().cpu().double()
        selected = knots[mask].detach().cpu().double().sort().values
    fit = refit_bspline_control_points(parameters, points.double(), selected,
                                      degree=model.degree, smoothness_weight=0.0,
                                      control_ridge=0.0, interpolate_endpoints=True)
    return fit, parameters


def validation_panel(data_root, real_count, seed):
    cases = []
    # Deliberately independent screening seeds, not the saved test seed 20000.
    # K here is the generating source count, not a certified minimal label.
    for k in range(4, 25):
        sample = SyntheticCubicBSplineDataset(
            size=1, num_points=192, min_control_points=k + 4,
            max_control_points=k + 4, seed=seed + k,
            noise_std=0.001, canonical_knot_tolerance=0.0,
            certified_minimal_source=False, cache_samples=False,
        )[0]
        cases.append(dict(dataset="Synthetic", sample_id=f"screen_seed{seed+k}",
                          group_id=f"screen_seed{seed+k}", source_k=k,
                          split="independent_screening", points=sample["points"]))
    for source, manifest in default_manifests(data_root).items():
        dataset = RealWorldCurveDataset(manifest, split="val", num_points=192)
        if len(dataset) < real_count:
            raise ValueError(f"Insufficient validation cases: {source}: {len(dataset)}")
        indices = list(range(len(dataset)))
        random.Random(f"{seed}:{source}").shuffle(indices)
        picked, groups = [], set()
        for index in indices:
            group = dataset.records[index]["group_id"]
            if group not in groups:
                picked.append(index)
                groups.add(group)
            if len(picked) == real_count:
                break
        if len(picked) < real_count:
            picked.extend(i for i in indices if i not in picked)
            picked = picked[:real_count]
        for index in picked:
            sample = dataset[index]
            cases.append(dict(dataset=source, sample_id=sample["curve_id"],
                              group_id=sample["group_id"], source_k=None,
                              split="val", points=sample["points"]))
    return cases


def panel_metadata(cases):
    return [{**{k: v for k, v in case.items() if k != "points"},
             "points_sha256": hashlib.sha256(case["points"].double().numpy().tobytes()).hexdigest()}
            for case in cases]


def selection_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
    per_source = {}
    for source, items in groups.items():
        valid = [r for r in items if r["status"] == "ok"]
        per_source[source] = dict(
            n=len(items), failures=len(items)-len(valid),
            dual_pass_rate=sum(r.get("joint_pass", False) for r in items)/len(items),
            mse_pass_rate=sum(r.get("fit_pass", False) for r in items)/len(items),
            mean_k=float(np.mean([r["k"] for r in valid])) if valid else None,
            mean_mse=float(np.mean([r["mse"] for r in valid])) if valid else None,
        )
    valid = [r for r in rows if r["status"] == "ok"]
    rates = [s["dual_pass_rate"] for s in per_source.values()]
    return dict(per_source=per_source, macro_dual_pass=float(np.mean(rates)),
                worst_source_dual_pass=min(rates),
                pooled_dual_pass=sum(r.get("joint_pass", False) for r in rows)/len(rows),
                mean_k=float(np.mean([r["k"] for r in valid])) if valid else None,
                mean_mse=float(np.mean([r["mse"] for r in valid])) if valid else None,
                failures=len(rows)-len(valid))


def rank_key(record):
    s = record["summary"]
    return (-s["macro_dual_pass"], -s["worst_source_dual_pass"],
            s["failures"], s["mean_k"] if s["mean_k"] is not None else float("inf"),
            s["mean_mse"] if s["mean_mse"] is not None else float("inf"), record["path"])


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, default=ROOT / "outputs/diagnostics/historical_checkpoint_metadata_20260922.json")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/diagnostics/historical_common_validation_20260922.json")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-internal-knots", type=int, default=32)
    p.add_argument("--real-count", type=int, default=8)
    p.add_argument("--seed", type=int, default=731000000)
    p.add_argument("--mse-tolerance", type=float, default=5e-5)
    p.add_argument("--max-squared-error-tolerance", type=float, default=5e-4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--shortlist-from", type=Path)
    p.add_argument("--shortlist-size", type=int, default=5)
    args = p.parse_args()
    if args.output.exists() and not args.resume:
        p.error("Output exists; use --resume or a new output path")
    torch.set_num_threads(4)
    torch.manual_seed(20260922)
    device = torch.device(args.device)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    eligible = [r for r in inventory["records"] if r.get("screening_eligible", False)
                and 0 < r["capacity"] <= args.max_internal_knots]
    if args.shortlist_from:
        prior = json.loads(args.shortlist_from.read_text(encoding="utf-8"))
        shortlist = {r["path"] for r in prior["ranking"][:args.shortlist_size]}
        eligible = [r for r in eligible if r["path"] in shortlist]
    cases = validation_panel(args.data_root, args.real_count, args.seed)
    protocol = dict(mse_tolerance=args.mse_tolerance,
                    max_squared_error_tolerance=args.max_squared_error_tolerance,
                    max_internal_knots=args.max_internal_knots, panel=panel_metadata(cases),
                    inventory_sha256=file_sha256(args.inventory),
                    shortlist_source_sha256=file_sha256(args.shortlist_from) if args.shortlist_from else None,
                    eligible_paths=[r["path"] for r in eligible],
                    selection="raw common-refit validation only: source-macro dual pass, worst-source dual pass, failures, K, MSE; no repair or test metrics",
                    no_global_best_claim="Best on this finite screening panel among available compatible <=32 weights; not a general optimality claim")
    result = dict(schema_version=1, protocol=protocol, screened=[], excluded=[])
    if args.resume and args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if result["protocol"] != protocol:
            raise ValueError("Resume protocol/panel changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = {r["sha256"]: r["path"] for r in result["screened"]}
    done_paths = {r["path"] for r in result["screened"] + result["excluded"]}
    print(f"Screening {len(eligible)} eligible files on {len(cases)} fixed validation curves", flush=True)
    for number, entry in enumerate(eligible, 1):
        if entry["path"] in done_paths:
            continue
        path = ROOT / entry["path"]
        digest = file_sha256(path)
        if digest in seen:
            result["excluded"].append(dict(path=entry["path"], reason="identical_checkpoint_sha256", same_as=seen[digest]))
            write_json(args.output, result)
            continue
        started = time.perf_counter()
        model = checkpoint = None
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model, config, _ = build_model_from_checkpoint(checkpoint)
            model.to(device).eval()
            rows = []
            for index, case in enumerate(cases):
                row = {k: v for k, v in case.items() if k != "points"}
                try:
                    fit, parameters = raw_network_fit(model, checkpoint, case["points"], device, args.mse_tolerance)
                    squared = (fit.evaluate(parameters) - case["points"].double()).square().sum(-1)
                    if not torch.isfinite(squared).all():
                        raise ValueError("Non-finite fit")
                    mse, peak, k = float(squared.mean()), float(squared.max()), fit.internal_knots.numel()
                    if k > args.max_internal_knots:
                        raise ValueError("Output exceeds shared capacity")
                    row.update(status="ok", mse=mse, max_squared_error=peak, k=k,
                               fit_pass=mse <= args.mse_tolerance,
                               joint_pass=mse <= args.mse_tolerance and peak <= args.max_squared_error_tolerance)
                except (ValueError, RuntimeError, KeyError, TypeError) as error:
                    row.update(status="failed", fit_pass=False, joint_pass=False, error=f"{type(error).__name__}: {error}")
                rows.append(row)
                if (index + 1) % 20 == 0:
                    print(f"[{number}/{len(eligible)}] {path.name}: {index+1}/{len(cases)}", flush=True)
            record = dict(path=entry["path"], sha256=digest, epoch=checkpoint.get("epoch"),
                          capacity=entry["capacity"], objective_version=checkpoint.get("objective_version"),
                          rows=rows, summary=selection_summary(rows), wall_seconds=time.perf_counter()-started)
            result["screened"].append(record)
            seen[digest] = entry["path"]
            s = record["summary"]
            print(f"[{number}/{len(eligible)}] {path.name}: dual macro={s['macro_dual_pass']:.3f} worst={s['worst_source_dual_pass']:.3f} K={s['mean_k']} failures={s['failures']}", flush=True)
        except (ValueError, RuntimeError, KeyError, TypeError) as error:
            result["excluded"].append(dict(path=entry["path"], reason=f"strict_restore_failed: {type(error).__name__}: {error}"))
            print(f"EXCLUDED {path.name}: {error}", flush=True)
        finally:
            del model, checkpoint
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        ranked = sorted((r for r in result["screened"] if r["summary"]["failures"] < len(cases)), key=rank_key)
        result["ranking"] = [{k: r[k] for k in ("path", "sha256", "epoch", "capacity", "summary")} for r in ranked]
        result["winner"] = result["ranking"][0] if ranked else None
        write_json(args.output, result)
    print(f"Saved: {args.output}\nWinner: {result.get('winner', {}).get('path')}", flush=True)
    return 0 if result.get("winner") else 1


if __name__ == "__main__":
    raise SystemExit(main())
