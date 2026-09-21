"""Paired, read-only learned versus chord-parameterization diagnostics.

The chord arm transports every knot with the same piecewise-linear t -> chord
map before the common float64 refit. It never reselects the mask or tunes a
checkpoint. This is not a deployment policy or a test-set oracle to train on.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from spline_fitting.checkpointing import (  # noqa: E402
    V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
    build_model_from_checkpoint,
)
from spline_fitting.data.synthetic import SyntheticCubicBSplineDataset  # noqa: E402
from spline_fitting.data.real_world import RealWorldCurveDataset  # noqa: E402
from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points  # noqa: E402
from spline_fitting.evaluation.gradient_knot_pruning import chord_length_parameters  # noqa: E402


def content_hash(points, reference=None):
    raw = points.contiguous().cpu().numpy().tobytes()
    if reference is not None:
        raw += reference.contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def warp_knots(knots, old_parameters, new_parameters):
    """Transport knots, including empty/singleton sets, without endpoint hacks."""
    if any(x.ndim != 1 for x in (knots, old_parameters, new_parameters)):
        raise ValueError("knots and parameter grids must be one-dimensional")
    if old_parameters.shape != new_parameters.shape or old_parameters.numel() < 2:
        raise ValueError("paired parameter grids must have equal length >= 2")
    if not all(torch.isfinite(x).all() for x in (knots, old_parameters, new_parameters)):
        raise ValueError("knots and grids must be finite")
    if torch.any(old_parameters[1:] <= old_parameters[:-1]):
        raise ValueError("old parameters must be strictly increasing")
    if torch.any(new_parameters[1:] < new_parameters[:-1]):
        raise ValueError("new parameters must be non-decreasing")
    if torch.any((knots < old_parameters[0]) | (knots > old_parameters[-1])):
        raise ValueError("knots must lie inside the old parameter domain")
    right = torch.searchsorted(old_parameters.contiguous(), knots.contiguous(), right=True)
    right = right.clamp(1, old_parameters.numel() - 1)
    left = right - 1
    fraction = (knots - old_parameters[left]) / (old_parameters[right] - old_parameters[left])
    return new_parameters[left] + fraction * (new_parameters[right] - new_parameters[left])


def refit_metrics(points, parameters, knots, *, degree, tolerance):
    fit = refit_bspline_control_points(
        parameters, points, knots, degree=degree, smoothness_weight=0.0,
        control_ridge=0.0, interpolate_endpoints=True,
    )
    squared = (fit.evaluate(parameters) - points).square().sum(-1)
    if not torch.isfinite(squared).all():
        raise ValueError("non-finite diagnostic fit")
    return dict(mse=float(squared.mean()), max_squared_error=float(squared.max()),
                fit_pass=bool(squared.mean() <= tolerance), k=int(knots.numel()))


def compare_geometry(points, parameters, knots, *, degree, tolerance):
    """One geometry and one mask: only its parameter coordinate is changed."""
    points, parameters, knots = (x.detach().cpu().double() for x in (points, parameters, knots))
    chord = chord_length_parameters(points)
    transported = warp_knots(knots, parameters, chord)
    learned = refit_metrics(points, parameters, knots, degree=degree, tolerance=tolerance)
    counterfactual = refit_metrics(points, chord, transported, degree=degree, tolerance=tolerance)
    return dict(
        learned=learned, chord_warped=counterfactual,
        chord_minus_learned_mse=counterfactual["mse"] - learned["mse"],
        chord_mse_better=counterfactual["mse"] < learned["mse"],
        chord_rescues_failure=not learned["fit_pass"] and counterfactual["fit_pass"],
        chord_breaks_pass=learned["fit_pass"] and not counterfactual["fit_pass"],
        parameter_chord_rmse=float((parameters - chord).square().mean().sqrt()),
        parameter_chord_max_abs=float((parameters - chord).abs().max()),
        learned_parameters=parameters.tolist(), chord_parameters=chord.tolist(),
        learned_knots=knots.tolist(), chord_warped_knots=transported.tolist(),
    )


@torch.inference_mode()
def diagnose_case(model, case, tolerance):
    weight = next(model.parameters())
    points = case["points"].to(device=weight.device, dtype=weight.dtype).unsqueeze(0)
    context = model.encode_candidates(points, mse_tolerance=tolerance)
    mask = model.select_mask(context)
    output = model.decode_subset(context, mask)
    return dict(
        dataset=case["dataset"], sample_id=case["sample_id"],
        source_k=case.get("source_k"), sample_content_sha256=case["content_sha256"],
        remote_sample_content_sha256=case.get("remote_content_sha256"),
        remote_content_hash_matches=case.get("remote_content_hash_matches"),
        learned_keep_mask=mask[0].cpu().tolist(),
        proposal_parameter_trust=float(context["proposal_parameter_trust"].mean()),
        final=compare_geometry(points[0], output["params"][0],
                               output["internal_knots"][0][mask[0]].sort().values,
                               degree=model.degree, tolerance=tolerance),
        dense=compare_geometry(points[0], context["proposal_params"][0],
                               context["proposal_internal_knots"][0].sort().values,
                               degree=model.degree, tolerance=tolerance),
    )


def load_cases(cases_path, comparison_path, limit, data_root=None, input_audit=None,
               allow_content_mismatch=False):
    saved = (json.loads(cases_path.read_text(encoding="utf-8-sig"))
             if cases_path is not None else {"records": []})
    cases, counts = [], defaultdict(int)
    for row in saved["records"]:
        if counts[row["dataset"]] >= limit:
            continue
        counts[row["dataset"]] += 1
        points = torch.tensor(row["network_input_points_normalized"], dtype=torch.float32)
        raw_reference = row.get("original_reference_points_normalized")
        reference = None if raw_reference is None else torch.tensor(raw_reference, dtype=torch.float64)
        cases.append(dict(dataset=row["dataset"], sample_id=row["sample_id"], points=points,
                          content_sha256=content_hash(points, reference)))
    if comparison_path is not None:
        comparison = json.loads(comparison_path.read_text(encoding="utf-8-sig"))
        ours = [r for r in comparison["measurements"] if r["method"] == "ours"]
        hashes = comparison["metadata"]["sample_content_sha256"]
        if len(ours) != len(hashes):
            raise ValueError("comparison row/hash lengths disagree")
        expected = {(r["dataset"], r["sample_id"]): h for r, h in zip(ours, hashes)}
        if data_root is not None:
            from overnight_datasets import default_manifests

            manifests = default_manifests(data_root)
            for provenance in comparison["metadata"]["datasets"]:
                name = provenance["dataset"]
                if name == "Synthetic":
                    continue
                dataset = RealWorldCurveDataset(manifests[name], split="test",
                    num_points=comparison["metadata"]["num_points"])
                # Local snapshots can differ in row count/order. Match saved
                # sample IDs, not remote row indices; hash every resulting input.
                index_by_id = {r["sample_id"]: i for i, r in enumerate(dataset.records)}
                desired = [r["sample_id"] for r in ours if r["dataset"] == name]
                available = [sample_id for sample_id in desired if sample_id in index_by_id]
                absent = [sample_id for sample_id in desired if sample_id not in index_by_id]
                if input_audit is not None:
                    input_audit.append(dict(dataset=name, local_test_count=len(dataset),
                        benchmark_count=len(desired), available_benchmark_count=len(available),
                        selected_count=min(limit, len(available)), absent_benchmark_sample_ids=absent))
                print(f"{name}: {len(available)}/{len(desired)} benchmark IDs locally available; "
                      f"using {min(limit, len(available))}", flush=True)
                for sample_id in available[:limit]:
                    index = index_by_id[sample_id]
                    sample = dataset[index]
                    raw = dataset.load_reference_points(index).double()
                    reference = (raw - sample["center"].double()) / sample["scale"].double()
                    points = sample["points"]
                    cases.append(dict(dataset=name, sample_id=sample["curve_id"], points=points,
                                      content_sha256=content_hash(points, reference)))
        for provenance in comparison["metadata"]["datasets"]:
            if provenance["dataset"] != "Synthetic":
                continue
            indices = provenance["selected_indices"]
            # Select by saved order, spanning source K; never use outcome/MSE.
            positions = torch.linspace(0, len(indices) - 1, min(limit, len(indices))).round().long().tolist()
            chosen = [indices[p] for p in positions]
            dataset = SyntheticCubicBSplineDataset(size=max(chosen) + 1,
                seed=provenance["seed"], cache_samples=False, **provenance["config"])
            for index in chosen:
                sample = dataset[index]
                points = sample["points"]
                cases.append(dict(dataset="Synthetic", sample_id=f"seed{provenance['seed']}_sample{index}",
                    source_k=int(sample["source_internal_knot_count"]), points=points,
                    content_sha256=content_hash(points)))
        for case in cases:
            key = case["dataset"], case["sample_id"]
            case["remote_content_sha256"] = expected.get(key)
            case["remote_content_hash_matches"] = expected.get(key) == case["content_sha256"]
            if not case["remote_content_hash_matches"] and not allow_content_mismatch:
                raise ValueError(f"saved/regenerated input+reference hash mismatch: {key}")
    if not cases:
        raise ValueError("no cases selected")
    if len({(r["dataset"], r["sample_id"]) for r in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    return cases


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["checkpoint_label"], row["dataset"]].append(row)
    summaries = []
    for (checkpoint, dataset), rows in groups.items():
        for geometry in ("final", "dense"):
            arms = [r[geometry] for r in rows]
            entry = dict(checkpoint_label=checkpoint, dataset=dataset, geometry=geometry, n=len(rows))
            for arm in ("learned", "chord_warped"):
                for metric in ("mse", "max_squared_error", "k"):
                    entry[f"{arm}_{metric}_mean"] = sum(r[arm][metric] for r in arms) / len(rows)
                entry[f"{arm}_pass_count"] = sum(r[arm]["fit_pass"] for r in arms)
            for flag in ("chord_mse_better", "chord_rescues_failure", "chord_breaks_pass"):
                entry[f"{flag}_count"] = sum(r[flag] for r in arms)
            summaries.append(entry)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cases-json", type=Path)
    source.add_argument("--data-root", type=Path,
                        help="Rebuild exact comparison test IDs from a complete local data tree")
    parser.add_argument("--comparison", type=Path,
                        help="Verify saved input+reference hashes and regenerate saved synthetic test IDs")
    parser.add_argument("--allow-content-mismatch", action="store_true",
                        help="Explicit diagnostic only: disclose local/remote hash mismatches; never claim exact benchmark reproduction")
    parser.add_argument("--samples-per-dataset", type=int, default=8)
    parser.add_argument("--mse-tolerance", type=float, default=5e-5)
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.samples_per_dataset <= 8:
        parser.error("samples-per-dataset must be 1..8 for this bounded diagnostic")
    if args.torch_num_threads < 1 or not 0 < args.mse_tolerance < float("inf"):
        parser.error("threads and finite tolerance must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}; choose a new path")
    if len(set(p.resolve() for p in args.checkpoint)) != len(args.checkpoint):
        parser.error("duplicate checkpoint paths")
    if args.data_root is not None and args.comparison is None:
        parser.error("--data-root requires --comparison for exact saved case IDs")
    torch.set_num_threads(args.torch_num_threads)
    input_audit = []
    cases = load_cases(args.cases_json, args.comparison, args.samples_per_dataset, args.data_root, input_audit,
                       args.allow_content_mismatch)
    records, checkpoints = [], []
    for path in args.checkpoint:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("objective_version") != V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION:
            parser.error(f"native v16 checkpoint required: {path}")
        model, config, _ = build_model_from_checkpoint(checkpoint)
        model.to(args.device).eval()
        checkpoints.append(dict(path=str(path.resolve()), label=path.stem,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), epoch=checkpoint.get("epoch"),
            model_config=config))
        for case in cases:
            row = diagnose_case(model, case, args.mse_tolerance)
            row["checkpoint_label"] = path.stem
            records.append(row)
            print(f"{path.stem}/{case['dataset']}/{case['sample_id']}: "
                  f"final learned/chord={row['final']['learned']['mse']:.6g}/"
                  f"{row['final']['chord_warped']['mse']:.6g}", flush=True)
    report = dict(
        role="diagnostic_only_not_training_targets_checkpoint_selection_or_deployment",
        scope="normalized INPUT-point squared Euclidean residuals; no reference or continuous-curve claim",
        coordinate_policy="same mask; knots transported by piecewise-linear learned t -> chord grid; no relocation or reselection",
        interpretation="Non-affine parameter transport changes the spline function space; controls are refitted. This is not exact curve-preserving reparameterization.",
        solver="common CPU float64 endpoint-constrained least squares; zero smoothness/ridge",
        limitations="Small saved-case subset, not a random population sample; no timing comparison or causal capacity/depth/peak ablation.",
        selection="First locally available saved benchmark IDs per source (missing IDs disclosed); synthetic evenly spaced over saved K-ordered IDs, independent of errors",
        content_hash_verified_against_comparison=(args.comparison is not None and
            all(case["remote_content_hash_matches"] for case in cases)),
        remote_content_hash_match_count=sum(case.get("remote_content_hash_matches", False) for case in cases),
        unique_case_count=len(cases), allow_content_mismatch=args.allow_content_mismatch,
        data_root=None if args.data_root is None else str(args.data_root.resolve()),
        cases_json=None if args.cases_json is None else str(args.cases_json.resolve()),
        cases_json_sha256=None if args.cases_json is None else hashlib.sha256(args.cases_json.read_bytes()).hexdigest(),
        comparison=None if args.comparison is None else str(args.comparison.resolve()),
        mse_tolerance=args.mse_tolerance, checkpoints=checkpoints, local_input_availability=input_audit,
        runtime=dict(torch=torch.__version__, device=args.device, torch_num_threads=args.torch_num_threads),
        code_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), ROOT / "src/spline_fitting/models/v16_network.py",
                         ROOT / "src/spline_fitting/evaluation/bspline_inference.py",
                         ROOT / "src/spline_fitting/data/synthetic.py",
                         ROOT / "src/spline_fitting/data/real_world.py"]},
        input_cases=[dict(dataset=case["dataset"], sample_id=case["sample_id"],
                          source_k=case.get("source_k"),
                          network_input_points_normalized=case["points"].tolist(),
                          sample_content_sha256=case["content_sha256"],
                          remote_content_hash_matches=case.get("remote_content_hash_matches"))
                     for case in cases],
        summary=summarize(records), records=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved diagnostic: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
