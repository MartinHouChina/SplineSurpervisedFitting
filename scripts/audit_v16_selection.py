"""Paired diagnostic of v16 proposal, hard selection and subset refinement.

Every checkpoint sees the same independently seeded synthetic curves.  The
three native measurements are: all proposal knots; the deployed mask with
unchanged proposal geometry; that same mask after the learned subset decoder.
Optional numerical teachers are diagnostic references, never deployment repair.
"""
from __future__ import annotations

# ruff: noqa: E402

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from quick_compare_v16_checkpoints import _synthetic_cases
from spline_fitting.checkpointing import V16_OBJECTIVE_VERSIONS, build_model_from_checkpoint
from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
from spline_fitting.evaluation.knot_diagnostics import warp_internal_knots_to_parameterization
from spline_fitting.evaluation.minimal_knot_pruning import prune_knots_to_rms_tolerance
from spline_fitting.training.one_shot_teacher import OneShotTeacherConfig, load_one_shot_teacher_cache
from spline_fitting.training.v16_feasible_teacher import _proposal_fingerprint
from visualize_batch_comparison import _dataset_config_from_checkpoint


def _source_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("source-k must be comma-separated integers") from error
    if not counts or len(set(counts)) != len(counts) or any(k < 4 or k > 56 for k in counts):
        raise argparse.ArgumentTypeError("source-k must contain distinct integers in 4..56")
    return counts


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-checkpoint", type=Path, required=True)
    p.add_argument("--new-checkpoint", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--source-k", type=_source_counts, default=_source_counts("4,8,16,24,32,40,48,56"))
    p.add_argument("--samples-per-k", type=int, default=2)
    p.add_argument("--seed", type=int, default=9_100_000_000)
    p.add_argument("--max-generate-attempts", type=int, default=8)
    p.add_argument("--mse-tolerance", type=float, default=1e-4)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--torch-num-threads", type=int, default=4)
    p.add_argument(
        "--teacher-cache", type=Path,
        help="Read training-mask statistics only; these rows are not labels for new diagnostic seeds",
    )
    p.add_argument(
        "--build-greedy-teacher", action="store_true",
        help="Explicitly search numerical teacher masks on each diagnostic proposal (additional solves)",
    )
    return p


def _cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().double()


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _finite_list(value):
    return value.tolist() if bool(torch.isfinite(value).all()) else None


def _geometry(parameters: torch.Tensor, knots: torch.Tensor) -> dict:
    finite = bool(torch.isfinite(parameters).all() and torch.isfinite(knots).all())
    parameters_strict = finite and bool((parameters.diff() > 0).all())
    knots_strict = finite and bool((knots.diff() > 0).all())
    internal = finite and bool(((knots > parameters[0]) & (knots < parameters[-1])).all())
    endpoints = finite and bool(parameters[0] == 0 and parameters[-1] == 1)
    gaps = torch.cat([parameters[:1], knots, parameters[-1:]]).diff()
    return {
        "finite": finite, "parameters_strict": parameters_strict,
        "knots_strict": knots_strict, "knots_internal": internal,
        "unit_endpoints": endpoints,
        "minimum_knot_gap": float(gaps.min()) if finite else None,
        "valid": finite and parameters_strict and knots_strict and internal and endpoints,
    }


def _fit(parameters, points, knots, *, degree, tolerance) -> dict:
    geometry = _geometry(parameters, knots)
    result = {"k": int(knots.numel()), "mse": None, "pass": False, "geometry": geometry}
    if not geometry["valid"]:
        result["error"] = "invalid predicted geometry; no sorting, clipping or repair applied"
        return result
    try:
        fit = refit_bspline_control_points(
            parameters, points, knots, degree=degree,
            smoothness_weight=0.0, control_ridge=0.0, interpolate_endpoints=True,
        )
        mse = float(fit.fit_mse)
        if not math.isfinite(mse):
            raise ValueError("non-finite standard-refit MSE")
    except (RuntimeError, ValueError) as error:
        result["error"] = str(error)
        return result
    result.update(mse=mse, **{"pass": mse <= tolerance})
    return result


def _change(before: dict, after: dict) -> dict:
    delta = (
        after["mse"] - before["mse"]
        if before["mse"] is not None and after["mse"] is not None else None
    )
    return {
        "mse_change": delta, "improved": delta is not None and delta < 0,
        "worsened": delta is not None and delta > 0,
        "rescued_pass": not before["pass"] and after["pass"],
        "lost_pass": before["pass"] and not after["pass"],
    }


def _refinement_motion(original_params, original_knots, decoded_params, decoded_knots) -> dict:
    """Compare corresponding retained slots in a shared observation-index domain."""
    if not (_geometry(original_params, original_knots)["valid"] and
            _geometry(decoded_params, decoded_knots)["valid"]):
        return {"valid": False, "domain": "normalized_observation_index", "mean_abs": None, "max_abs": None}
    reference = torch.linspace(0, 1, original_params.numel(), dtype=original_params.dtype)
    before = warp_internal_knots_to_parameterization(original_knots, original_params, reference)
    after = warp_internal_knots_to_parameterization(decoded_knots, decoded_params, reference)
    motion = after - before
    return {
        "valid": True, "domain": "normalized_observation_index",
        "before": before.tolist(), "after": after.tolist(),
        "signed_displacement": motion.tolist(),
        "mean_abs": float(motion.abs().mean()) if motion.numel() else 0.0,
        "max_abs": float(motion.abs().max()) if motion.numel() else 0.0,
        "parameter_mean_abs_change": float((decoded_params - original_params).abs().mean()),
    }


def _parameter_knot_ablations(parameters, knots, decoded_parameters, decoded_knots, points, **options):
    """Keep all knots in the parameterization used by the corresponding solve."""
    valid = (_geometry(parameters, knots)["valid"] and
             _geometry(decoded_parameters, decoded_knots)["valid"])
    if not valid:
        invalid = {"k": int(knots.numel()), "mse": None, "pass": False,
                   "geometry": {"valid": False}, "error": "invalid geometry prevents parameter-domain warp"}
        return {"parameter_only": dict(invalid), "knots_only": dict(invalid)}
    unchanged_knots_in_decoded_domain = warp_internal_knots_to_parameterization(
        knots, parameters, decoded_parameters,
    )
    refined_knots_in_proposal_domain = warp_internal_knots_to_parameterization(
        decoded_knots, decoded_parameters, parameters,
    )
    return {
        "parameter_only": _fit(decoded_parameters, points, unchanged_knots_in_decoded_domain, **options),
        "knots_only": _fit(parameters, points, refined_knots_in_proposal_domain, **options),
    }


@torch.inference_mode()
def audit_model(model, points, *, device, tolerance, build_greedy_teacher=False) -> dict:
    model_input = points.unsqueeze(0).to(device=device, dtype=next(model.parameters()).dtype)
    context = model.encode_candidates(model_input, mse_tolerance=tolerance)
    mask_device = model.select_mask(context)
    output = model.decode_subset(context, mask_device)
    mask = mask_device[0].detach().cpu()
    if mask.dtype != torch.bool or not torch.equal(output["learned_keep_mask"], mask_device):
        raise RuntimeError("decoder changed the deployed Boolean subset")
    parameters = _cpu(context["proposal_params"][0])
    knots = _cpu(context["proposal_internal_knots"][0])
    decoded_parameters = _cpu(output["params"][0])
    decoded_knots = _cpu(output["internal_knots"][0])[mask]
    points = _cpu(points)
    options = dict(degree=model.degree, tolerance=tolerance)
    full = _fit(parameters, points, knots, **options)
    selected_original = _fit(parameters, points, knots[mask], **options)
    selected_decoded = _fit(decoded_parameters, points, decoded_knots, **options)
    ablations = _parameter_knot_ablations(
        parameters, knots[mask], decoded_parameters, decoded_knots, points, **options,
    )
    probabilities = _cpu(context["keep_probabilities"][0])
    probability_finite = bool(torch.isfinite(probabilities).all())
    result = {
        "proposal_full": full,
        "free_mask_original": selected_original,
        "free_mask_decoded": selected_decoded,
        "free_mask_parameter_only": ablations["parameter_only"],
        "free_mask_knots_only": ablations["knots_only"],
        "removed_count": int((~mask).sum()),
        "selected_slots": mask.nonzero(as_tuple=False).flatten().tolist(),
        "keep_probabilities": probabilities.tolist() if probability_finite else None,
        "keep_probability_std": float(probabilities.std(unbiased=False)) if probability_finite else None,
        "probability_mass": float(probabilities.sum()) if probability_finite else None,
        "geometry_values": {
            "proposal_parameters": _finite_list(parameters),
            "proposal_internal_knots": _finite_list(knots),
            "decoded_parameters": _finite_list(decoded_parameters),
            "decoded_selected_knots": _finite_list(decoded_knots),
        },
        "selection_change": _change(full, selected_original),
        "refinement_change": _change(selected_original, selected_decoded),
        "refinement_motion": _refinement_motion(
            parameters, knots[mask], decoded_parameters, decoded_knots,
        ),
        "teacher": None,
        "teacher_count_topk": None,
    }
    if build_greedy_teacher and full["geometry"]["valid"]:
        teacher = prune_knots_to_rms_tolerance(
            parameters, points, knots, error_tolerance=math.sqrt(tolerance),
            min_internal_knots=int(model.min_selected_knots), degree=model.degree,
            smoothness_weight=0.0, control_ridge=0.0, interpolate_endpoints=True,
        )
        teacher_mask = torch.isin(knots, teacher.final_internal_knots)
        if int(teacher_mask.sum()) != teacher.final_count:
            raise RuntimeError("greedy reference no longer matches fixed proposal slots")
        teacher_output = model.decode_subset(context, teacher_mask.unsqueeze(0).to(device))
        teacher_original = _fit(parameters, points, knots[teacher_mask], **options)
        teacher_decoded_params = _cpu(teacher_output["params"][0])
        teacher_decoded_knots = _cpu(teacher_output["internal_knots"][0])[teacher_mask]
        teacher_decoded = _fit(teacher_decoded_params, points, teacher_decoded_knots, **options)
        result["teacher"] = {
            "kind": "greedy_fixed_proposal_reference_not_globally_minimal",
            "matches_training_anchor_counterfactual_teacher": False,
            "selected_slots": teacher_mask.nonzero(as_tuple=False).flatten().tolist(),
            "original": teacher_original, "decoded": teacher_decoded,
            "decoded_parameters": _finite_list(teacher_decoded_params),
            "decoded_selected_knots": _finite_list(teacher_decoded_knots),
            "accepted_deletions": len(teacher.accepted_steps),
            "free_mask_intersection_count": int((teacher_mask & mask).sum()),
            "refinement_change": _change(teacher_original, teacher_decoded),
            "refinement_motion": _refinement_motion(
                parameters, knots[teacher_mask], teacher_decoded_params, teacher_decoded_knots,
            ),
        }
        count_mask_device = model.select_mask_at_count(
            context, torch.tensor([teacher.final_count], device=device),
        )
        count_mask = count_mask_device[0].detach().cpu()
        count_output = model.decode_subset(context, count_mask_device)
        count_params = _cpu(count_output["params"][0])
        count_knots = _cpu(count_output["internal_knots"][0])[count_mask]
        count_original = _fit(parameters, points, knots[count_mask], **options)
        count_decoded = _fit(count_params, points, count_knots, **options)
        result["teacher_count_topk"] = {
            "kind": "deployed_ranking_and_coverage_at_numerical_teacher_count",
            "requested_count": teacher.final_count,
            "teacher_original_pass": teacher_original["pass"],
            "selected_slots": count_mask.nonzero(as_tuple=False).flatten().tolist(),
            "teacher_mask_intersection_count": int((teacher_mask & count_mask).sum()),
            "original": count_original, "decoded": count_decoded,
            "decoded_parameters": _finite_list(count_params),
            "decoded_selected_knots": _finite_list(count_knots),
            "refinement_change": _change(count_original, count_decoded),
            "refinement_motion": _refinement_motion(
                parameters, knots[count_mask], count_params, count_knots,
            ),
        }
    return result


def teacher_cache_statistics(path: Path, model, *, tolerance: float) -> dict:
    """Validate cache contents, then summarize only their original training rows."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = OneShotTeacherConfig.from_dict(payload["config"])
    batch = load_one_shot_teacher_cache(
        path, expected_config=config, expected_sample_indices=payload["sample_indices"],
        expected_input_shapes=payload["input_shapes"],
    )
    counts = batch.teacher_count.cpu()
    mask = batch.teacher_retained_mask.cpu()
    mse = batch.teacher_fit_mse.cpu().double()
    probabilities = batch.teacher_soft_keep_risk.cpu().double()
    result = {
        "path": str(path.resolve()),
        "scope": "cached_training_rows_only_not_paired_diagnostic_samples",
        "used_as_diagnostic_case_labels": False,
        "proposal_fingerprint_matches_new_model": config.proposal_fingerprint == _proposal_fingerprint(model),
        "sample_count": int(counts.numel()), "candidate_count": int(mask.shape[1]),
        "count_histogram": {str(k): n for k, n in sorted(Counter(counts.tolist()).items())},
        "count_mean": float(counts.double().mean()),
        "retained_fraction_by_slot": mask.double().mean(0).tolist(),
        "mse_mean": float(mse.mean()),
        "cached_tolerance_mse": config.error_tolerance ** 2,
        "cached_threshold_pass_rate": float(batch.teacher_threshold_satisfied.double().mean()),
        "requested_threshold_pass_rate": float((mse <= tolerance).double().mean()),
        "keep_risk_std_mean": float(probabilities.std(-1, unbiased=False).mean()),
    }
    if batch.teacher_counterfactual_probe_mask is not None:
        probes = batch.teacher_counterfactual_probe_mask.cpu()
        swaps = batch.teacher_counterfactual_swap_feasible.cpu()
        result["counterfactual_probe_count"] = int(probes.sum())
        result["counterfactual_feasible_swap_fraction"] = (
            float(swaps[probes].double().mean()) if bool(probes.any()) else None
        )
    return result


def summarize(rows: list[dict]) -> dict:
    def group(selected):
        result = {"n": len(selected)}
        for label in ("old", "new"):
            models = [row[label] for row in selected]
            metrics = {}
            for stage in ("proposal_full", "free_mask_original", "free_mask_parameter_only",
                          "free_mask_knots_only", "free_mask_decoded"):
                values = [model[stage] for model in models]
                metrics[stage] = {
                    "mse_mean": _mean(value["mse"] for value in values),
                    "pass_rate": _mean(value["pass"] for value in values),
                    "k_mean": _mean(value["k"] for value in values),
                    "valid_geometry_fraction": _mean(value["geometry"]["valid"] for value in values),
                    "valid_refit_count": sum(value["mse"] is not None for value in values),
                }
            metrics["removed_count_mean"] = _mean(model["removed_count"] for model in models)
            metrics["keep_probability_std_mean"] = _mean(model["keep_probability_std"] for model in models)
            for kind in ("selection_change", "refinement_change"):
                metrics[kind] = {
                    "mse_change_mean": _mean(model[kind]["mse_change"] for model in models),
                    "improved_count": sum(model[kind]["improved"] for model in models),
                    "worsened_count": sum(model[kind]["worsened"] for model in models),
                    "rescued_pass_count": sum(model[kind]["rescued_pass"] for model in models),
                    "lost_pass_count": sum(model[kind]["lost_pass"] for model in models),
                }
            metrics["refinement_motion_mean_abs"] = _mean(
                model["refinement_motion"]["mean_abs"] for model in models
            )
            for mask_kind in ("teacher", "teacher_count_topk"):
                teachers = [model[mask_kind] for model in models if model[mask_kind] is not None]
                metrics[mask_kind] = None if not teachers else {
                    stage: {
                        "mse_mean": _mean(teacher[stage]["mse"] for teacher in teachers),
                        "pass_rate": _mean(teacher[stage]["pass"] for teacher in teachers),
                        "k_mean": _mean(teacher[stage]["k"] for teacher in teachers),
                    } for stage in ("original", "decoded")
                }
            result[label] = metrics
        result["new_minus_old_decoded"] = {
            "mse_change_mean": _mean(
                row["new"]["free_mask_decoded"]["mse"] - row["old"]["free_mask_decoded"]["mse"]
                for row in selected
                if row["new"]["free_mask_decoded"]["mse"] is not None
                and row["old"]["free_mask_decoded"]["mse"] is not None
            ),
            "k_change_mean": _mean(
                row["new"]["free_mask_decoded"]["k"] - row["old"]["free_mask_decoded"]["k"]
                for row in selected
            ),
            "rescued_pass_count": sum(
                row["new"]["free_mask_decoded"]["pass"] and not row["old"]["free_mask_decoded"]["pass"]
                for row in selected
            ),
            "lost_pass_count": sum(
                row["old"]["free_mask_decoded"]["pass"] and not row["new"]["free_mask_decoded"]["pass"]
                for row in selected
            ),
        }
        return result

    return {
        "overall": group(rows),
        "by_source_k": {str(k): group([row for row in rows if row["source_k"] == k])
                        for k in sorted({row["source_k"] for row in rows})},
        "high_k_ge_40": group([row for row in rows if row["source_k"] >= 40]),
    }


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.samples_per_k < 1 or args.max_generate_attempts < 1 or args.torch_num_threads < 1:
        raise ValueError("sample counts, generation attempts and thread count must be positive")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if not math.isfinite(args.mse_tolerance) or args.mse_tolerance <= 0:
        raise ValueError("mse-tolerance must be positive and finite")
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_json}")
    torch.set_num_threads(args.torch_num_threads)
    checkpoints, models, configs = [], [], []
    for path in (args.old_checkpoint, args.new_checkpoint):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("objective_version") not in V16_OBJECTIVE_VERSIONS:
            raise ValueError(f"not a v16 checkpoint: {path}")
        model, config, _ = build_model_from_checkpoint(checkpoint)
        checkpoints.append(checkpoint)
        models.append(model)
        configs.append(config)
    data = [_dataset_config_from_checkpoint(checkpoint, config)
            for checkpoint, config in zip(checkpoints, configs)]
    if any(data[0][key] != data[1][key] for key in ("num_points", "point_dim")):
        raise ValueError("paired checkpoints must agree on num_points and point_dim")
    if models[0].degree != 3 or models[1].degree != 3:
        raise ValueError("synthetic exact-K diagnostic requires cubic models")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    teacher_statistics = (
        teacher_cache_statistics(args.teacher_cache, models[1], tolerance=args.mse_tolerance)
        if args.teacher_cache else None
    )
    for model in models:
        model.to(device).eval()
    rows = []
    total = len(args.source_k) * args.samples_per_k
    for k in args.source_k:
        generation = SimpleNamespace(
            min_source_k=k, max_source_k=k, samples_per_k=args.samples_per_k,
            max_generate_attempts=args.max_generate_attempts, seed=args.seed + k * 100_003,
        )
        for case in _synthetic_cases(generation, *checkpoints, configs[0]):
            points = case["points"]
            row = {key: value for key, value in case.items() if key != "points"}
            row["points_sha256"] = hashlib.sha256(points.contiguous().numpy().tobytes()).hexdigest()
            for label, model in zip(("old", "new"), models):
                row[label] = audit_model(
                    model, points, device=device, tolerance=args.mse_tolerance,
                    build_greedy_teacher=args.build_greedy_teacher,
                )
            rows.append(row)
            print(
                f"[{len(rows)}/{total}] {case['case_id']}: "
                f"old {row['old']['free_mask_decoded']['mse']} / K{row['old']['free_mask_decoded']['k']}; "
                f"new {row['new']['free_mask_decoded']['mse']} / K{row['new']['free_mask_decoded']['k']}",
                flush=True,
            )
    report = {
        "protocol": "paired_proposal_selection_refinement_audit_v1",
        "diagnostic_only": True, "model_based_sample_selection": False,
        "mse_definition": "mean squared Euclidean distance at observed ordered points",
        "mse_tolerance": args.mse_tolerance,
        "motion_domain": "normalized_observation_index_shared_by_both_models",
        "free_mask_geometry_ablations": {
            "A_free_mask_original": "proposal parameters, selected proposal knots",
            "B_free_mask_parameter_only": "decoded parameters, proposal knots warped into decoded parameter domain",
            "C_free_mask_knots_only": "proposal parameters, decoded knots warped back into proposal parameter domain",
            "D_free_mask_decoded": "decoded parameters and decoded selected knots",
        },
        "refit_device_dtype": "cpu_float64_standard_production_bspline_solver",
        "old_checkpoint": str(args.old_checkpoint.resolve()),
        "new_checkpoint": str(args.new_checkpoint.resolve()),
        "synthetic_generator_checkpoint": str(args.old_checkpoint.resolve()),
        "source_k": list(args.source_k), "samples_per_k": args.samples_per_k,
        "seed": args.seed, "device": str(device),
        "greedy_teacher_requested": args.build_greedy_teacher,
        "teacher_training_cache": teacher_statistics,
        "summary": summarize(rows), "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved selection/refinement diagnostic: {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
