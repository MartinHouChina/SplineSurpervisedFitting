"""Audit a frozen v16 Teacher cache against CPU deployment refits.

The cache is training data, not an independent test set. This script checks its
sample and Proposal fingerprints before re-solving cached masks with the same
CPU float64 standard B-spline refit used by deployment. Optionally, a Joint
checkpoint adds same-mask Proposal-versus-decoder diagnostics without search.
"""
from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import build_model_from_checkpoint
from spline_fitting.data.v16_mixed import MixedTrainingCurves
from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
from spline_fitting.training.one_shot_teacher import (
    OneShotTeacherConfig,
    load_one_shot_teacher_cache,
)
from spline_fitting.training.v16_feasible_teacher import (
    _fixed_dataset_points,
    _proposal_fingerprint,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--proposal-checkpoint", type=Path,
        default=Path("outputs/self_validation/v16_keep_state_20260917_r1/recovery.proposal.pt"),
    )
    p.add_argument(
        "--teacher-cache", type=Path,
        default=Path("outputs/self_validation/v16_keep_state_20260917_r1/teacher/train.pt"),
    )
    p.add_argument("--joint-checkpoint", type=Path)
    p.add_argument("--indices", type=str, default="0:16",
                   help="Half-open start:stop slice or comma-separated row ids")
    p.add_argument("--output-json", type=Path,
                   default=Path("outputs/self_validation/v16_keep_state_20260917_r1/teacher_cache_audit_0_15.json"))
    p.add_argument("--torch-num-threads", type=int, default=4)
    return p


def parse_indices(spec: str, sample_count: int) -> list[int]:
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError("indices slice must be start:stop")
        start, stop = (int(part) for part in parts)
        indices = list(range(start, stop))
    else:
        indices = [int(part) for part in spec.split(",")]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("indices must be nonempty and unique")
    if min(indices) < 0 or max(indices) >= sample_count:
        raise ValueError("indices lie outside the Teacher cache")
    return indices


def _load_checkpoint(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model, _, _ = build_model_from_checkpoint(payload)
    model.cpu().eval()
    return payload, model


def _fit(parameters: torch.Tensor, points: torch.Tensor,
         knots: torch.Tensor, *, degree: int) -> float:
    fit = refit_bspline_control_points(
        parameters.detach().cpu().double(), points.detach().cpu().double(),
        knots.detach().cpu().double(), degree=degree, smoothness_weight=0.0,
        control_ridge=0.0, interpolate_endpoints=True,
    )
    value = float(fit.fit_mse)
    if not math.isfinite(value):
        raise RuntimeError("CPU production refit returned non-finite MSE")
    return value


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _require_matching_fingerprints(
    proposal_model, expected_proposal: str,
    dataset_fingerprint: str, expected_dataset: str,
) -> None:
    if _proposal_fingerprint(proposal_model) != expected_proposal:
        raise ValueError("Proposal checkpoint weights do not match Teacher cache")
    if dataset_fingerprint != expected_dataset:
        raise ValueError("Joint dataset point/anchor fingerprint differs from Teacher cache")


def _mask_diagnostics(model, context: dict, mask: torch.Tensor,
                      points: torch.Tensor, *, tolerance: float) -> dict:
    if mask.shape != context["proposal_internal_knots"].shape:
        raise ValueError("mask shape does not match Proposal candidates")
    output = model.decode_subset(context, mask)
    if not torch.equal(output["learned_keep_mask"], mask):
        raise RuntimeError("decoder changed a supplied Boolean mask")
    proposal_mse = _fit(
        context["proposal_params"][0], points,
        context["proposal_internal_knots"][0][mask[0]], degree=model.degree,
    )
    decoded_mse = _fit(
        output["params"][0], points,
        output["internal_knots"][0][mask[0]], degree=model.degree,
    )
    return {
        "count": int(mask[0].sum()),
        "proposal_mse": proposal_mse,
        "proposal_pass": proposal_mse <= tolerance,
        "decoded_mse": decoded_mse,
        "decoded_pass": decoded_mse <= tolerance,
        "decoded_minus_proposal_mse": decoded_mse - proposal_mse,
    }


def _summarize(rows: list[dict], *, tolerance: float) -> dict:
    deltas = [row["cpu_minus_cached_mse"] for row in rows]
    cached_pass = [row["cached_pass"] for row in rows]
    cpu_pass = [row["cpu_pass"] for row in rows]
    summary = {
        "sample_count": len(rows),
        "mse_tolerance": tolerance,
        "cached_pass_fraction": _mean([float(x) for x in cached_pass]),
        "cpu_pass_fraction": _mean([float(x) for x in cpu_pass]),
        "pass_disagreement_count": sum(a != b for a, b in zip(cached_pass, cpu_pass)),
        "mean_abs_mse_difference": _mean([abs(x) for x in deltas]),
        "max_abs_mse_difference": max(abs(x) for x in deltas),
        "max_abs_selected_knot_difference": max(
            row["max_abs_selected_knot_difference"] for row in rows
        ),
    }
    if "joint" in rows[0]:
        for key in ("teacher_mask", "teacher_count_topk", "free_mask"):
            items = [row["joint"][key] for row in rows]
            summary[key] = {
                "mean_count": _mean([float(item["count"]) for item in items]),
                "proposal_mean_mse": _mean([item["proposal_mse"] for item in items]),
                "proposal_pass_fraction": _mean([float(item["proposal_pass"]) for item in items]),
                "decoded_mean_mse": _mean([item["decoded_mse"] for item in items]),
                "decoded_pass_fraction": _mean([float(item["decoded_pass"]) for item in items]),
                "mean_decoder_mse_change": _mean([
                    item["decoded_minus_proposal_mse"] for item in items
                ]),
            }
        summary["teacher_count_topk_recall"] = _mean([
            row["joint"]["teacher_count_topk_recall"] for row in rows
        ])
        summary["teacher_count_topk_exact_fraction"] = _mean([
            float(row["joint"]["teacher_count_topk_exact"]) for row in rows
        ])
        feasible_rows = [row for row in rows if row["cached_pass"]]
        summary["teacher_feasible_subset"] = {
            "sample_count": len(feasible_rows),
            "teacher_count_topk_recall": _mean([
                row["joint"]["teacher_count_topk_recall"] for row in feasible_rows
            ]),
            "teacher_count_topk_exact_fraction": _mean([
                float(row["joint"]["teacher_count_topk_exact"])
                for row in feasible_rows
            ]),
            "teacher_mask_decoded_pass_fraction": _mean([
                float(row["joint"]["teacher_mask"]["decoded_pass"])
                for row in feasible_rows
            ]),
            "teacher_count_topk_proposal_pass_fraction": _mean([
                float(row["joint"]["teacher_count_topk"]["proposal_pass"])
                for row in feasible_rows
            ]),
            "teacher_count_topk_decoded_pass_fraction": _mean([
                float(row["joint"]["teacher_count_topk"]["decoded_pass"])
                for row in feasible_rows
            ]),
        }
    return summary


@torch.inference_mode()
def audit(args: argparse.Namespace) -> dict:
    if args.torch_num_threads < 1:
        raise ValueError("torch-num-threads must be positive")
    torch.set_num_threads(args.torch_num_threads)
    proposal_payload, proposal_model = _load_checkpoint(args.proposal_checkpoint)
    train_config = proposal_payload["training_config"]
    if train_config.get("resample_train_each_epoch", True):
        raise ValueError("Teacher cache audit requires fixed, non-resampling Joint data")
    if train_config.get("joint_supervision") != "offline_feasible_teacher":
        raise ValueError("checkpoint did not use offline feasible Teacher supervision")
    raw_cache = torch.load(args.teacher_cache, map_location="cpu", weights_only=True)
    teacher_config = OneShotTeacherConfig.from_dict(raw_cache["config"])
    tolerance = float(train_config["mse_tolerance"])
    if not math.isclose(teacher_config.error_tolerance ** 2, tolerance,
                        rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("Teacher tolerance differs from checkpoint training tolerance")
    if any((teacher_config.smoothness_weight, teacher_config.control_ridge)):
        raise ValueError("Teacher uses nonzero regularization; production refit differs")
    if teacher_config.relocation_strategy != "none" or teacher_config.rcond is not None:
        raise ValueError("Teacher relocation/rcond differs from production refit")
    dataset = MixedTrainingCurves(
        proposal_payload["dataset_config"], (), size=int(train_config["train_size"]),
        seed=int(train_config["seed"]), real_fraction=0.0, epoch=0, resample=False,
        synthetic_high_k_fraction=float(train_config.get("joint_high_k_fraction", 0.0)),
        synthetic_high_k_min_knots=train_config.get("joint_high_k_min_knots"),
    )
    strategy = train_config["feasible_teacher_strategy"]
    points, dataset_fingerprint, _ = _fixed_dataset_points(
        dataset, teacher_strategy=strategy,
        anchor_match_tolerance=float(train_config["feasible_teacher_anchor_match_tolerance"]),
        anchor_fallback_to_greedy=bool(train_config["feasible_teacher_anchor_fallback_to_greedy"]),
        counterfactual_max_probes=int(train_config["feasible_teacher_counterfactual_max_probes"]),
    )
    _require_matching_fingerprints(
        proposal_model, teacher_config.proposal_fingerprint,
        dataset_fingerprint, teacher_config.dataset_fingerprint,
    )
    batch = load_one_shot_teacher_cache(
        args.teacher_cache, expected_config=teacher_config,
        expected_sample_indices=torch.arange(len(dataset)),
        expected_input_shapes={
            "parameters": (len(dataset), points.shape[1]),
            "points": tuple(points.shape),
            "candidate_knots": (len(dataset), proposal_model.max_internal_knots),
        },
    )
    indices = parse_indices(args.indices, len(dataset))
    joint_model = None
    joint_payload = None
    if args.joint_checkpoint is not None:
        joint_payload, joint_model = _load_checkpoint(args.joint_checkpoint)
        if _proposal_fingerprint(joint_model) != teacher_config.proposal_fingerprint:
            raise ValueError("Joint checkpoint moved its frozen Proposal; cached slot labels mismatch")

    rows = []
    model_dtype = next(proposal_model.parameters()).dtype
    for index in indices:
        point = points[index]
        context = proposal_model.encode_candidates(
            point.unsqueeze(0).to(dtype=model_dtype), mse_tolerance=tolerance,
        )
        mask = batch.teacher_retained_mask[index].unsqueeze(0)
        selected = context["proposal_internal_knots"][0][mask[0]]
        packed = batch.teacher_internal_knots[index][batch.teacher_internal_knot_mask[index]]
        if selected.numel() != packed.numel():
            raise RuntimeError(f"row {index}: selected and packed knot counts differ")
        knot_difference = float((selected.double() - packed).abs().max()) if selected.numel() else 0.0
        cached_mse = float(batch.teacher_fit_mse[index])
        cpu_mse = _fit(context["proposal_params"][0], point, selected,
                       degree=proposal_model.degree)
        row = {
            "index": index,
            "source_internal_knots": int(dataset[index]["target_internal_knot_count"]),
            "teacher_count": int(batch.teacher_count[index]),
            "cached_mse": cached_mse,
            "cpu_mse": cpu_mse,
            "cpu_minus_cached_mse": cpu_mse - cached_mse,
            "cached_pass": bool(batch.teacher_threshold_satisfied[index]),
            "cpu_pass": cpu_mse <= tolerance,
            "max_abs_selected_knot_difference": knot_difference,
        }
        if row["cached_pass"] != (cached_mse <= tolerance):
            raise RuntimeError(f"row {index}: cached pass flag disagrees with cached MSE")
        if joint_model is not None:
            joint_context = joint_model.encode_candidates(
                point.unsqueeze(0).to(dtype=next(joint_model.parameters()).dtype),
                mse_tolerance=tolerance,
            )
            joint_knots = joint_context["proposal_internal_knots"][0]
            if not torch.allclose(joint_knots.double(),
                                  context["proposal_internal_knots"][0].double(),
                                  atol=1e-6, rtol=1e-6):
                raise RuntimeError(f"row {index}: Joint Proposal differs from frozen checkpoint")
            topk_mask = joint_model.select_mask_at_count(
                joint_context, batch.teacher_count[index].reshape(1),
            )
            free_mask = joint_model.select_mask(joint_context)
            row["joint"] = {
                "teacher_mask": _mask_diagnostics(
                    joint_model, joint_context, mask, point, tolerance=tolerance,
                ),
                "teacher_count_topk": _mask_diagnostics(
                    joint_model, joint_context, topk_mask, point, tolerance=tolerance,
                ),
                "free_mask": _mask_diagnostics(
                    joint_model, joint_context, free_mask, point, tolerance=tolerance,
                ),
                "teacher_count_topk_recall": float(
                    (topk_mask & mask).sum() / mask.sum().clamp_min(1)
                ),
                "teacher_count_topk_exact": bool(torch.equal(topk_mask, mask)),
                "beta": float(joint_context["adaptive_keep_threshold"][0]),
                "keep_mass": float(joint_context["one_shot_probability_mass"][0]),
            }
        rows.append(row)
        print(f"[{len(rows)}/{len(indices)}] row={index} cached={cached_mse:.6e} "
              f"CPU={cpu_mse:.6e} delta={cpu_mse-cached_mse:+.3e}", flush=True)

    return {
        "scope": "fixed_joint_training_rows_only_not_heldout_generalization",
        "proposal_checkpoint": str(args.proposal_checkpoint.resolve()),
        "teacher_cache": str(args.teacher_cache.resolve()),
        "joint_checkpoint": (
            str(args.joint_checkpoint.resolve()) if args.joint_checkpoint else None
        ),
        "joint_checkpoint_recorded_epoch": (
            int(joint_payload["epoch"]) if joint_payload is not None else None
        ),
        "joint_checkpoint_planned_epochs": (
            int(joint_payload["training_config"]["epochs"])
            if joint_payload is not None else None
        ),
        "joint_checkpoint_is_final_epoch": (
            int(joint_payload["epoch"]) >= int(joint_payload["training_config"]["epochs"])
            if joint_payload is not None else None
        ),
        "proposal_fingerprint": teacher_config.proposal_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "teacher_strategy": strategy,
        "indices": indices,
        "summary": _summarize(rows, tolerance=tolerance),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {args.output_json}")
    result = audit(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"Saved {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
