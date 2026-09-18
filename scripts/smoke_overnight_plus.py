"""Paired CPU training sanity check on two repeated toy curves, not a benchmark.

The baseline and enhanced runs share initialization, per-step RNG seeds, data,
optimizer and all original objective settings. Every step is retained, including
temporary regressions. The JSON is created exclusively and never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from smoke_v16_subset_learning import _RecordingLoss, _snapshot  # noqa: E402
from spline_fitting.losses.v16_subset_loss import select_best_subset  # noqa: E402
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork  # noqa: E402


ENHANCEMENTS = {
    "teacher_refinement_steps": 1,
    "teacher_refinement_candidates": 2,
    "boundary_ranking_weight": 0.5,
    "boundary_ranking_candidates": 4,
}


def _state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _finite_scalar(value):
    if isinstance(value, torch.Tensor):
        value = value.detach()
    value = float(value)
    return value if math.isfinite(value) else str(value)


def _safe_snapshot(model, objective, points):
    try:
        return _snapshot(model, objective, points)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


@torch.no_grad()
def _fixed_context_teacher_audit(model, options, points):
    """Compare before/after teachers with identical, frozen model and geometry."""
    objective = _RecordingLoss(**options, **ENHANCEMENTS)
    context = model.encode_candidates(points, mse_tolerance=objective.mse_tolerance)
    probability = context["keep_logits"].sigmoid()
    tolerance = torch.full((points.shape[0],), objective.mse_tolerance, dtype=torch.float64)
    minimum = model.min_selected_knots
    before, before_mse, _, _ = objective._minimum_feasible_ranked_prefix(
        model, context, probability, points, model.degree, tolerance, minimum,
    )
    deployed = model.select_mask(context)
    masks = [before, deployed]
    errors = [before_mse, objective._decode_mse(model, context, deployed, points, model.degree)]
    for index, trial in enumerate(objective._counterfactual_masks(
        deployed, probability, minimum,
        constrain=lambda mask: model.constrain_selection_mask(context, mask),
    )):
        if index >= objective.counterfactual_edits:
            break
        masks.append(trial)
        errors.append(objective._decode_mse(model, context, trial, points, model.degree))
    pool_mse = torch.stack(errors)
    selected, indices = select_best_subset(torch.stack(masks), pool_mse, tolerance)
    columns = torch.arange(points.shape[0])
    any_feasible = (pool_mse <= tolerance).any(0)
    before = torch.where(any_feasible[:, None], selected, before)
    before_mse = torch.where(any_feasible, pool_mse[indices, columns], before_mse)
    after, after_mse, diagnostics = objective._refine_teacher(
        model, context, before, before_mse, probability, points,
        model.degree, tolerance, minimum,
    )
    before_k, after_k = before.sum(-1), after.sum(-1)
    before_feasible, after_feasible = before_mse <= tolerance, after_mse <= tolerance
    unchanged = (before == after).all(-1) & (before_mse == after_mse)
    non_regression = unchanged | (
        after_feasible & (
            ~before_feasible | (after_k < before_k)
            | ((after_k == before_k) & (after_mse <= before_mse))
        )
    )
    return {
        "comparison": "same frozen context; feasibility first, then K, then MSE",
        "before": {"per_curve_k": before_k.tolist(), "per_curve_mse": before_mse.tolist(),
                   "per_curve_feasible": before_feasible.tolist(), "mask": before.tolist()},
        "after": {"per_curve_k": after_k.tolist(), "per_curve_mse": after_mse.tolist(),
                  "per_curve_feasible": after_feasible.tolist(), "mask": after.tolist()},
        "per_curve_non_regression": non_regression.tolist(),
        "all_non_regressing": bool(non_regression.all()),
        "diagnostics": {name: _finite_scalar(value) for name, value in diagnostics.items()},
    }


def _run(name, template, points, args, options):
    started = time.perf_counter()
    model = copy.deepcopy(template)
    objective = _RecordingLoss(**options, **(ENHANCEMENTS if name == "enhanced" else {}))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    initial = _safe_snapshot(model, objective, points)
    history = []
    for step in range(args.steps):
        # Common random numbers at every step, even after the model trajectories
        # diverge; neither method receives a more favorable random stream.
        rng_seed = args.seed + 1 + step
        torch.manual_seed(rng_seed)
        stage = "proposal" if step < args.proposal_steps else "joint"
        record = {"step": step, "stage": stage, "rng_seed": rng_seed}
        optimizer.zero_grad(set_to_none=True)
        try:
            loss, metrics = objective(model, points, stage=stage)
            record.update({key: _finite_scalar(value) for key, value in metrics.items()})
            record["production_before_update"] = _safe_snapshot(model, objective, points)
            loss.backward()
            gradients_finite = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            gradients_finite = gradients_finite and bool(torch.isfinite(norm))
            record["gradients_finite"] = gradients_finite
            record["gradient_norm_before_clipping"] = _finite_scalar(norm)
            record["update_applied"] = gradients_finite and bool(torch.isfinite(loss))
            if record["update_applied"]:
                optimizer.step()
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            record["update_applied"] = False
        history.append(record)
        if step % 10 == 0 or step in (args.proposal_steps, args.steps - 1):
            print(
                f"{name} step={step} stage={stage} loss={record.get('loss')} "
                f"MSE={record.get('deployment_mse')} K={record.get('keep_count')} "
                f"teacherF1={record.get('teacher_keep_f1')} "
                f"finite={record.get('gradients_finite')} error={record.get('error')}",
                flush=True,
            )
    final = _safe_snapshot(model, objective, points)
    optimizer.zero_grad(set_to_none=True)
    try:
        torch.manual_seed(args.seed + args.steps + 1)
        final_loss, _ = objective(model, points, stage="joint")
        final_loss.backward()
        final_gradients = {
            "loss": _finite_scalar(final_loss),
            "all_gradients_finite": all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ),
            "parameters_with_gradients": sum(parameter.grad is not None for parameter in model.parameters()),
        }
    except Exception as error:
        final_gradients = {"error": f"{type(error).__name__}: {error}", "all_gradients_finite": False}
    try:
        teacher_audit = _fixed_context_teacher_audit(model, options, points)
    except Exception as error:
        teacher_audit = {"error": f"{type(error).__name__}: {error}", "all_non_regressing": False}
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "initial_state_sha256": _state_digest(template),
        "objective_enhancements": ENHANCEMENTS if name == "enhanced" else {},
        "initial_deployment": initial,
        "final_deployment": final,
        "final_gradient_check": final_gradients,
        "fixed_context_teacher_audit": teacher_audit,
        "diagnostics": {
            "failed_or_skipped_update_steps": [row["step"] for row in history if not row["update_applied"]],
            "production_infeasible_steps": [
                row["step"] for row in history
                if row.get("production_before_update", {}).get("pass_rate", 0.0) < 1.0
            ],
            "joint_teacher_false_remove_rates": [
                row.get("teacher_false_remove_rate") for row in history if row["stage"] == "joint"
            ],
        },
        "history_before_each_update": history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/tmp/_smoke_overnight_plus.json"))
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--proposal-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--mse-tolerance", type=float, default=2.5e-5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if not 0 <= args.proposal_steps < args.steps:
        parser.error("steps must exceed non-negative proposal-steps")
    if any(not math.isfinite(value) or value <= 0 for value in (args.learning_rate, args.mse_tolerance)):
        parser.error("learning-rate and mse-tolerance must be finite and positive")
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    t = torch.linspace(0, 1, 32)
    points = torch.stack([
        torch.stack([t, 0.2 * torch.sin(8 * t)], -1),
        torch.stack([t, 0.2 * torch.cos(6 * t)], -1),
    ])
    template = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1, mse_tolerance=args.mse_tolerance,
        relocation_blend=0.0, one_shot_selection_policy="mass_topk",
        one_shot_adaptive_threshold=True, one_shot_safety_sigma=0.25,
        one_shot_safety_knots=1, one_shot_coverage_bins=2,
    )
    options = {
        "mse_tolerance": args.mse_tolerance, "policy_samples": 3,
        "counterfactual_edits": 4, "policy_weight": 0.25, "distillation_weight": 3.0,
    }
    print("Fixed seed paired toy-training sanity check; no held-out/generalization claim.", flush=True)
    runs = {name: _run(name, template, points, args, options) for name in ("baseline", "enhanced")}
    result = {
        "role": "paired_cpu_toy_training_sanity_only",
        "limitation": "Two repeatedly trained analytic curves; not held-out, not a generalization or real-data comparison. Every step and any regression retained.",
        "elapsed_seconds": time.perf_counter() - started,
        "configuration": {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "device": "cpu", "torch_num_threads": 1, "torch_version": str(torch.__version__),
            "optimizer": "Adam", "gradient_clip_norm": 10.0, "model": template.get_config(),
            "common_objective_options": options, "enhancements": ENHANCEMENTS,
            "curves": ["(t, 0.2*sin(8*t))", "(t, 0.2*cos(6*t))"],
            "num_points": 32, "normalization": "none",
            "rng_rule": "same model initialization seed; reseed both runs to seed+1+step before every training step",
            "source_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__), ROOT / "scripts/smoke_v16_subset_learning.py",
                             ROOT / "src/spline_fitting/models/v16_network.py",
                             ROOT / "src/spline_fitting/losses/v16_subset_loss.py")
            },
        },
        "same_initialization_verified": runs["baseline"]["initial_deployment"] == runs["enhanced"]["initial_deployment"],
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "saved": str(args.output), "elapsed_seconds": result["elapsed_seconds"],
        "same_initialization_verified": result["same_initialization_verified"],
        "runs": {name: {key: run[key] for key in (
            "final_deployment", "final_gradient_check", "fixed_context_teacher_audit", "diagnostics",
        )} for name, run in runs.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
