"""Reproducible CPU toy optimization, not a generalization benchmark.

Fits the same two analytic training curves throughout. The JSON retains every
step, including temporary feasibility drops, and a fresh final deployed refit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.losses.v16_subset_loss import V16SubsetLoss  # noqa: E402
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork  # noqa: E402
from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points  # noqa: E402


class _RecordingLoss(V16SubsetLoss):
    def _decode_mse(self, model, context, mask, points, degree):
        self.last_decoded_mask = mask.detach().clone()
        return super()._decode_mse(model, context, mask, points, degree)


def _snapshot(model, objective, points):
    with torch.no_grad():
        context = model.encode_candidates(points)
        mask = model.select_mask(context)
        output = model.decode_subset(context, mask)
        # Evaluate the actual production solver, independently of the training
        # least-squares guard rows. These snapshots contain no solver jitter.
        fits = [
            refit_bspline_control_points(
                output["params"][row].to(device="cpu", dtype=torch.float64),
                points[row].to(device="cpu", dtype=torch.float64),
                output["internal_knots"][row, mask[row]].to(device="cpu", dtype=torch.float64),
                degree=model.degree, smoothness_weight=0.0, control_ridge=0.0,
                interpolate_endpoints=True,
            )
            for row in range(points.shape[0])
        ]
        mse = torch.stack([fit.fit_mse for fit in fits])
        probability = context["keep_probabilities"]
        return {
            "solver": "production refit_bspline_control_points, CPU float64 gelsd, endpoint-constrained, no regularization or jitter",
            "per_curve_mse": mse.tolist(),
            "mean_mse": float(mse.mean()),
            "pass_rate": float((mse <= objective.mse_tolerance).double().mean()),
            "mean_k": float(mask.sum(-1).double().mean()),
            "per_curve_k": mask.sum(-1).tolist(),
            "mask": mask.tolist(),
            "keep_probability_mean": float(probability.mean()),
            "keep_probability_min": float(probability.min()),
            "keep_probability_max": float(probability.max()),
            "keep_probabilities": probability.tolist(),
            "adaptive_beta": context["adaptive_keep_threshold"].tolist(),
            "probability_mass": context["one_shot_probability_mass"].tolist(),
            "requested_count_score": context["one_shot_requested_count_score"].tolist(),
            "selected_knots": [output["internal_knots"][row, mask[row]].tolist()
                               for row in range(points.shape[0])],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tmp/_smoke_v16_subset_learning.json"),
    )
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing output JSON.")
    parser.add_argument("--steps", type=int, default=181)
    parser.add_argument("--proposal-steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--mse-tolerance", type=float, default=2.5e-5)
    parser.add_argument("--policy-weight", type=float, default=0.25)
    parser.add_argument("--distillation-weight", type=float, default=3.0)
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; use --overwrite to replace it")
    if not 0 <= args.proposal_steps < args.steps:
        parser.error("steps must exceed non-negative proposal-steps")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    t = torch.linspace(0, 1, 32)
    points = torch.stack([
        torch.stack([t, 0.2 * torch.sin(8 * t)], -1),
        torch.stack([t, 0.2 * torch.cos(6 * t)], -1),
    ])
    model = V16CandidateSelectionNetwork(
        hidden_dim=16, encoder_layers=1, max_internal_knots=6,
        attention_heads=2, selector_layers=1, mse_tolerance=args.mse_tolerance,
        relocation_blend=0.0, one_shot_selection_policy="mass_topk",
        one_shot_adaptive_threshold=True, one_shot_safety_sigma=0.25,
        one_shot_safety_knots=1, one_shot_coverage_bins=2,
    )
    objective = _RecordingLoss(
        mse_tolerance=args.mse_tolerance, policy_samples=3, counterfactual_edits=4,
        policy_weight=args.policy_weight, distillation_weight=args.distillation_weight,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    initial = _snapshot(model, objective, points)
    history = []
    print("Toy training curves only; MSE threshold=" + str(args.mse_tolerance), flush=True)
    for step in range(args.steps):
        stage = "proposal" if step < args.proposal_steps else "joint"
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = objective(model, points, stage=stage)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("Nonfinite toy optimization gradient")
        # Record before the update: fit, count, mask and probabilities therefore
        # refer to the exact same model state and objective evaluation.
        probability = model.encode_candidates(points)["keep_probabilities"].detach()
        record = {
            "step": step, "stage": stage,
            **{name: float(value) for name, value in metrics.items()},
            "gradient_norm_before_clipping": float(gradient_norm),
            "keep_probability_mean": float(probability.mean()),
            "keep_probability_min": float(probability.min()),
            "keep_probability_max": float(probability.max()),
            "online_best_mask": objective.last_decoded_mask.tolist() if stage == "joint" else None,
        }
        history.append(record)
        optimizer.step()
        if step % 20 == 0 or step in (args.proposal_steps, args.steps - 1):
            print(
                f"step={step} stage={stage} K={record['keep_count']:.2f} "
                f"MSE={record['deployment_mse']:.4e} pass={record['deployment_pass_rate']:.3f} "
                f"p={record['keep_probability_mean']:.3f} "
                f"policy={record['policy_loss']:.3f} BCE={record['mask_distillation_loss']:.3f}",
                flush=True,
            )
    final = _snapshot(model, objective, points)
    joint_history = [row for row in history if row["stage"] == "joint"]
    files = [Path(__file__), ROOT / "src/spline_fitting/models/v16_network.py",
             ROOT / "src/spline_fitting/losses/v16_subset_loss.py",
             ROOT / "src/spline_fitting/evaluation/bspline_inference.py"]
    result = {
        "role": "toy_training_curve_fit_only",
        "limitation": "Two repeatedly optimized analytic curves; no held-out/generalization or real-data claim.",
        "configuration": {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "device": "cpu", "torch_num_threads": 1, "torch_version": str(torch.__version__),
            "optimizer": "Adam", "gradient_clip_norm": 10.0,
            "num_curves": 2, "num_points": 32,
            "curves": ["(t, 0.2*sin(8*t))", "(t, 0.2*cos(6*t))"],
            "parameter_samples": "32 uniform t values in [0,1]",
            "normalization": "none (analytic coordinates as specified)",
            "policy_samples": 3, "counterfactual_edits": 4,
            "history_solver": "differentiable standard B-spline fit, CPU float64 gels, solver_jitter=1e-10",
            "model": model.get_config(),
            "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in files},
        },
        "initial_deployment": initial,
        "final_deployment": final,
        "transient_diagnostics": {
            "minimum_joint_pass_rate": min(row["deployment_pass_rate"] for row in joint_history),
            "minimum_joint_mean_k": min(row["keep_count"] for row in joint_history),
            "maximum_joint_mean_mse": max(row["deployment_mse"] for row in joint_history),
        },
        "history_before_each_update": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"final_deployment": final, "saved": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
