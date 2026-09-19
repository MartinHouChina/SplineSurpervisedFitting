"""Train v16 dense feasibility, then online counterfactual subset selection."""
# Imports below the local src bootstrap are intentional for direct script use.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import (
    V16_ADAPTIVE_SELECTION_REVISION,
    V16_CERTIFIED_SYNTHETIC_CONTRACT,
    V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
    V16_FORMAL_PASS_RATE,
    V16_FORMAL_SYNTHETIC_COUNT_MAE_MAX,
    V16_FORMAL_SYNTHETIC_KNOT_F1_MIN,
    V16_FORMAL_SYNTHETIC_MATCHED_MAE_MAX,
    V16_MINIMALITY_AUDIT_POINTS,
    V16_MINIMALITY_MARGIN,
    V16_SIMPLIFICATION_CONTRACT,
    assess_v16_checkpoint,
    build_model_from_checkpoint,
)
from spline_fitting.data.v16_mixed import EPOCH_SEED_STRIDE, MixedTrainingCurves, ValidationCurves, load_real_sources
from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
from spline_fitting.evaluation.knot_diagnostics import (
    match_internal_knots,
    warp_internal_knots_to_parameterization,
)
from spline_fitting.losses.v16_subset_loss import V16SubsetLoss
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork


# Missing entries in historical checkpoints mean these opt-in features were off.
ENHANCED_TRAINING_DEFAULTS = {
    "teacher_refinement_steps": 0,
    "teacher_refinement_candidates": 2,
    "boundary_ranking_weight": 0.0,
    "boundary_ranking_candidates": 4,
    "joint_proposal_lr_scale": 1.0,
    "joint_decoder_lr_scale": 1.0,
    "joint_final_lr_ratio": 1.0,
    "warm_start_checkpoint": None,
    "parameter_trust_enabled": False,
    "parameter_trust_initial": 0.25,
    "parameter_counterfactual_weight": 0.0,
    "teacher_geometry_candidates": 0,
    "local_fit_weight": 0.0,
    "proposal_ordered_weight": 0.0,
    "simplification_controller": "worst_source",
    "feasible_objective": False,
    "feasible_fit_margin": 0.8,
    "feasible_fit_weight": 0.02,
    "teacher_greedy_steps": 0,
    "teacher_greedy_max_curves": 2,
    "teacher_geometry_distillation_weight": 0.0,
    "count_reserve_alignment": False,
    "synthetic_simple_fraction": 0.0,
    "synthetic_shape_fraction": 0.0,
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=60, help="Total proposal plus joint epochs")
    p.add_argument(
        "--proposal-epochs", type=int, default=20,
        help="Last proposal-stage epoch (total, not extra epochs when resuming)",
    )
    p.add_argument("--train-size", type=int, default=2400, help="Mixture draws per epoch")
    p.add_argument("--val-size", type=int, default=500, help="Synthetic validation curves")
    p.add_argument("--real-val-size", type=int, default=100, help="Maximum val curves per real source")
    p.add_argument("--real-manifest", action="append", type=Path, default=[])
    p.add_argument("--real-fraction", type=float, default=0.5)
    p.add_argument("--synthetic-simple-fraction", type=float, default=0.0,
                   help="Training-only fraction of synthetic draws biased to low source K")
    p.add_argument("--synthetic-shape-fraction", type=float, default=0.0,
                   help="Training-only procedural shapes; no fabricated true-knot labels")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-points", type=int, default=192)
    p.add_argument("--point-dim", type=int, choices=(2, 3), default=2)
    p.add_argument("--min-control-points", type=int, default=8)
    p.add_argument(
        "--max-control-points", type=int, default=24,
        help="Maximum synthetic source control points; this does not change network Kc",
    )
    p.add_argument(
        "--candidate-knots", type=int, default=None,
        help=("Internal candidate capacity Kc (default: 64 internal knots). "
              "Changing it requires a new experiment"),
    )
    p.add_argument(
        "--full-knot-vector-size", type=int,
        help=("Alternative cubic-spline notation including the eight clamped "
              "endpoint entries; for example 64 means Kc=56 internal knots"),
    )
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--encoder-layers", type=int, default=3)
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--selector-layers", type=int, default=2)
    p.add_argument("--noise-std", type=float, default=0.001)
    p.add_argument(
        "--certified-minimal-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Require every synthetic source knot set to be threshold-minimal "
              "within all subsets of its original knots"),
    )
    p.add_argument("--minimality-margin", type=float, default=V16_MINIMALITY_MARGIN)
    p.add_argument("--minimality-max-attempts", type=int, default=16)
    p.add_argument(
        "--minimality-audit-points", type=int, default=V16_MINIMALITY_AUDIT_POINTS,
        help="Uniform clean-curve samples used by the source minimality certificate",
    )
    p.add_argument("--oscillation-amplitude", type=float, default=0.3)
    p.add_argument("--mse-tolerance", type=float, default=2.5e-5)
    p.add_argument("--knot-match-tolerance", type=float, default=0.01)
    p.add_argument("--tolerance-factor-min", type=float, default=0.75)
    p.add_argument("--tolerance-factor-max", type=float, default=1.0)
    p.add_argument(
        "--proposal-pass-target", type=float, default=V16_FORMAL_PASS_RATE,
        help=(f"Worst-source proposal gate; values below {V16_FORMAL_PASS_RATE:g} "
              "are diagnostic/ablation settings"),
    )
    p.add_argument(
        "--deployment-pass-target", type=float, default=V16_FORMAL_PASS_RATE,
        help=(f"Worst-source checkpoint-selection target; values below "
              f"{V16_FORMAL_PASS_RATE:g} are not engineering-qualified"),
    )
    p.add_argument("--allow-infeasible-proposals", action="store_true",
                   help="Explicit ablation: continue even when the proposal gate fails")
    p.add_argument("--policy-samples", type=int, default=4)
    p.add_argument("--counterfactual-edits", type=int, default=4)
    p.add_argument(
        "--teacher-prefix-search-steps", type=int, default=7,
        help=("Training-only ranked-prefix feasibility search depth; seven "
              "steps resolve a Kc=96 count boundary without deployment search"),
    )
    p.add_argument("--teacher-refinement-steps", type=int, default=0,
                   help="Opt-in training-only greedy deletion rounds after the online teacher")
    p.add_argument("--teacher-refinement-candidates", type=int, default=2,
                   help="Maximum deletion candidates tested per teacher refinement round")
    p.add_argument("--boundary-ranking-weight", type=float, default=0.0,
                   help="Opt-in online teacher keep/remove boundary ranking loss weight")
    p.add_argument("--boundary-ranking-candidates", type=int, default=4)
    p.add_argument("--parameter-trust-enabled", action="store_true",
                   help="Learn bounded chord/proposal parameter update gates (opt-in)")
    p.add_argument("--parameter-trust-initial", type=float, default=0.25)
    p.add_argument("--parameter-counterfactual-weight", type=float, default=0.0,
                   help="Training-only same-mask chord comparison; no inference search")
    p.add_argument("--teacher-geometry-candidates", type=int, default=0,
                   help="Bounded student-ranking-independent teacher proposals per joint step")
    p.add_argument("--local-fit-weight", type=float, default=0.0,
                   help="Auxiliary local/endpoint residual supervision")
    p.add_argument("--proposal-ordered-weight", type=float, default=0.0,
                   help="Additional ordered one-to-one candidate supervision")
    p.add_argument("--feasible-objective", action="store_true",
                   help="Opt-in threshold-aware fit loss with weak reward inside the safe region")
    p.add_argument("--feasible-fit-margin", type=float, default=0.8)
    p.add_argument("--feasible-fit-weight", type=float, default=0.02)
    p.add_argument("--teacher-greedy-steps", type=int, default=0,
                   help="Training-only fixed-geometry full single-deletion search rounds")
    p.add_argument("--teacher-greedy-max-curves", type=int, default=2,
                   help="Maximum curves per batch receiving full deletion search")
    p.add_argument("--teacher-geometry-distillation-weight", type=float, default=0.0)
    p.add_argument("--count-reserve-alignment", action="store_true",
                   help="Align mask-probability mass targets with deployed count/safety reserve")
    p.add_argument("--count-weight", type=float, default=2.0)
    p.add_argument("--supervised-count-weight", type=float, default=1.0)
    p.add_argument("--supervised-over-count-weight", type=float, default=1.0)
    p.add_argument("--complexity-weight", type=float, default=0.05)
    p.add_argument("--true-parameter-weight", type=float, default=0.1)
    p.add_argument("--proposal-knot-coverage-weight", type=float, default=1.0)
    p.add_argument("--selected-knot-position-weight", type=float, default=1.0)
    p.add_argument("--knot-position-beta", type=float, default=0.01)
    p.add_argument(
        "--one-shot-selection-policy", choices=("threshold", "mass_topk"),
        default="mass_topk",
        help="Use adaptive probability mass for cardinality; threshold is a legacy ablation",
    )
    p.add_argument("--one-shot-safety-sigma", type=float, default=0.25)
    p.add_argument("--one-shot-safety-knots", type=int, default=2)
    p.add_argument("--final-safety-sigma", type=float, default=0.05)
    p.add_argument("--final-safety-knots", type=int, default=0)
    p.add_argument("--safety-anneal-epochs", type=int, default=10)
    p.add_argument("--one-shot-coverage-bins", type=int, default=4)
    p.add_argument("--min-selected-knots", type=int, default=4)
    p.add_argument(
        "--relocation-blend", type=float, default=0.0,
        help="Initial uniform-rank relocation blend; zero starts from identity",
    )
    p.add_argument(
        "--complexity-ramp-epochs", type=int, default=10,
        help=("Safe validation epochs needed to reach unit complexity pressure; "
              "the feedback controller may continue up to complexity-max-scale"),
    )
    p.add_argument(
        "--complexity-max-scale", type=float, default=4.0,
        help="Maximum validation-controlled multiplier on the complexity loss",
    )
    p.add_argument("--complexity-pass-margin", type=float, default=0.02)
    p.add_argument("--simplification-controller", choices=("worst_source", "per_curve"),
                   default="worst_source",
                   help="per_curve ramps independently; only feasible training curves receive complexity pressure")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--joint-lr", type=float, default=5e-5)
    p.add_argument("--joint-proposal-lr-scale", type=float, default=1.0,
                   help="Joint-stage encoder/parameter/proposal LR multiplier")
    p.add_argument("--joint-decoder-lr-scale", type=float, default=1.0,
                   help="Joint-stage subset decoder LR multiplier; selector stays at joint-lr")
    p.add_argument("--joint-final-lr-ratio", type=float, default=1.0,
                   help="Cosine joint LR end/start ratio in (0,1]; 1 preserves constant LR")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-seed", type=int, default=1_000_000)
    p.add_argument("--resample-train-each-epoch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--torch-num-threads", type=int, default=4)
    p.add_argument("--log-every-batches", type=int, default=10)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument(
        "--init-checkpoint", type=Path,
        help=("Copy compatible encoder, ParameterHead and candidate-proposal tensors; "
              "not optimizer/selector/subset-decoder resume"),
    )
    p.add_argument(
        "--resume", type=Path,
        help=("Resume .last.pt with the original data/targets/output; only total epochs, "
              "runtime options, and an unfinished proposal-stage end may be extended"),
    )
    p.add_argument(
        "--warm-start-checkpoint", type=Path,
        help=("Strictly copy all compatible native-v16 weights into a fresh run; "
              "reset optimizer/history/controllers and recheck the proposal gate"),
    )
    p.add_argument("--output", type=Path, default=Path("outputs/checkpoints/candidate_selection_v16.pt"))
    return p


def validate_args(args):
    if args.full_knot_vector_size is not None:
        if args.candidate_knots is not None:
            raise ValueError(
                "use either --candidate-knots or --full-knot-vector-size, not both"
            )
        # Cubic open clamping contributes four zeros and four ones.
        args.candidate_knots = args.full_knot_vector_size - 8
    elif args.candidate_knots is None:
        args.candidate_knots = 64
    if not 1 <= args.proposal_epochs < args.epochs:
        raise ValueError("require 1 <= proposal-epochs < epochs")
    for key in ("train_size", "val_size", "real_val_size", "batch_size", "hidden_dim",
                "encoder_layers", "attention_heads", "selector_layers", "torch_num_threads", "log_every_batches"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if args.num_workers < 0 or args.seed < 0 or args.val_seed < 0:
        raise ValueError("workers and seeds must be nonnegative")
    if args.policy_samples < 2 or args.counterfactual_edits < 0:
        raise ValueError("policy-samples must be >= 2 and counterfactual-edits >= 0")
    for key in (
        "one_shot_safety_knots", "one_shot_coverage_bins",
        "min_selected_knots", "complexity_ramp_epochs", "final_safety_knots",
        "safety_anneal_epochs", "teacher_prefix_search_steps",
        "minimality_max_attempts",
        "teacher_refinement_steps",
        "teacher_geometry_candidates",
        "teacher_greedy_steps",
    ):
        value = getattr(args, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    for key in ("teacher_refinement_candidates", "boundary_ranking_candidates",
                "teacher_greedy_max_curves"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if not 4 <= args.min_control_points <= args.max_control_points:
        raise ValueError("control-point range must satisfy 4 <= min <= max")
    if not 1 <= args.candidate_knots <= args.num_points - 4:
        raise ValueError("candidate-knots must be between 1 and num-points - 4")
    if (
        args.certified_minimal_source
        and args.candidate_knots < args.max_control_points - 4
    ):
        raise ValueError(
            "certified training requires candidate-knots >= the maximum "
            "synthetic internal-knot count (max-control-points - 4)"
        )
    if args.min_selected_knots > args.candidate_knots:
        raise ValueError("min-selected-knots cannot exceed candidate-knots")
    if (
        args.certified_minimal_source
        and args.min_selected_knots > args.min_control_points - 4
    ):
        raise ValueError(
            "certified training requires min-selected-knots <= the minimum "
            "synthetic internal-knot count (min-control-points - 4)"
        )
    if args.one_shot_coverage_bins > args.candidate_knots:
        raise ValueError("one-shot-coverage-bins cannot exceed candidate-knots")
    if args.max_control_points > args.num_points:
        raise ValueError("max-control-points must not exceed num-points")
    for key in (
        "mse_tolerance", "knot_match_tolerance", "lr", "joint_lr",
        "grad_clip", "tolerance_factor_min", "tolerance_factor_max",
        "joint_proposal_lr_scale", "joint_decoder_lr_scale", "joint_final_lr_ratio",
    ):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in (
        "noise_std", "weight_decay", "one_shot_safety_sigma",
        "final_safety_sigma", "complexity_pass_margin", "minimality_margin",
        "count_weight", "supervised_count_weight",
        "supervised_over_count_weight", "complexity_weight",
        "true_parameter_weight", "proposal_knot_coverage_weight",
        "selected_knot_position_weight", "knot_position_beta",
        "complexity_max_scale",
        "boundary_ranking_weight",
        "parameter_counterfactual_weight", "local_fit_weight", "proposal_ordered_weight",
        "teacher_geometry_distillation_weight", "feasible_fit_weight",
    ):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if args.joint_final_lr_ratio > 1:
        raise ValueError("joint-final-lr-ratio must lie in (0,1]")
    if not math.isfinite(args.feasible_fit_margin) or not 0 < args.feasible_fit_margin <= 1:
        raise ValueError("feasible-fit-margin must lie in (0,1]")
    if args.feasible_fit_weight > 1:
        raise ValueError("feasible-fit-weight must lie in [0,1]")
    if args.teacher_geometry_distillation_weight and not args.teacher_greedy_steps:
        raise ValueError("teacher-geometry-distillation-weight requires teacher-greedy-steps > 0")
    if args.simplification_controller == "per_curve" and not args.feasible_objective:
        raise ValueError("per_curve controller requires --feasible-objective")
    if args.minimality_max_attempts < 1:
        raise ValueError("minimality-max-attempts must be positive")
    if args.minimality_audit_points not in (0,) and args.minimality_audit_points < 2:
        raise ValueError("minimality-audit-points must be 0 or at least 2")
    if not math.isfinite(args.oscillation_amplitude) or args.oscillation_amplitude <= 0:
        raise ValueError("oscillation-amplitude must be finite and positive")
    if not math.isfinite(args.parameter_trust_initial) or not 0 < args.parameter_trust_initial < 1:
        raise ValueError("parameter-trust-initial must be strictly between zero and one")
    if args.safety_anneal_epochs < 1:
        raise ValueError("safety-anneal-epochs must be positive")
    if args.teacher_prefix_search_steps < 1:
        raise ValueError("teacher-prefix-search-steps must be positive")
    if args.complexity_max_scale <= 0:
        raise ValueError("complexity-max-scale must be positive")
    if args.knot_position_beta <= 0:
        raise ValueError("knot-position-beta must be positive")
    if args.final_safety_sigma > args.one_shot_safety_sigma:
        raise ValueError("final-safety-sigma cannot exceed one-shot-safety-sigma")
    if args.final_safety_knots > args.one_shot_safety_knots:
        raise ValueError("final-safety-knots cannot exceed one-shot-safety-knots")
    for key in ("real_fraction", "proposal_pass_target", "deployment_pass_target",
                "synthetic_simple_fraction", "synthetic_shape_fraction"):
        if not math.isfinite(getattr(args, key)) or not 0 <= getattr(args, key) <= 1:
            raise ValueError(f"{key} must lie in [0,1]")
    if args.synthetic_simple_fraction + args.synthetic_shape_fraction > 1:
        raise ValueError("synthetic-simple-fraction + synthetic-shape-fraction must be <= 1")
    if args.synthetic_simple_fraction and not args.certified_minimal_source:
        raise ValueError("synthetic-simple-fraction requires certified-minimal-source")
    if args.synthetic_simple_fraction and max(4, args.min_control_points - 4) > min(8, args.max_control_points - 4):
        raise ValueError("synthetic-simple-fraction requires a source range overlapping K=4..8")
    if args.tolerance_factor_min > args.tolerance_factor_max:
        raise ValueError("tolerance-factor-min must be <= tolerance-factor-max")
    if not math.isfinite(args.relocation_blend) or not 0 <= args.relocation_blend <= 1:
        raise ValueError("relocation-blend must lie in [0,1]")
    if args.deployment_pass_target + args.complexity_pass_margin > 1:
        raise ValueError("deployment-pass-target + complexity-pass-margin cannot exceed one")
    if sum(bool(value) for value in (
        args.resume, args.init_checkpoint, args.warm_start_checkpoint,
    )) > 1:
        raise ValueError("resume, init-checkpoint and warm-start-checkpoint are mutually exclusive")
    if args.train_size >= EPOCH_SEED_STRIDE:
        raise ValueError("train-size exceeds epoch seed stride")
    for epoch in range(args.epochs if args.resample_train_each_epoch else 1):
        first = args.seed + epoch * EPOCH_SEED_STRIDE
        if first < args.val_seed + args.val_size and args.val_seed < first + args.train_size:
            raise ValueError("synthetic train/validation seed ranges overlap")


def progress(current, total, title, extra="", every=10):
    if current == total or current == 1 or current % every == 0:
        filled = int(24 * current / max(total, 1))
        print(f"{title} [{'=' * filled}{'.' * (24-filled)}] {current}/{total} {extra}", flush=True)


def selection_safety(args, scale):
    """Interpolate the validation-controlled one-shot safety reserve."""
    if not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError("selection safety scale must lie in [0,1]")
    sigma = args.final_safety_sigma + scale * (
        args.one_shot_safety_sigma - args.final_safety_sigma
    )
    knot_span = args.one_shot_safety_knots - args.final_safety_knots
    knots = args.final_safety_knots + math.ceil(scale * knot_span - 1e-12)
    return float(sigma), int(knots)


def simplification_is_ready(
    args, *, epoch, stage, applied_safety_scale, applied_complexity_scale,
):
    """Require the final deployment reserve and mature complexity curriculum."""
    return bool(
        stage == "joint"
        and epoch - args.proposal_epochs
        >= max(args.complexity_ramp_epochs, args.safety_anneal_epochs)
        and applied_safety_scale <= 1e-12
        and applied_complexity_scale >= 1.0
    )


def update_simplification_controller(
    args, *, pass_rate, complexity_scale, safety_scale,
):
    """Advance the curriculum; per-curve mode leaves feasibility gating to loss.

    Worst-source validation still selects checkpoints in either mode. In the
    opt-in mode, an unseen validation domain must not switch off complexity
    gradients for every already-feasible training curve.
    """
    for name, value in {
        "pass_rate": pass_rate,
        "complexity_scale": complexity_scale,
        "safety_scale": safety_scale,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0 <= pass_rate <= 1 or not 0 <= safety_scale <= 1:
        raise ValueError("pass rate and safety scale must lie in [0,1]")
    if not 0 <= complexity_scale <= args.complexity_max_scale:
        raise ValueError("complexity scale lies outside its configured range")

    safe_target = min(
        1.0, args.deployment_pass_target + args.complexity_pass_margin
    )
    mode = getattr(args, "simplification_controller", "worst_source")
    if mode not in {"worst_source", "per_curve"}:
        raise ValueError("unknown simplification controller")
    if mode == "per_curve":
        speed = 1.0
    elif pass_rate >= safe_target:
        speed = 1.0
    elif pass_rate >= args.deployment_pass_target:
        # Cautious progress in the hysteresis band prevents a permanent
        # 90--92% deadlock at the initial safety reserve.
        speed = 0.5
    else:
        speed = -2.0
    complexity_scale = min(
        args.complexity_max_scale,
        max(
            0.0,
            complexity_scale
            + speed / max(args.complexity_ramp_epochs, 1),
        ),
    )
    safety_scale = min(
        1.0,
        max(
            0.0,
            safety_scale - speed / max(args.safety_anneal_epochs, 1),
        ),
    )
    return float(complexity_scale), float(safety_scale)


def summarize(rows, tolerance):
    groups = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)
    def group(values):
        result = dict(
            n=len(values),
            dense_mse=statistics.fmean(r["dense_mse"] for r in values),
            dense_pass_rate=sum(r["dense_mse"] <= tolerance for r in values)/len(values),
            deployment_mse=statistics.fmean(r["mse"] for r in values),
            deployment_pass_rate=sum(r["mse"] <= tolerance for r in values)/len(values),
            deployment_mse_p95=float(np.quantile([r["mse"] for r in values], 0.95)),
            keep_count=statistics.fmean(r["k"] for r in values),
            keep_probability_mass=statistics.fmean(r["probability_mass"] for r in values),
            adaptive_keep_threshold=statistics.fmean(r["adaptive_threshold"] for r in values),
        )
        for field in ("proposal_parameter_trust", "subset_parameter_trust"):
            gates = [r[field] for r in values if field in r]
            if gates:
                result.update({field + "_mean": statistics.fmean(gates),
                               field + "_min": min(gates), field + "_max": max(gates)})
        supervised = [r for r in values if r.get("target_k") is not None]
        if supervised:
            errors = [r["k"] - r["target_k"] for r in supervised]
            result.update(
                supervised_count_n=len(supervised),
                target_count_mean=statistics.fmean(r["target_k"] for r in supervised),
                count_mae=statistics.fmean(abs(error) for error in errors),
                count_bias=statistics.fmean(errors),
                over_count_mean=statistics.fmean(max(error, 0) for error in errors),
                under_count_mean=statistics.fmean(max(-error, 0) for error in errors),
                count_exact_rate=sum(error == 0 for error in errors) / len(errors),
                count_within_one_rate=sum(abs(error) <= 1 for error in errors) / len(errors),
            )
        geometry = [r for r in values if "parameter_rmse" in r]
        if geometry:
            matched = sum(r["knot_match_matched"] for r in geometry)
            predicted = sum(r["knot_match_predicted"] for r in geometry)
            true = sum(r["knot_match_true"] for r in geometry)
            precision = matched / predicted if predicted else float(true == 0)
            recall = matched / true if true else float(predicted == 0)
            result.update(
                parameter_rmse=statistics.fmean(
                    r["parameter_rmse"] for r in geometry
                ),
                knot_match_precision=precision,
                knot_match_recall=recall,
                knot_match_f1=(
                    2 * precision * recall / (precision + recall)
                    if precision + recall else 0.0
                ),
                knot_matched_mae=(
                    sum(r["knot_match_error_sum"] for r in geometry) / matched
                    if matched else None
                ),
            )
        return result
    if not rows:
        raise ValueError("validation cannot be empty")
    by_source = {name: group(values) for name, values in groups.items()}
    result = group(rows)
    result.update(by_source=by_source,
                  worst_dense_pass_rate=min(v["dense_pass_rate"] for v in by_source.values()),
                  worst_deployment_pass_rate=min(v["deployment_pass_rate"] for v in by_source.values()))
    synthetic = by_source.get("Synthetic", {})
    if "count_mae" in synthetic:
        result["synthetic_count_mae"] = synthetic["count_mae"]
        result["synthetic_count_bias"] = synthetic["count_bias"]
    if "knot_match_f1" in synthetic:
        result["synthetic_knot_match_f1"] = synthetic["knot_match_f1"]
        result["synthetic_knot_matched_mae"] = synthetic["knot_matched_mae"]
    return result


@torch.no_grad()
def validate(
    model, loader, device, tolerance, *, stage="joint",
    knot_match_tolerance=0.01, log_every=10,
):
    if stage not in ("proposal", "joint"):
        raise ValueError("stage must be 'proposal' or 'joint'")
    model.eval()
    rows = []
    for step, batch in enumerate(loader, 1):
        points = batch["points"].to(device, non_blocking=device.type == "cuda")
        context = model.encode_candidates(points, mse_tolerance=tolerance)
        output = None
        if stage == "joint":
            output = model.decode_subset(context, model.select_mask(context))
        # Transfer each batch tensor only once.  Calling .cpu() separately for
        # every curve introduces a GPU synchronization per validation sample.
        points_cpu = points.cpu().double()
        proposal_params_cpu = context["proposal_params"].cpu().double()
        proposal_knots_cpu = context["proposal_internal_knots"].cpu().double()
        if output is not None:
            output_params_cpu = output["params"].cpu().double()
            output_knots_cpu = output["internal_knots"].cpu().double()
            output_masks_cpu = output["learned_keep_mask"].cpu()
        probability_mass_cpu = context["one_shot_probability_mass"].cpu()
        adaptive_threshold_cpu = context["adaptive_keep_threshold"].cpu()
        trust_values = {}
        if getattr(model, "parameter_trust_enabled", False):
            trust_values["proposal_parameter_trust"] = context["proposal_parameter_trust"].cpu()
            if output is not None:
                trust_values["subset_parameter_trust"] = output["subset_parameter_trust"].cpu()
        target_count = batch.get("target_internal_knot_count")
        target_valid = batch.get("target_internal_knot_count_valid")
        target_params = batch.get("target_params")
        target_knots = batch.get("target_internal_knots")
        target_knot_mask = batch.get("target_internal_knot_mask")
        target_geometry_valid = batch.get("target_geometry_valid")
        if target_count is not None:
            target_count = target_count.cpu()
        if target_valid is not None:
            target_valid = target_valid.cpu().bool()
        if target_params is not None:
            target_params = target_params.cpu().double()
        if target_knots is not None:
            target_knots = target_knots.cpu().double()
        if target_knot_mask is not None:
            target_knot_mask = target_knot_mask.cpu().bool()
        if target_geometry_valid is not None:
            target_geometry_valid = target_geometry_valid.cpu().bool()
        for i, source in enumerate(batch["source"]):
            q = points_cpu[i]
            dense = refit_bspline_control_points(proposal_params_cpu[i], q,
                       proposal_knots_cpu[i], degree=model.degree,
                       smoothness_weight=0.0, control_ridge=0.0, interpolate_endpoints=True)
            if output is None:
                # The selector is deliberately untrained during proposal
                # learning.  Its selected refit is irrelevant to the only
                # stage gate (dense feasibility), so avoid a duplicate solve.
                fit = dense
                retained = proposal_knots_cpu.shape[1]
            else:
                mask = output_masks_cpu[i]
                selected = output_knots_cpu[i][mask].sort().values
                fit = refit_bspline_control_points(output_params_cpu[i], q, selected,
                           degree=model.degree, smoothness_weight=0.0, control_ridge=0.0,
                           interpolate_endpoints=True)
                retained = int(mask.sum())
            if not math.isfinite(float(fit.fit_mse)) or not math.isfinite(float(dense.fit_mse)):
                raise RuntimeError("non-finite validation fit")
            supervised_target = None
            if target_count is not None and target_valid is not None and bool(target_valid[i]):
                supervised_target = int(target_count[i])
            row = dict(
                source=source,
                mse=float(fit.fit_mse),
                dense_mse=float(dense.fit_mse),
                k=retained,
                target_k=supervised_target,
                probability_mass=float(probability_mass_cpu[i]),
                adaptive_threshold=float(adaptive_threshold_cpu[i]),
            )
            row.update({key: float(value[i].reshape(())) for key, value in trust_values.items()})
            if (
                target_params is not None
                and target_knots is not None
                and target_knot_mask is not None
                and target_geometry_valid is not None
                and bool(target_geometry_valid[i])
            ):
                predicted_params = (
                    proposal_params_cpu[i]
                    if output is None else output_params_cpu[i]
                )
                predicted_knots = (
                    proposal_knots_cpu[i]
                    if output is None else output_knots_cpu[i][output_masks_cpu[i]]
                )
                true_params = target_params[i]
                true_knots = target_knots[i][target_knot_mask[i]]
                warped = warp_internal_knots_to_parameterization(
                    predicted_knots,
                    predicted_params,
                    true_params,
                )
                matching = match_internal_knots(
                    warped,
                    true_knots,
                    tolerance=knot_match_tolerance,
                )
                row.update(
                    parameter_rmse=float(
                        (predicted_params - true_params).square().mean().sqrt()
                    ),
                    knot_match_predicted=matching.predicted_count,
                    knot_match_true=matching.true_count,
                    knot_match_matched=matching.matched_count,
                    knot_match_error_sum=(
                        matching.matched_mae * matching.matched_count
                        if matching.matched_count else 0.0
                    ),
                )
            rows.append(row)
        progress(step, len(loader), "  validation", every=log_every)
    return summarize(rows, tolerance)


def checkpoint_rank(
    metrics, target, safety_margin=0.0, *, simplification_ready=True,
):
    safe = metrics["worst_deployment_pass_rate"] >= target + safety_margin
    feasible = metrics["worst_deployment_pass_rate"] >= target
    tail = metrics.get("deployment_mse_p95", metrics["deployment_mse"])
    if feasible and simplification_ready:
        # Once the curriculum is mature, prefer a formally reportable
        # count/position solution, then reliability margin and fine-grained
        # geometry quality.  This prevents blind under-pruning from looking
        # better merely because it has fewer knots.
        count_mae = metrics.get("synthetic_count_mae")
        count_mae = (
            float("inf")
            if count_mae is None or not math.isfinite(count_mae)
            else count_mae
        )
        knot_f1 = metrics.get("synthetic_knot_match_f1")
        knot_f1 = (
            -float("inf")
            if knot_f1 is None or not math.isfinite(knot_f1)
            else knot_f1
        )
        knot_mae = metrics.get("synthetic_knot_matched_mae")
        knot_mae = (
            float("inf")
            if knot_mae is None or not math.isfinite(knot_mae)
            else knot_mae
        )
        # Quarter-knot buckets prevent a numerically tiny count-MAE change from
        # outranking a material node-localization improvement.
        count_mae_bucket = (
            math.ceil(count_mae * 4.0) / 4.0
            if math.isfinite(count_mae) else float("inf")
        )
        reporting_geometry_ready = bool(
            metrics.get("worst_dense_pass_rate", 0.0) >= target
            and count_mae <= V16_FORMAL_SYNTHETIC_COUNT_MAE_MAX
            and knot_f1 >= V16_FORMAL_SYNTHETIC_KNOT_F1_MIN
            and knot_mae <= V16_FORMAL_SYNTHETIC_MATCHED_MAE_MAX
        )
        return (
            3,
            int(reporting_geometry_ready),
            int(safe),
            -count_mae_bucket,
            knot_f1,
            -knot_mae,
            -count_mae,
            -metrics["keep_count"],
            -tail,
            -metrics["deployment_mse"],
        )
    if safe:
        return (2, metrics["worst_deployment_pass_rate"], -tail, -metrics["deployment_mse"])
    if feasible:
        # Inside the target-to-safety band, improve reliability before trying
        # to shave another knot from a statistically marginal checkpoint.
        return (1, metrics["worst_deployment_pass_rate"], -tail, -metrics["deployment_mse"])
    return (0, metrics["worst_deployment_pass_rate"], -tail, -metrics["deployment_mse"])


_TRAINING_TRUST_EXTREMA = {
    f"{gate}_{statistic}": reducer
    for gate in ("proposal_parameter_trust", "subset_parameter_trust")
    for statistic, reducer in (("min", min), ("max", max))
}


def accumulate_training_metrics(total, metrics, sample_count):
    """Accumulate sample-weighted means and true cross-batch gate extrema."""
    for key, value in metrics.items():
        numeric = float(value)
        reducer = _TRAINING_TRUST_EXTREMA.get(key)
        if reducer is not None:
            total[key] = reducer(total.get(key, numeric), numeric)
        else:
            total[key] = total.get(key, 0.0) + numeric * sample_count


def finalize_training_metrics(total, sample_count):
    """Normalize mean metrics only; extrema already describe the whole epoch."""
    if sample_count <= 0:
        raise ValueError("training metric aggregation requires at least one sample")
    return {key: value if key in _TRAINING_TRUST_EXTREMA else value / sample_count
            for key, value in total.items()}


def serial_args(args):
    return {key: ([str(v.resolve()) for v in value] if isinstance(value, list) else
                  str(value.resolve()) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key not in ENHANCED_TRAINING_DEFAULTS
            or value != ENHANCED_TRAINING_DEFAULTS[key]}


def training_config_changes(current, previous, ignored):
    """Compare optional settings symmetrically, including legacy absent defaults."""
    normalized_current = {**ENHANCED_TRAINING_DEFAULTS, **current}
    normalized_previous = {**ENHANCED_TRAINING_DEFAULTS, **previous}
    return [key for key, value in normalized_current.items()
            if key not in ignored and normalized_previous.get(key) != value]


def build_optimizer(model, args, *, stage):
    """Keep the historical one-group layout unless differential LRs are requested."""
    lr = args.lr if stage == "proposal" else args.joint_lr
    if stage == "proposal" or (
        args.joint_proposal_lr_scale == 1.0 and args.joint_decoder_lr_scale == 1.0
    ):
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    proposal_prefixes = ("encoder.", "parameter_head.", "candidate_head.",
                         "proposal_parameter_trust_head.")
    decoder_prefixes = (
        "subset_geometry.", "survivor_attention.", "parameter_attention.",
        "survivor_norm.", "parameter_norm.", "parameter_update.",
        "relocation_update.", "relocation_blend_logit",
        "subset_parameter_trust_head.",
    )
    groups = {"proposal": [], "selector": [], "decoder": []}
    for name, parameter in model.named_parameters():
        group = ("proposal" if name.startswith(proposal_prefixes) else
                 "decoder" if name.startswith(decoder_prefixes) else "selector")
        groups[group].append(parameter)
    scales = {"proposal": args.joint_proposal_lr_scale, "selector": 1.0,
              "decoder": args.joint_decoder_lr_scale}
    return torch.optim.AdamW([
        {"params": parameters, "lr": lr * scales[name],
         "group_name": name, "lr_scale": scales[name]}
        for name, parameters in groups.items()
    ], lr=lr, weight_decay=args.weight_decay)


def apply_joint_learning_rate(optimizer, args, *, epoch, end_epoch):
    """Epoch-based cosine decay; extensions retain the original decay horizon."""
    first_joint = args.proposal_epochs + 1
    progress_fraction = min(1.0, max(0.0,
        (epoch - first_joint) / max(1, end_epoch - first_joint)))
    ratio = args.joint_final_lr_ratio + (1.0 - args.joint_final_lr_ratio) * (
        1.0 + math.cos(math.pi * progress_fraction)
    ) / 2.0
    for group in optimizer.param_groups:
        group["lr"] = args.joint_lr * group.get("lr_scale", 1.0) * ratio
    return ratio


def synthetic_dataset_config(args):
    """Build the auditable synthetic-data contract stored in every checkpoint."""
    return dict(
        num_points=args.num_points,
        point_dim=args.point_dim,
        min_control_points=args.min_control_points,
        max_control_points=args.max_control_points,
        noise_std=args.noise_std,
        normalize=True,
        return_ground_truth=False,
        certified_minimal_source=args.certified_minimal_source,
        minimality_margin=args.minimality_margin,
        minimality_max_attempts=args.minimality_max_attempts,
        minimality_audit_points=args.minimality_audit_points,
        oscillation_amplitude=args.oscillation_amplitude,
        # The dataset certificate is expressed in RMS, whereas v16's public
        # engineering budget is mean squared Euclidean error.
        canonical_knot_tolerance=(
            math.sqrt(args.mse_tolerance)
            if args.certified_minimal_source else 0.0
        ),
    )


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def transfer_proposal_weights(model, checkpoint):
    """Warm-start every shape-compatible tensor needed for dense proposals."""
    old_state = checkpoint["model_state_dict"]
    state = model.state_dict()
    prefixes = ("encoder.", "parameter_head.", "candidate_head.")
    copied = {
        key: value for key, value in old_state.items()
        if key.startswith(prefixes) and key in state and state[key].shape == value.shape
    }
    if not copied:
        raise ValueError("initial checkpoint has no compatible encoder/parameter/proposal tensors")
    state.update(copied)
    model.load_state_dict(state, strict=True)
    return tuple(sorted(copied))


def transfer_all_weights(model, checkpoint):
    """Transfer all old tensors; only explicit off->on trust migration adds weights."""
    contracts = {
        "objective_version": V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
        "architecture_revision": V16_ADAPTIVE_SELECTION_REVISION,
        "simplification_contract": V16_SIMPLIFICATION_CONTRACT,
    }
    for name, expected in contracts.items():
        if checkpoint.get(name) != expected:
            raise ValueError(f"full warm start requires matching native v16 {name}")
    source_model, _, _ = build_model_from_checkpoint(checkpoint)
    source_config, target_config = source_model.get_config(), model.get_config()
    controlled_safety = {"one_shot_safety_sigma", "one_shot_safety_knots"}
    add_trust = (not source_config.get("parameter_trust_enabled", False)
                 and target_config.get("parameter_trust_enabled", False))
    if add_trust:
        controlled_safety |= {"parameter_trust_enabled", "parameter_trust_initial"}
    changed = [key for key in set(source_config) | set(target_config)
               if key not in controlled_safety
               and source_config.get(key) != target_config.get(key)]
    if changed:
        raise ValueError(f"full warm-start model configuration mismatch: {sorted(changed)}")
    source_state = source_model.state_dict()
    target_state = model.state_dict()
    missing = set(target_state) - set(source_state)
    trust_prefixes = ("proposal_parameter_trust_head.", "subset_parameter_trust_head.")
    if set(source_state) - set(target_state) or (
        missing and not (add_trust and all(key.startswith(trust_prefixes) for key in missing))
    ):
        raise ValueError("full warm start has unexpected missing or extra state tensors")
    target_state.update(source_state)
    model.load_state_dict(target_state, strict=True)
    return tuple(sorted(source_state))


def initializer_record(path, checkpoint, *, mode, copied):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return dict(
        mode=mode, path=str(path.resolve()), sha256=digest,
        epoch=checkpoint.get("epoch"), stage=checkpoint.get("stage"),
        copied_tensor_count=len(copied),
        training_config=checkpoint.get("training_config", {}),
        real_data_provenance=checkpoint.get("real_data_provenance", []),
        ancestor_initializer_provenance=checkpoint.get("initializer_provenance"),
        note="Weights only; fresh optimizer, history and validation controllers. Ancestor exposure is retained.",
    )


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as error:
        p.error(str(error))
    output = args.output.resolve()
    last_path = output.with_name(output.stem + ".last.pt")
    proposal_path = output.with_name(output.stem + ".proposal.pt")
    history_path = output.with_suffix(".history.json")
    if args.warm_start_checkpoint and args.warm_start_checkpoint.resolve() in {
        output, last_path, proposal_path,
    }:
        p.error("warm-start checkpoint must be a different experiment from the new output")
    if not args.resume and any(path.exists() for path in (output, last_path, proposal_path, history_path)):
        p.error("output artifacts already exist; choose a new output or explicitly --resume the .last.pt")
    torch.set_num_threads(args.torch_num_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    dataset_config = synthetic_dataset_config(args)
    print("Checking real manifests and train/val/test group separation...", flush=True)
    sources, provenance = load_real_sources(args.real_manifest, num_points=args.num_points,
                                           point_dim=args.point_dim,
                                           progress=lambda message: print(message, flush=True))
    validation = ValidationCurves(dataset_config, sources, size=args.val_size, seed=args.val_seed,
                                 real_per_source=args.real_val_size)
    loader_runtime = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_runtime = dict(loader_runtime)
    if args.num_workers > 0:
        val_runtime["persistent_workers"] = True
    val_loader = DataLoader(validation, batch_size=args.batch_size, **val_runtime)
    current_config = serial_args(args)
    current_config.update(train_seed=args.seed, train_seed_stride=EPOCH_SEED_STRIDE)
    model = V16CandidateSelectionNetwork(point_dim=args.point_dim, hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers, max_internal_knots=args.candidate_knots,
        attention_heads=args.attention_heads, selector_layers=args.selector_layers,
        mse_tolerance=args.mse_tolerance, relocation_blend=args.relocation_blend,
        parameter_trust_enabled=args.parameter_trust_enabled,
        parameter_trust_initial=args.parameter_trust_initial,
        one_shot_selection_policy=args.one_shot_selection_policy,
        one_shot_adaptive_threshold=True,
        one_shot_safety_sigma=args.one_shot_safety_sigma,
        one_shot_safety_knots=args.one_shot_safety_knots,
        one_shot_coverage_bins=args.one_shot_coverage_bins,
        min_selected_knots=args.min_selected_knots)
    history, start_epoch, best_rank, proposal_rank, proposal_ready = [], 1, None, None, False
    complexity_scale, feasible_streak = 0.0, 0
    selection_safety_scale = 1.0
    initializer_provenance = None
    joint_schedule_end_epoch = args.epochs
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        if resume_payload.get("objective_version") != V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION:
            p.error("resume requires a v16 checkpoint; use --init-checkpoint for proposal transfer")
        if resume_payload.get("architecture_revision") != V16_ADAPTIVE_SELECTION_REVISION:
            p.error(
                "resume checkpoint uses an older v16 selection revision; "
                "start a new run or use --init-checkpoint for proposal transfer"
            )
        if resume_payload.get("simplification_contract") != V16_SIMPLIFICATION_CONTRACT:
            p.error(
                "resume checkpoint uses an older simplification contract; "
                "start a new run or use --init-checkpoint"
            )
        ignored = {"epochs", "resume", "init_checkpoint", "warm_start_checkpoint", "output", "device", "num_workers",
                   "torch_num_threads", "log_every_batches"}
        previous_config = resume_payload["training_config"]
        if Path(previous_config["output"]).resolve() != output:
            p.error("resume must use the original --output so its best/proposal artifacts remain available")
        if (resume_payload.get("stage") == "proposal"
                and args.proposal_epochs >= previous_config["proposal_epochs"]):
            ignored.add("proposal_epochs")
        changed = training_config_changes(current_config, previous_config, ignored)
        if changed or resume_payload.get("real_data_provenance") != provenance:
            p.error(f"resume data/training configuration mismatch: {changed or 'manifest fingerprints'}")
        model, _, _ = build_model_from_checkpoint(resume_payload)
        history = resume_payload.get("history", [])
        start_epoch = int(resume_payload["epoch"]) + 1
        best_rank = resume_payload.get("best_joint_rank")
        proposal_rank = resume_payload.get("best_proposal_rank")
        proposal_ready = bool(resume_payload.get("proposal_ready", False))
        complexity_scale = float(resume_payload.get("complexity_scale", 0.0))
        feasible_streak = int(resume_payload.get("feasible_streak", 0))
        selection_safety_scale = float(
            resume_payload.get("next_selection_safety_scale", 1.0)
        )
        initializer_provenance = resume_payload.get("initializer_provenance")
        joint_schedule_end_epoch = int(resume_payload.get(
            "training_schedule", {},
        ).get("joint_end_epoch", previous_config["epochs"]))
        if not 0 <= complexity_scale <= args.complexity_max_scale:
            p.error("resume checkpoint has an invalid complexity controller state")
        if not 0 <= selection_safety_scale <= 1:
            p.error("resume checkpoint has an invalid safety controller state")
        if start_epoch > args.epochs:
            p.error("checkpoint already completed the requested epochs")
        if not proposal_path.exists() or (resume_payload.get("stage") == "joint" and not output.exists()):
            p.error("resume requires the saved best proposal and, for joint training, the best model artifact")
    elif args.warm_start_checkpoint:
        source_checkpoint = torch.load(args.warm_start_checkpoint, map_location="cpu", weights_only=True)
        try:
            copied = transfer_all_weights(model, source_checkpoint)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            p.error(str(error))
        initializer_provenance = initializer_record(
            args.warm_start_checkpoint, source_checkpoint, mode="full_model", copied=copied,
        )
        initializer_provenance["initialized_tensor_names"] = sorted(set(model.state_dict()) - set(copied))
        print(f"Transferred all {len(copied)} model tensors including selector/decoder; "
              "fresh optimizer, history, safety curriculum and proposal gate.", flush=True)
        if initializer_provenance["initialized_tensor_names"]:
            print("Initialized new parameter trust gates; this migration changes the forward function.", flush=True)
    elif args.init_checkpoint:
        source_checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        try:
            copied = transfer_proposal_weights(model, source_checkpoint)
        except (KeyError, TypeError, ValueError) as error:
            p.error(str(error))
        print(f"Transferred {len(copied)} encoder/parameter/proposal tensors; "
              "selector and subset decoder start fresh.", flush=True)
        initializer_provenance = initializer_record(
            args.init_checkpoint, source_checkpoint, mode="proposal_only", copied=copied,
        )
    model.to(device)
    optimizer = build_optimizer(model, args, stage=resume_payload["stage"] if resume_payload else "proposal")
    if resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        torch.set_rng_state(resume_payload["rng_state"])
        if device.type == "cuda" and resume_payload.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state"])
    loss_options = dict(
        mse_tolerance=args.mse_tolerance,
        policy_samples=args.policy_samples,
        counterfactual_edits=args.counterfactual_edits,
        teacher_prefix_search_steps=args.teacher_prefix_search_steps,
        teacher_refinement_steps=args.teacher_refinement_steps,
        teacher_refinement_candidates=args.teacher_refinement_candidates,
        boundary_ranking_weight=args.boundary_ranking_weight,
        boundary_ranking_candidates=args.boundary_ranking_candidates,
        parameter_counterfactual_weight=args.parameter_counterfactual_weight,
        teacher_geometry_candidates=args.teacher_geometry_candidates,
        local_fit_weight=args.local_fit_weight,
        proposal_ordered_weight=args.proposal_ordered_weight,
        feasible_objective=args.feasible_objective,
        feasible_fit_margin=args.feasible_fit_margin,
        feasible_fit_weight=args.feasible_fit_weight,
        teacher_greedy_steps=args.teacher_greedy_steps,
        teacher_greedy_max_curves=args.teacher_greedy_max_curves,
        teacher_geometry_distillation_weight=args.teacher_geometry_distillation_weight,
        count_reserve_alignment=args.count_reserve_alignment,
        count_weight=args.count_weight,
        supervised_count_weight=args.supervised_count_weight,
        supervised_over_count_weight=args.supervised_over_count_weight,
        complexity_weight=args.complexity_weight,
        true_parameter_weight=args.true_parameter_weight,
        proposal_knot_coverage_weight=args.proposal_knot_coverage_weight,
        selected_knot_position_weight=args.selected_knot_position_weight,
        knot_position_beta=args.knot_position_beta,
    )
    if resume_payload:
        saved_loss = resume_payload.get("loss_config", {})
        loss_options.update(saved_loss.get("weights", {}))
        if "solver_jitter" in saved_loss:
            loss_options["solver_jitter"] = saved_loss["solver_jitter"]
    objective = V16SubsetLoss(**loss_options)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"v16 adaptive {args.one_shot_selection_policy}: {device}, "
          f"Kc={args.candidate_knots} internal "
          f"(full cubic knot vector at all-keep={args.candidate_knots + 8}), "
          f"MSE tolerance={args.mse_tolerance:g}; "
           f"{args.train_size} train draws, {len(validation)} validation curves; real sources={len(sources)}", flush=True)
    print(
        "Synthetic source contract: "
        + (
            "clean source-subset threshold-minimal certificate "
            f"at RMS={math.sqrt(args.mse_tolerance):.6g}, margin={args.minimality_margin:.0%}, "
            f"audit_points={args.minimality_audit_points}"
            if args.certified_minimal_source
            else "UNCERTIFIED random source representation (diagnostic ablation)"
        ),
        flush=True,
    )
    print(
        "Simplification curriculum: training-only ranked-prefix teacher "
        f"steps={args.teacher_prefix_search_steps}; certified over-count "
        f"weight={args.supervised_over_count_weight:g}, symmetric true-count "
        f"weight={args.supervised_count_weight:g}; complexity multiplier "
        f"0..{args.complexity_max_scale:g}; safety "
        f"{args.one_shot_safety_knots}+{args.one_shot_safety_sigma:g}sigma -> "
        f"{args.final_safety_knots}+{args.final_safety_sigma:g}sigma. "
        "Deployment still uses one network forward and one final refit.",
        flush=True,
    )
    print("Validation checkpoint quality uses the worst source pass rate. Additional subset fits run only during training.", flush=True)
    if args.simplification_controller == "per_curve":
        print("Per-curve simplification: scheduled complexity ramp; infeasible training curves receive no complexity pressure. "
              "Worst-source validation still governs checkpoint selection, not the training ramp.", flush=True)
    if args.synthetic_simple_fraction or args.synthetic_shape_fraction:
        print(f"Training-only synthetic mixture: low-K={args.synthetic_simple_fraction:.0%}, "
              f"procedural-shape={args.synthetic_shape_fraction:.0%}, "
              f"historical={1-args.synthetic_simple_fraction-args.synthetic_shape_fraction:.0%}; "
              "procedural shapes have no true-knot/count labels; validation distribution is unchanged.", flush=True)
    if sources and args.real_fraction == 0:
        if args.synthetic_shape_fraction:
            print("Real sources are VALIDATION ONLY; training uses labeled synthetic splines and unlabeled procedural shapes, "
                  "and test splits are excluded from model selection.", flush=True)
        else:
            print("Real sources are VALIDATION ONLY; training uses synthetic labels, and test splits are excluded from model selection.", flush=True)
    if args.teacher_refinement_steps or args.boundary_ranking_weight:
        print(
            f"Online teacher refinement: {args.teacher_refinement_steps} rounds x "
            f"{args.teacher_refinement_candidates} deletion candidates; boundary ranking "
            f"weight={args.boundary_ranking_weight:g}, candidates={args.boundary_ranking_candidates}. "
            "Teacher mask agreement below is a training signal, not ground-truth knot F1.",
            flush=True,
        )
    for epoch in range(start_epoch, args.epochs + 1):
        stage = "proposal" if epoch <= args.proposal_epochs else "joint"
        applied_safety_scale = selection_safety_scale
        safety_sigma, safety_knots = selection_safety(args, applied_safety_scale)
        model.set_selection_safety(sigma=safety_sigma, knots=safety_knots)
        if stage == "joint" and epoch == args.proposal_epochs + 1:
            if not proposal_path.exists():
                p.error("missing best proposal checkpoint for the stage transition")
            best_proposal = torch.load(proposal_path, map_location="cpu", weights_only=True)
            model.load_state_dict(best_proposal["model_state_dict"], strict=True)
            proposal_gate = args.proposal_pass_target
            proposal_ready = (
                best_proposal["validation_metrics"]["worst_dense_pass_rate"]
                >= proposal_gate
            )
            if not proposal_ready and not args.allow_infeasible_proposals:
                print(f"STOP: dense proposal did not reach the safety gate "
                      f"{proposal_gate:.1%} in every source. "
                      f"Inspect {history_path}; increase proposal training/candidate budget before simplification.", flush=True)
                return 2
            optimizer = build_optimizer(model, args, stage="joint")
        if stage == "joint":
            apply_joint_learning_rate(
                optimizer, args, epoch=epoch, end_epoch=joint_schedule_end_epoch,
            )
        learning_rates = {group.get("group_name", "all"): group["lr"]
                          for group in optimizer.param_groups}
        train_data = MixedTrainingCurves(dataset_config, sources, size=args.train_size, seed=args.seed,
            real_fraction=args.real_fraction, epoch=epoch-1, resample=args.resample_train_each_epoch,
            synthetic_simple_fraction=args.synthetic_simple_fraction,
            synthetic_shape_fraction=args.synthetic_shape_fraction)
        loader_generator = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            generator=loader_generator, **loader_runtime,
        )
        model.train()
        started = time.perf_counter()
        total, samples = defaultdict(float), 0
        training_family_counts = defaultdict(int)
        for step, batch in enumerate(train_loader, 1):
            for family in batch.get("synthetic_family", ()):
                training_family_counts[family] += 1
            points = batch["points"].to(device, non_blocking=device.type == "cuda")
            lower, upper = math.log(args.tolerance_factor_min), math.log(args.tolerance_factor_max)
            tolerance = args.mse_tolerance * (lower + (upper-lower)*torch.rand(points.shape[0], device=device)).exp()
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = objective(
                model, points, stage=stage, mse_tolerance=tolerance,
                complexity_scale=complexity_scale if stage == "joint" else 0.0,
                synthetic_target_count=batch["target_internal_knot_count"].to(
                    device, non_blocking=device.type == "cuda"
                ),
                synthetic_target_valid=batch["target_internal_knot_count_valid"].to(
                    device, non_blocking=device.type == "cuda"
                ),
                target_params=batch["target_params"].to(
                    device, non_blocking=device.type == "cuda"
                ),
                target_internal_knots=batch["target_internal_knots"].to(
                    device, non_blocking=device.type == "cuda"
                ),
                target_internal_knot_mask=batch["target_internal_knot_mask"].to(
                    device, non_blocking=device.type == "cuda"
                ),
                target_geometry_valid=batch["target_geometry_valid"].to(
                    device, non_blocking=device.type == "cuda"
                ),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            size = len(points)
            samples += size
            accumulate_training_metrics(total, metrics, size)
            progress(step, len(train_loader), f"Epoch {epoch:03}/{args.epochs} {stage}",
                     f"MSE={metrics['deployment_mse']:.3e} pass={metrics['deployment_pass_rate']:.1%} "
                     f"K={metrics['keep_count']:.1f} teacherK={metrics['subset_best_count']:.1f} "
                     f"countMAE={metrics['supervised_count_mae']:.2f}",
                     args.log_every_batches)
        measured = validate(
            model, val_loader, device, args.mse_tolerance,
            stage=stage,
            knot_match_tolerance=args.knot_match_tolerance,
            log_every=args.log_every_batches,
        )
        train_metrics = finalize_training_metrics(total, samples)
        applied_complexity_scale = complexity_scale
        if stage == "joint":
            observed_pass = measured["worst_deployment_pass_rate"]
            if observed_pass >= args.deployment_pass_target:
                feasible_streak += 1
            else:
                feasible_streak = 0
            complexity_scale, selection_safety_scale = (
                update_simplification_controller(
                    args,
                    pass_rate=observed_pass,
                    complexity_scale=complexity_scale,
                    safety_scale=selection_safety_scale,
                )
            )
        entry = dict(
            epoch=epoch, stage=stage, train=train_metrics, validation=measured,
            learning_rates=learning_rates,
            applied_complexity_scale=applied_complexity_scale,
            next_complexity_scale=complexity_scale,
            applied_selection_safety_scale=applied_safety_scale,
            next_selection_safety_scale=selection_safety_scale,
            applied_selection_safety_sigma=safety_sigma,
            applied_selection_safety_knots=safety_knots,
            feasible_streak=feasible_streak,
            seconds=time.perf_counter()-started,
        )
        if training_family_counts:
            entry["training_family_counts"] = dict(training_family_counts)
        history.append(entry)
        improved = False
        if stage == "proposal":
            rank = (measured["worst_dense_pass_rate"], -measured["dense_mse"])
            if proposal_rank is None or rank > tuple(proposal_rank):
                proposal_rank, improved = rank, True
            proposal_ready = proposal_rank[0] >= min(
                1.0, args.proposal_pass_target
            )
        else:
            simplification_ready = simplification_is_ready(
                args,
                epoch=epoch,
                stage=stage,
                applied_safety_scale=applied_safety_scale,
                applied_complexity_scale=applied_complexity_scale,
            )
            rank = checkpoint_rank(
                measured, args.deployment_pass_target,
                args.complexity_pass_margin,
                simplification_ready=simplification_ready,
            )
            if best_rank is None or rank > tuple(best_rank):
                best_rank, improved = rank, True
        accepted = stage == "joint" and measured["worst_deployment_pass_rate"] >= args.deployment_pass_target
        payload = dict(objective_version=V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
            model_config=model.get_config(), model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            optimizer_state_dict=optimizer.state_dict(), epoch=epoch, stage=stage,
            training_config=current_config, dataset_config=dataset_config, history=history,
            training_schedule=dict(joint_end_epoch=joint_schedule_end_epoch,
                                   joint_final_lr_ratio=args.joint_final_lr_ratio,
                                   **({"simplification_controller": args.simplification_controller}
                                      if args.simplification_controller != "worst_source" else {})),
            initializer_provenance=initializer_provenance,
            dataset_type=(
                "synthetic_bspline_and_procedural_shape_mixture"
                if args.synthetic_shape_fraction and args.real_fraction == 0 else
                (
                    "certified_synthetic_geometry_and_real_unlabeled"
                    if args.certified_minimal_source
                    else "uncertified_synthetic_and_real_unlabeled"
                )
                if sources and args.real_fraction > 0
                else (
                    "certified_synthetic_open_cubic_bspline"
                    if args.certified_minimal_source
                    else "uncertified_synthetic_open_cubic_bspline"
                )
            ),
            real_data_provenance=provenance, validation_real_ids=validation.selected_real_ids,
            real_data_role=("training_and_validation" if args.real_fraction > 0 and sources
                            else "validation_only" if sources else "not_used"),
            synthetic_data_contract=(
                V16_CERTIFIED_SYNTHETIC_CONTRACT
                if args.certified_minimal_source
                else "random_source_uncertified"
            ),
            synthetic_training_mixture=dict(
                simple_fraction=args.synthetic_simple_fraction,
                shape_fraction=args.synthetic_shape_fraction,
                historical_fraction=1-args.synthetic_simple_fraction-args.synthetic_shape_fraction,
                applies_to="training synthetic draws only",
                shape_geometry_targets_valid=False,
                validation_distribution="unchanged historical certified B-splines and held-out real val splits",
            ),
            validation_metrics=measured, train_metrics=train_metrics,
            best_joint_rank=best_rank, best_proposal_rank=proposal_rank, proposal_ready=proposal_ready,
            architecture_revision=(
                V16_ADAPTIVE_SELECTION_REVISION
                if args.one_shot_selection_policy == "mass_topk"
                else "v16_adaptive_beta_threshold_ablation"
            ),
            simplification_contract=V16_SIMPLIFICATION_CONTRACT,
            simplification_ready=simplification_is_ready(
                args,
                epoch=epoch,
                stage=stage,
                applied_safety_scale=applied_safety_scale,
                applied_complexity_scale=applied_complexity_scale,
            ),
            complexity_scale=complexity_scale,
            applied_complexity_scale=applied_complexity_scale,
            next_selection_safety_scale=selection_safety_scale,
            applied_selection_safety_scale=applied_safety_scale,
            feasible_streak=feasible_streak,
            best_deployment_pass_constraint_satisfied=accepted,
            current_deployment_pass_constraint_satisfied=accepted,
            checkpoint_quality="deployment_target_met" if accepted else "target_not_met",
            checkpoint_selection=(
                "formal_geometry_gate_then_safety_then_bucketed_count_f1_mae"
            ),
            deployment_config=dict(mse_tolerance=args.mse_tolerance, error_tolerance=math.sqrt(args.mse_tolerance),
                knot_match_tolerance=args.knot_match_tolerance,
                one_shot_selection_policy=args.one_shot_selection_policy,
                adaptive_keep_threshold=True,
                one_shot_safety_sigma=safety_sigma,
                one_shot_safety_knots=safety_knots,
                one_shot_coverage_bins=args.one_shot_coverage_bins,
                min_selected_knots=args.min_selected_knots,
                smoothness_weight=0.0, control_ridge=0.0,
                interpolate_endpoints=True, network_forwards=1, final_refits=1),
            rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            loss_config=dict(policy_samples=args.policy_samples, counterfactual_edits=args.counterfactual_edits,
                             teacher_prefix_search_steps=args.teacher_prefix_search_steps,
                             teacher_refinement_steps=args.teacher_refinement_steps,
                             teacher_refinement_candidates=args.teacher_refinement_candidates,
                             teacher_geometry_candidates=args.teacher_geometry_candidates,
                             teacher_greedy_steps=args.teacher_greedy_steps,
                             teacher_greedy_max_curves=args.teacher_greedy_max_curves,
                             feasible_objective=args.feasible_objective,
                             feasible_fit_margin=args.feasible_fit_margin,
                             feasible_fit_weight=args.feasible_fit_weight,
                             count_reserve_alignment=args.count_reserve_alignment,
                             boundary_ranking_candidates=args.boundary_ranking_candidates,
                             ranked_prefix_teacher=objective.ranked_prefix_teacher,
                             knot_position_beta=objective.knot_position_beta,
                             mse_tolerance=args.mse_tolerance, solver_jitter=objective.solver_jitter,
                             weights={name: getattr(objective, name) for name in (
                                 "fit_weight", "policy_weight", "distillation_weight", "dense_weight",
                                  "count_weight", "ranking_weight", "false_remove_weight",
                                  "ranking_margin", "entropy_weight", "complexity_weight",
                                  "complexity_activation_ratio", "tail_weight", "tail_fraction",
                                  "supervised_count_weight",
                                  "supervised_over_count_weight", "true_parameter_weight",
                                  "proposal_knot_coverage_weight",
                                  "selected_knot_position_weight", "boundary_ranking_weight",
                                  "parameter_counterfactual_weight", "local_fit_weight",
                                  "proposal_ordered_weight", "teacher_geometry_distillation_weight")}))
        payload["qualification"] = assess_v16_checkpoint(
            payload,
            required_pass_rate=V16_FORMAL_PASS_RATE,
            required_mse_tolerance=args.mse_tolerance,
        )
        atomic_save(payload, last_path)
        if improved:
            atomic_save(payload, proposal_path if stage == "proposal" else output)
        history_path.write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Epoch {epoch:03} val dense={measured['dense_pass_rate']:.1%} "
              f"deployment={measured['deployment_pass_rate']:.1%} worst-source={measured['worst_deployment_pass_rate']:.1%} "
              f"MSE={measured['deployment_mse']:.3e} K={measured['keep_count']:.2f} "
              f"mass={measured['keep_probability_mass']:.2f} beta={measured['adaptive_keep_threshold']:.2f} "
              f"complexity={applied_complexity_scale:.2f}->{complexity_scale:.2f} "
              f"safety={safety_knots}+{safety_sigma:.2f}sigma "
              f"scale={applied_safety_scale:.2f}->{selection_safety_scale:.2f} "
              f"target_met={accepted}", flush=True)
        if (args.joint_proposal_lr_scale != 1.0 or args.joint_decoder_lr_scale != 1.0
                or args.joint_final_lr_ratio != 1.0):
            print("  learning rates: " + ", ".join(
                f"{name}={value:.3e}" for name, value in learning_rates.items()
            ), flush=True)
        if stage == "joint" and (args.teacher_refinement_steps or args.boundary_ranking_weight):
            print(
                f"  training teacher: keep_P/R/F1="
                f"{train_metrics['teacher_keep_precision']:.3f}/"
                f"{train_metrics['teacher_keep_recall']:.3f}/"
                f"{train_metrics['teacher_keep_f1']:.3f}, "
                f"false_remove={train_metrics['teacher_false_remove_rate']:.1%}, "
                f"refinement_delta_K={train_metrics['teacher_refinement_count_reduction']:.3f}, "
                f"improved={train_metrics['teacher_refinement_improved_fraction']:.1%}, "
                f"extra_fits={train_metrics['teacher_refinement_evaluations']:.1f}, "
                f"boundary_loss={train_metrics['teacher_boundary_ranking_loss']:.4f}",
                flush=True,
            )
        if args.parameter_trust_enabled or args.parameter_counterfactual_weight or args.local_fit_weight:
            print(
                f"  parameter/local: counterfactual={train_metrics['parameter_counterfactual_loss']:.4f}, "
                f"local={train_metrics['local_fit_loss']:.4f}, "
                f"geometry_teacher_delta_K={train_metrics['teacher_geometry_count_reduction']:.3f}, "
                f"geometry_fits={train_metrics['teacher_geometry_evaluations']:.1f}", flush=True,
            )
        if args.teacher_greedy_steps and stage == "joint":
            print(
                f"  compact teacher: fixed_K={train_metrics['teacher_greedy_fixed_count']:.2f}, "
                f"fixed_pass={train_metrics['teacher_greedy_fixed_pass_rate']:.1%}, "
                f"decoded_pass={train_metrics['teacher_greedy_decoded_pass_rate']:.1%}, "
                f"accepted_delta_K={train_metrics['teacher_greedy_accepted_delta_k']:.3f}, "
                f"geometry_violation={train_metrics['teacher_geometry_fit_violation']:.4f}, "
                f"extra_refits={train_metrics['teacher_greedy_refits']:.1f}, "
                f"complexity_active={train_metrics['complexity_active_fraction']:.1%}", flush=True,
            )
        if training_family_counts:
            print(f"  training families: {dict(training_family_counts)}", flush=True)
        for name, values in measured["by_source"].items():
            count_detail = (
                f", targetK={values['target_count_mean']:.2f}, "
                f"count_MAE={values['count_mae']:.2f}, bias={values['count_bias']:+.2f}"
                if "count_mae" in values else ""
            )
            position_detail = (
                f", knot_F1@{args.knot_match_tolerance:g}="
                f"{values['knot_match_f1']:.3f}, param_RMSE="
                f"{values['parameter_rmse']:.3e}"
                if "knot_match_f1" in values else ""
            )
            trust_detail = (
                f", trust_proposal={values['proposal_parameter_trust_mean']:.3f}"
                if "proposal_parameter_trust_mean" in values else ""
            )
            if "subset_parameter_trust_mean" in values:
                trust_detail += f", trust_subset={values['subset_parameter_trust_mean']:.3f}"
            print(f"  {name}: dense={values['dense_pass_rate']:.1%}, deployment={values['deployment_pass_rate']:.1%}, "
                  f"MSE={values['deployment_mse']:.3e}, K={values['keep_count']:.2f}"
                  f"{count_detail}{position_detail}{trust_detail}", flush=True)
    best = torch.load(output, map_location="cpu", weights_only=True)
    print(f"Saved best v16: {output}; quality={best['checkpoint_quality']}. "
          f"Last/resume: {last_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
