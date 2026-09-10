"""Train v16 dense feasibility, then online counterfactual subset selection."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from torch.nn import functional as F
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


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=60, help="Total proposal plus joint epochs")
    p.add_argument(
        "--proposal-epochs", type=int, default=20,
        help="Last proposal-stage epoch (total, not extra epochs when resuming)",
    )
    p.add_argument("--train-size", type=int, default=2400, help="Mixture draws per epoch")
    p.add_argument("--val-size", type=int, default=500, help="Synthetic validation curves")
    p.add_argument(
        "--synthetic-boundary-val-size",
        type=int,
        default=32,
        help=("Synthetic validation slots fixed at the maximum source knot "
              "count; these slots are included inside --val-size"),
    )
    p.add_argument("--real-val-size", type=int, default=100, help="Maximum val curves per real source")
    p.add_argument("--real-manifest", action="append", type=Path, default=[])
    p.add_argument("--real-fraction", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-points", type=int, default=192)
    p.add_argument("--point-dim", type=int, choices=(2, 3), default=2)
    p.add_argument("--min-control-points", type=int, default=8)
    p.add_argument(
        "--max-control-points", type=int, default=60,
        help="Maximum synthetic source control points; this does not change network Kc",
    )
    p.add_argument(
        "--candidate-knots", type=int, default=None,
        help=("Internal candidate capacity Kc (default: 56 internal knots, "
              "equivalent to 64 entries in a full cubic clamped knot vector). "
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
        "--knot-min-span", type=float, default=0.01,
        help=("Minimum parameter span between adjacent synthetic source knots. "
              "The v16 default supports K=56; historical datasets retain 0.02"),
    )
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
              "steps resolve the default Kc=56 count boundary without "
              "deployment search"),
    )
    p.add_argument(
        "--teacher-low-count-sweep", type=int, default=16,
        help=("Exactly evaluate every deployed ranked prefix up to this count "
              "during training; 0 disables the simple-curve sweep"),
    )
    p.add_argument(
        "--synthetic-count-role", choices=("exact", "upper_bound"),
        default="upper_bound",
        help=("Treat certified source K as an upper bound when survivor knots "
              "may relocate; 'exact' preserves the historical v16 ablation"),
    )
    p.add_argument(
        "--synthetic-geometry-oracle-teacher",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Training-only: add a score-independent monotone candidate/true-"
              "knot assignment and its expanded mask to the teacher pool"),
    )
    p.add_argument(
        "--oracle-teacher-extra-knots", type=int, default=2,
        help="Extra high-score candidates in the oracle bootstrap safety mask",
    )
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
        "--initial-keep-fraction", type=float, default=30 / 56,
        help=("Initial selector probability mass as a fraction of Kc; proposal "
              "training does not update the selector. The default is 30/56, "
              "matching the mean K of the default synthetic K=4..56 range"),
    )
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
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--joint-lr", type=float, default=5e-5)
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
        args.candidate_knots = 56
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
        "teacher_low_count_sweep", "minimality_max_attempts",
        "oracle_teacher_extra_knots", "synthetic_boundary_val_size",
    ):
        value = getattr(args, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")
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
    if not math.isfinite(args.knot_min_span) or args.knot_min_span <= 0:
        raise ValueError("knot-min-span must be finite and positive")
    if args.knot_min_span * (args.max_control_points - 3) >= 1.0:
        raise ValueError(
            "knot-min-span is too large for max-control-points; cubic source "
            "generation requires knot-min-span * (max-control-points - 3) < 1"
        )
    for key in (
        "mse_tolerance", "knot_match_tolerance", "lr", "joint_lr",
        "grad_clip", "tolerance_factor_min", "tolerance_factor_max",
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
    ):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if args.minimality_max_attempts < 1:
        raise ValueError("minimality-max-attempts must be positive")
    if args.minimality_audit_points not in (0,) and args.minimality_audit_points < 2:
        raise ValueError("minimality-audit-points must be 0 or at least 2")
    if not math.isfinite(args.oscillation_amplitude) or args.oscillation_amplitude <= 0:
        raise ValueError("oscillation-amplitude must be finite and positive")
    if args.safety_anneal_epochs < 1:
        raise ValueError("safety-anneal-epochs must be positive")
    if args.teacher_prefix_search_steps < 1:
        raise ValueError("teacher-prefix-search-steps must be positive")
    # Both teacher budgets are safely clamped to Kc inside the objective. This
    # keeps tiny unit-test/ablation capacities compatible with production
    # defaults instead of requiring unrelated command-line overrides.
    if args.complexity_max_scale <= 0:
        raise ValueError("complexity-max-scale must be positive")
    if args.knot_position_beta <= 0:
        raise ValueError("knot-position-beta must be positive")
    if args.final_safety_sigma > args.one_shot_safety_sigma:
        raise ValueError("final-safety-sigma cannot exceed one-shot-safety-sigma")
    if args.final_safety_knots > args.one_shot_safety_knots:
        raise ValueError("final-safety-knots cannot exceed one-shot-safety-knots")
    for key in ("real_fraction", "proposal_pass_target", "deployment_pass_target"):
        if not math.isfinite(getattr(args, key)) or not 0 <= getattr(args, key) <= 1:
            raise ValueError(f"{key} must lie in [0,1]")
    if args.tolerance_factor_min > args.tolerance_factor_max:
        raise ValueError("tolerance-factor-min must be <= tolerance-factor-max")
    if not math.isfinite(args.relocation_blend) or not 0 <= args.relocation_blend <= 1:
        raise ValueError("relocation-blend must lie in [0,1]")
    if (
        not math.isfinite(args.initial_keep_fraction)
        or not 0.0 < args.initial_keep_fraction < 1.0
    ):
        raise ValueError("initial-keep-fraction must lie strictly inside (0,1)")
    if args.deployment_pass_target + args.complexity_pass_margin > 1:
        raise ValueError("deployment-pass-target + complexity-pass-margin cannot exceed one")
    if args.resume and args.init_checkpoint:
        raise ValueError("resume and init-checkpoint are mutually exclusive")
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
    """Advance the pass-feedback complexity/safety curriculum by one epoch."""
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
    if pass_rate >= safe_target:
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


def summarize(rows, tolerance, *, synthetic_boundary_knot_count=None):
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
    worst_source_dense = min(v["dense_pass_rate"] for v in by_source.values())
    worst_source_deployment = min(
        v["deployment_pass_rate"] for v in by_source.values()
    )
    result.update(
        by_source=by_source,
        worst_dense_pass_rate=worst_source_dense,
        worst_deployment_pass_rate=worst_source_deployment,
    )
    synthetic = by_source.get("Synthetic", {})
    if "count_mae" in synthetic:
        result["synthetic_count_mae"] = synthetic["count_mae"]
        result["synthetic_count_bias"] = synthetic["count_bias"]
    if "knot_match_f1" in synthetic:
        result["synthetic_knot_match_f1"] = synthetic["knot_match_f1"]
        result["synthetic_knot_matched_mae"] = synthetic["knot_matched_mae"]
    # The upper end of a source-count range is also the proposal-capacity
    # boundary in the formal K=4..56 protocol.  An aggregate Synthetic rate can
    # hide a complete failure of that sparse stratum, so retain an explicit
    # boundary audit and use it together with the per-data-source minimum for
    # training gates and checkpoint selection.
    boundary_dense = None
    boundary_deployment = None
    if synthetic_boundary_knot_count is not None:
        if (
            isinstance(synthetic_boundary_knot_count, bool)
            or not isinstance(synthetic_boundary_knot_count, int)
            or synthetic_boundary_knot_count < 0
        ):
            raise ValueError(
                "synthetic_boundary_knot_count must be a non-negative integer"
            )
        boundary_rows = [
            row for row in rows
            if row["source"] == "Synthetic"
            and row.get("target_k") == synthetic_boundary_knot_count
        ]
        result["synthetic_boundary_knot_count"] = synthetic_boundary_knot_count
        result["synthetic_boundary_sample_count"] = len(boundary_rows)
        if boundary_rows:
            boundary_summary = group(boundary_rows)
            boundary_dense = boundary_summary["dense_pass_rate"]
            boundary_deployment = boundary_summary["deployment_pass_rate"]
            result["synthetic_boundary_dense_pass_rate"] = boundary_dense
            result[
                "synthetic_boundary_deployment_pass_rate"
            ] = boundary_deployment
            for source_name, output_name in (
                ("count_mae", "synthetic_boundary_count_mae"),
                ("knot_match_f1", "synthetic_boundary_knot_match_f1"),
                ("knot_matched_mae", "synthetic_boundary_knot_matched_mae"),
            ):
                if source_name in boundary_summary:
                    result[output_name] = boundary_summary[source_name]
        else:
            result["synthetic_boundary_dense_pass_rate"] = None
            result["synthetic_boundary_deployment_pass_rate"] = None
            result["synthetic_boundary_count_mae"] = None
            result["synthetic_boundary_knot_match_f1"] = None
            result["synthetic_boundary_knot_matched_mae"] = None
    result["qualification_dense_pass_rate"] = (
        min(worst_source_dense, boundary_dense)
        if boundary_dense is not None else worst_source_dense
    )
    result["qualification_deployment_pass_rate"] = (
        min(worst_source_deployment, boundary_deployment)
        if boundary_deployment is not None else worst_source_deployment
    )
    return result


@torch.no_grad()
def validate(
    model, loader, device, tolerance, *, stage="joint",
    knot_match_tolerance=0.01, log_every=10,
    synthetic_boundary_knot_count=None,
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
    return summarize(
        rows,
        tolerance,
        synthetic_boundary_knot_count=synthetic_boundary_knot_count,
    )


def checkpoint_rank(
    metrics, target, safety_margin=0.0, *, simplification_ready=True,
):
    deployment_pass = metrics.get(
        "qualification_deployment_pass_rate",
        metrics["worst_deployment_pass_rate"],
    )
    dense_pass = metrics.get(
        "qualification_dense_pass_rate",
        metrics.get("worst_dense_pass_rate", 0.0),
    )
    safe = deployment_pass >= target + safety_margin
    feasible = deployment_pass >= target
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
            dense_pass >= target
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
        return (2, deployment_pass, -tail, -metrics["deployment_mse"])
    if feasible:
        # Inside the target-to-safety band, improve reliability before trying
        # to shave another knot from a statistically marginal checkpoint.
        return (1, deployment_pass, -tail, -metrics["deployment_mse"])
    return (0, deployment_pass, -tail, -metrics["deployment_mse"])


def serial_args(args):
    return {key: ([str(v.resolve()) for v in value] if isinstance(value, list) else
                  str(value.resolve()) if isinstance(value, Path) else value)
            for key, value in vars(args).items()}


def synthetic_dataset_config(args):
    """Build the auditable synthetic-data contract stored in every checkpoint."""
    return dict(
        num_points=args.num_points,
        point_dim=args.point_dim,
        min_control_points=args.min_control_points,
        max_control_points=args.max_control_points,
        noise_std=args.noise_std,
        knot_min_span=args.knot_min_span,
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


@dataclass(frozen=True)
class ProposalTransferReport:
    """Exact audit trail for a proposal-only warm start."""

    copied: tuple[str, ...]
    resized: tuple[str, ...]
    retained_target: tuple[str, ...]
    skipped_mismatched: tuple[str, ...]

    @property
    def transferred_count(self) -> int:
        return len(self.copied) + len(self.resized)


def _resize_interval_queries(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor | None:
    """Linearly resample only the ordered interval-query rank axis."""
    if (
        source.ndim != 2
        or target.ndim != 2
        or source.shape[1] != target.shape[1]
    ):
        return None
    # ``interpolate`` treats the left-to-right query rank as a one-dimensional
    # signal. ``align_corners=False`` maps cell centres to cell centres, which
    # matches CandidateKnotHead's ``(rank + 0.5) / interval_count`` anchors.
    resized = F.interpolate(
        source.transpose(0, 1).unsqueeze(0),
        size=target.shape[0],
        mode="linear",
        align_corners=False,
    )
    return resized.squeeze(0).transpose(0, 1).to(
        device=target.device,
        dtype=target.dtype,
    )


def transfer_proposal_weights(model, checkpoint) -> ProposalTransferReport:
    """Warm-start proposal modules, explicitly adapting an ordered query table.

    The selector and subset decoder are never transferred.  Of all mismatched
    proposal tensors, only ``candidate_head.interval_queries`` is rank-resizable.
    Candidate anchors remain the deterministic buffers constructed for the
    target Kc, so a Kc=64 proposal checkpoint can safely initialize Kc=56.
    """
    source_objective = checkpoint.get("objective_version")
    if (
        source_objective is not None
        and source_objective != V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION
    ):
        raise ValueError(
            "initial checkpoint objective is not compatible with the v16 "
            "counterfactual proposal network"
        )

    source_config = checkpoint.get("model_config")
    target_config = model.get_config()
    # Capacity and selector/deployment fields may intentionally differ. These
    # keys define the encoder, ParameterHead and CandidateKnotHead tensors that
    # are actually transferred.
    proposal_contract_keys = (
        "point_dim",
        "degree",
        "hidden_dim",
        "encoder_layers",
        "min_parameter_gap",
        "min_knot_gap",
        "attention_heads",
        "parameter_residual_limit",
        "structure_mode",
    )
    if source_config is not None:
        mismatched = [
            key for key in proposal_contract_keys
            if key in source_config
            and key in target_config
            and source_config[key] != target_config[key]
        ]
        if mismatched:
            details = ", ".join(
                f"{key}={source_config[key]!r} (target "
                f"{target_config[key]!r})"
                for key in mismatched
            )
            raise ValueError(
                "initial checkpoint has an incompatible proposal contract: "
                + details
            )

    old_state = checkpoint["model_state_dict"]
    state = model.state_dict()
    prefixes = ("encoder.", "parameter_head.", "candidate_head.")
    required = {key for key in state if key.startswith(prefixes)}
    missing = sorted(required.difference(old_state))
    if missing:
        preview = ", ".join(missing[:3])
        suffix = "..." if len(missing) > 3 else ""
        raise ValueError(
            "initial checkpoint is missing required proposal tensors: "
            + preview
            + suffix
        )
    copied: dict[str, torch.Tensor] = {}
    resized: dict[str, torch.Tensor] = {}
    retained_target: list[str] = []
    skipped_mismatched: list[str] = []
    for key, value in old_state.items():
        if not key.startswith(prefixes) or key not in state:
            continue
        target = state[key]
        # Anchors are a deterministic function of the target candidate count.
        # Never trust or copy a checkpoint value, even when its shape happens
        # to match (the non-persistent candidate-position anchors are absent).
        if key == "candidate_head.interval_query_anchors":
            retained_target.append(key)
            continue
        if target.shape == value.shape:
            copied[key] = value
            continue
        if key == "candidate_head.interval_queries":
            adapted = _resize_interval_queries(value, target)
            if adapted is not None:
                resized[key] = adapted
                continue
        skipped_mismatched.append(key)
    transferred = {**copied, **resized}
    if not transferred:
        raise ValueError("initial checkpoint has no compatible encoder/parameter/proposal tensors")
    state.update(transferred)
    model.load_state_dict(state, strict=True)
    return ProposalTransferReport(
        copied=tuple(sorted(copied)),
        resized=tuple(sorted(resized)),
        retained_target=tuple(sorted(retained_target)),
        skipped_mismatched=tuple(sorted(skipped_mismatched)),
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
    if not args.resume and any(path.exists() for path in (output, last_path, proposal_path, history_path)):
        p.error("output artifacts already exist; choose a new output or explicitly --resume the .last.pt")
    torch.set_num_threads(args.torch_num_threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    dataset_config = synthetic_dataset_config(args)
    print("Checking real manifests and train/val/test group separation...", flush=True)
    sources, provenance = load_real_sources(args.real_manifest, num_points=args.num_points,
                                           point_dim=args.point_dim,
                                           progress=lambda message: print(message, flush=True))
    validation = ValidationCurves(
        dataset_config,
        sources,
        size=args.val_size,
        seed=args.val_seed,
        real_per_source=args.real_val_size,
        synthetic_boundary_samples=args.synthetic_boundary_val_size,
    )
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
        one_shot_selection_policy=args.one_shot_selection_policy,
        one_shot_adaptive_threshold=True,
        one_shot_safety_sigma=args.one_shot_safety_sigma,
        one_shot_safety_knots=args.one_shot_safety_knots,
        one_shot_coverage_bins=args.one_shot_coverage_bins,
        min_selected_knots=args.min_selected_knots,
        initial_keep_fraction=args.initial_keep_fraction)
    history, start_epoch, best_rank, proposal_rank, proposal_ready = [], 1, None, None, False
    complexity_scale, feasible_streak = 0.0, 0
    selection_safety_scale = 1.0
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
        expected_synthetic_contract = (
            V16_CERTIFIED_SYNTHETIC_CONTRACT
            if args.certified_minimal_source
            else "random_source_uncertified"
        )
        if (
            resume_payload.get("synthetic_data_contract")
            != expected_synthetic_contract
        ):
            p.error(
                "resume checkpoint uses an older synthetic source-range "
                "contract; start a new K=4..56 run or use --init-checkpoint "
                "for proposal-only transfer"
            )
        ignored = {"epochs", "resume", "init_checkpoint", "output", "device", "num_workers",
                   "torch_num_threads", "log_every_batches", "initial_keep_fraction"}
        previous_config = dict(resume_payload["training_config"])
        # A legacy .last.pt must continue with the exact loss that created its
        # optimizer state even though new-run CLI defaults are more aggressive.
        # Perform this migration automatically so unattended/overnight resume
        # commands do not need to know which refinement fields postdate them.
        legacy_refinement_defaults = {
            "teacher_low_count_sweep": 0,
            "synthetic_count_role": "exact",
            "synthetic_geometry_oracle_teacher": False,
            "oracle_teacher_extra_knots": 2,
            "initial_keep_fraction": 0.95,
        }
        for key, legacy_value in legacy_refinement_defaults.items():
            if key not in previous_config:
                previous_config[key] = legacy_value
                setattr(args, key, legacy_value)
                current_config[key] = legacy_value
        # This value only initializes a fresh selector. A resumed model already
        # contains the learned beta bias, so preserve the original provenance
        # instead of allowing a harmless CLI default change to rewrite the
        # saved training metadata.
        args.initial_keep_fraction = previous_config["initial_keep_fraction"]
        current_config["initial_keep_fraction"] = previous_config[
            "initial_keep_fraction"
        ]
        if Path(previous_config["output"]).resolve() != output:
            p.error("resume must use the original --output so its best/proposal artifacts remain available")
        if (resume_payload.get("stage") == "proposal"
                and args.proposal_epochs >= previous_config["proposal_epochs"]):
            ignored.add("proposal_epochs")
        changed = [k for k, v in current_config.items() if k not in ignored and previous_config.get(k) != v]
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
        if not 0 <= complexity_scale <= args.complexity_max_scale:
            p.error("resume checkpoint has an invalid complexity controller state")
        if not 0 <= selection_safety_scale <= 1:
            p.error("resume checkpoint has an invalid safety controller state")
        if start_epoch > args.epochs:
            p.error("checkpoint already completed the requested epochs")
        if not proposal_path.exists() or (resume_payload.get("stage") == "joint" and not output.exists()):
            p.error("resume requires the saved best proposal and, for joint training, the best model artifact")
    elif args.init_checkpoint:
        source_checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        try:
            transfer = transfer_proposal_weights(model, source_checkpoint)
        except (KeyError, TypeError, ValueError) as error:
            p.error(str(error))
        print(
            f"Transferred {transfer.transferred_count} proposal tensors "
            f"({len(transfer.copied)} exact, {len(transfer.resized)} rank-resized); "
            f"retained {len(transfer.retained_target)} target anchor buffer(s), "
            f"skipped {len(transfer.skipped_mismatched)} other shape mismatch(es). "
            "Selector and subset decoder start fresh.",
            flush=True,
        )
        if transfer.resized:
            print(
                "  rank-resized: " + ", ".join(transfer.resized),
                flush=True,
            )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        teacher_low_count_sweep=args.teacher_low_count_sweep,
        synthetic_count_role=args.synthetic_count_role,
        synthetic_geometry_oracle_teacher=(
            args.synthetic_geometry_oracle_teacher
        ),
        oracle_teacher_extra_knots=args.oracle_teacher_extra_knots,
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
        f"steps={args.teacher_prefix_search_steps}, exact low-K sweep through "
        f"K={args.teacher_low_count_sweep}; certified source K role="
        f"{args.synthetic_count_role}; geometry-oracle bootstrap="
        f"{args.synthetic_geometry_oracle_teacher} (+"
        f"{args.oracle_teacher_extra_knots}); certified over-count "
        f"weight={args.supervised_over_count_weight:g}, symmetric true-count "
        f"weight={args.supervised_count_weight:g}; complexity multiplier "
        f"0..{args.complexity_max_scale:g}; safety "
        f"{args.one_shot_safety_knots}+{args.one_shot_safety_sigma:g}sigma -> "
        f"{args.final_safety_knots}+{args.final_safety_sigma:g}sigma. "
        "Deployment still uses one network forward and one final refit.",
        flush=True,
    )
    print(
        "Validation checkpoint quality uses the worst data-source rate and "
        f"an explicit K={args.max_control_points - 4} boundary audit "
        f"(n={min(args.synthetic_boundary_val_size, args.val_size)}). "
        "Additional subset fits run only during training.",
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
                best_proposal["validation_metrics"].get(
                    "qualification_dense_pass_rate",
                    best_proposal["validation_metrics"]["worst_dense_pass_rate"],
                )
                >= proposal_gate
            )
            if not proposal_ready and not args.allow_infeasible_proposals:
                print(f"STOP: dense proposal did not reach the safety gate "
                      f"{proposal_gate:.1%} in every source. "
                      f"Inspect {history_path}; increase proposal training/candidate budget before simplification.", flush=True)
                return 2
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.joint_lr, weight_decay=args.weight_decay)
        train_data = MixedTrainingCurves(dataset_config, sources, size=args.train_size, seed=args.seed,
            real_fraction=args.real_fraction, epoch=epoch-1, resample=args.resample_train_each_epoch)
        loader_generator = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            generator=loader_generator, **loader_runtime,
        )
        model.train(); started = time.perf_counter(); total, samples = defaultdict(float), 0
        for step, batch in enumerate(train_loader, 1):
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
            size = len(points); samples += size
            for key, value in metrics.items():
                total[key] += float(value)*size
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
            synthetic_boundary_knot_count=args.max_control_points - 4,
        )
        train_metrics = {k: v/samples for k, v in total.items()}
        applied_complexity_scale = complexity_scale
        if stage == "joint":
            observed_pass = measured["qualification_deployment_pass_rate"]
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
            applied_complexity_scale=applied_complexity_scale,
            next_complexity_scale=complexity_scale,
            applied_selection_safety_scale=applied_safety_scale,
            next_selection_safety_scale=selection_safety_scale,
            applied_selection_safety_sigma=safety_sigma,
            applied_selection_safety_knots=safety_knots,
            feasible_streak=feasible_streak,
            seconds=time.perf_counter()-started,
        )
        history.append(entry)
        improved = False
        if stage == "proposal":
            rank = (
                measured["qualification_dense_pass_rate"],
                -measured["dense_mse"],
            )
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
        accepted = (
            stage == "joint"
            and measured["qualification_deployment_pass_rate"]
            >= args.deployment_pass_target
        )
        payload = dict(objective_version=V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
            model_config=model.get_config(), model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            optimizer_state_dict=optimizer.state_dict(), epoch=epoch, stage=stage,
            training_config=current_config, dataset_config=dataset_config, history=history,
            dataset_type=(
                (
                    "certified_synthetic_geometry_and_real_unlabeled"
                    if args.certified_minimal_source
                    else "uncertified_synthetic_and_real_unlabeled"
                )
                if sources
                else (
                    "certified_synthetic_open_cubic_bspline"
                    if args.certified_minimal_source
                    else "uncertified_synthetic_open_cubic_bspline"
                )
            ),
            real_data_provenance=provenance, validation_real_ids=validation.selected_real_ids,
            synthetic_data_contract=(
                V16_CERTIFIED_SYNTHETIC_CONTRACT
                if args.certified_minimal_source
                else "random_source_uncertified"
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
                             teacher_low_count_sweep=args.teacher_low_count_sweep,
                             synthetic_count_role=args.synthetic_count_role,
                             synthetic_geometry_oracle_teacher=(
                                 args.synthetic_geometry_oracle_teacher
                             ),
                             oracle_teacher_extra_knots=(
                                 args.oracle_teacher_extra_knots
                             ),
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
                                  "selected_knot_position_weight")}))
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
              f"qualification={measured['qualification_deployment_pass_rate']:.1%} "
              f"MSE={measured['deployment_mse']:.3e} K={measured['keep_count']:.2f} "
              f"mass={measured['keep_probability_mass']:.2f} beta={measured['adaptive_keep_threshold']:.2f} "
              f"complexity={applied_complexity_scale:.2f}->{complexity_scale:.2f} "
              f"safety={safety_knots}+{safety_sigma:.2f}sigma "
              f"scale={applied_safety_scale:.2f}->{selection_safety_scale:.2f} "
              f"target_met={accepted}", flush=True)
        boundary_n = measured["synthetic_boundary_sample_count"]
        if boundary_n:
            print(
                f"  Synthetic K={measured['synthetic_boundary_knot_count']}: "
                f"n={boundary_n}, "
                f"dense={measured['synthetic_boundary_dense_pass_rate']:.1%}, "
                "deployment="
                f"{measured['synthetic_boundary_deployment_pass_rate']:.1%}",
                flush=True,
            )
        else:
            print(
                f"  Synthetic K={measured['synthetic_boundary_knot_count']}: "
                "boundary audit unavailable (uncertified diagnostic data)",
                flush=True,
            )
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
            print(f"  {name}: dense={values['dense_pass_rate']:.1%}, deployment={values['deployment_pass_rate']:.1%}, "
                  f"MSE={values['deployment_mse']:.3e}, K={values['keep_count']:.2f}"
                  f"{count_detail}{position_detail}", flush=True)
    best = torch.load(output, map_location="cpu", weights_only=True)
    print(f"Saved best v16: {output}; quality={best['checkpoint_quality']}. "
          f"Last/resume: {last_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
