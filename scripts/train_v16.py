"""Train v16 Proposal and Joint stages from labelled synthetic curves.

The optional offline-feasible Joint target is searched once on a frozen
Proposal candidate/parameter frame. Real curves are held-out validation only.
"""
from __future__ import annotations

# Standalone entry point intentionally imports the local package after adding
# ``src`` to ``sys.path``.
# ruff: noqa: E402

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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import (
    V16_ADAPTIVE_SELECTION_REVISION,
    V16_CERTIFIED_SYNTHETIC_CONTRACT,
    V16_CHECKPOINT_SELECTION_CONTRACT,
    V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
    V16_FEASIBLE_TEACHER_CONTRACT,
    V16_FEASIBLE_TEACHER_OBJECTIVE_VERSION,
    V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION,
    V16_FORMAL_CANDIDATE_INTERNAL_KNOTS,
    V16_FORMAL_PASS_RATE,
    V16_JOINT_CHECKPOINT_QUALITY,
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
from spline_fitting.losses.v16_subset_loss import V16SubsetLoss, subset_cost
from spline_fitting.models.v16_network import V16CandidateSelectionNetwork
from spline_fitting.training.v16_feasible_teacher import (
    build_or_load_v16_feasible_teacher_cache,
)


V16_CHECKPOINT_SELECTION = V16_CHECKPOINT_SELECTION_CONTRACT


class IndexedTrainingDataset(Dataset):
    """Attach the stable row id used by offline numerical teacher labels."""

    def __init__(self, source: Dataset) -> None:
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        row = dict(self.source[index])
        row["feasible_teacher_index"] = index
        return row


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=128, help="Total proposal plus joint epochs")
    p.add_argument(
        "--proposal-epochs", type=int, default=64,
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
    p.add_argument(
        "--real-fraction", type=float, default=0.0,
        help=("Training fraction of unlabelled real curves. The formal supervised "
              "v16 protocol requires zero; real curves remain in validation."),
    )
    p.add_argument(
        "--proposal-high-k-fraction",
        type=float,
        default=0.5,
        help=("Exact fraction (up to integer rounding) of proposal-stage "
              "synthetic draws sampled from the high-K stratum; joint "
              "training always restores the original full-range distribution"),
    )
    p.add_argument(
        "--proposal-high-k-min-knots",
        type=int,
        default=None,
        help=("First internal-knot count in the proposal high-K stratum "
              "(default: min(40, maximum source K))"),
    )
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
        help=("Internal candidate capacity Kc (default: 72 internal knots, "
              "equivalent to 80 entries in a full cubic clamped knot vector). "
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
    p.add_argument("--tolerance-factor-min", type=float, default=1.0)
    p.add_argument("--tolerance-factor-max", type=float, default=1.0)
    p.add_argument(
        "--proposal-pass-target", type=float, default=V16_FORMAL_PASS_RATE,
        help=("Reporting reference only; Proposal always transitions to Joint "
              "after --proposal-epochs"),
    )
    p.add_argument(
        "--deployment-pass-target", type=float, default=V16_FORMAL_PASS_RATE,
        help=("Reporting/qualification reference only; it does not gate the "
              "curriculum or checkpoint selection"),
    )
    p.add_argument("--allow-infeasible-proposals", action="store_true",
                   help="Deprecated compatibility flag; Joint transition is always scheduled")
    p.add_argument("--policy-samples", type=int, default=4)
    p.add_argument("--counterfactual-edits", type=int, default=4)
    p.add_argument(
        "--teacher-prefix-search-steps", type=int, default=7,
        help=("Training-only ranked-prefix feasibility search depth; seven "
              "steps resolve the default Kc=72 count boundary without "
              "deployment search"),
    )
    p.add_argument(
        "--teacher-low-count-sweep", type=int, default=16,
        help=("Exactly evaluate every deployed ranked prefix up to this count "
              "during training; 0 disables the simple-curve sweep"),
    )
    p.add_argument(
        "--synthetic-count-role", choices=("exact", "upper_bound", "reference_only"),
        default="exact",
        help=("Treat certified source K as an upper bound when survivor knots "
              "may relocate; 'exact' preserves the historical v16 ablation"),
    )
    p.add_argument(
        "--joint-supervision",
        choices=("synthetic_ground_truth", "offline_feasible_teacher", "online_teacher"),
        default="synthetic_ground_truth",
        help=("Formal mode directly supervises KeepMask, count and relocation "
              "from certified synthetic labels. online_teacher is a legacy "
              "ablation and permits mixed real-data training."),
    )
    p.add_argument(
        "--feasible-teacher-cache-dir", type=Path,
        help="Required with offline_feasible_teacher; fixed Proposal labels are cached here",
    )
    p.add_argument(
        "--feasible-teacher-batch-size", type=int, default=8,
        help="Offline numerical Teacher batch; independent of Joint optimizer batch size",
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
    p.add_argument(
        "--complexity-weight", type=float, default=0.0,
        help=("Legacy online-teacher ablation only. Direct exact-K supervision "
              "uses zero because an extra free complexity penalty conflicts "
              "with the labelled knot count."),
    )
    p.add_argument("--true-parameter-weight", type=float, default=0.1)
    p.add_argument("--proposal-knot-coverage-weight", type=float, default=1.0)
    p.add_argument(
        "--proposal-knot-assignment-weight",
        type=float,
        default=1.0,
        help=("Ordered one-to-one proposal/ground-truth knot supervision; "
              "kept separate from recall-direction proposal coverage"),
    )
    p.add_argument(
        "--proposal-multiscale-recall-weight", type=float, default=0.25,
        help=("CVaR-aware candidate recall supervision at parameter tolerances "
              "0.0025, 0.005 and 0.01"),
    )
    p.add_argument("--selected-knot-position-weight", type=float, default=1.0)
    p.add_argument("--keep-dice-weight", type=float, default=0.5)
    p.add_argument("--keep-cdf-weight", type=float, default=0.25)
    p.add_argument("--parameter-gap-weight", type=float, default=0.05)
    p.add_argument("--parameter-bias-weight", type=float, default=0.1)
    p.add_argument(
        "--fine-teacher-weight", type=float, default=0.5,
        help=("Use certified per-knot deletion MSE to emphasize critical "
              "positive Keep slots; adds no online spline solves"),
    )
    p.add_argument("--fine-teacher-ranking-weight", type=float, default=0.25)
    p.add_argument("--fine-teacher-temperature", type=float, default=0.5)
    p.add_argument("--keep-fuzzy-negative-radius", type=float, default=0.01)
    p.add_argument("--keep-fuzzy-negative-floor", type=float, default=0.1)
    p.add_argument(
        "--proposal-parameter-warp-gradient-scale", type=float, default=0.0,
        help="Cross-task gradient from proposal knot matching into ParameterHead",
    )
    p.add_argument(
        "--joint-parameter-warp-gradient-scale", type=float, default=0.1,
        help="Bounded cross-task gradient from relocated knot matching into ParameterHead",
    )
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
    p.add_argument(
        "--one-shot-coverage-bins",
        type=int,
        default=0,
        help=("Legacy deployment coverage anchors. Direct synthetic KeepMask "
              "supervision requires zero because forced bin anchors need not "
              "belong to the exact labelled subset"),
    )
    p.add_argument("--min-selected-knots", type=int, default=4)
    p.add_argument(
        "--initial-keep-fraction", type=float, default=30 / 72,
        help=("Initial selector probability mass as a fraction of Kc; proposal "
              "training does not update the selector. The default is 30/72, "
              "matching the mean K of the default synthetic K=4..56 range"),
    )
    p.add_argument(
        "--relocation-blend", type=float, default=0.03,
        help=("Initial uniform-rank relocation blend. A small nonzero value "
              "keeps survivor relocation trainable from the first Joint step"),
    )
    p.add_argument(
        "--complexity-ramp-epochs", type=int, default=10,
        help=("Joint-stage epochs for a deterministic linear ramp from zero to "
              "complexity-max-scale"),
    )
    p.add_argument(
        "--complexity-max-scale", type=float, default=4.0,
        help="Final deterministic multiplier on the complexity loss",
    )
    p.add_argument(
        "--complexity-pass-margin", type=float, default=0.02,
        help="Deprecated compatibility field; aggregate pass no longer controls the curriculum",
    )
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--joint-lr", type=float, default=5e-5)
    p.add_argument(
        "--selector-warmup-epochs",
        type=int,
        default=8,
        help=("First Joint epochs with Encoder, ParameterHead and CandidateHead "
              "frozen while Selector and subset decoder learn stable labels"),
    )
    p.add_argument(
        "--selector-lr",
        type=float,
        default=2e-4,
        help="Joint learning rate for selection blocks, KeepHead and adaptive beta",
    )
    p.add_argument(
        "--proposal-joint-lr",
        type=float,
        default=1e-5,
        help="Post-warmup Joint learning rate for Encoder and CandidateHead",
    )
    p.add_argument(
        "--parameter-joint-lr",
        type=float,
        default=5e-5,
        help="Post-warmup Joint learning rate for ParameterHead",
    )
    p.add_argument(
        "--decoder-joint-lr",
        type=float,
        default=5e-5,
        help="Joint learning rate for subset parameter/relocation decoder",
    )
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
        args.candidate_knots = V16_FORMAL_CANDIDATE_INTERNAL_KNOTS
    if not 1 <= args.proposal_epochs < args.epochs:
        raise ValueError("require 1 <= proposal-epochs < epochs")
    for key in ("train_size", "val_size", "real_val_size", "batch_size",
                "feasible_teacher_batch_size", "hidden_dim",
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
        "selector_warmup_epochs",
    ):
        value = getattr(args, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    if not 4 <= args.min_control_points <= args.max_control_points:
        raise ValueError("control-point range must satisfy 4 <= min <= max")
    minimum_source_knots = args.min_control_points - 4
    maximum_source_knots = args.max_control_points - 4
    if args.proposal_high_k_min_knots is None:
        args.proposal_high_k_min_knots = min(40, maximum_source_knots)
    if (
        isinstance(args.proposal_high_k_min_knots, bool)
        or not isinstance(args.proposal_high_k_min_knots, int)
        or not minimum_source_knots
        <= args.proposal_high_k_min_knots
        <= maximum_source_knots
    ):
        raise ValueError(
            "proposal-high-k-min-knots must lie inside the synthetic "
            "internal-knot range"
        )
    if (
        args.proposal_high_k_fraction < 1.0
        and args.proposal_high_k_fraction > 0.0
        and args.proposal_high_k_min_knots <= minimum_source_knots
    ):
        raise ValueError(
            "a partial proposal high-K mixture requires a nonempty low-K range"
        )
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
        "selector_lr", "decoder_joint_lr",
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
        "proposal_knot_assignment_weight",
        "proposal_multiscale_recall_weight",
        "selected_knot_position_weight", "keep_dice_weight", "keep_cdf_weight",
        "parameter_gap_weight", "parameter_bias_weight", "fine_teacher_weight",
        "fine_teacher_ranking_weight", "fine_teacher_temperature",
        "keep_fuzzy_negative_radius", "knot_position_beta",
        "complexity_max_scale", "proposal_joint_lr", "parameter_joint_lr",
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
    if args.fine_teacher_temperature <= 0:
        raise ValueError("fine-teacher-temperature must be positive")
    if args.keep_fuzzy_negative_radius <= 0:
        raise ValueError("keep-fuzzy-negative-radius must be positive")
    if not 0 <= args.keep_fuzzy_negative_floor <= 1:
        raise ValueError("keep-fuzzy-negative-floor must lie in [0,1]")
    for key in (
        "proposal_parameter_warp_gradient_scale",
        "joint_parameter_warp_gradient_scale",
    ):
        if not 0 <= getattr(args, key) <= 1:
            raise ValueError(f"{key} must lie in [0,1]")
    if args.final_safety_sigma > args.one_shot_safety_sigma:
        raise ValueError("final-safety-sigma cannot exceed one-shot-safety-sigma")
    if args.final_safety_knots > args.one_shot_safety_knots:
        raise ValueError("final-safety-knots cannot exceed one-shot-safety-knots")
    for key in (
        "real_fraction",
        "proposal_high_k_fraction",
        "proposal_pass_target",
        "deployment_pass_target",
    ):
        if not math.isfinite(getattr(args, key)) or not 0 <= getattr(args, key) <= 1:
            raise ValueError(f"{key} must lie in [0,1]")
    if args.joint_supervision in {"synthetic_ground_truth", "offline_feasible_teacher"}:
        if args.real_fraction != 0.0:
            raise ValueError(
                "labelled Joint requires --real-fraction 0; "
                "real manifests are validation/test data only"
            )
        if not args.certified_minimal_source:
            raise ValueError(
                "labelled Joint requires --certified-minimal-source"
            )
        if args.joint_supervision == "synthetic_ground_truth" and args.synthetic_count_role != "exact":
            raise ValueError(
                "synthetic_ground_truth Joint requires --synthetic-count-role exact"
            )
        if args.synthetic_geometry_oracle_teacher:
            raise ValueError(
                "synthetic_ground_truth Joint does not use an oracle teacher; "
                "remove --synthetic-geometry-oracle-teacher"
            )
        if args.one_shot_coverage_bins != 0:
            raise ValueError(
                "synthetic_ground_truth requires --one-shot-coverage-bins 0; "
                "forced spatial anchors can conflict with the exact KeepMask label"
            )
        if args.tolerance_factor_min != 1.0 or args.tolerance_factor_max != 1.0:
            raise ValueError(
                "synthetic_ground_truth labels require fixed training tolerance; "
                "set --tolerance-factor-min 1 --tolerance-factor-max 1"
            )
        if args.complexity_weight != 0.0:
            raise ValueError(
                ("synthetic_ground_truth uses the exact labelled knot count; "
                 if args.joint_supervision == "synthetic_ground_truth"
                 else "offline feasible Teacher supervises its own knot count; ")
                + "set --complexity-weight 0 to avoid a conflicting objective"
            )
    if args.joint_supervision == "offline_feasible_teacher":
        if args.feasible_teacher_cache_dir is None:
            raise ValueError("offline_feasible_teacher requires --feasible-teacher-cache-dir")
        if args.resample_train_each_epoch:
            raise ValueError(
                "offline teacher labels require fixed training samples; "
                "set --no-resample-train-each-epoch"
            )
        if args.synthetic_count_role != "reference_only":
            raise ValueError(
                "source K is diagnostic, not a teacher upper bound; "
                "set --synthetic-count-role reference_only"
            )
        if args.proposal_joint_lr != 0 or args.parameter_joint_lr != 0:
            raise ValueError(
                "fixed Proposal Teacher requires --proposal-joint-lr 0 "
                "and --parameter-joint-lr 0"
            )
    elif args.feasible_teacher_cache_dir is not None:
        raise ValueError("--feasible-teacher-cache-dir requires offline_feasible_teacher")
    if args.tolerance_factor_min > args.tolerance_factor_max:
        raise ValueError("tolerance-factor-min must be <= tolerance-factor-max")
    if not math.isfinite(args.relocation_blend) or not 0 <= args.relocation_blend <= 1:
        raise ValueError("relocation-blend must lie in [0,1]")
    if (
        not math.isfinite(args.initial_keep_fraction)
        or not 0.0 < args.initial_keep_fraction < 1.0
    ):
        raise ValueError("initial-keep-fraction must lie strictly inside (0,1)")
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


JOINT_PARAMETER_GROUP_NAMES = ("proposal", "parameter", "selector", "decoder")


def joint_parameter_groups(model):
    """Partition every v16 parameter exactly once for Joint optimization."""
    groups = {name: [] for name in JOINT_PARAMETER_GROUP_NAMES}
    grouped_names = {name: [] for name in JOINT_PARAMETER_GROUP_NAMES}
    selector_prefixes = (
        "tolerance_embedding.",
        "coverage_embedding.",
        "selection_blocks.",
        "keep_head.",
        "adaptive_threshold_norm.",
        "adaptive_threshold_head.",
    )
    decoder_prefixes = (
        "subset_geometry.",
        "survivor_attention.",
        "parameter_attention.",
        "survivor_norm.",
        "parameter_norm.",
        "parameter_update.",
        "relocation_update.",
    )
    for name, parameter in model.named_parameters():
        if name.startswith(("encoder.", "candidate_head.")):
            group = "proposal"
        elif name.startswith("parameter_head."):
            group = "parameter"
        elif name.startswith(selector_prefixes):
            group = "selector"
        elif name == "relocation_blend_logit" or name.startswith(decoder_prefixes):
            group = "decoder"
        else:
            raise ValueError(f"unclassified v16 Joint parameter: {name}")
        groups[group].append(parameter)
        grouped_names[group].append(name)
    empty = [name for name, values in groups.items() if not values]
    if empty:
        raise ValueError(f"empty v16 Joint parameter group(s): {', '.join(empty)}")
    identifiers = [id(parameter) for values in groups.values() for parameter in values]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("v16 Joint parameter groups overlap")
    if len(identifiers) != sum(1 for _ in model.parameters()):
        raise RuntimeError("v16 Joint parameter grouping lost model parameters")
    return groups, grouped_names


def joint_training_phase(args, epoch):
    """Return the explicit optimization phase for one scheduled epoch."""
    if epoch <= args.proposal_epochs:
        return "proposal"
    joint_epoch = epoch - args.proposal_epochs
    if joint_epoch <= args.selector_warmup_epochs:
        return "selector_warmup"
    return "joint_finetune"


def configure_joint_trainability(model, *, selector_warmup, fixed_proposal=False):
    """Freeze proposal geometry only during the stable-label selector warmup."""
    groups, grouped_names = joint_parameter_groups(model)
    for group_name, parameters in groups.items():
        trainable = (
            group_name in {"selector", "decoder"}
            if fixed_proposal else
            not selector_warmup or group_name in {"selector", "decoder"}
        )
        for parameter in parameters:
            parameter.requires_grad_(trainable)
    return {
        name: tuple(grouped_names[name])
        for name in JOINT_PARAMETER_GROUP_NAMES
    }


def _configured_joint_learning_rates(args, *, selector_warmup):
    return {
        "proposal": 0.0 if selector_warmup else float(args.proposal_joint_lr),
        "parameter": 0.0 if selector_warmup else float(args.parameter_joint_lr),
        "selector": float(args.selector_lr),
        "decoder": float(args.decoder_joint_lr),
    }


def build_joint_optimizer(model, args, *, selector_warmup):
    """Build auditable named Joint groups while retaining frozen parameters."""
    groups, _ = joint_parameter_groups(model)
    learning_rates = _configured_joint_learning_rates(
        args, selector_warmup=selector_warmup,
    )
    parameter_groups = [
        {
            "params": groups[name],
            "lr": learning_rates[name],
            "name": name,
        }
        for name in JOINT_PARAMETER_GROUP_NAMES
    ]
    return torch.optim.AdamW(
        parameter_groups,
        lr=args.joint_lr,
        weight_decay=args.weight_decay,
    )


def set_joint_optimizer_learning_rates(optimizer, args, *, selector_warmup):
    """Apply warmup/post-warmup LRs without discarding Adam moments."""
    expected = _configured_joint_learning_rates(
        args, selector_warmup=selector_warmup,
    )
    names = [group.get("name") for group in optimizer.param_groups]
    if set(names) != set(JOINT_PARAMETER_GROUP_NAMES) or len(names) != len(set(names)):
        raise ValueError("Joint optimizer does not contain the four named groups")
    for group in optimizer.param_groups:
        group["lr"] = expected[group["name"]]


def optimizer_learning_rates(optimizer):
    """Serialize the effective LR of every optimizer group."""
    result = {}
    for index, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group_{index}")
        if name in result:
            raise ValueError(f"duplicate optimizer group name: {name}")
        result[name] = float(group["lr"])
    return result


def clip_joint_gradients(model, max_norm):
    """Clip the four Joint tasks independently and return pre-clip norms."""
    groups, _ = joint_parameter_groups(model)
    norms = {}
    for name, parameters in groups.items():
        active = [parameter for parameter in parameters if parameter.grad is not None]
        if active:
            norm = torch.nn.utils.clip_grad_norm_(
                active, max_norm, error_if_nonfinite=True,
            )
            norms[name] = float(norm)
        else:
            norms[name] = 0.0
    return norms


def selection_safety(args, scale):
    """Interpolate the deterministic Joint-stage one-shot safety reserve."""
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
    """Require completed warmup, final reserve and a mature curriculum."""
    return bool(
        stage == "joint"
        and joint_training_phase(args, epoch) == "joint_finetune"
        and epoch > args.proposal_epochs + args.selector_warmup_epochs
        and epoch - args.proposal_epochs
        >= max(args.complexity_ramp_epochs, args.safety_anneal_epochs)
        and applied_safety_scale <= 1e-12
        and applied_complexity_scale >= args.complexity_max_scale - 1e-12
    )


def simplification_schedule(args, *, epoch, stage):
    """Return deterministic complexity/safety scales for one training epoch.

    The first Joint epoch retains the complete initial safety reserve and zero
    complexity pressure. Each scale then advances linearly by Joint epoch,
    independently of any aggregate validation pass rate. The loss itself still
    activates complexity only for samples satisfying their own MSE threshold.
    """
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("curriculum epoch must be a positive integer")
    if stage not in {"proposal", "joint"}:
        raise ValueError("curriculum stage must be 'proposal' or 'joint'")
    if stage == "proposal":
        return 0.0, 1.0
    joint_epoch = epoch - args.proposal_epochs
    if joint_epoch < 1:
        raise ValueError("joint stage cannot precede the configured proposal stage")

    def progress(duration):
        if duration <= 1:
            return 1.0
        return min(1.0, max(0.0, (joint_epoch - 1) / (duration - 1)))

    complexity_scale = args.complexity_max_scale * progress(
        args.complexity_ramp_epochs
    )
    safety_scale = 1.0 - progress(args.safety_anneal_epochs)
    return float(complexity_scale), float(safety_scale)


def summarize(
    rows, tolerance, *, candidate_capacity, synthetic_boundary_knot_count=None,
):
    if (
        isinstance(candidate_capacity, bool)
        or not isinstance(candidate_capacity, int)
        or candidate_capacity < 1
    ):
        raise ValueError("candidate_capacity must be a positive integer")
    groups = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)
    def group(values):
        mse_values = torch.tensor(
            [r["mse"] for r in values], dtype=torch.float64,
        )
        dense_mse_values = torch.tensor(
            [r["dense_mse"] for r in values], dtype=torch.float64,
        )
        retained_counts = torch.tensor(
            [r["k"] for r in values], dtype=torch.float64,
        )
        tolerance_values = torch.full_like(mse_values, float(tolerance))
        deployment_cost = subset_cost(
            mse_values, retained_counts, tolerance_values, candidate_capacity,
        )
        dense_cost = subset_cost(
            dense_mse_values,
            torch.full_like(retained_counts, float(candidate_capacity)),
            tolerance_values,
            candidate_capacity,
        )
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
            deployment_subset_cost=float(deployment_cost.mean()),
            dense_subset_cost=float(dense_cost.mean()),
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
        result["synthetic_knot_match_recall"] = synthetic["knot_match_recall"]
        result["synthetic_knot_matched_mae"] = synthetic["knot_matched_mae"]
    if "parameter_rmse" in synthetic:
        result["synthetic_parameter_rmse"] = synthetic["parameter_rmse"]
    # The upper end of the source-count range is the hardest labelled stratum
    # in the formal K=4..56 protocol.  Kc=72 now leaves 16 redundant proposal
    # slots there, but an aggregate Synthetic rate can still hide a complete
    # failure on K=56, so retain an explicit boundary audit for diagnostics and
    # formal reporting.
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
                # learning. Its selected refit is irrelevant to the Proposal
                # objective and diagnostics, so avoid a duplicate solve.
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
        candidate_capacity=int(model.max_internal_knots),
        synthetic_boundary_knot_count=synthetic_boundary_knot_count,
    )


def checkpoint_selection_snapshot(metrics, *, stage, candidate_capacity, tolerance):
    """Expose every quantity used by checkpoint selection.

    ``subset_cost`` is averaged over validation curves: feasible curves trade
    retained K against a bounded MSE tie-break, while infeasible curves pay an
    unbounded logarithmic MSE penalty. Aggregate pass rates remain diagnostics,
    not a feasibility gate for training or model selection.
    """
    if stage not in {"proposal", "joint"}:
        raise ValueError("checkpoint stage must be 'proposal' or 'joint'")
    prefix = "dense" if stage == "proposal" else "deployment"
    pass_key = (
        "qualification_dense_pass_rate"
        if stage == "proposal" else "qualification_deployment_pass_rate"
    )
    snapshot = {
        "strategy": V16_CHECKPOINT_SELECTION,
        "stage": stage,
        "mean_subset_cost": float(metrics[f"{prefix}_subset_cost"]),
        "selection_score": -float(metrics[f"{prefix}_subset_cost"]),
        "pass_rate_diagnostic": float(metrics[pass_key]),
        "mean_mse": float(metrics[f"{prefix}_mse"]),
        "mse_p95": float(
            metrics.get(f"{prefix}_mse_p95", metrics[f"{prefix}_mse"])
        ),
        "mean_retained_knots": float(
            candidate_capacity if stage == "proposal" else metrics["keep_count"]
        ),
        "candidate_capacity": int(candidate_capacity),
        "per_curve_mse_tolerance": float(tolerance),
    }
    numeric = [
        value for key, value in snapshot.items()
        if key not in {"strategy", "stage"}
    ]
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("checkpoint selection metrics must be finite")
    return snapshot


def checkpoint_rank(metrics, *, simplification_ready=True):
    """Pareto-consistent rank led by mean per-curve fit/complexity cost."""
    deployment_pass = metrics.get(
        "qualification_deployment_pass_rate",
        metrics["worst_deployment_pass_rate"],
    )
    soft_cost = metrics["deployment_subset_cost"]
    tail = metrics.get("deployment_mse_p95", metrics["deployment_mse"])
    values = (
        soft_cost, deployment_pass, tail,
        metrics["deployment_mse"], metrics["keep_count"],
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("checkpoint ranking metrics must be finite")
    # Positive-weight subset_cost is monotone in feasibility, error and K on
    # every curve, so a mature checkpoint dominated on all three objectives
    # cannot outrank its dominator. Curriculum maturity comes first only to
    # guarantee that the saved deployment uses the completed deterministic
    # reserve/ramp contract; it is an epoch-state check, never a pass-rate gate.
    return (
        int(bool(simplification_ready)),
        -float(soft_cost),
        float(deployment_pass),
        -float(tail),
        -float(metrics["deployment_mse"]),
        -float(metrics["keep_count"]),
    )


def proposal_checkpoint_rank(metrics):
    """Rank Proposal checkpoints without a saturated boundary-pass gate.

    ``qualification_dense_pass_rate`` is the minimum of every source and the
    fixed maximum-K boundary audit.  It can therefore remain exactly zero for
    an entire Proposal stage, after which the former rank accidentally let a
    marginal recall@0.01 fluctuation select a much worse dense initializer.
    Continuous dense subset cost now leads, source/aggregate pass rates protect
    feasibility, and continuous matched-position error precedes the looser
    tolerance recall diagnostic.
    """
    worst_pass_rate = metrics.get(
        "worst_dense_pass_rate", metrics["qualification_dense_pass_rate"]
    )
    aggregate_pass_rate = metrics.get("dense_pass_rate", worst_pass_rate)
    dense_mse = metrics["dense_mse"]
    dense_cost = metrics["dense_subset_cost"]
    recall = metrics.get("synthetic_knot_match_recall", 0.0)
    knot_f1 = metrics.get("synthetic_knot_match_f1", 0.0)
    matched_mae = metrics.get("synthetic_knot_matched_mae")
    parameter_rmse = metrics.get("synthetic_parameter_rmse")
    matched_mae = 1.0 if matched_mae is None else matched_mae
    parameter_rmse = 1.0 if parameter_rmse is None else parameter_rmse
    values = (
        worst_pass_rate,
        aggregate_pass_rate,
        dense_mse,
        dense_cost,
        recall,
        knot_f1,
        matched_mae,
        parameter_rmse,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("proposal checkpoint ranking metrics must be finite")
    return (
        -float(dense_cost),
        float(worst_pass_rate),
        float(aggregate_pass_rate),
        -float(matched_mae),
        -float(parameter_rmse),
        -float(dense_mse),
        float(knot_f1),
        float(recall),
    )


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
    # Windows can briefly retain a read handle after loading a resume/best
    # checkpoint (and antivirus scanners may do the same).  The replacement is
    # still atomic; retry only the transient sharing violation instead of
    # turning a completed epoch into a failed run.
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (2**attempt))


def saved_checkpoint_rank(
    path,
    *,
    rank_key,
    objective_version,
    architecture_revision,
    simplification_contract,
    expected_output,
):
    """Read and validate a separately committed best-checkpoint rank.

    Best and resume artifacts cannot be replaced as one filesystem
    transaction. Saving best first prevents ``.last.pt`` from claiming a rank
    that was never materialized; this reader handles the inverse crash window
    (best committed, last still one epoch behind) by protecting the newer rank
    when training resumes.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("objective_version") != objective_version:
        raise ValueError(f"{path.name} objective does not match the resume run")
    if payload.get("architecture_revision") != architecture_revision:
        raise ValueError(f"{path.name} architecture revision does not match")
    if payload.get("simplification_contract") != simplification_contract:
        raise ValueError(f"{path.name} simplification contract does not match")
    artifact_output = payload.get("training_config", {}).get("output")
    if artifact_output is None or Path(artifact_output).resolve() != expected_output:
        raise ValueError(f"{path.name} belongs to a different output run")
    rank = payload.get(rank_key)
    if not isinstance(rank, (tuple, list)) or not rank:
        raise ValueError(f"{path.name} has no valid {rank_key}")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in rank
    ):
        raise ValueError(f"{path.name} contains a non-finite {rank_key}")
    return tuple(rank)


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
        and source_objective not in {
            V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
            V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION,
        }
    ):
        raise ValueError(
            "initial checkpoint objective is not compatible with the v16 "
            "candidate proposal network"
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
    run_objective_version = {
        "synthetic_ground_truth": V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION,
        "offline_feasible_teacher": V16_FEASIBLE_TEACHER_OBJECTIVE_VERSION,
        "online_teacher": V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
    }[args.joint_supervision]
    run_simplification_contract = (
        V16_FEASIBLE_TEACHER_CONTRACT
        if args.joint_supervision == "offline_feasible_teacher"
        else V16_SIMPLIFICATION_CONTRACT
    )
    output = args.output.resolve()
    last_path = output.with_name(output.stem + ".last.pt")
    proposal_path = output.with_name(output.stem + ".proposal.pt")
    proposal_final_path = output.with_name(output.stem + ".proposal.final.pt")
    history_path = output.with_suffix(".history.json")
    artifacts = (
        output,
        last_path,
        proposal_path,
        proposal_final_path,
        history_path,
    )
    if not args.resume and any(path.exists() for path in artifacts):
        p.error(
            "output artifacts already exist; choose a new output or explicitly "
            "--resume the .last.pt"
        )
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
    reporting_target_streak = 0
    resume_payload = None
    legacy_joint_optimizer_contract = False
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        if resume_payload.get("objective_version") != run_objective_version:
            p.error(
                "resume objective does not match --joint-supervision; use "
                "--init-checkpoint for proposal-only transfer"
            )
        if resume_payload.get("architecture_revision") != V16_ADAPTIVE_SELECTION_REVISION:
            p.error(
                "resume checkpoint uses an older v16 selection revision; "
                "start a new run or use --init-checkpoint for proposal transfer"
            )
        if resume_payload.get("simplification_contract") != run_simplification_contract:
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
                   "feasible_teacher_batch_size",
                   "torch_num_threads", "log_every_batches", "initial_keep_fraction"}
        previous_config = dict(resume_payload["training_config"])
        legacy_named_groups_missing = "selector_warmup_epochs" not in previous_config
        legacy_joint_optimizer_contract = (
            resume_payload.get("stage") == "joint" and legacy_named_groups_missing
        )
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
        # A checkpoint already inside the historical one-group Joint optimizer
        # must preserve that exact layout so Adam moments remain loadable.  A
        # Proposal-stage checkpoint has no Joint moments yet and may safely enter
        # the new warmup/grouped-LR schedule using the current CLI values.
        legacy_joint_lr = float(previous_config.get("joint_lr", args.joint_lr))
        if legacy_joint_optimizer_contract:
            legacy_refinement_defaults.update(
                selector_warmup_epochs=0,
                selector_lr=legacy_joint_lr,
                proposal_joint_lr=legacy_joint_lr,
                parameter_joint_lr=legacy_joint_lr,
                decoder_joint_lr=legacy_joint_lr,
            )
        elif legacy_named_groups_missing:
            legacy_refinement_defaults.update(
                selector_warmup_epochs=args.selector_warmup_epochs,
                selector_lr=args.selector_lr,
                proposal_joint_lr=args.proposal_joint_lr,
                parameter_joint_lr=args.parameter_joint_lr,
                decoder_joint_lr=args.decoder_joint_lr,
            )
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
        selection_compatible = (
            resume_payload.get("checkpoint_selection")
            == V16_CHECKPOINT_SELECTION
        )
        best_rank = (
            resume_payload.get("best_joint_rank")
            if selection_compatible else None
        )
        proposal_rank = (
            resume_payload.get("best_proposal_rank")
            if selection_compatible else None
        )
        proposal_ready = bool(resume_payload.get("proposal_ready", False))
        reporting_target_streak = int(
            resume_payload.get(
                "reporting_target_streak",
                resume_payload.get("feasible_streak", 0),
            )
        )
        if reporting_target_streak < 0:
            p.error("resume checkpoint has an invalid reporting streak")
        if start_epoch > args.epochs:
            p.error("checkpoint already completed the requested epochs")
        if not proposal_path.exists() or (resume_payload.get("stage") == "joint" and not output.exists()):
            p.error("resume requires the saved best proposal and, for joint training, the best model artifact")
        try:
            committed_proposal_rank = saved_checkpoint_rank(
                proposal_path,
                rank_key="best_proposal_rank",
                objective_version=run_objective_version,
                architecture_revision=V16_ADAPTIVE_SELECTION_REVISION,
                simplification_contract=run_simplification_contract,
                expected_output=output,
            )
            if (
                proposal_rank is None
                or committed_proposal_rank > tuple(proposal_rank)
            ):
                proposal_rank = committed_proposal_rank
            if resume_payload.get("stage") == "joint":
                committed_joint_rank = saved_checkpoint_rank(
                    output,
                    rank_key="best_joint_rank",
                    objective_version=run_objective_version,
                    architecture_revision=V16_ADAPTIVE_SELECTION_REVISION,
                    simplification_contract=run_simplification_contract,
                    expected_output=output,
                )
                if best_rank is None or committed_joint_rank > tuple(best_rank):
                    best_rank = committed_joint_rank
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
            p.error(f"resume best-checkpoint consistency check failed: {error}")
        if resume_payload.get("stage") == "proposal":
            # A process can be interrupted after .last.pt is replaced but just
            # before the independent final-Proposal replacement.  Repair both a
            # legacy missing artifact and this one-save lag from the authoritative
            # resume payload; never silently overwrite a genuinely newer final.
            resume_epoch = int(resume_payload["epoch"])
            final_epoch = None
            if proposal_final_path.exists():
                try:
                    final_payload = torch.load(
                        proposal_final_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    final_epoch = int(final_payload["epoch"])
                except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
                    p.error(f"cannot read Proposal final artifact: {error}")
                if final_epoch > resume_epoch:
                    p.error(
                        "Proposal final artifact is newer than the requested "
                        "resume checkpoint"
                    )
            if final_epoch is None or final_epoch < resume_epoch:
                proposal_final_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_save(resume_payload, proposal_final_path)
    elif args.init_checkpoint:
        source_checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        if args.joint_supervision in {"synthetic_ground_truth", "offline_feasible_teacher"}:
            source_training = source_checkpoint.get("training_config", {})
            source_real_fraction = source_training.get("real_fraction")
            if (
                isinstance(source_real_fraction, bool)
                or not isinstance(source_real_fraction, (int, float))
                or float(source_real_fraction) != 0.0
            ):
                p.error(
                    "formal synthetic-only training can initialize only from a "
                    "proposal checkpoint whose training_config.real_fraction is 0"
                )
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
    proposal_initializer_path_used = None
    proposal_initializer_epoch = None
    optimizer_regime = "proposal_single_group"
    if resume_payload and resume_payload.get("stage") == "joint":
        saved_optimizer = resume_payload["optimizer_state_dict"]
        if len(saved_optimizer.get("param_groups", ())) == len(
            JOINT_PARAMETER_GROUP_NAMES
        ):
            resumed_phase = joint_training_phase(
                args, int(resume_payload["epoch"]),
            )
            selector_warmup = resumed_phase == "selector_warmup"
            configure_joint_trainability(
                model, selector_warmup=selector_warmup,
                fixed_proposal=args.joint_supervision == "offline_feasible_teacher",
            )
            optimizer = build_joint_optimizer(
                model, args, selector_warmup=selector_warmup,
            )
            optimizer_regime = "joint_named_groups"
        else:
            # Historical Joint checkpoints used one AdamW group.  Preserve its
            # exact layout so load_state_dict can restore moments losslessly.
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.joint_lr,
                weight_decay=args.weight_decay,
            )
            optimizer_regime = "legacy_joint_single_group"
        proposal_initializer_path_used = resume_payload.get(
            "proposal_initializer_path"
        )
        proposal_initializer_epoch = resume_payload.get(
            "proposal_initializer_epoch"
        )
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
    if resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        torch.set_rng_state(resume_payload["rng_state"])
        if device.type == "cuda" and resume_payload.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state"])
    loss_options = dict(
        mse_tolerance=args.mse_tolerance,
        policy_samples=args.policy_samples,
        counterfactual_edits=args.counterfactual_edits,
        joint_supervision=args.joint_supervision,
        ranked_prefix_teacher=args.joint_supervision == "online_teacher",
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
        proposal_knot_assignment_weight=args.proposal_knot_assignment_weight,
        proposal_multiscale_recall_weight=(
            args.proposal_multiscale_recall_weight
        ),
        selected_knot_position_weight=args.selected_knot_position_weight,
        keep_dice_weight=args.keep_dice_weight,
        keep_cdf_weight=args.keep_cdf_weight,
        parameter_gap_weight=args.parameter_gap_weight,
        parameter_bias_weight=args.parameter_bias_weight,
        fine_teacher_weight=args.fine_teacher_weight,
        fine_teacher_ranking_weight=args.fine_teacher_ranking_weight,
        fine_teacher_temperature=args.fine_teacher_temperature,
        keep_fuzzy_negative_radius=args.keep_fuzzy_negative_radius,
        keep_fuzzy_negative_floor=args.keep_fuzzy_negative_floor,
        proposal_parameter_warp_gradient_scale=(
            args.proposal_parameter_warp_gradient_scale
        ),
        joint_parameter_warp_gradient_scale=(
            args.joint_parameter_warp_gradient_scale
        ),
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
    if args.joint_supervision == "synthetic_ground_truth":
        print(
            "Joint supervision: certified synthetic ground truth only; ordered "
            "one-to-one candidate assignment directly labels KeepMask, exact K, "
            "parameters and survivor relocation. Certified per-knot deletion "
            "MSE provides fine-grained criticality, multi-scale recall and "
            "log-gap/bias losses refine Proposal/Parameter learning. Online "
            "prefix/counterfactual Teacher is disabled. Real manifests are "
            "validation/test only. "
            "Deployment uses one network forward and one final refit.",
            flush=True,
        )
    elif args.joint_supervision == "offline_feasible_teacher":
        print(
            "Joint supervision: certified synthetic geometry labels plus "
            "one offline Hard-RMS feasible-subset Teacher on the frozen "
            "Proposal parameter/candidate frame. Source K is a reference only. "
            "The Teacher mask/count/risk supervise Selector and subset decoder; "
            "no subset search runs in Joint steps. Proposal and ParameterHead "
            "remain frozen for cache validity. Raw deployment is one-shot; "
            "MSE verification/repair is a separate numerical method.",
            flush=True,
        )
    else:
        print(
            "LEGACY Joint supervision: online ranked-prefix/counterfactual "
            f"teacher steps={args.teacher_prefix_search_steps}, low-K sweep="
            f"{args.teacher_low_count_sweep}. This mode is not eligible for "
            "the current formal supervised protocol.",
            flush=True,
        )
    print(
        "Checkpoint selection minimizes mean per-curve subset cost: feasible "
        "curves trade K against a bounded MSE tie-break; infeasible curves use "
        "a logarithmic MSE penalty. Aggregate pass rates and the explicit "
        f"K={args.max_control_points - 4} boundary audit "
        f"(n={min(args.synthetic_boundary_val_size, args.val_size)}) are "
        "reported but never gate training. Labelled Joint performs "
        "only dense, deployed-mask and labelled-mask fits per batch.",
        flush=True,
    )
    if args.proposal_high_k_fraction > 0.0:
        proposal_sampling = (
            f"{args.proposal_high_k_fraction:.1%} from K>="
            f"{args.proposal_high_k_min_knots}"
        )
        if args.proposal_high_k_fraction < 1.0:
            proposal_sampling += (
                f"; the remainder uses K={args.min_control_points - 4}.."
                f"{args.proposal_high_k_min_knots - 1}"
            )
    else:
        proposal_sampling = "disabled (original full-range distribution)"
    print(
        "Proposal-stage synthetic stratification: "
        + proposal_sampling
        + ". Joint training always uses the original "
        f"K={args.min_control_points - 4}.."
        f"{args.max_control_points - 4} distribution.",
        flush=True,
    )
    feasible_teacher_cache = None
    for epoch in range(start_epoch, args.epochs + 1):
        stage = "proposal" if epoch <= args.proposal_epochs else "joint"
        training_phase = joint_training_phase(args, epoch)
        applied_complexity_scale, applied_safety_scale = simplification_schedule(
            args, epoch=epoch, stage=stage,
        )
        next_epoch = epoch + 1
        next_stage = (
            "proposal" if next_epoch <= args.proposal_epochs else "joint"
        )
        next_complexity_scale, next_safety_scale = simplification_schedule(
            args, epoch=next_epoch, stage=next_stage,
        )
        safety_sigma, safety_knots = selection_safety(args, applied_safety_scale)
        model.set_selection_safety(sigma=safety_sigma, knots=safety_knots)
        if stage == "joint" and epoch == args.proposal_epochs + 1:
            initializer_path = (
                proposal_path
                if proposal_path.exists()
                else proposal_final_path
            )
            if not initializer_path.exists():
                p.error("missing Proposal checkpoint for the stage transition")
            if initializer_path == proposal_final_path:
                print(
                    "WARNING: validation-best proposal.pt is unavailable; "
                    "falling back to the stage-final Proposal artifact for "
                    "legacy compatibility.",
                    flush=True,
                )
            proposal_initializer = torch.load(
                initializer_path, map_location="cpu", weights_only=True,
            )
            model.load_state_dict(
                proposal_initializer["model_state_dict"], strict=True,
            )
            proposal_dense_pass = proposal_initializer["validation_metrics"].get(
                "worst_dense_pass_rate",
                proposal_initializer["validation_metrics"]["dense_pass_rate"],
            )
            proposal_initializer_path_used = str(initializer_path)
            proposal_initializer_epoch = int(proposal_initializer["epoch"])
            proposal_ready = True
            selector_warmup = training_phase == "selector_warmup"
            if legacy_joint_optimizer_contract:
                for parameter in model.parameters():
                    parameter.requires_grad_(True)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=args.joint_lr,
                    weight_decay=args.weight_decay,
                )
                optimizer_regime = "legacy_joint_single_group"
            else:
                configure_joint_trainability(
                    model, selector_warmup=selector_warmup,
                    fixed_proposal=args.joint_supervision == "offline_feasible_teacher",
                )
                optimizer = build_joint_optimizer(
                    model, args, selector_warmup=selector_warmup,
                )
                optimizer_regime = "joint_named_groups"
            print(
                "Proposal schedule complete; loading the validation-ranked "
                "best dense "
                f"initializer from epoch {proposal_initializer_epoch} "
                f"(worst-source pass={proposal_dense_pass:.1%}, reporting "
                f"reference={args.proposal_pass_target:.1%}). Joint training "
                f"starts unconditionally in {training_phase}.",
                flush=True,
            )
        elif stage == "joint" and optimizer_regime == "joint_named_groups":
            selector_warmup = training_phase == "selector_warmup"
            configure_joint_trainability(
                model, selector_warmup=selector_warmup,
                fixed_proposal=args.joint_supervision == "offline_feasible_teacher",
            )
            set_joint_optimizer_learning_rates(
                optimizer, args, selector_warmup=selector_warmup,
            )
        elif stage == "joint":
            # Legacy one-group resume keeps its historical all-trainable state.
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        current_learning_rates = optimizer_learning_rates(optimizer)
        training_sources = (
            () if args.joint_supervision in {"synthetic_ground_truth", "offline_feasible_teacher"}
            else sources
        )
        training_real_fraction = (
            0.0
            if args.joint_supervision in {"synthetic_ground_truth", "offline_feasible_teacher"}
            else args.real_fraction
        )
        train_data = MixedTrainingCurves(
            dataset_config,
            training_sources,
            size=args.train_size,
            seed=args.seed,
            real_fraction=training_real_fraction,
            epoch=epoch - 1,
            resample=args.resample_train_each_epoch,
            synthetic_high_k_fraction=(
                args.proposal_high_k_fraction if stage == "proposal" else 0.0
            ),
            synthetic_high_k_min_knots=args.proposal_high_k_min_knots,
        )
        if stage == "joint" and args.joint_supervision == "offline_feasible_teacher":
            if feasible_teacher_cache is None:
                teacher_initializer_path = proposal_path
                if not teacher_initializer_path.exists():
                    teacher_initializer_path = proposal_final_path
                if not teacher_initializer_path.exists():
                    p.error("offline feasible Teacher requires the fixed Proposal artifact")
                teacher_initializer = torch.load(
                    teacher_initializer_path, map_location="cpu", weights_only=True,
                )
                teacher_proposal_model, _, _ = build_model_from_checkpoint(
                    teacher_initializer
                )
                teacher_proposal_model.to(device)
                teacher_proposal_model.eval()
                cache_path = args.feasible_teacher_cache_dir / "train.pt"
                print(
                    "Building/checking fixed-Proposal feasible-subset labels at "
                    f"{cache_path}; this is offline work, not Joint per-step search.",
                    flush=True,
                )
                feasible_teacher_cache = build_or_load_v16_feasible_teacher_cache(
                    teacher_proposal_model, train_data, cache_path,
                    mse_tolerance=args.mse_tolerance,
                    device=device, batch_size=args.feasible_teacher_batch_size,
                    smoothness_weight=0.0, control_ridge=0.0,
                    progress=lambda done, total: progress(
                        done, total, "offline feasible Teacher",
                        every=args.feasible_teacher_batch_size,
                    ),
                )
                del teacher_proposal_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if stage == "joint" and feasible_teacher_cache is not None:
            feasible_teacher_cache.assert_proposal_unchanged(model)
            train_data = IndexedTrainingDataset(train_data)
        loader_generator = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            generator=loader_generator, **loader_runtime,
        )
        model.train()
        started = time.perf_counter()
        total, samples = defaultdict(float), 0
        for step, batch in enumerate(train_loader, 1):
            if args.joint_supervision in {"synthetic_ground_truth", "offline_feasible_teacher"}:
                if not bool(batch["target_internal_knot_count_valid"].all()):
                    raise RuntimeError(
                        "unlabelled row entered supervised-only training"
                    )
                if not bool(batch["target_geometry_valid"].all()):
                    raise RuntimeError(
                        "row without parameter/knot labels entered supervised-only training"
                    )
                if (
                    (args.fine_teacher_weight > 0
                     or args.fine_teacher_ranking_weight > 0)
                    and not bool(batch["target_single_deletion_valid"].all())
                ):
                    raise RuntimeError(
                        "row without certified single-deletion MSE entered "
                        "fine-teacher training"
                    )
            points = batch["points"].to(device, non_blocking=device.type == "cuda")
            lower, upper = math.log(args.tolerance_factor_min), math.log(args.tolerance_factor_max)
            tolerance = args.mse_tolerance * (lower + (upper-lower)*torch.rand(points.shape[0], device=device)).exp()
            target_single_deletion_mse = batch.get("target_single_deletion_mse")
            if target_single_deletion_mse is not None:
                target_single_deletion_mse = target_single_deletion_mse.to(
                    device, non_blocking=device.type == "cuda"
                )
            target_single_deletion_mask = batch.get("target_single_deletion_mask")
            if target_single_deletion_mask is not None:
                target_single_deletion_mask = target_single_deletion_mask.to(
                    device, non_blocking=device.type == "cuda"
                )
            target_single_deletion_valid = batch.get("target_single_deletion_valid")
            if target_single_deletion_valid is not None:
                target_single_deletion_valid = target_single_deletion_valid.to(
                    device, non_blocking=device.type == "cuda"
                )
            optimizer.zero_grad(set_to_none=True)
            feasible_labels = (
                feasible_teacher_cache.labels_for_indices(
                    batch["feasible_teacher_index"], device=device,
                )
                if stage == "joint" and feasible_teacher_cache is not None
                else {}
            )
            loss, metrics = objective(
                model, points, stage=stage, mse_tolerance=tolerance,
                complexity_scale=(
                    applied_complexity_scale if stage == "joint" else 0.0
                ),
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
                target_single_deletion_mse=target_single_deletion_mse,
                target_single_deletion_mask=target_single_deletion_mask,
                target_single_deletion_valid=target_single_deletion_valid,
                feasible_teacher_mask=feasible_labels.get("teacher_retained_mask"),
                feasible_teacher_knots=feasible_labels.get("teacher_internal_knots"),
                feasible_teacher_knot_mask=feasible_labels.get("teacher_internal_knot_mask"),
                feasible_teacher_count=feasible_labels.get("teacher_count"),
                feasible_teacher_mse=feasible_labels.get("teacher_fit_mse"),
                feasible_teacher_pass=feasible_labels.get("teacher_threshold_satisfied"),
                feasible_teacher_risk=feasible_labels.get("teacher_soft_keep_risk"),
            )
            loss.backward()
            if optimizer_regime == "joint_named_groups":
                gradient_norms = clip_joint_gradients(model, args.grad_clip)
            else:
                global_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                    error_if_nonfinite=True,
                )
                gradient_norms = {"global": float(global_norm)}
            optimizer.step()
            if (
                feasible_teacher_cache is not None
                and step == len(train_loader)
            ):
                feasible_teacher_cache.assert_proposal_unchanged(model)
            size = len(points)
            samples += size
            for key, value in metrics.items():
                total[key] += float(value)*size
            for group_name, value in gradient_norms.items():
                total[f"gradient_norm_{group_name}"] += value * size
            subset_label = (
                "denseK" if stage == "proposal" else
                "targetK" if args.joint_supervision == "synthetic_ground_truth"
                else "feasibleK" if args.joint_supervision == "offline_feasible_teacher"
                else "teacherK"
            )
            lr_display = "/".join(
                f"{name}:{value:.1e}"
                for name, value in current_learning_rates.items()
            )
            progress(step, len(train_loader),
                     f"Epoch {epoch:03}/{args.epochs} {training_phase}",
                     f"MSE={metrics['deployment_mse']:.3e} pass={metrics['deployment_pass_rate']:.1%} "
                     f"K={metrics['keep_count']:.1f} {subset_label}={metrics['subset_best_count']:.1f} "
                     f"countMAE={metrics['supervised_count_mae']:.2f} lr={lr_display}",
                     args.log_every_batches)
        measured = validate(
            model, val_loader, device, args.mse_tolerance,
            stage=stage,
            knot_match_tolerance=args.knot_match_tolerance,
            log_every=args.log_every_batches,
            synthetic_boundary_knot_count=args.max_control_points - 4,
        )
        train_metrics = {k: v/samples for k, v in total.items()}
        reporting_target_met = (
            stage == "joint"
            and measured["qualification_deployment_pass_rate"]
            >= args.deployment_pass_target
        )
        if stage == "joint":
            if reporting_target_met:
                reporting_target_streak += 1
            else:
                reporting_target_streak = 0
        entry = dict(
            epoch=epoch, stage=stage, train=train_metrics, validation=measured,
            training_phase=training_phase,
            optimizer_regime=optimizer_regime,
            learning_rates=current_learning_rates,
            gradient_clipping=(
                "per_named_group"
                if optimizer_regime == "joint_named_groups"
                else "global"
            ),
            applied_complexity_scale=applied_complexity_scale,
            next_complexity_scale=next_complexity_scale,
            applied_selection_safety_scale=applied_safety_scale,
            next_selection_safety_scale=next_safety_scale,
            applied_selection_safety_sigma=safety_sigma,
            applied_selection_safety_knots=safety_knots,
            reporting_target_met=reporting_target_met,
            reporting_target_streak=reporting_target_streak,
            seconds=time.perf_counter()-started,
        )
        history.append(entry)
        improved = False
        if stage == "proposal":
            rank = (
                proposal_checkpoint_rank(measured)
                if args.joint_supervision in {"synthetic_ground_truth", "offline_feasible_teacher"}
                else (
                    -measured["dense_subset_cost"],
                    -measured["dense_mse"],
                    measured["qualification_dense_pass_rate"],
                )
            )
            if proposal_rank is None or rank > tuple(proposal_rank):
                proposal_rank, improved = rank, True
            # Readiness means that the scheduled Proposal stage has produced a
            # usable best initializer; it no longer encodes an aggregate pass gate.
            proposal_ready = True
        else:
            simplification_ready = simplification_is_ready(
                args,
                epoch=epoch,
                stage=stage,
                applied_safety_scale=applied_safety_scale,
                applied_complexity_scale=applied_complexity_scale,
            )
            rank = checkpoint_rank(
                measured, simplification_ready=simplification_ready,
            )
            if best_rank is None or rank > tuple(best_rank):
                best_rank, improved = rank, True
        selection_metrics = checkpoint_selection_snapshot(
            measured,
            stage=stage,
            candidate_capacity=args.candidate_knots,
            tolerance=args.mse_tolerance,
        )
        payload = dict(objective_version=run_objective_version,
            model_config=model.get_config(), model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            optimizer_state_dict=optimizer.state_dict(), epoch=epoch, stage=stage,
            training_phase=training_phase,
            optimizer_config=dict(
                regime=optimizer_regime,
                learning_rates=current_learning_rates,
                gradient_clipping=(
                    "per_named_group"
                    if optimizer_regime == "joint_named_groups"
                    else "global"
                ),
                max_gradient_norm=args.grad_clip,
                selector_warmup_epochs=args.selector_warmup_epochs,
            ),
            training_config=current_config, dataset_config=dataset_config, history=history,
            dataset_type=(
                (
                    "certified_synthetic_supervised_train_real_validation_only"
                    if args.certified_minimal_source
                    else "uncertified_synthetic_train_real_validation_only"
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
            proposal_best_path=str(proposal_path),
            proposal_final_path=str(proposal_final_path),
            proposal_initializer_path=proposal_initializer_path_used,
            proposal_initializer_epoch=proposal_initializer_epoch,
            proposal_ready_role=(
                "scheduled_stage_complete_with_saved_initializer"
            ),
            proposal_reporting_target_met=(
                measured["qualification_dense_pass_rate"]
                >= args.proposal_pass_target
            ),
            architecture_revision=(
                V16_ADAPTIVE_SELECTION_REVISION
                if args.one_shot_selection_policy == "mass_topk"
                else "v16_adaptive_beta_threshold_ablation"
            ),
            simplification_contract=run_simplification_contract,
            offline_feasible_teacher=(
                dict(
                    cache_path=str(feasible_teacher_cache.path),
                    loaded_from_cache=feasible_teacher_cache.loaded,
                    sample_count=feasible_teacher_cache.sample_count,
                    candidate_count=feasible_teacher_cache.candidate_count,
                    numerical_pass_fraction=feasible_teacher_cache.feasible_fraction,
                    proposal_fingerprint=(
                        feasible_teacher_cache.batch.config.proposal_fingerprint
                    ),
                    dataset_fingerprint=(
                        feasible_teacher_cache.batch.config.dataset_fingerprint
                    ),
                    rms_tolerance=(
                        feasible_teacher_cache.batch.config.error_tolerance
                    ),
                    smoothness_weight=(
                        feasible_teacher_cache.batch.config.smoothness_weight
                    ),
                    greedy_not_globally_minimal=True,
                )
                if feasible_teacher_cache is not None else None
            ),
            simplification_ready=simplification_is_ready(
                args,
                epoch=epoch,
                stage=stage,
                applied_safety_scale=applied_safety_scale,
                applied_complexity_scale=applied_complexity_scale,
            ),
            complexity_scale=next_complexity_scale,
            applied_complexity_scale=applied_complexity_scale,
            next_selection_safety_scale=next_safety_scale,
            applied_selection_safety_scale=applied_safety_scale,
            reporting_target_streak=reporting_target_streak,
            feasible_streak=reporting_target_streak,
            aggregate_pass_role="reporting_reference_only",
            best_deployment_pass_constraint_satisfied=reporting_target_met,
            current_deployment_pass_constraint_satisfied=reporting_target_met,
            deployment_pass_flags_role="reporting_reference_only",
            checkpoint_quality=(
                V16_JOINT_CHECKPOINT_QUALITY
                if stage == "joint" else "dense_proposal_selected"
            ),
            reporting_checkpoint_quality=(
                "deployment_target_met"
                if reporting_target_met else "target_not_met"
            ),
            checkpoint_selection=V16_CHECKPOINT_SELECTION,
            checkpoint_selection_metrics=selection_metrics,
            simplification_curriculum=dict(
                kind="deterministic_linear_by_joint_epoch",
                aggregate_pass_feedback=False,
                complexity_ramp_epochs=args.complexity_ramp_epochs,
                complexity_max_scale=args.complexity_max_scale,
                safety_anneal_epochs=args.safety_anneal_epochs,
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
            loss_config=dict(joint_supervision=args.joint_supervision,
                             online_teacher=args.joint_supervision == "online_teacher",
                             fine_grained_teacher=(
                                 (
                                     "frozen_proposal_feasible_subset_risk"
                                     if args.joint_supervision == "offline_feasible_teacher"
                                     else "certified_source_single_deletion_mse"
                                 )
                                 if args.fine_teacher_weight > 0
                                 or args.fine_teacher_ranking_weight > 0
                                 else None
                             ),
                             fine_teacher_additional_spline_solves_per_batch=0,
                             fine_teacher_error_unit="mean_squared_euclidean",
                             ground_truth_keep_assignment=(
                                 "offline_hard_rms_feasible_slot_mask"
                                 if args.joint_supervision == "offline_feasible_teacher"
                                 else "ordered_one_to_one_minimum_l1"
                                 if args.joint_supervision == "synthetic_ground_truth"
                                 else None
                             ),
                             policy_samples=args.policy_samples, counterfactual_edits=args.counterfactual_edits,
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
                                  "proposal_knot_assignment_weight",
                                  "proposal_multiscale_recall_weight",
                                  "selected_knot_position_weight",
                                  "keep_dice_weight", "keep_cdf_weight",
                                  "parameter_gap_weight", "parameter_bias_weight",
                                  "fine_teacher_weight",
                                  "fine_teacher_ranking_weight")},
                             fine_teacher_temperature=(
                                 objective.fine_teacher_temperature
                             ),
                             keep_fuzzy_negative_radius=(
                                 objective.keep_fuzzy_negative_radius
                             ),
                             keep_fuzzy_negative_floor=(
                                 objective.keep_fuzzy_negative_floor
                             ),
                             proposal_parameter_warp_gradient_scale=(
                                 objective.proposal_parameter_warp_gradient_scale
                             ),
                             joint_parameter_warp_gradient_scale=(
                                 objective.joint_parameter_warp_gradient_scale
                             )))
        payload["qualification"] = assess_v16_checkpoint(
            payload,
            required_pass_rate=V16_FORMAL_PASS_RATE,
            required_mse_tolerance=args.mse_tolerance,
        )
        # Commit a newly selected best artifact first.  ``.last.pt`` is the
        # transaction marker: once it advertises the new rank, the matching
        # best model is guaranteed to exist.  Resume also reconciles the
        # inverse crash window in which best was written but last was not.
        if improved:
            atomic_save(payload, proposal_path if stage == "proposal" else output)
        atomic_save(payload, last_path)
        if stage == "proposal":
            # This rolling artifact is deliberately independent of the
            # validation-ranked best Proposal.  It records the latest Proposal
            # state for audit/recovery; Joint normally loads ``proposal.pt``,
            # whose corrected continuous rank protects against both an early
            # loose-recall winner and a late-stage regression.
            atomic_save(payload, proposal_final_path)
        history_path.write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
        epoch_lr_display = "/".join(
            f"{name}:{value:.1e}"
            for name, value in current_learning_rates.items()
        )
        print(f"Epoch {epoch:03} phase={training_phase} lr={epoch_lr_display} "
              f"val dense={measured['dense_pass_rate']:.1%} "
              f"deployment={measured['deployment_pass_rate']:.1%} worst-source={measured['worst_deployment_pass_rate']:.1%} "
              f"qualification={measured['qualification_deployment_pass_rate']:.1%} "
              f"MSE={measured['deployment_mse']:.3e} K={measured['keep_count']:.2f} "
              f"mass={measured['keep_probability_mass']:.2f} beta={measured['adaptive_keep_threshold']:.2f} "
              f"soft_cost={selection_metrics['mean_subset_cost']:.4f} "
              f"selection_score={selection_metrics['selection_score']:.4f} "
              f"complexity={applied_complexity_scale:.2f}->{next_complexity_scale:.2f} "
              f"safety={safety_knots}+{safety_sigma:.2f}sigma "
              f"scale={applied_safety_scale:.2f}->{next_safety_scale:.2f} "
              f"pass_reference_only={measured['qualification_deployment_pass_rate']:.1%}",
              flush=True)
        if stage == "proposal":
            print(
                "  Proposal train geometry: coverage_MAE="
                f"{train_metrics['proposal_knot_nearest_mae']:.4e}, "
                "ordered_assignment_MAE="
                f"{train_metrics['proposal_knot_assignment_mae']:.4e}, "
                "matched_K="
                f"{train_metrics['proposal_knot_assignment_count']:.2f}, "
                "R@.005/.01/.02="
                f"{train_metrics['proposal_recall_at_005']:.3f}/"
                f"{train_metrics['proposal_recall_at_010']:.3f}/"
                f"{train_metrics['proposal_recall_at_020']:.3f}, "
                "parameter_bias="
                f"{train_metrics['parameter_bias_mae']:.3e}",
                flush=True,
            )
        else:
            count_calibration_detail = ""
            if "count_calibration_score" in train_metrics:
                count_calibration_detail = (
                    ", count-cal loss="
                    f"{train_metrics['structured_count_loss']:.3e}, "
                    "score/target/bias/MAE="
                    f"{train_metrics['count_calibration_score']:.2f}/"
                    f"{train_metrics['count_calibration_target']:.2f}/"
                    f"{train_metrics['structured_count_bias']:+.2f}/"
                    f"{train_metrics['structured_count_mae']:.2f}"
                )
            fine_risk_detail = ""
            if "fine_teacher_risk_std" in train_metrics:
                fine_risk_detail = (
                    ", fine-risk mean/std/range="
                    f"{train_metrics['fine_teacher_mean_risk']:.3f}/"
                    f"{train_metrics['fine_teacher_risk_std']:.3f}/"
                    f"{train_metrics['fine_teacher_risk_range']:.3f}, "
                    "fine-loss/rank="
                    f"{train_metrics['fine_teacher_loss']:.3e}/"
                    f"{train_metrics['fine_teacher_ranking_loss']:.3e}"
                )
            print(
                "  Joint train structure: Keep P/R/F1="
                f"{train_metrics['keep_mask_precision']:.3f}/"
                f"{train_metrics['keep_mask_recall']:.3f}/"
                f"{train_metrics['keep_mask_f1']:.3f}, "
                "critical_false_delete="
                f"{train_metrics['critical_false_delete_rate']:.3f}, "
                "proposal R@.005/.01/.02="
                f"{train_metrics['proposal_recall_at_005']:.3f}/"
                f"{train_metrics['proposal_recall_at_010']:.3f}/"
                f"{train_metrics['proposal_recall_at_020']:.3f}, "
                "parameter_bias="
                f"{train_metrics['parameter_bias_mae']:.3e}"
                f"{count_calibration_detail}{fine_risk_detail}",
                flush=True,
            )
            if args.joint_supervision == "offline_feasible_teacher":
                print(
                    "  Feasibility chain: numerical Teacher pass/MSE="
                    f"{train_metrics['offline_teacher_numerical_pass_rate']:.1%}/"
                    f"{train_metrics['offline_teacher_numerical_mse']:.3e}, "
                    "student teacher-mask pass/MSE="
                    f"{train_metrics['supervised_target_pass_rate']:.1%}/"
                    f"{train_metrics['supervised_target_mse']:.3e}, "
                    "student one-shot pass/MSE="
                    f"{train_metrics['deployment_pass_rate']:.1%}/"
                    f"{train_metrics['deployment_mse']:.3e}, "
                    "teacher-source |K gap|="
                    f"{train_metrics['offline_teacher_source_count_gap']:.2f}",
                    flush=True,
                )
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
          f"Last/resume: {last_path}; Proposal best/final: "
          f"{proposal_path} / {proposal_final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
