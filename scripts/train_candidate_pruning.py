from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import (
    V11_JOINT_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
)
from spline_fitting.data.synthetic import SyntheticCubicBSplineDataset
from spline_fitting.losses import CandidatePruningLoss, CandidatePruningLossWeights
from spline_fitting.models import SplineFittingNetwork
from spline_fitting.training import (
    OneShotTeacherBatch,
    OneShotTeacherConfig,
    TeacherAugmentedDataset,
    Trainer,
    build_one_shot_teacher_batch,
    load_one_shot_teacher_cache,
    save_one_shot_teacher_cache,
)


def _sample_progress_line(
    label: str,
    completed: int,
    total: int,
    elapsed: float,
    *,
    width: int = 28,
) -> str:
    fraction = completed / max(total, 1)
    filled = min(width, max(0, int(round(width * fraction))))
    rate = completed / elapsed if elapsed > 0.0 else 0.0
    remaining = (total - completed) / rate if rate > 0.0 else float("inf")
    if remaining == float("inf"):
        eta = "--:--"
    else:
        total_seconds = int(round(max(remaining, 0.0)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        eta = (
            f"{hours:d}:{minutes:02d}:{seconds:02d}"
            if hours
            else f"{minutes:02d}:{seconds:02d}"
        )
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"  {label:<18} [{bar}] {completed:>5}/{total:<5} "
        f"{100.0 * fraction:6.2f}% | {rate:6.2f} sample/s | ETA {eta}"
    )


def _show_sample_progress(
    label: str,
    completed: int,
    total: int,
    started_at: float,
) -> None:
    interactive = sys.stdout.isatty()
    report_interval = 1 if interactive else max(total // 20, 1)
    if completed % report_interval and completed != total:
        return
    line = _sample_progress_line(
        label,
        completed,
        total,
        max(time.perf_counter() - started_at, 1e-9),
    )
    if interactive:
        print(f"\r{line:<130}", end="", flush=True)
        if completed == total:
            print(flush=True)
    else:
        print(line, flush=True)


def _checkpoint_with_metadata(
    checkpoint: dict[str, object],
    *,
    model_config: dict[str, object],
    dataset_config: dict[str, object],
    loss_config: dict[str, object],
    training_config: dict[str, object],
    deployment_config: dict[str, object],
    histories: dict[str, list[dict[str, float]]],
) -> dict[str, object]:
    checkpoint.update(
        {
            "model_config": model_config,
            "dataset_config": dataset_config,
            "dataset_type": "synthetic_open_cubic_bspline",
            "objective_version": V11_JOINT_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
            "loss_config": loss_config,
            "training_config": training_config,
            "deployment_config": deployment_config,
            "selected_stage": checkpoint.get("stage", "one_shot_distillation"),
            "stage_histories": histories,
        }
    )
    return checkpoint


def _model_fingerprint(model: torch.nn.Module) -> str:
    """Bind an offline teacher cache to the exact proposal-producing weights."""
    digest = hashlib.sha256()
    proposal_prefixes = (
        "encoder.",
        "parameter_head.",
        "candidate_head.",
        "pruning_head.analytic_projection.",
        "pruning_head.input_norm.",
        "pruning_head.self_attention.",
        "pruning_head.self_norm.",
        "pruning_head.feed_forward.",
        "pruning_head.output_norm.",
        "pruning_head.position_residual_head.",
    )
    proposal_metadata = {
        "degree": int(model.degree),
        "lambda_poly": float(model.lambda_poly),
        "lambda_knot": float(model.lambda_knot),
        "pruning_residual_bandwidth": float(model.pruning_residual_bandwidth),
        "parameter_min_gap": float(model.parameter_head.min_gap),
        "parameter_gap_parameterization": str(
            model.parameter_head.gap_parameterization
        ),
        "geometry_feature_mode": str(model.encoder.feature_mode),
        "candidate_count": int(model.candidate_head.num_candidates),
        "candidate_attention_heads": int(model.candidate_head.attention_heads),
        "candidate_min_gap": float(model.candidate_head.min_gap),
        "candidate_local_attention_bandwidth": float(
            model.candidate_head.local_attention_bandwidth
        ),
        "proposal_position_fraction": float(model.pruning_head.max_position_fraction),
        "pruning_attention_heads": int(model.pruning_head.self_attention.num_heads),
        "pruning_min_gap": float(model.pruning_head.min_gap),
        "fixed_proposal_geometry": bool(
            model.pruning_head.one_shot_fixed_proposal_geometry
        ),
    }
    digest.update(
        json.dumps(proposal_metadata, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith(proposal_prefixes):
            continue
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        if tensor.is_floating_point():
            # CPU linear algebra can differ by a few float32 ulps across
            # otherwise identical runs.  Such sub-micro changes do not alter
            # teacher geometry, so hash a stable 1e-6 quantisation instead of
            # making cache reuse depend on thread-level reduction order.
            tensor = torch.round(tensor.to(torch.float64) * 1e6).to(torch.int64)
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


_PROPOSAL_SEMANTIC_KEYS = (
    "point_dim",
    "degree",
    "hidden_dim",
    "encoder_layers",
    "max_internal_knots",
    "min_parameter_gap",
    "min_knot_gap",
    "gap_parameterization",
    "lambda_poly",
    "lambda_knot",
    "structure_mode",
    "structure_attention_heads",
    "geometry_feature_mode",
    "pruning_residual_bandwidth",
    "one_shot_fixed_proposal_geometry",
    "compute_first_derivative",
)


def _proposal_semantic_mismatches(
    requested: Mapping[str, object],
    saved: Mapping[str, object],
) -> list[str]:
    """Find non-weight settings that can change fixed proposal geometry."""

    mismatches: list[str] = []
    for key in _PROPOSAL_SEMANTIC_KEYS:
        if key not in saved or key not in requested:
            continue
        requested_value = requested[key]
        saved_value = saved[key]
        if isinstance(requested_value, float) or isinstance(saved_value, float):
            equal = abs(float(requested_value) - float(saved_value)) <= 1e-12
        else:
            equal = requested_value == saved_value
        if not equal:
            mismatches.append(
                f"{key}: requested={requested_value!r}, checkpoint={saved_value!r}"
            )
    return mismatches


def _dataset_fingerprint(
    dataset: SyntheticCubicBSplineDataset,
    *,
    split: str,
    seed: int,
    dataset_config: dict[str, object],
) -> str:
    """Hash the actual fixed samples, not only their reusable row indices."""
    digest = hashlib.sha256()
    metadata = {
        "split": split,
        "seed": int(seed),
        "size": len(dataset),
        "dataset_config": dataset_config,
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    identity_keys = (
        "sample_id",
        "points",
        "true_params",
        "true_internal_knots",
        "true_internal_knot_mask",
    )
    started_at = time.perf_counter()
    total = len(dataset)
    for index in range(total):
        sample = dataset[index]
        for key in identity_keys:
            value = torch.as_tensor(sample[key]).detach().cpu().contiguous()
            digest.update(key.encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.numpy().tobytes())
        _show_sample_progress(
            f"fingerprint {split}",
            index + 1,
            total,
            started_at,
        )
    return digest.hexdigest()


def _concatenate_teacher_batches(
    batches: list[OneShotTeacherBatch],
    config: OneShotTeacherConfig,
) -> OneShotTeacherBatch:
    if not batches:
        raise ValueError("cannot concatenate an empty teacher batch list")
    labels = {
        key: torch.cat([batch.as_loss_kwargs()[key] for batch in batches], dim=0)
        for key in batches[0].as_loss_kwargs()
    }
    sample_indices = torch.cat([batch.sample_indices for batch in batches], dim=0)
    first_shapes = batches[0].input_shapes
    total = int(sample_indices.numel())
    return OneShotTeacherBatch(
        sample_indices=sample_indices,
        input_shapes={
            "parameters": (total, *first_shapes["parameters"][1:]),
            "points": (total, *first_shapes["points"][1:]),
            "candidate_knots": (total, *first_shapes["candidate_knots"][1:]),
        },
        config=config,
        **labels,
    )


def _build_or_load_teacher_cache(
    *,
    model: SplineFittingNetwork,
    dataset: SyntheticCubicBSplineDataset,
    cache_path: Path,
    config: OneShotTeacherConfig,
    batch_size: int,
    device: torch.device,
    reuse: bool,
) -> OneShotTeacherBatch:
    sample_count = len(dataset)
    expected_indices = torch.arange(sample_count, dtype=torch.long)
    sample = dataset[0]
    expected_shapes = {
        "parameters": (sample_count, int(sample["points"].shape[0])),
        "points": (sample_count, *tuple(sample["points"].shape)),
        "candidate_knots": (
            sample_count,
            int(model.candidate_head.num_candidates),
        ),
    }
    if reuse and cache_path.exists():
        print(f"Loading verified offline teacher cache: {cache_path}", flush=True)
        return load_one_shot_teacher_cache(
            cache_path,
            expected_config=config,
            expected_sample_indices=expected_indices,
            expected_input_shapes=expected_shapes,
        )

    print(f"Generating offline Hard-RMS teacher cache: {cache_path}", flush=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batches: list[OneShotTeacherBatch] = []
    model.eval()
    processed = 0
    started_at = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device)
            output = model(points)
            teacher_batch = build_one_shot_teacher_batch(
                output["params"].detach().cpu().to(torch.float64),
                batch["points"].detach().cpu().to(torch.float64),
                # Refined candidates are strictly ordered without sorting, so
                # their left-to-right slot identity remains stable after the
                # proposal backbone is frozen for distillation.
                output.get("proposal_internal_knots", output["internal_knots"])
                .detach()
                .cpu()
                .to(torch.float64),
                sample_indices=batch["sample_id"],
                config=config,
            )
            batches.append(teacher_batch)
            processed += points.shape[0]
            _show_sample_progress(
                "Hard-RMS teacher",
                processed,
                sample_count,
                started_at,
            )
    teacher = _concatenate_teacher_batches(batches, config)
    if not torch.equal(teacher.sample_indices, expected_indices):
        raise RuntimeError("offline teacher rows are not in dataset index order")
    save_one_shot_teacher_cache(cache_path, teacher)
    return teacher


def _set_one_shot_trainable(
    model: SplineFittingNetwork,
    *,
    calibrate_positions: bool,
) -> list[torch.nn.Parameter]:
    """Freeze proposal geometry and expose cache-compatible v11 student modules."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    module_names = ["keep_head", "adaptive_threshold_head"]
    fixed_geometry = bool(
        getattr(model.pruning_head, "one_shot_fixed_proposal_geometry", False)
    )
    joint_refinement = bool(
        getattr(
            model.pruning_head,
            "one_shot_joint_position_refinement",
            False,
        )
    )
    if fixed_geometry:
        module_names.extend(
            [
                "one_shot_selector_attention",
                "one_shot_selector_attention_norm",
                "one_shot_selector_feed_forward",
                "one_shot_selector_output_norm",
                "one_shot_selector_extra_attention",
                "one_shot_selector_extra_attention_norm",
                "one_shot_selector_extra_feed_forward",
                "one_shot_selector_extra_output_norm",
            ]
        )
    else:
        module_names.append("position_to_keep_feedback")
    if calibrate_positions and joint_refinement:
        module_names.extend(
            [
                "joint_preliminary_position_projection",
                "joint_preliminary_position_norm",
                "joint_preliminary_position_head",
                "joint_position_to_keep_feedback",
                "joint_position_to_keep_norm",
                "joint_final_position_projection",
                "joint_final_position_norm",
                "joint_final_position_head",
            ]
        )
    elif calibrate_positions and not fixed_geometry:
        module_names.extend(
            [
                "keep_context_projection",
                "keep_context_norm",
                "position_residual_head",
            ]
        )
    trainable: list[torch.nn.Parameter] = []
    for module_name in module_names:
        module = getattr(model.pruning_head, module_name, None)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("one-shot head exposes no trainable selection parameters")
    return trainable


def _reset_one_shot_selector_to_neutral(
    model: SplineFittingNetwork,
    *,
    initial_probability: float,
) -> None:
    """Discard conservative proposal-time Keep logits before distillation.

    Candidate pretraining forces every gate open, so its selector weights carry
    no learned pruning information.  A deterministic neutral reset avoids
    starting the offline-teacher student at p=0.9 (the all-keep basin) and also
    makes proposal-checkpoint/cache reuse reproducible.
    """
    head = model.pruning_head
    if not getattr(head, "one_shot_adaptive", False):
        raise ValueError("neutral selector reset requires an adaptive one-shot head")
    if not 0.5 < initial_probability < 1.0:
        raise ValueError("initial_probability must lie strictly in (0.5, 1)")
    initial_logit = torch.logit(torch.tensor(initial_probability)).item()
    with torch.no_grad():
        head.keep_head.weight.zero_()
        head.keep_head.bias.fill_(initial_logit)
        for module_name in (
            "adaptive_threshold_head",
            "position_to_keep_feedback",
            "joint_position_to_keep_feedback",
            "joint_preliminary_position_head",
            "joint_final_position_head",
        ):
            module = getattr(head, module_name, None)
            if module is None:
                continue
            final_linear = (
                module[-1] if isinstance(module, torch.nn.Sequential) else module
            )
            final_linear.weight.zero_()
            final_linear.bias.zero_()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train high-recall proposals and a one-shot adaptive LearnedKeep "
            "student distilled from an offline Hard-RMS teacher."
        )
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--candidate-pretrain-epochs", type=int, default=20)
    parser.add_argument(
        "--proposal-checkpoint",
        type=Path,
        default=None,
        help=(
            "Load a previously selected proposal checkpoint. This is required "
            "when candidate pretraining epochs are zero and makes teacher-cache "
            "reuse reproducible."
        ),
    )
    parser.add_argument(
        "--selector-calibration-epochs",
        "--keep-position-calibration-epochs",
        "--joint-finetune-epochs",
        dest="keep_position_calibration_epochs",
        type=int,
        default=10,
        help=(
            "Final low-learning-rate joint Keep/position calibration epochs. "
            "The immutable proposal slots remain teacher-compatible while the "
            "separate deployment positions are updated."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--log-every-batches", type=int, default=20)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show live train/validation batch progress bars in an interactive terminal.",
    )
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--val-seed", type=int, default=10000)
    parser.add_argument("--num-points", type=int, default=192)
    parser.add_argument("--point-dim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--min-control-points", type=int, default=8)
    parser.add_argument("--max-control-points", type=int, default=24)
    parser.add_argument("--candidate-knots", type=int, default=28)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.001)
    parser.add_argument("--knot-nonuniformity", type=float, default=0.65)
    parser.add_argument("--sampling-nonuniformity", type=float, default=0.45)
    parser.add_argument("--turn-strength", type=float, default=0.45)
    parser.add_argument(
        "--fit-tolerance",
        type=float,
        default=5e-3,
        help=(
            "Normalized mean Euclidean RMS threshold. It defines canonical "
            "labels, offline teacher pruning and the one-shot training target."
        ),
    )
    parser.add_argument("--candidate-match-tolerance", type=float, default=0.01)
    parser.add_argument("--min-knot-gap", type=float, default=1e-3)
    parser.add_argument("--pruning-residual-bandwidth", type=float, default=0.05)
    parser.add_argument("--lambda-poly", type=float, default=1e-6)
    parser.add_argument("--lambda-knot", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--selector-lr",
        type=float,
        default=5e-4,
        help=(
            "Learning rate for the small frozen-proposal one-shot selector. "
            "It is intentionally higher than the proposal learning rate so "
            "the mask can leave its conservative initialization."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-fit", type=float, default=0.25)
    parser.add_argument("--lambda-threshold-violation", type=float, default=5.0)
    parser.add_argument(
        "--lambda-one-shot-surrogate-fit",
        type=float,
        default=0.0,
        help=(
            "Truncated-power fit weight during one-shot distillation/calibration. "
            "Keep at zero: standard-B-spline Hard-RMS teacher labels supervise "
            "selection, while the surrogate remains diagnostic only."
        ),
    )
    parser.add_argument(
        "--lambda-one-shot-surrogate-threshold",
        type=float,
        default=0.0,
        help=(
            "Truncated-power threshold-violation weight during one-shot stages. "
            "Keep at zero to prevent the all-keep shortcut."
        ),
    )
    parser.add_argument("--lambda-true-params", type=float, default=0.1)
    parser.add_argument("--lambda-candidate-coverage", type=float, default=5.0)
    parser.add_argument("--lambda-candidate-repulsion", type=float, default=0.05)
    parser.add_argument("--lambda-keep", type=float, default=1.0)
    parser.add_argument(
        "--lambda-remove-action",
        type=float,
        default=0.0,
        help="Compatibility option; v11 one-shot training does not use sequential actions.",
    )
    parser.add_argument("--lambda-knot-position", type=float, default=2.0)
    parser.add_argument(
        "--lambda-deletion-cost",
        type=float,
        default=0.0,
        help="Compatibility option; offline teacher risk replaces this v7 term.",
    )
    parser.add_argument("--lambda-teacher-risk", type=float, default=0.5)
    parser.add_argument("--lambda-teacher-ranking", type=float, default=1.0)
    parser.add_argument("--lambda-teacher-distribution", type=float, default=2.0)
    parser.add_argument("--lambda-teacher-critical-recall", type=float, default=1.0)
    parser.add_argument("--lambda-teacher-false-positive", type=float, default=1.0)
    parser.add_argument("--teacher-ranking-margin", type=float, default=1.0)
    parser.add_argument("--lambda-teacher-count", type=float, default=4.0)
    parser.add_argument("--lambda-policy-count", type=float, default=4.0)
    parser.add_argument("--lambda-canonical-selection", type=float, default=0.5)
    parser.add_argument("--lambda-joint-position", type=float, default=1.0)
    parser.add_argument(
        "--lambda-joint-fit",
        type=float,
        default=0.05,
        help=(
            "Normalized truncated-power fit weight used only during joint "
            "Keep/position calibration; its mask path is detached."
        ),
    )
    parser.add_argument(
        "--lambda-joint-threshold-violation",
        type=float,
        default=0.5,
        help=(
            "Surrogate RMS-threshold penalty used only during joint position "
            "calibration to discourage deployment-fit regressions."
        ),
    )
    parser.add_argument("--lambda-complexity", type=float, default=0.1)
    parser.add_argument("--positive-keep-weight", type=float, default=1.5)
    parser.add_argument(
        "--one-shot-selection-policy",
        choices=("threshold", "mass_topk"),
        default="mass_topk",
        help=(
            "Discrete LearnedKeep construction. mass_topk derives cardinality "
            "from probability mass and selects a global top-K subset."
        ),
    )
    parser.add_argument(
        "--one-shot-safety-sigma",
        type=float,
        default=0.25,
        help=(
            "Bernoulli uncertainty reserve used by mass_topk. It adds sigma*std "
            "to probability mass before rounding the retained count."
        ),
    )
    parser.add_argument(
        "--one-shot-selector-layers",
        type=int,
        default=2,
        help="Number of position-aware candidate interaction blocks.",
    )
    parser.add_argument(
        "--one-shot-coverage-bins",
        type=int,
        default=0,
        help=(
            "Reserve the highest-probability candidate in each parameter-domain "
            "bin before filling the remaining mass_topk budget; zero disables it."
        ),
    )
    parser.add_argument(
        "--candidate-local-attention-bandwidth",
        type=float,
        default=0.08,
        help=(
            "Gaussian parameter-space bandwidth for anchor-centred candidate "
            "cross-attention; zero restores the historical global attention."
        ),
    )
    parser.add_argument(
        "--candidate-coverage-tolerances",
        type=float,
        nargs="+",
        default=(0.005, 0.01, 0.02),
        help="Multi-scale proposal-recall margins used during pretraining.",
    )
    parser.add_argument(
        "--one-shot-max-position-shift",
        type=float,
        default=0.05,
        help="Maximum absolute v11 deployment-knot correction in [0,1].",
    )
    parser.add_argument(
        "--initial-keep-probability",
        type=float,
        default=0.55,
        help=(
            "Neutral-but-conservative selector initialization. Values near 0.5 "
            "avoid both the historical p=0.9 all-keep basin and an immediate "
            "all-remove mask at the hard cutoff."
        ),
    )
    parser.add_argument(
        "--deployment-pass-rate-target",
        type=float,
        default=0.97,
        help=(
            "Validation feasibility target. Before reaching it v11 ranks by "
            "real standard-B-spline pass rate and mean/P95 RMS; afterwards it "
            "minimizes retained knots, with fit and localization as tie-breakers."
        ),
    )
    parser.add_argument(
        "--selector-checkpoint-warmup-epochs",
        type=int,
        default=10,
        help=(
            "Ignore the selector's conservative initialization when choosing "
            "the best distillation checkpoint. The effective value is clipped "
            "to leave at least one selectable epoch."
        ),
    )
    parser.add_argument("--knot-position-beta", type=float, default=0.01)
    parser.add_argument(
        "--exact-deletion-supervision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate exact standard-B-spline Hard-RMS labels offline. "
            "Disabling this is unsupported by the v11 one-shot objective."
        ),
    )
    parser.add_argument("--teacher-risk-temperature", type=float, default=0.1)
    parser.add_argument("--teacher-batch-size", type=int, default=4)
    parser.add_argument(
        "--teacher-cache-dir",
        type=Path,
        default=None,
        help="Directory for train/validation offline teacher .pt caches.",
    )
    parser.add_argument(
        "--reuse-teacher-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse only a cache whose config, proposal fingerprint and shapes match.",
    )
    parser.add_argument(
        "--resample-train-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Regenerate curves each epoch. Unsupported with an index-aligned "
            "offline teacher cache; keep this disabled for v11."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "candidate_pruning_one_shot_v11.pt",
    )
    args = parser.parse_args()

    if args.epochs < 2:
        parser.error("--epochs must be at least 2")
    if not 0 <= args.candidate_pretrain_epochs < args.epochs:
        parser.error("--candidate-pretrain-epochs must lie in [0, epochs)")
    if args.candidate_pretrain_epochs == 0 and args.proposal_checkpoint is None:
        parser.error("zero candidate pretraining requires --proposal-checkpoint")
    if args.proposal_checkpoint is not None and not args.proposal_checkpoint.is_file():
        checkpoint_parent = args.proposal_checkpoint.parent
        nearby = (
            sorted(checkpoint_parent.glob("*.pt")) if checkpoint_parent.is_dir() else []
        )
        nearby_text = (
            "\nAvailable checkpoints in that directory:\n  "
            + "\n  ".join(str(path) for path in nearby)
            if nearby
            else ""
        )
        parser.error(
            f"proposal checkpoint does not exist: {args.proposal_checkpoint}"
            f"{nearby_text}"
        )
    if (
        not 0
        <= args.keep_position_calibration_epochs
        < (args.epochs - args.candidate_pretrain_epochs)
    ):
        parser.error(
            "--keep-position-calibration-epochs must leave at least one "
            "distillation epoch"
        )
    if args.train_size <= 0 or args.val_size <= 0 or args.batch_size <= 0:
        parser.error("dataset and batch sizes must be positive")
    if args.log_every_batches < 0:
        parser.error("--log-every-batches must be non-negative")
    if args.selector_checkpoint_warmup_epochs < 0:
        parser.error("--selector-checkpoint-warmup-epochs must be non-negative")
    if args.fit_tolerance <= 0.0:
        parser.error("--fit-tolerance must be positive")
    if args.teacher_risk_temperature <= 0.0 or args.teacher_batch_size <= 0:
        parser.error("teacher temperature and batch size must be positive")
    if not args.exact_deletion_supervision:
        parser.error("v11 requires --exact-deletion-supervision for offline labels")
    if args.resample_train_each_epoch:
        parser.error("offline teacher labels require --no-resample-train-each-epoch")
    if args.candidate_match_tolerance <= 0.0:
        parser.error("--candidate-match-tolerance must be positive")
    if args.attention_heads <= 0 or args.hidden_dim % args.attention_heads:
        parser.error("--attention-heads must divide --hidden-dim")
    max_source_knots = args.max_control_points - 4
    if args.candidate_knots < max_source_knots:
        parser.error(
            "--candidate-knots must cover the maximum source knot count "
            f"({max_source_knots})"
        )
    if args.min_knot_gap * (args.candidate_knots + 1) >= 1.0:
        parser.error("--min-knot-gap leaves no free parameter interval")
    nonnegative = (
        args.lambda_poly,
        args.lambda_knot,
        args.weight_decay,
        args.selector_lr,
        args.lambda_fit,
        args.lambda_threshold_violation,
        args.lambda_one_shot_surrogate_fit,
        args.lambda_one_shot_surrogate_threshold,
        args.lambda_true_params,
        args.lambda_candidate_coverage,
        args.lambda_candidate_repulsion,
        args.lambda_keep,
        args.lambda_remove_action,
        args.lambda_knot_position,
        args.lambda_deletion_cost,
        args.lambda_teacher_risk,
        args.lambda_teacher_ranking,
        args.lambda_teacher_distribution,
        args.lambda_teacher_critical_recall,
        args.lambda_teacher_false_positive,
        args.lambda_teacher_count,
        args.lambda_policy_count,
        args.lambda_canonical_selection,
        args.lambda_joint_position,
        args.lambda_joint_fit,
        args.lambda_joint_threshold_violation,
        args.lambda_complexity,
    )
    if any(value < 0.0 for value in nonnegative):
        parser.error("regularization and loss weights must be non-negative")
    if not 0.0 <= args.deployment_pass_rate_target <= 1.0:
        parser.error("--deployment-pass-rate-target must lie in [0, 1]")
    if not 0.5 < args.initial_keep_probability < 1.0:
        parser.error("--initial-keep-probability must lie strictly in (0.5, 1)")
    if args.teacher_ranking_margin < 0.0:
        parser.error("--teacher-ranking-margin must be non-negative")
    if args.one_shot_safety_sigma < 0.0:
        parser.error("--one-shot-safety-sigma must be non-negative")
    if args.one_shot_selector_layers < 1:
        parser.error("--one-shot-selector-layers must be positive")
    if args.one_shot_coverage_bins < 0:
        parser.error("--one-shot-coverage-bins must be non-negative")
    if args.candidate_local_attention_bandwidth < 0.0:
        parser.error("--candidate-local-attention-bandwidth must be non-negative")
    if any(value <= 0.0 for value in args.candidate_coverage_tolerances):
        parser.error("--candidate-coverage-tolerances must be positive")
    if not 0.0 < args.one_shot_max_position_shift < 0.5:
        parser.error("--one-shot-max-position-shift must lie in (0,0.5)")

    effective_lambda_policy_count = args.lambda_policy_count
    if args.one_shot_selection_policy == "threshold":
        effective_lambda_policy_count = 0.0
        if args.lambda_policy_count:
            print(
                "Threshold selection does not use probability-mass cardinality; "
                "disabling --lambda-policy-count.",
                flush=True,
            )

    torch.manual_seed(args.train_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_config: dict[str, object] = {
        "num_points": args.num_points,
        "point_dim": args.point_dim,
        "min_control_points": args.min_control_points,
        "max_control_points": args.max_control_points,
        "noise_std": args.noise_std,
        "knot_nonuniformity": args.knot_nonuniformity,
        "sampling_nonuniformity": args.sampling_nonuniformity,
        "turn_strength": args.turn_strength,
        # Canonical proposal labels and the offline teacher share one RMS scale.
        "canonical_knot_tolerance": args.fit_tolerance,
        "normalize": True,
        "return_ground_truth": True,
    }
    train_set = SyntheticCubicBSplineDataset(
        size=args.train_size,
        seed=args.train_seed,
        resample_each_epoch=args.resample_train_each_epoch,
        **dataset_config,
    )
    val_set = SyntheticCubicBSplineDataset(
        size=args.val_size,
        seed=args.val_seed,
        resample_each_epoch=False,
        **dataset_config,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    proposal_initializer_checkpoint: dict[str, object] | None = None
    saved_proposal_model_config: Mapping[str, object] = {}
    saved_proposal_bandwidth = 0.0
    if args.proposal_checkpoint is not None:
        proposal_initializer_checkpoint = torch.load(
            args.proposal_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        saved_model_config = proposal_initializer_checkpoint.get("model_config", {})
        if isinstance(saved_model_config, Mapping):
            saved_proposal_model_config = saved_model_config
            saved_proposal_bandwidth = float(
                saved_model_config.get("candidate_local_attention_bandwidth", 0.0)
            )

    effective_local_attention_bandwidth = args.candidate_local_attention_bandwidth
    if args.candidate_pretrain_epochs == 0:
        # No adaptation means the proposal forward must remain byte-for-byte
        # compatible with its source checkpoint.  Applying v11's Gaussian
        # bias to old global-attention weights silently degrades recall.
        effective_local_attention_bandwidth = saved_proposal_bandwidth
        if (
            effective_local_attention_bandwidth
            != args.candidate_local_attention_bandwidth
        ):
            print(
                "Proposal adaptation is disabled; preserving checkpoint local "
                "attention bandwidth "
                f"{effective_local_attention_bandwidth:g} instead of requested "
                f"{args.candidate_local_attention_bandwidth:g}.",
                flush=True,
            )

    model_config: dict[str, object] = {
        "point_dim": args.point_dim,
        "degree": 3,
        "hidden_dim": args.hidden_dim,
        "encoder_layers": args.encoder_layers,
        "max_internal_knots": args.candidate_knots,
        "min_parameter_gap": 1e-4,
        "min_knot_gap": args.min_knot_gap,
        "gap_parameterization": "strict",
        "lambda_poly": args.lambda_poly,
        "lambda_knot": args.lambda_knot,
        "structure_mode": "candidate_pruning_one_shot",
        "structure_attention_heads": args.attention_heads,
        "geometry_feature_mode": "chord_derivatives",
        "pruning_residual_bandwidth": args.pruning_residual_bandwidth,
        "pruning_initial_keep_probability": args.initial_keep_probability,
        "one_shot_fixed_proposal_geometry": True,
        "one_shot_selection_policy": args.one_shot_selection_policy,
        "one_shot_safety_sigma": args.one_shot_safety_sigma,
        "one_shot_selector_layers": args.one_shot_selector_layers,
        "one_shot_coverage_bins": args.one_shot_coverage_bins,
        "candidate_local_attention_bandwidth": effective_local_attention_bandwidth,
        "one_shot_joint_position_refinement": True,
        "one_shot_max_position_shift": args.one_shot_max_position_shift,
        "compute_first_derivative": False,
    }
    if args.candidate_pretrain_epochs == 0:
        if saved_proposal_model_config:
            semantic_mismatches = _proposal_semantic_mismatches(
                model_config,
                saved_proposal_model_config,
            )
            if semantic_mismatches:
                parser.error(
                    "fixed proposal checkpoint semantics do not match the requested "
                    "model; use matching arguments or enable proposal adaptation:\n  "
                    + "\n  ".join(semantic_mismatches)
                )
        else:
            print(
                "Warning: proposal checkpoint has no model_config; assuming "
                "historical global attention and unable to verify other proposal "
                "semantics.",
                flush=True,
            )
    model = SplineFittingNetwork(**model_config)
    if proposal_initializer_checkpoint is not None and args.candidate_pretrain_epochs:
        # A v10 checkpoint is a useful initializer, but v11's local proposal
        # attention still needs a short adaptation stage.  Previously the
        # checkpoint argument was silently ignored whenever pretraining was
        # enabled, which made migration unexpectedly start from scratch.
        initializer_state = proposal_initializer_checkpoint.get(
            "model_state_dict", proposal_initializer_checkpoint
        )
        model.load_state_dict(initializer_state, strict=True)
        print(
            f"Initialized v11 proposal adaptation from: {args.proposal_checkpoint}",
            flush=True,
        )
    proposal_weights = CandidatePruningLossWeights(
        fit=args.lambda_fit,
        threshold_violation=args.lambda_threshold_violation,
        true_parameter=args.lambda_true_params,
        candidate_coverage=args.lambda_candidate_coverage,
        candidate_repulsion=args.lambda_candidate_repulsion,
        keep=args.lambda_keep,
        remove_action=0.0,
        knot_position=args.lambda_knot_position,
        count_consistency=0.0,
        deletion_cost=0.0,
        teacher_risk=args.lambda_teacher_risk,
        teacher_ranking=args.lambda_teacher_ranking,
        teacher_distribution=args.lambda_teacher_distribution,
        teacher_critical_recall=args.lambda_teacher_critical_recall,
        teacher_false_positive=args.lambda_teacher_false_positive,
        teacher_count=args.lambda_teacher_count,
        policy_count=effective_lambda_policy_count,
        canonical_selection=args.lambda_canonical_selection,
        complexity=args.lambda_complexity,
    )
    pretrain_weights = replace(
        proposal_weights,
        keep=0.0,
        remove_action=0.0,
        count_consistency=0.0,
        deletion_cost=0.0,
        teacher_risk=0.0,
        teacher_ranking=0.0,
        teacher_distribution=0.0,
        teacher_critical_recall=0.0,
        teacher_false_positive=0.0,
        teacher_count=0.0,
        policy_count=0.0,
        canonical_selection=0.0,
        complexity=0.0,
    )
    one_shot_weights = replace(
        proposal_weights,
        # The selected mask is supervised by an offline teacher computed with
        # the exact deployment B-spline solver.  The truncated-power curve is
        # deliberately diagnostic-only here: its ST gradient previously made
        # retaining every candidate the easiest way to reduce violation loss.
        fit=args.lambda_one_shot_surrogate_fit,
        threshold_violation=args.lambda_one_shot_surrogate_threshold,
        true_parameter=0.0,
        candidate_coverage=0.0,
        candidate_repulsion=0.0,
        knot_position=0.0,
    )
    calibration_weights = replace(
        one_shot_weights,
        fit=args.lambda_joint_fit,
        threshold_violation=args.lambda_joint_threshold_violation,
        knot_position=args.lambda_joint_position,
    )

    def make_loss(
        weights: CandidatePruningLossWeights,
        *,
        exact_deletion_supervision: bool,
    ) -> CandidatePruningLoss:
        return CandidatePruningLoss(
            weights,
            knot_position_beta=args.knot_position_beta,
            candidate_match_tolerance=args.candidate_match_tolerance,
            fit_tolerance=args.fit_tolerance,
            positive_keep_weight=args.positive_keep_weight,
            exact_deletion_supervision=exact_deletion_supervision,
            deletion_smoothness_weight=1e-6,
            deletion_control_ridge=0.0,
            teacher_ranking_margin=args.teacher_ranking_margin,
            candidate_coverage_tolerances=tuple(args.candidate_coverage_tolerances),
            position_aware_distribution=True,
            joint_position_supervision=True,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    trainer = Trainer(
        model,
        make_loss(pretrain_weights, exact_deletion_supervision=False),
        optimizer,
        device,
        knot_match_tolerance=args.candidate_match_tolerance,
        log_every_batches=args.log_every_batches,
        deployment_pass_rate_target=args.deployment_pass_rate_target,
        show_progress=args.progress,
    )

    print(
        "v11 objective: high-recall local proposals plus joint one-shot "
        "Keep/position prediction "
        "distilled from "
        f"an offline Hard-RMS teacher at epsilon={args.fit_tolerance:g}",
        flush=True,
    )
    print(
        "Data: "
        f"train={args.train_size}, val={args.val_size}, points={args.num_points}, "
        f"source K={args.min_control_points - 4}..{max_source_knots}, "
        f"candidate Kc={args.candidate_knots}",
        flush=True,
    )
    print(
        "Hard-RMS search runs once offline to create cached masks and deletion "
        "risks. Deployment uses one LearnedKeep mask and one B-spline refit.",
        flush=True,
    )

    histories: dict[str, list[dict[str, float]]] = {}
    if args.candidate_pretrain_epochs:
        # Keep scores stay at their conservative all-open initialization while
        # proposal coverage, parameters and position refinement are pretrained.
        for module_name in (
            "keep_head",
            "adaptive_threshold_head",
            "position_to_keep_feedback",
            "one_shot_selector_attention",
            "one_shot_selector_attention_norm",
            "one_shot_selector_feed_forward",
            "one_shot_selector_output_norm",
            "one_shot_selector_extra_attention",
            "one_shot_selector_extra_attention_norm",
            "one_shot_selector_extra_feed_forward",
            "one_shot_selector_extra_output_norm",
            "joint_preliminary_position_projection",
            "joint_preliminary_position_norm",
            "joint_preliminary_position_head",
            "joint_position_to_keep_feedback",
            "joint_position_to_keep_norm",
            "joint_final_position_projection",
            "joint_final_position_norm",
            "joint_final_position_head",
        ):
            module = getattr(model.pruning_head, module_name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        proposal_path = args.output.with_name(
            args.output.stem + "_proposal" + args.output.suffix
        )
        histories["candidate_pretrain"] = trainer.fit(
            train_loader,
            val_loader,
            args.candidate_pretrain_epochs,
            gate_warmup_epochs=args.candidate_pretrain_epochs,
            checkpoint_path=proposal_path,
            stage_name="candidate_pretrain",
        )
        proposal_checkpoint = torch.load(
            proposal_path, map_location=device, weights_only=True
        )
        model.load_state_dict(proposal_checkpoint["model_state_dict"], strict=True)
        # Trainer checkpoints are intentionally generic.  Persist proposal
        # semantics here so a later --candidate-pretrain-epochs 0 run can
        # reproduce this exact local-attention forward and safely fingerprint
        # its offline teacher cache.
        proposal_checkpoint.update(
            {
                "model_config": dict(model_config),
                "dataset_config": dict(dataset_config),
                "dataset_type": "synthetic_open_cubic_bspline",
                "objective_version": V11_JOINT_ONE_SHOT_PRUNING_OBJECTIVE_VERSION,
                "loss_config": {
                    "weights": asdict(pretrain_weights),
                    "candidate_match_tolerance": args.candidate_match_tolerance,
                    "candidate_coverage_tolerances": list(
                        args.candidate_coverage_tolerances
                    ),
                    "fit_tolerance": args.fit_tolerance,
                },
                "training_config": {
                    "stage": "candidate_pretrain",
                    "candidate_pretrain_epochs": args.candidate_pretrain_epochs,
                    "candidate_local_attention_bandwidth": (
                        effective_local_attention_bandwidth
                    ),
                },
                "deployment_config": {"role": "fixed_proposal_for_offline_teacher"},
                "selected_stage": "candidate_pretrain",
                "stage_histories": {
                    "candidate_pretrain": histories["candidate_pretrain"]
                },
            }
        )
        torch.save(proposal_checkpoint, proposal_path)
        print(
            f"Restored best proposal checkpoint from epoch "
            f"{proposal_checkpoint['epoch']} before teacher generation.",
            flush=True,
        )
    else:
        if proposal_initializer_checkpoint is None:
            raise RuntimeError("proposal checkpoint was not loaded")
        proposal_checkpoint = proposal_initializer_checkpoint
        state = proposal_checkpoint.get("model_state_dict", proposal_checkpoint)
        model.load_state_dict(state, strict=True)
        print(
            f"Loaded fixed proposal checkpoint: {args.proposal_checkpoint}",
            flush=True,
        )

    model.set_force_open_gates(False)
    _reset_one_shot_selector_to_neutral(
        model,
        initial_probability=args.initial_keep_probability,
    )
    for parameter in model.pruning_head.parameters():
        parameter.requires_grad_(True)

    proposal_fingerprint = _model_fingerprint(model)
    print("Fingerprinting the fixed train/validation samples...", flush=True)
    train_dataset_fingerprint = _dataset_fingerprint(
        train_set,
        split="train",
        seed=args.train_seed,
        dataset_config=dataset_config,
    )
    val_dataset_fingerprint = _dataset_fingerprint(
        val_set,
        split="validation",
        seed=args.val_seed,
        dataset_config=dataset_config,
    )
    teacher_base_config = {
        "error_tolerance": args.fit_tolerance,
        "temperature": args.teacher_risk_temperature,
        "min_internal_knots": 0,
        "degree": 3,
        "smoothness_weight": 1e-6,
        "control_ridge": 0.0,
        "interpolate_endpoints": True,
        "proposal_fingerprint": proposal_fingerprint,
    }
    teacher_config = OneShotTeacherConfig(
        **teacher_base_config,
        dataset_fingerprint=train_dataset_fingerprint,
    )
    val_teacher_config = OneShotTeacherConfig(
        **teacher_base_config,
        dataset_fingerprint=val_dataset_fingerprint,
    )
    teacher_cache_dir = args.teacher_cache_dir or (
        args.output.parent / f"{args.output.stem}_teacher"
    )
    train_teacher = _build_or_load_teacher_cache(
        model=model,
        dataset=train_set,
        cache_path=teacher_cache_dir / "train.pt",
        config=teacher_config,
        batch_size=args.teacher_batch_size,
        device=device,
        reuse=args.reuse_teacher_cache,
    )
    val_teacher = _build_or_load_teacher_cache(
        model=model,
        dataset=val_set,
        cache_path=teacher_cache_dir / "val.pt",
        config=val_teacher_config,
        batch_size=args.teacher_batch_size,
        device=device,
        reuse=args.reuse_teacher_cache,
    )
    teacher_train_set = TeacherAugmentedDataset(train_set, train_teacher)
    teacher_val_set = TeacherAugmentedDataset(val_set, val_teacher)
    teacher_train_loader = DataLoader(
        teacher_train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )
    teacher_val_loader = DataLoader(
        teacher_val_set,
        batch_size=args.batch_size,
    )

    # Offline labels bind to immutable proposal geometry.  Initial distillation
    # learns only the selection rule; the later low-LR stage activates the
    # separate deployment-position path without changing teacher slot identity.
    selection_parameters = _set_one_shot_trainable(
        model,
        calibrate_positions=False,
    )
    trainer.optimizer = torch.optim.AdamW(
        selection_parameters,
        lr=args.selector_lr,
        weight_decay=args.weight_decay,
    )
    trainer.loss_fn = make_loss(
        one_shot_weights,
        exact_deletion_supervision=False,
    ).to(device)
    distill_epochs = (
        args.epochs
        - args.candidate_pretrain_epochs
        - args.keep_position_calibration_epochs
    )
    distill_path = (
        args.output
        if args.keep_position_calibration_epochs == 0
        else args.output.with_name(args.output.stem + "_distill" + args.output.suffix)
    )
    effective_selector_warmup = min(
        args.selector_checkpoint_warmup_epochs,
        max(distill_epochs - 1, 0),
    )
    histories["one_shot_distillation"] = trainer.fit(
        teacher_train_loader,
        teacher_val_loader,
        distill_epochs,
        checkpoint_path=distill_path,
        epoch_offset=args.candidate_pretrain_epochs,
        stage_name="one_shot_selection_distillation",
        deployment_validation=True,
        checkpoint_selection_start_epoch=effective_selector_warmup,
    )

    candidate_checkpoints = [distill_path]
    if args.keep_position_calibration_epochs:
        distill_checkpoint = torch.load(
            distill_path, map_location=device, weights_only=True
        )
        model.load_state_dict(distill_checkpoint["model_state_dict"], strict=True)
        calibration_parameters = _set_one_shot_trainable(
            model,
            calibrate_positions=True,
        )
        trainer.optimizer = torch.optim.AdamW(
            calibration_parameters,
            lr=args.selector_lr * 0.1,
            weight_decay=args.weight_decay,
        )
        trainer.loss_fn = make_loss(
            calibration_weights,
            exact_deletion_supervision=False,
        ).to(device)
        calibration_path = args.output.with_name(
            args.output.stem + "_calibrated" + args.output.suffix
        )
        histories["one_shot_selector_calibration"] = trainer.fit(
            teacher_train_loader,
            teacher_val_loader,
            args.keep_position_calibration_epochs,
            checkpoint_path=calibration_path,
            epoch_offset=args.candidate_pretrain_epochs + distill_epochs,
            stage_name="one_shot_selector_calibration",
            deployment_validation=True,
        )
        candidate_checkpoints.append(calibration_path)

    best_candidates = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in candidate_checkpoints
    ]
    best = max(
        best_candidates,
        key=lambda checkpoint: tuple(checkpoint.get("selection_rank", [])),
    )
    selected_position_calibration = best.get("stage") == "one_shot_selector_calibration"
    selected_loss_weights = (
        calibration_weights if selected_position_calibration else one_shot_weights
    )

    loss_config: dict[str, object] = {
        "weights": asdict(selected_loss_weights),
        "proposal_weights": asdict(pretrain_weights),
        "knot_position_beta": args.knot_position_beta,
        "candidate_match_tolerance": args.candidate_match_tolerance,
        "fit_tolerance": args.fit_tolerance,
        "positive_keep_weight": args.positive_keep_weight,
        "exact_deletion_supervision": False,
        "deletion_smoothness_weight": 1e-6,
        "deletion_control_ridge": 0.0,
        "one_shot_teacher": True,
        "fixed_proposal_geometry": True,
        "selector_adapter": "two_pass_keep_position_feedback",
        "selector_layers": args.one_shot_selector_layers,
        "straight_through_keep_gate": True,
        "surrogate_selection_role": (
            "diagnostic_during_distillation_position_feasibility_during_calibration"
        ),
        "teacher_mask_loss": (
            "teacher_bce_dice_ranking_plus_canonical_set_and_policy_count"
        ),
        "selection_policy": args.one_shot_selection_policy,
        "selection_safety_sigma": args.one_shot_safety_sigma,
        "selection_coverage_bins": args.one_shot_coverage_bins,
        "teacher_risk_temperature": args.teacher_risk_temperature,
        "teacher_ranking_margin": args.teacher_ranking_margin,
        "candidate_coverage_tolerances": list(args.candidate_coverage_tolerances),
        "position_aware_distribution": True,
        "joint_position_supervision": True,
        "selected_checkpoint_position_calibrated": selected_position_calibration,
        "candidate_local_attention_bandwidth": (effective_local_attention_bandwidth),
        "joint_position_max_shift": args.one_shot_max_position_shift,
        "joint_calibration_fit_weight": args.lambda_joint_fit,
        "joint_calibration_threshold_weight": (args.lambda_joint_threshold_violation),
        "count_consistency_is_deployment_rule": False,
        "analytic_deletion_cost_is_auxiliary_only": True,
    }
    training_config: dict[str, object] = {
        "structure_mode": "candidate_pruning_one_shot",
        "epochs": args.epochs,
        "candidate_pretrain_epochs": args.candidate_pretrain_epochs,
        "proposal_checkpoint": (
            str(args.proposal_checkpoint)
            if args.proposal_checkpoint is not None
            else None
        ),
        "one_shot_distillation_epochs": distill_epochs,
        "selector_calibration_epochs": args.keep_position_calibration_epochs,
        "keep_position_calibration_epochs": args.keep_position_calibration_epochs,
        "joint_finetune_epochs": args.keep_position_calibration_epochs,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "train_seed": args.train_seed,
        "val_seed": args.val_seed,
        "resample_train_each_epoch": args.resample_train_each_epoch,
        "fit_tolerance": args.fit_tolerance,
        "deployment_pass_rate_target": args.deployment_pass_rate_target,
        "candidate_match_tolerance": args.candidate_match_tolerance,
        "offline_hard_rms_teacher": True,
        "teacher_cache_dir": str(teacher_cache_dir),
        "teacher_config": teacher_config.as_dict(),
        "train_teacher_config": teacher_config.as_dict(),
        "val_teacher_config": val_teacher_config.as_dict(),
        "proposal_fingerprint": proposal_fingerprint,
        "weight_decay": args.weight_decay,
        "proposal_learning_rate": args.lr,
        "selector_learning_rate": args.selector_lr,
        "selector_checkpoint_warmup_epochs": effective_selector_warmup,
        "selection_policy": args.one_shot_selection_policy,
        "selection_safety_sigma": args.one_shot_safety_sigma,
        "selection_coverage_bins": args.one_shot_coverage_bins,
        "candidate_local_attention_bandwidth": (effective_local_attention_bandwidth),
        "joint_position_max_shift": args.one_shot_max_position_shift,
        "requested_policy_count_weight": args.lambda_policy_count,
        "effective_policy_count_weight": effective_lambda_policy_count,
        "joint_calibration_fit_weight": args.lambda_joint_fit,
        "joint_calibration_threshold_weight": (args.lambda_joint_threshold_violation),
    }
    deployment_config: dict[str, object] = {
        "selection_rule": (
            "v11_joint_position_mass_topk_then_single_refit"
            if args.one_shot_selection_policy == "mass_topk"
            else "v11_joint_position_threshold_then_single_refit"
        ),
        "error_tolerance": args.fit_tolerance,
        "min_internal_knots": 0,
        "smoothness_weight": 1e-6,
        "control_ridge": 0.0,
        "interpolate_endpoints": True,
        "network_forward_passes": 1,
        "standard_bspline_refits": 1,
        "learned_keep_threshold_is_final": (
            args.one_shot_selection_policy == "threshold"
        ),
        "learned_keep_selection_policy": args.one_shot_selection_policy,
        "uncertainty_safety_sigma": args.one_shot_safety_sigma,
        "parameter_domain_coverage_bins": args.one_shot_coverage_bins,
        "hard_rms_pruning_at_deployment": False,
        "proposal_geometry_frozen_after_teacher": True,
        "deployment_positions_jointly_refined": True,
        "selected_checkpoint_position_calibrated": selected_position_calibration,
        "threshold_guarantee": "statistical_not_per_sample_exact",
        "checkpoint_selection": (
            "min_retained_knots_subject_to_validation_pass_rate_target"
        ),
        "global_minimum_guaranteed": False,
        "forward_internal_surrogate_solves": 2,
    }

    best = _checkpoint_with_metadata(
        best,
        model_config=model_config,
        dataset_config=dataset_config,
        loss_config=loss_config,
        training_config=training_config,
        deployment_config=deployment_config,
        histories=histories,
    )
    torch.save(best, args.output)

    last_path = args.output.with_name(args.output.stem + "_last" + args.output.suffix)
    last = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "stage": "one_shot_training_last",
        "selection_metric": "last_epoch_not_selected",
        "history": histories[
            "one_shot_selector_calibration"
            if args.keep_position_calibration_epochs
            else "one_shot_distillation"
        ],
    }
    last_position_calibrated = bool(args.keep_position_calibration_epochs)
    last_loss_config = {
        **loss_config,
        "weights": asdict(
            calibration_weights if last_position_calibrated else one_shot_weights
        ),
        "selected_checkpoint_position_calibrated": last_position_calibrated,
    }
    last_deployment_config = {
        **deployment_config,
        "selected_checkpoint_position_calibrated": last_position_calibrated,
    }
    last = _checkpoint_with_metadata(
        last,
        model_config=model_config,
        dataset_config=dataset_config,
        loss_config=last_loss_config,
        training_config=training_config,
        deployment_config=last_deployment_config,
        histories=histories,
    )
    torch.save(last, last_path)
    print(
        f"Selected epoch {best['epoch']} | one-shot deployment pass="
        f"{best.get('best_threshold_satisfied_rate', float('nan')):.3f} | "
        f"K={best.get('best_deployment_retained_knot_count', float('nan')):.2f} | "
        f"teacher mask F1="
        f"{best.get('best_teacher_mask_f1', float('nan')):.3f}",
        flush=True,
    )
    print(f"Saved best checkpoint: {args.output}", flush=True)
    print(f"Saved last checkpoint: {last_path}", flush=True)


if __name__ == "__main__":
    main()
