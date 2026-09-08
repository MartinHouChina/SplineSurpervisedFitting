from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import (  # noqa: E402
    V14_PARAMETER_FEEDBACK_LOSS_CONFIG,
    V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION,
    V15_DEPLOYMENT_ALIGNED_LOSS_CONFIG,
    V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION,
    migrate_model_config,
)
from spline_fitting.data.synthetic import (  # noqa: E402
    SyntheticCubicBSplineDataset,
)
from spline_fitting.data.real_world import RealWorldCurveDataset  # noqa: E402
from spline_fitting.evaluation.bspline_inference import (  # noqa: E402
    refit_hard_gated_bspline_batch,
)
from spline_fitting.losses import (  # noqa: E402
    ParameterFeedbackLoss,
    ParameterFeedbackLossWeights,
)
from spline_fitting.models import SplineFittingNetwork  # noqa: E402
from spline_fitting.training import Trainer  # noqa: E402


_FEEDBACK_PREFIX = "parameter_feedback_head."
_JOINT_PREFIX = "joint_parameter_structure_head."
_DATASET_RUNTIME_KEYS = {
    "size",
    "seed",
    "cache_samples",
    "resample_each_epoch",
    "epoch_seed_stride",
    "dtype",
}


class _TrainerCompatibleFeedbackLoss(nn.Module):
    """Expose refit configuration consumed by the generic Trainer."""

    def __init__(
        self,
        loss: ParameterFeedbackLoss,
        *,
        canonical_structure_supervision: bool = False,
    ) -> None:
        super().__init__()
        self.loss = loss
        self.canonical_structure_supervision = bool(
            canonical_structure_supervision
        )
        self.fit_tolerance = loss.fit_tolerance
        self.deletion_smoothness_weight = 1e-6
        self.deletion_control_ridge = 0.0

    def forward(
        self,
        output: dict[str, torch.Tensor],
        points: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        if (
            self.canonical_structure_supervision
            and "teacher_retained_mask" not in kwargs
        ):
            true_knots = kwargs.get("true_internal_knots")
            true_mask = kwargs.get("true_internal_knot_mask")
            if isinstance(true_knots, torch.Tensor) and isinstance(
                true_mask, torch.Tensor
            ):
                candidates = output.get("proposal_internal_knots")
                if candidates is None:
                    raise KeyError(
                        "canonical joint supervision requires proposal_internal_knots"
                    )
                teacher_keep = torch.zeros_like(candidates, dtype=torch.bool)
                padded_knots = torch.zeros_like(candidates)
                padded_mask = torch.zeros_like(candidates, dtype=torch.bool)
                for batch_index in range(candidates.shape[0]):
                    target = true_knots[batch_index, true_mask[batch_index]]
                    pairs = ParameterFeedbackLoss._ordered_pair_indices(
                        candidates[batch_index], target
                    )
                    if pairs:
                        candidate_indices = torch.tensor(
                            [left for left, _ in pairs],
                            dtype=torch.long,
                            device=candidates.device,
                        )
                        teacher_keep[batch_index, candidate_indices] = True
                    count = min(int(target.numel()), candidates.shape[-1])
                    if count:
                        padded_knots[batch_index, :count] = target[:count]
                        padded_mask[batch_index, :count] = True
                kwargs = {
                    **kwargs,
                    "teacher_retained_mask": teacher_keep,
                    "teacher_internal_knots": padded_knots,
                    "teacher_internal_knot_mask": padded_mask,
                    "teacher_count": padded_mask.sum(dim=-1),
                }
        return self.loss(output, points, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "No offline teacher is rebuilt. Attach or continue calibrating "
            "the v15 deployment-aligned late parameter/structure feedback "
            "loop on an existing one-shot checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--metadata-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional fully packaged proposal checkpoint supplying model/dataset "
            "metadata when --checkpoint is a generic Trainer structure checkpoint "
            "such as *_calibrated.pt from an interrupted multi-stage run. The "
            "structure weights come from --checkpoint; any pre-attached feedback "
            "head is reset before calibration."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Default: <checkpoint stem>_v15_deployment_aligned.pt; the "
            "legacy no-joint path uses _v14_parameter_feedback.pt."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--val-seed", type=int, default=10000)
    parser.add_argument(
        "--real-world-manifest",
        type=Path,
        action="append",
        default=None,
        help=(
            "Use one or more prepared real-world JSONL manifests instead of "
            "synthetic curves. Repeat the option to mix sources. Feedback and "
            "node positions adapt from fit/threshold losses; no knot labels or "
            "KeepMask targets are invented."
        ),
    )
    parser.add_argument("--real-world-train-split", default="train")
    parser.add_argument("--real-world-val-split", default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--resample-train-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--feedback-attention-heads",
        type=int,
        default=None,
        help="Default: checkpoint structure_attention_heads.",
    )
    parser.add_argument(
        "--parameter-feedback-fusion-mode",
        choices=(
            "fast_global",
            "fast_structural",
            "gaussian_pool",
            "cross_attention",
        ),
        default=None,
        help=(
            "Default: cross_attention with joint feedback; fast_global with "
            "--no-joint-parameter-structure-feedback."
        ),
    )
    parser.add_argument("--feedback-max-logit-shift", type=float, default=0.5)
    parser.add_argument(
        "--joint-parameter-structure-feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After t0->t1 correction, jointly update Keep logits and candidate "
            "positions, select again, and relocate the final survivor set."
        ),
    )
    parser.add_argument("--joint-feedback-local-bandwidth", type=float, default=0.08)
    parser.add_argument(
        "--joint-feedback-max-keep-logit-shift", type=float, default=2.0
    )
    parser.add_argument(
        "--final-relocation-lr-scale",
        type=float,
        default=0.25,
        help=(
            "Learning-rate multiplier for the survivor-relocation block used by "
            "the final joint pass."
        ),
    )
    parser.add_argument(
        "--initial-chord-blend",
        type=float,
        default=0.6,
        help=(
            "Chord-gap weight used to seed the trainable candidate after the "
            "strict identity baseline is measured."
        ),
    )
    parser.add_argument("--fit-tolerance", type=float, default=None)
    parser.add_argument("--parameter-error-scale", type=float, default=0.02)
    parser.add_argument("--lambda-feedback-fit", type=float, default=0.0)
    parser.add_argument(
        "--lambda-feedback-threshold-violation", type=float, default=0.0
    )
    parser.add_argument("--lambda-feedback-deployment-fit", type=float, default=0.25)
    parser.add_argument(
        "--lambda-feedback-deployment-threshold", type=float, default=2.0
    )
    parser.add_argument("--lambda-feedback-true-params", type=float, default=1.00)
    parser.add_argument("--lambda-feedback-chord-prior", type=float, default=0.05)
    parser.add_argument("--lambda-feedback-identity", type=float, default=0.01)
    parser.add_argument("--lambda-feedback-joint-keep", type=float, default=1.0)
    parser.add_argument(
        "--lambda-feedback-joint-critical-recall", type=float, default=0.5
    )
    parser.add_argument("--lambda-feedback-joint-position", type=float, default=2.0)
    parser.add_argument(
        "--lambda-feedback-joint-set-position", type=float, default=1.0
    )
    parser.add_argument("--lambda-feedback-joint-spacing", type=float, default=0.5)
    parser.add_argument("--lambda-feedback-joint-count", type=float, default=2.0)
    parser.add_argument("--joint-knot-position-beta", type=float, default=0.01)
    parser.add_argument("--joint-positive-keep-weight", type=float, default=1.5)
    parser.add_argument("--deployment-pass-rate-target", type=float, default=0.97)
    parser.add_argument("--knot-match-tolerance", type=float, default=0.01)
    parser.add_argument("--log-every-batches", type=int, default=20)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(name)


def compose_structure_checkpoint(
    state_checkpoint: Mapping[str, Any],
    metadata_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Package generic Trainer weights as a feedback-free structure checkpoint.

    The main v14 trainer creates the feedback module before the earlier structure
    stages, so interrupted ``*_calibrated.pt`` files contain an unused feedback
    state but no model/dataset metadata.  Recovery must retain the calibrated
    proposal/selector/relocation weights while deliberately reattaching a fresh
    identity feedback head.
    """

    state = state_checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise KeyError("state checkpoint is missing model_state_dict")
    raw_config = metadata_checkpoint.get("model_config")
    if not isinstance(raw_config, Mapping):
        raise KeyError("metadata checkpoint is missing model_config")
    if not isinstance(metadata_checkpoint.get("dataset_config"), Mapping):
        raise KeyError("metadata checkpoint is missing dataset_config")

    packaged = dict(metadata_checkpoint)
    packaged["model_state_dict"] = {
        name: value
        for name, value in state.items()
        if not name.startswith((_FEEDBACK_PREFIX, _JOINT_PREFIX))
    }
    config = dict(raw_config)
    config["parameter_feedback_fusion"] = False
    config["joint_parameter_structure_feedback"] = False
    for name in (
        "parameter_feedback_attention_heads",
        "parameter_feedback_max_logit_shift",
        "parameter_feedback_fusion_mode",
        "joint_parameter_structure_local_bandwidth",
        "joint_parameter_structure_max_keep_logit_shift",
    ):
        config.pop(name, None)
    packaged["model_config"] = config
    packaged["stage"] = state_checkpoint.get("stage", "recovered_structure")
    packaged["epoch"] = state_checkpoint.get("epoch")
    packaged["recovered_structure_checkpoint"] = True
    return packaged


def build_real_world_split(
    manifests: list[Path],
    *,
    split: str,
    num_points: int,
    maximum_size: int,
    seed: int,
) -> Dataset:
    """Build a deterministic, optionally capped union of manifest splits."""

    parts: list[Dataset] = []
    for manifest in manifests:
        resolved = manifest.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"real-world manifest does not exist: {resolved}")
        dataset = RealWorldCurveDataset(
            resolved,
            split=split,
            num_points=num_points,
            normalize=True,
        )
        if len(dataset):
            parts.append(dataset)
    if not parts:
        raise ValueError(f"no real-world samples found for split {split!r}")
    combined: Dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    if len(combined) <= maximum_size:
        return combined
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(combined), generator=generator)[:maximum_size].tolist()
    return Subset(combined, indices)


def build_feedback_loss_weights(
    args: argparse.Namespace,
    *,
    real_world_unlabeled: bool,
) -> ParameterFeedbackLossWeights:
    """Resolve supervision without inventing labels for real-world curves."""

    joint_supervision = bool(
        args.joint_parameter_structure_feedback and not real_world_unlabeled
    )
    return ParameterFeedbackLossWeights(
        fit=args.lambda_feedback_fit,
        threshold_violation=args.lambda_feedback_threshold_violation,
        deployment_fit=args.lambda_feedback_deployment_fit,
        deployment_threshold_violation=(
            args.lambda_feedback_deployment_threshold
        ),
        true_parameter=(
            0.0 if real_world_unlabeled else args.lambda_feedback_true_params
        ),
        chord_prior=args.lambda_feedback_chord_prior,
        identity=args.lambda_feedback_identity,
        joint_keep=(args.lambda_feedback_joint_keep if joint_supervision else 0.0),
        joint_critical_recall=(
            args.lambda_feedback_joint_critical_recall
            if joint_supervision
            else 0.0
        ),
        joint_position=(
            args.lambda_feedback_joint_position if joint_supervision else 0.0
        ),
        joint_set_position=(
            args.lambda_feedback_joint_set_position if joint_supervision else 0.0
        ),
        joint_spacing=(
            args.lambda_feedback_joint_spacing if joint_supervision else 0.0
        ),
        joint_count=(args.lambda_feedback_joint_count if joint_supervision else 0.0),
    )


def _resolve_fit_tolerance(
    checkpoint: Mapping[str, Any],
    override: float | None,
) -> float:
    if override is not None:
        if override <= 0.0:
            raise ValueError("fit tolerance must be positive")
        return float(override)
    deployment = checkpoint.get("deployment_config", {})
    if isinstance(deployment, Mapping):
        for key in ("error_tolerance", "fit_tolerance", "normalized_rms_tolerance"):
            value = deployment.get(key)
            if value is not None and float(value) > 0.0:
                return float(value)
    loss_config = checkpoint.get("loss_config", {})
    if isinstance(loss_config, Mapping):
        value = loss_config.get("fit_tolerance")
        if value is not None and float(value) > 0.0:
            return float(value)
    dataset = checkpoint.get("dataset_config", {})
    if isinstance(dataset, Mapping):
        value = dataset.get("canonical_knot_tolerance")
        if value is not None and float(value) > 0.0:
            return float(value)
    return 5e-3


def _resolved_dataset_config(
    checkpoint: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    saved = checkpoint.get("dataset_config")
    if not isinstance(saved, Mapping):
        raise KeyError("checkpoint is missing dataset_config")
    accepted = set(inspect.signature(SyntheticCubicBSplineDataset).parameters)
    accepted.difference_update(_DATASET_RUNTIME_KEYS)
    config = {key: value for key, value in saved.items() if key in accepted}
    config.setdefault("num_points", int(saved.get("num_points", 192)))
    config.setdefault("point_dim", int(model_config.get("point_dim", 2)))
    degree = int(model_config.get("degree", 3))
    config.setdefault("min_control_points", degree + 5)
    config.setdefault(
        "max_control_points",
        int(model_config.get("max_internal_knots", 28)) + degree + 1,
    )
    config["normalize"] = True
    config["return_ground_truth"] = True
    return config


def build_feedback_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    attention_heads: int | None,
    max_logit_shift: float,
    joint_parameter_structure_feedback: bool = True,
    fusion_mode: str | None = None,
    joint_local_bandwidth: float = 0.08,
    joint_max_keep_logit_shift: float = 2.0,
) -> tuple[SplineFittingNetwork, dict[str, Any], list[str]]:
    """Attach the requested v14 feedback heads to a compatible checkpoint.

    Enabling the joint loop defaults to full cross-attention.  A legacy v14
    ``fast_global`` head cannot be shape-migrated into that architecture, so it
    is deliberately reattached at its zero/identity initialisation while all
    proposal, selector and relocation weights remain exact.
    """

    source_config, _ = migrate_model_config(checkpoint)
    if source_config.get("structure_mode") != "candidate_pruning_one_shot":
        raise ValueError("source checkpoint must use candidate_pruning_one_shot")
    if source_config.get("gap_parameterization", "strict") != "strict":
        raise ValueError("parameter feedback requires strict gap parameterization")
    existing_feedback = bool(source_config.get("parameter_feedback_fusion", False))
    existing_joint = bool(
        source_config.get("joint_parameter_structure_feedback", False)
    )
    if max_logit_shift <= 0.0:
        raise ValueError("feedback max logit shift must be positive")
    if joint_local_bandwidth <= 0.0:
        raise ValueError("joint local bandwidth must be positive")
    if joint_max_keep_logit_shift <= 0.0:
        raise ValueError("joint max Keep-logit shift must be positive")

    effective_heads = (
        int(source_config.get("structure_attention_heads", 4))
        if attention_heads is None
        else int(attention_heads)
    )
    target_config = dict(source_config)
    effective_fusion_mode = fusion_mode or (
        "cross_attention" if joint_parameter_structure_feedback else "fast_global"
    )
    # These are the only v14 model-semantic additions. In particular, stable
    # pilot descriptors and every proposal/selector option stay untouched.
    target_config.update(
        {
            "parameter_feedback_fusion": True,
            "parameter_feedback_attention_heads": effective_heads,
            "parameter_feedback_max_logit_shift": float(max_logit_shift),
            "parameter_feedback_fusion_mode": effective_fusion_mode,
            "joint_parameter_structure_feedback": bool(
                joint_parameter_structure_feedback
            ),
            "joint_parameter_structure_local_bandwidth": float(
                joint_local_bandwidth
            ),
            "joint_parameter_structure_max_keep_logit_shift": float(
                joint_max_keep_logit_shift
            ),
        }
    )
    model = SplineFittingNetwork(**target_config)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise KeyError("checkpoint is missing model_state_dict")
    source_fusion_mode = str(
        source_config.get("parameter_feedback_fusion_mode", "fast_global")
    )
    reattach_feedback = existing_feedback and (
        source_fusion_mode != effective_fusion_mode
    )
    reattach_joint = existing_joint and (
        not joint_parameter_structure_feedback or reattach_feedback
    )
    filtered_state = {
        name: value
        for name, value in state.items()
        if not (reattach_feedback and name.startswith(_FEEDBACK_PREFIX))
        and not (reattach_joint and name.startswith(_JOINT_PREFIX))
    }
    incompatible = model.load_state_dict(filtered_state, strict=False)
    missing = list(incompatible.missing_keys)
    allowed_missing_prefixes = []
    if not existing_feedback or reattach_feedback:
        allowed_missing_prefixes.append(_FEEDBACK_PREFIX)
    if joint_parameter_structure_feedback and (not existing_joint or reattach_joint):
        allowed_missing_prefixes.append(_JOINT_PREFIX)
    invalid_missing = [
        key for key in missing if not key.startswith(tuple(allowed_missing_prefixes))
    ]
    if invalid_missing or incompatible.unexpected_keys:
        details = []
        if invalid_missing:
            details.append("invalid missing keys=" + ", ".join(invalid_missing))
        if incompatible.unexpected_keys:
            details.append("unexpected keys=" + ", ".join(incompatible.unexpected_keys))
        raise RuntimeError("source checkpoint is incompatible: " + "; ".join(details))
    if existing_feedback and not reattach_feedback and any(
        key.startswith(_FEEDBACK_PREFIX) for key in missing
    ):
        raise RuntimeError(
            "checkpoint config enables parameter feedback but its state is incomplete: "
            + ", ".join(missing)
        )
    if not existing_feedback and not any(
        key.startswith(_FEEDBACK_PREFIX) for key in missing
    ):
        raise RuntimeError("source state unexpectedly already contains the v14 head")
    zero_init_prefixes = (
        "gap_logit_head.",
        "fast_gap_logit_head.",
        "fast_chord_adjust_head.",
        "global_chord_adjust_head.",
    )
    zero_initialized = (
        all(
            torch.count_nonzero(parameter.detach()) == 0
            for name, parameter in model.parameter_feedback_head.named_parameters()
            if name.startswith(zero_init_prefixes)
        )
        and int(
            torch.count_nonzero(
                model.parameter_feedback_head.chord_blend_weight.detach()
            )
        )
        == 0
    )
    if (not existing_feedback or reattach_feedback) and not zero_initialized:
        raise RuntimeError("new parameter-feedback output projection is not identity")
    if joint_parameter_structure_feedback and (
        not existing_joint or reattach_feedback
    ):
        joint_head = model.joint_parameter_structure_head
        terminal_names = (
            "token_delta_head.",
            "keep_delta_head.",
            "position_delta_head.",
        )
        joint_zero = all(
            torch.count_nonzero(parameter.detach()) == 0
            for name, parameter in joint_head.named_parameters()
            if name.startswith(terminal_names)
        ) and int(torch.count_nonzero(joint_head.relocation_scale.detach())) == 0
        if not joint_zero:
            raise RuntimeError("new joint feedback output projections are not identity")
    model.set_force_open_gates(False)
    return model, target_config, missing


_FINAL_RELOCATION_MODULES = (
    "survivor_relative_projection",
    "survivor_relocation_input_norm",
    "survivor_relocation_attention",
    "survivor_relocation_attention_norm",
    "survivor_relocation_feed_forward",
    "survivor_relocation_output_norm",
    "survivor_relocation_head",
)


def feedback_training_parameter_groups(
    model: SplineFittingNetwork,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return feedback/joint parameters and separately scaled relocation ones."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = getattr(model, "parameter_feedback_head", None)
    if head is None:
        raise RuntimeError("model has no parameter-feedback head")
    if head.fusion_mode == "fast_global":
        trainable = [head.chord_blend_weight]
        for module in (
            head.global_input_projection,
            head.global_chord_adjust_head,
        ):
            trainable.extend(module.parameters())
    elif head.fusion_mode == "fast_structural":
        trainable = [head.chord_blend_weight]
        for module in (
            head.fast_input_projection,
            head.fast_gap_logit_head,
            head.fast_chord_adjust_head,
        ):
            trainable.extend(module.parameters())
    else:
        trainable = list(head.parameters())
    joint_head = getattr(model, "joint_parameter_structure_head", None)
    relocation: list[nn.Parameter] = []
    if joint_head is not None:
        trainable.extend(joint_head.parameters())
        # The relocation implementation is shared by the preliminary and final
        # pass. Its smaller learning-rate group prevents the already calibrated
        # first pass from being overwritten while adapting the final survivors.
        for module_name in _FINAL_RELOCATION_MODULES:
            module = getattr(model.pruning_head, module_name, None)
            if module is not None:
                relocation.extend(module.parameters())

    primary_ids = {id(parameter) for parameter in trainable}
    relocation = [
        parameter for parameter in relocation if id(parameter) not in primary_ids
    ]
    for parameter in (*trainable, *relocation):
        parameter.requires_grad_(True)
    if not trainable and not relocation:
        raise RuntimeError("parameter-feedback head has no trainable parameters")
    return trainable, relocation


def freeze_except_parameter_feedback(model: SplineFittingNetwork) -> list[nn.Parameter]:
    """Compatibility helper returning all parameters trained by this script."""

    primary, relocation = feedback_training_parameter_groups(model)
    return [*primary, *relocation]


def _structure_fingerprint_update(
    digest: Any,
    knots: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    quantized = torch.round(knots.detach().cpu().to(torch.float64) * 1e7).to(
        torch.int64
    )
    digest.update(quantized.contiguous().numpy().tobytes())
    digest.update(mask.detach().cpu().to(torch.uint8).contiguous().numpy().tobytes())


@torch.inference_mode()
def evaluate_standard_bspline_deployment(
    model: SplineFittingNetwork,
    loader: DataLoader,
    device: torch.device,
    *,
    fit_tolerance: float,
) -> dict[str, float | int | str]:
    """Evaluate the actual hard subset and standard B-spline endpoint refit."""

    model.eval()
    mse_values: list[torch.Tensor] = []
    rms_values: list[torch.Tensor] = []
    retained_counts: list[torch.Tensor] = []
    parameter_error_sum = 0.0
    proposal_parameter_error_sum = 0.0
    parameter_value_count = 0
    feedback_shift_sum = 0.0
    feedback_shift_count = 0
    chord_blend_sum = 0.0
    chord_blend_count = 0
    digest = hashlib.sha256()

    for batch in loader:
        points = batch["points"].to(device)
        output = model.forward_deployment(points)
        hard_mask = output.get("learned_keep_mask", output.get("knot_mask"))
        if hard_mask is None:
            raise KeyError("deployment output is missing learned_keep_mask")
        # Feedback transports the same survivors from the proposal t0 domain
        # into t1.  Fingerprint the pre-transport geometry plus KeepMask so a
        # legitimate parameter-domain warp is not mistaken for a structure
        # change.
        fingerprint_knots = output.get(
            "parameter_feedback_source_internal_knots",
            output["internal_knots"],
        )
        _structure_fingerprint_update(digest, fingerprint_knots, hard_mask)

        # Run the reporting refit on CPU float64, matching the robust deployment
        # evaluator rather than the differentiable truncated-power surrogate.
        deployed = refit_hard_gated_bspline_batch(
            parameters=output["params"].detach().cpu().to(torch.float64),
            candidate_knots=output["internal_knots"].detach().cpu().to(torch.float64),
            hard_gates=hard_mask.detach().cpu(),
            points=points.detach().cpu().to(torch.float64),
            degree=int(model.degree),
            smoothness_weight=1e-6,
            control_ridge=0.0,
            interpolate_endpoints=True,
        )
        mse_values.extend(item.fit_mse.cpu() for item in deployed)
        rms_values.extend(item.fit_rmse.cpu() for item in deployed)
        retained_counts.append(hard_mask.sum(dim=-1).detach().cpu().to(torch.float64))

        true_params = batch.get("true_params")
        if isinstance(true_params, torch.Tensor):
            predicted = output["params"].detach().cpu()
            proposal = output["proposal_params"].detach().cpu()
            parameter_error_sum += float((predicted - true_params).square().sum())
            proposal_parameter_error_sum += float(
                (proposal - true_params).square().sum()
            )
            parameter_value_count += int(true_params.numel())
        shift = output.get("parameter_feedback_gap_logit_delta")
        if shift is not None:
            feedback_shift_sum += float(shift.detach().abs().sum().cpu())
            feedback_shift_count += int(shift.numel())
        chord_blend = output.get("parameter_feedback_chord_blend_weight")
        if chord_blend is not None:
            chord_blend_sum += float(chord_blend.detach().sum().cpu())
            chord_blend_count += int(chord_blend.numel())

    if not mse_values:
        raise RuntimeError("validation loader produced no deployment samples")
    mse = torch.stack(mse_values).to(torch.float64)
    rms = torch.stack(rms_values).to(torch.float64)
    counts = torch.cat(retained_counts)
    return {
        "sample_count": int(mse.numel()),
        "standard_bspline_mse_mean": float(mse.mean()),
        "standard_bspline_mse_p95": float(torch.quantile(mse, 0.95)),
        "standard_bspline_rms_mean": float(rms.mean()),
        "standard_bspline_rms_p95": float(torch.quantile(rms, 0.95)),
        "threshold_satisfied_rate": float((rms <= fit_tolerance).to(rms.dtype).mean()),
        "retained_knot_count_mean": float(counts.mean()),
        "parameter_rmse": (
            (parameter_error_sum / parameter_value_count) ** 0.5
            if parameter_value_count
            else float("nan")
        ),
        "proposal_parameter_rmse": (
            (proposal_parameter_error_sum / parameter_value_count) ** 0.5
            if parameter_value_count
            else float("nan")
        ),
        "feedback_gap_logit_shift_mean_abs": (
            feedback_shift_sum / feedback_shift_count if feedback_shift_count else 0.0
        ),
        "chord_blend_weight_mean": (
            chord_blend_sum / chord_blend_count if chord_blend_count else 0.0
        ),
        "structure_fingerprint": digest.hexdigest(),
    }


def deployment_selection_rank(
    metrics: Mapping[str, float | int | str],
    *,
    pass_rate_target: float,
) -> tuple[float, ...]:
    pass_rate = float(metrics["threshold_satisfied_rate"])
    p95 = float(metrics["standard_bspline_mse_p95"])
    mean = float(metrics["standard_bspline_mse_mean"])
    parameter_rmse = float(metrics["parameter_rmse"])
    if parameter_rmse != parameter_rmse:
        parameter_rmse = float("inf")
    feasible = pass_rate >= pass_rate_target
    if feasible:
        return (1.0, -p95, -mean, -parameter_rmse)
    return (0.0, pass_rate, -p95, -mean, -parameter_rmse)


def choose_feedback_candidate(
    identity_metrics: Mapping[str, float | int | str],
    candidate_metrics: Mapping[str, float | int | str],
    *,
    pass_rate_target: float,
    preserve_structure: bool = True,
) -> bool:
    """Accept an exact-deployment improvement under the requested semantics."""

    if preserve_structure and (
        identity_metrics["structure_fingerprint"]
        != candidate_metrics["structure_fingerprint"]
    ):
        return False
    return deployment_selection_rank(
        candidate_metrics,
        pass_rate_target=pass_rate_target,
    ) > deployment_selection_rank(
        identity_metrics,
        pass_rate_target=pass_rate_target,
    )


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _print_metrics(label: str, metrics: Mapping[str, float | int | str]) -> None:
    print(
        f"{label}: pass={float(metrics['threshold_satisfied_rate']):.3f} | "
        f"MSE mean/P95={float(metrics['standard_bspline_mse_mean']):.6e}/"
        f"{float(metrics['standard_bspline_mse_p95']):.6e} | "
        f"RMS mean/P95={float(metrics['standard_bspline_rms_mean']):.6e}/"
        f"{float(metrics['standard_bspline_rms_p95']):.6e} | "
        f"K={float(metrics['retained_knot_count_mean']):.3f} | "
        f"param_RMSE={float(metrics['parameter_rmse']):.6e} | "
        f"chord={float(metrics['chord_blend_weight_mean']):.3f}",
        flush=True,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.train_size <= 0 or args.val_size <= 0:
        parser.error("dataset sizes must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0.0:
        parser.error("--weight-decay must be non-negative")
    if args.parameter_error_scale <= 0.0:
        parser.error("--parameter-error-scale must be positive")
    if args.feedback_max_logit_shift <= 0.0:
        parser.error("--feedback-max-logit-shift must be positive")
    if args.joint_feedback_local_bandwidth <= 0.0:
        parser.error("--joint-feedback-local-bandwidth must be positive")
    if args.joint_feedback_max_keep_logit_shift <= 0.0:
        parser.error("--joint-feedback-max-keep-logit-shift must be positive")
    if args.final_relocation_lr_scale <= 0.0:
        parser.error("--final-relocation-lr-scale must be positive")
    if args.joint_knot_position_beta <= 0.0:
        parser.error("--joint-knot-position-beta must be positive")
    if args.joint_positive_keep_weight <= 0.0:
        parser.error("--joint-positive-keep-weight must be positive")
    if not 0.0 <= args.initial_chord_blend <= 1.0:
        parser.error("--initial-chord-blend must lie in [0,1]")
    if not 0.0 <= args.deployment_pass_rate_target <= 1.0:
        parser.error("--deployment-pass-rate-target must lie in [0,1]")
    loss_values = (
        args.lambda_feedback_fit,
        args.lambda_feedback_threshold_violation,
        args.lambda_feedback_deployment_fit,
        args.lambda_feedback_deployment_threshold,
        args.lambda_feedback_true_params,
        args.lambda_feedback_chord_prior,
        args.lambda_feedback_identity,
        args.lambda_feedback_joint_keep,
        args.lambda_feedback_joint_critical_recall,
        args.lambda_feedback_joint_position,
        args.lambda_feedback_joint_set_position,
        args.lambda_feedback_joint_spacing,
        args.lambda_feedback_joint_count,
    )
    if any(value < 0.0 for value in loss_values):
        parser.error("feedback loss weights must be non-negative")
    if args.real_world_manifest and args.resample_train_each_epoch:
        parser.error(
            "--resample-train-each-epoch is synthetic-only; prepared real-world "
            "curves already use deterministic arc-length resampling"
        )

    source_path = args.checkpoint.resolve()
    if not source_path.is_file():
        parser.error(f"checkpoint does not exist: {source_path}")
    metadata_path = (
        args.metadata_checkpoint.resolve()
        if args.metadata_checkpoint is not None
        else None
    )
    if metadata_path is not None and not metadata_path.is_file():
        parser.error(f"metadata checkpoint does not exist: {metadata_path}")
    output_path = (
        args.output.resolve()
        if args.output is not None
        else source_path.with_name(
            source_path.stem
            + (
                "_v15_deployment_aligned"
                if args.joint_parameter_structure_feedback
                else "_v14_parameter_feedback"
            )
            + source_path.suffix
        )
    )
    if output_path == source_path:
        parser.error("--output must differ from --checkpoint")
    if output_path.exists() and not args.overwrite:
        parser.error(f"output already exists (use --overwrite): {output_path}")

    loaded_checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded_checkpoint, Mapping):
        raise TypeError("checkpoint root must be a mapping")
    if metadata_path is not None:
        metadata_checkpoint = torch.load(
            metadata_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(metadata_checkpoint, Mapping):
            raise TypeError("metadata checkpoint root must be a mapping")
        checkpoint = compose_structure_checkpoint(
            loaded_checkpoint,
            metadata_checkpoint,
        )
    else:
        checkpoint = loaded_checkpoint
    fit_tolerance = _resolve_fit_tolerance(checkpoint, args.fit_tolerance)
    device = _resolve_device(args.device)
    torch.manual_seed(args.train_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.train_seed)

    model, model_config, allowed_missing = build_feedback_model_from_checkpoint(
        checkpoint,
        attention_heads=args.feedback_attention_heads,
        max_logit_shift=args.feedback_max_logit_shift,
        joint_parameter_structure_feedback=(
            args.joint_parameter_structure_feedback
        ),
        fusion_mode=args.parameter_feedback_fusion_mode,
        joint_local_bandwidth=args.joint_feedback_local_bandwidth,
        joint_max_keep_logit_shift=args.joint_feedback_max_keep_logit_shift,
    )
    primary_trainable, relocation_trainable = feedback_training_parameter_groups(
        model
    )
    model.to(device)

    dataset_config = _resolved_dataset_config(checkpoint, model_config)
    real_world_manifests = list(args.real_world_manifest or [])
    if real_world_manifests:
        model_point_count = int(dataset_config["num_points"])
        try:
            train_set = build_real_world_split(
                real_world_manifests,
                split=args.real_world_train_split,
                num_points=model_point_count,
                maximum_size=args.train_size,
                seed=args.train_seed,
            )
            val_set = build_real_world_split(
                real_world_manifests,
                split=args.real_world_val_split,
                num_points=model_point_count,
                maximum_size=args.val_size,
                seed=args.val_seed,
            )
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
    else:
        train_set = SyntheticCubicBSplineDataset(
            size=args.train_size,
            seed=args.train_seed,
            cache_samples=True,
            resample_each_epoch=args.resample_train_each_epoch,
            **dataset_config,
        )
        val_set = SyntheticCubicBSplineDataset(
            size=args.val_size,
            seed=args.val_seed,
            cache_samples=True,
            resample_each_epoch=False,
            **dataset_config,
        )
    loader_generator = torch.Generator().manual_seed(args.train_seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=loader_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    weights = build_feedback_loss_weights(
        args,
        real_world_unlabeled=bool(real_world_manifests),
    )
    feedback_loss = ParameterFeedbackLoss(
        weights,
        fit_tolerance=fit_tolerance,
        parameter_error_scale=args.parameter_error_scale,
        knot_position_beta=args.joint_knot_position_beta,
        positive_keep_weight=args.joint_positive_keep_weight,
        degree=model.degree,
        deployment_smoothness_weight=1e-6,
        deployment_control_ridge=0.0,
    )
    training_loss = _TrainerCompatibleFeedbackLoss(
        feedback_loss,
        canonical_structure_supervision=(
            args.joint_parameter_structure_feedback and not real_world_manifests
        ),
    )
    optimizer_groups: list[dict[str, object]] = [
        {"params": primary_trainable, "lr": args.learning_rate}
    ]
    if relocation_trainable:
        optimizer_groups.append(
            {
                "params": relocation_trainable,
                "lr": args.learning_rate * args.final_relocation_lr_scale,
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=args.weight_decay,
    )
    trainer = Trainer(
        model,
        training_loss,
        optimizer,
        device,
        grad_clip=args.grad_clip,
        knot_match_tolerance=args.knot_match_tolerance,
        log_every_batches=args.log_every_batches,
        deployment_pass_rate_target=args.deployment_pass_rate_target,
        show_progress=args.progress,
    )

    print(f"Source: {source_path}", flush=True)
    if metadata_path is not None:
        print(
            f"Recovered structure metadata from: {metadata_path}",
            flush=True,
        )
    print(
        "Frozen proposal + preliminary selector; trainable feedback/joint "
        f"parameters={sum(parameter.numel() for parameter in primary_trainable):,}, "
        "scaled relocation parameters="
        f"{sum(parameter.numel() for parameter in relocation_trainable):,} "
        f"(lr scale={args.final_relocation_lr_scale:g})",
        flush=True,
    )
    if real_world_manifests:
        print(
            "Real-world unlabeled adaptation: "
            f"train={len(train_set)} ({args.real_world_train_split}), "
            f"val={len(val_set)} ({args.real_world_val_split}); true-knot and "
            "true-parameter supervision disabled; KeepMask/position/count "
            "supervision weights are exactly zero (no pseudo masks).",
            flush=True,
        )
    print(
        f"Stable pilot descriptors preserved: {model.stable_pilot_descriptors}",
        flush=True,
    )
    identity_state = _cpu_state_dict(model)
    identity_metrics = evaluate_standard_bspline_deployment(
        model,
        val_loader,
        device,
        fit_tolerance=fit_tolerance,
    )
    _print_metrics("Identity baseline", identity_metrics)
    with torch.no_grad():
        model.parameter_feedback_head.chord_blend_weight.fill_(args.initial_chord_blend)
    print(
        f"Seeded trainable chord-gap blend at {args.initial_chord_blend:.3f}.",
        flush=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path = output_path.with_name(
        output_path.stem + ".feedback_candidate" + output_path.suffix
    )
    finetune_stage = (
        "joint_parameter_structure_finetune"
        if args.joint_parameter_structure_feedback
        else "parameter_feedback_finetune"
    )
    history = trainer.fit(
        train_loader,
        val_loader,
        args.epochs,
        checkpoint_path=candidate_path,
        stage_name=finetune_stage,
        deployment_validation=True,
    )
    candidate_checkpoint = torch.load(
        candidate_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(candidate_checkpoint["model_state_dict"], strict=True)
    model.to(device)
    candidate_metrics = evaluate_standard_bspline_deployment(
        model,
        val_loader,
        device,
        fit_tolerance=fit_tolerance,
    )
    _print_metrics("Trained candidate", candidate_metrics)

    selected_feedback = choose_feedback_candidate(
        identity_metrics,
        candidate_metrics,
        pass_rate_target=args.deployment_pass_rate_target,
        preserve_structure=not args.joint_parameter_structure_feedback,
    )
    if selected_feedback:
        selected_state = _cpu_state_dict(model)
        selected_metrics = candidate_metrics
        selected_stage = finetune_stage
        print("Selected trained feedback candidate.", flush=True)
    else:
        selected_state = identity_state
        selected_metrics = identity_metrics
        selected_stage = (
            "joint_parameter_structure_identity_fallback"
            if args.joint_parameter_structure_feedback
            else "parameter_feedback_identity_fallback"
        )
        print(
            "Candidate regressed exact deployment; retained identity feedback.",
            flush=True,
        )

    objective_version = (
        V15_DEPLOYMENT_ALIGNED_OBJECTIVE_VERSION
        if args.joint_parameter_structure_feedback
        else V14_PARAMETER_FEEDBACK_OBJECTIVE_VERSION
    )
    loss_config = deepcopy(
        V15_DEPLOYMENT_ALIGNED_LOSS_CONFIG
        if args.joint_parameter_structure_feedback
        else V14_PARAMETER_FEEDBACK_LOSS_CONFIG
    )
    loss_config["parameter_feedback_loss"] = {
        **asdict(weights),
        "parameter_error_scale": args.parameter_error_scale,
        "fit_tolerance": fit_tolerance,
        "initial_chord_blend": args.initial_chord_blend,
        "knot_position_beta": args.joint_knot_position_beta,
        "positive_keep_weight": args.joint_positive_keep_weight,
        "supervision_source": (
            "fit_threshold_only_no_pseudo_keep_labels"
            if real_world_manifests
            else "synthetic_canonical_true_knot_labels"
        ),
    }
    source_histories = checkpoint.get("stage_histories", {})
    histories = dict(source_histories) if isinstance(source_histories, Mapping) else {}
    histories[finetune_stage] = history
    source_training_config = checkpoint.get("training_config", {})
    source_deployment_config = checkpoint.get("deployment_config", {})
    final_checkpoint = dict(checkpoint)
    final_checkpoint.update(
        {
            "model_state_dict": selected_state,
            "model_config": model_config,
            "dataset_config": dict(checkpoint["dataset_config"]),
            "dataset_type": (
                "real_world_unlabeled_parameter_feedback_adaptation"
                if real_world_manifests
                else checkpoint.get("dataset_type", "synthetic_open_cubic_bspline")
            ),
            "objective_version": objective_version,
            "loss_config": loss_config,
            "training_config": {
                "stage": selected_stage,
                "epochs": args.epochs,
                "train_size": args.train_size,
                "val_size": args.val_size,
                "actual_train_size": len(train_set),
                "actual_val_size": len(val_set),
                "train_seed": args.train_seed,
                "val_seed": args.val_seed,
                "batch_size": args.batch_size,
                "resample_train_each_epoch": args.resample_train_each_epoch,
                "real_world_manifests": [
                    str(path.resolve()) for path in real_world_manifests
                ],
                "real_world_train_split": args.real_world_train_split,
                "real_world_val_split": args.real_world_val_split,
                "learning_rate": args.learning_rate,
                "final_relocation_learning_rate": (
                    args.learning_rate * args.final_relocation_lr_scale
                ),
                "final_relocation_lr_scale": args.final_relocation_lr_scale,
                "initial_chord_blend": args.initial_chord_blend,
                "parameter_feedback_fusion_mode": model_config[
                    "parameter_feedback_fusion_mode"
                ],
                "joint_parameter_structure_feedback": (
                    args.joint_parameter_structure_feedback
                ),
                "real_world_keep_supervision_weight": (
                    0.0 if real_world_manifests else float(weights.joint_keep)
                ),
                "real_world_position_supervision_weight": (
                    0.0 if real_world_manifests else float(weights.joint_position)
                ),
                "real_world_count_supervision_weight": (
                    0.0 if real_world_manifests else float(weights.joint_count)
                ),
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "device": str(device),
                "source_training_config": source_training_config,
            },
            "deployment_config": {
                **(
                    dict(source_deployment_config)
                    if isinstance(source_deployment_config, Mapping)
                    else {}
                ),
                "error_tolerance": fit_tolerance,
                "parameter_feedback_fusion": True,
                "joint_parameter_structure_feedback": (
                    args.joint_parameter_structure_feedback
                ),
                "parameter_feedback_additional_spline_solves": 0,
                "selection": (
                    "identity_vs_joint_candidate_exact_standard_bspline"
                    if args.joint_parameter_structure_feedback
                    else "identity_vs_candidate_exact_standard_bspline"
                ),
            },
            "stage_histories": histories,
            "stage": selected_stage,
            "selected_stage": selected_stage,
            "selected_checkpoint_parameter_feedback": selected_feedback,
            "source_checkpoint": str(source_path),
            "source_metadata_checkpoint": (
                str(metadata_path) if metadata_path is not None else None
            ),
            "source_objective_version": checkpoint.get("objective_version"),
            "source_epoch": checkpoint.get("epoch"),
            "allowed_missing_state_keys": allowed_missing,
            "parameter_feedback_identity_metrics": identity_metrics,
            "parameter_feedback_candidate_metrics": candidate_metrics,
            "parameter_feedback_selected_metrics": selected_metrics,
            "parameter_feedback_selection_rank": list(
                deployment_selection_rank(
                    selected_metrics,
                    pass_rate_target=args.deployment_pass_rate_target,
                )
            ),
            "selection_metric": (
                "exact_standard_bspline_pass_then_p95_mse_mean_mse_parameter_rmse"
            ),
            "selection_value": float(selected_metrics["standard_bspline_mse_p95"]),
            "selection_rank": list(
                deployment_selection_rank(
                    selected_metrics,
                    pass_rate_target=args.deployment_pass_rate_target,
                )
            ),
            "best_threshold_satisfied_rate": float(
                selected_metrics["threshold_satisfied_rate"]
            ),
            "best_deployment_bspline_rms": float(
                selected_metrics["standard_bspline_rms_mean"]
            ),
            "best_deployment_bspline_rms_p95": float(
                selected_metrics["standard_bspline_rms_p95"]
            ),
            "best_deployment_bspline_mse": float(
                selected_metrics["standard_bspline_mse_mean"]
            ),
            "best_deployment_bspline_mse_p95": float(
                selected_metrics["standard_bspline_mse_p95"]
            ),
            "best_deployment_retained_knot_count": float(
                selected_metrics["retained_knot_count_mean"]
            ),
            "deployment_pass_rate_target": args.deployment_pass_rate_target,
            "epoch": int(checkpoint.get("epoch", 0)) + args.epochs,
        }
    )
    torch.save(final_checkpoint, output_path)
    if candidate_path.exists():
        candidate_path.unlink()
    version_label = "v15" if args.joint_parameter_structure_feedback else "v14"
    print(f"Saved {version_label} checkpoint: {output_path}", flush=True)


if __name__ == "__main__":
    main()
