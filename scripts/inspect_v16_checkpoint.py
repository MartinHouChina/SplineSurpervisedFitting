"""Inspect v16 checkpoint progress and formal-reporting eligibility."""
from __future__ import annotations

# Standalone entry point from an unpacked repository.
# ruff: noqa: E402
import argparse
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import (
    V16_FORMAL_PASS_RATE,
    assess_v16_checkpoint,
)


def _unit_interval(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number in [0,1]") from error
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be finite and lie in [0,1]")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument(
        "--required-pass-rate",
        type=_unit_interval,
        default=V16_FORMAL_PASS_RATE,
        help=f"Minimum formal worst-source pass rate (default: {V16_FORMAL_PASS_RATE:g})",
    )
    result.add_argument(
        "--mse-tolerance",
        type=_positive_float,
        default=None,
        help="Optionally require an exact normalized-MSE tolerance match",
    )
    return result


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"could not load checkpoint {path}: {error}") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    return checkpoint


def _format_rate(value: Any) -> str:
    if isinstance(value, bool):
        return "not recorded"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "not recorded"
    if not math.isfinite(numeric):
        return "not recorded"
    return f"{numeric:.3%} ({numeric:.6f})"


def _format_scalar(value: Any, *, scientific: bool = False) -> str:
    if isinstance(value, bool):
        return "not recorded"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "not recorded"
    if not math.isfinite(numeric):
        return "not recorded"
    return f"{numeric:.6e}" if scientific else f"{numeric:g}"


def _print_source_metrics(validation: Mapping[str, Any]) -> None:
    raw = validation.get("by_source")
    if not isinstance(raw, Mapping) or not raw:
        print("  by source: not recorded")
        return
    print("  by source:")
    for source in sorted(raw, key=str):
        metrics = raw[source]
        if not isinstance(metrics, Mapping):
            print(f"    {source}: invalid metrics")
            continue
        line = (
            f"    {source}: dense={_format_rate(metrics.get('dense_pass_rate'))}, "
            f"deployment={_format_rate(metrics.get('deployment_pass_rate'))}, "
            f"dense_MSE={_format_scalar(metrics.get('dense_mse'), scientific=True)}, "
            f"deployment_MSE={_format_scalar(metrics.get('deployment_mse'), scientific=True)}, "
            f"K={_format_scalar(metrics.get('keep_count'))}"
        )
        if "target_count_mean" in metrics:
            line += (
                f", target_K={_format_scalar(metrics.get('target_count_mean'))}, "
                f"count_MAE={_format_scalar(metrics.get('count_mae'))}, "
                f"count_bias={_format_scalar(metrics.get('count_bias'))}"
            )
        if "knot_match_f1" in metrics:
            line += (
                f", knot_F1={_format_scalar(metrics.get('knot_match_f1'))}, "
                "matched_MAE="
                + _format_scalar(
                    metrics.get("knot_matched_mae"), scientific=True
                )
            )
        print(line)


def inspect_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    required_pass_rate: float,
    mse_tolerance: float | None,
) -> bool:
    qualification = assess_v16_checkpoint(
        checkpoint,
        required_pass_rate=required_pass_rate,
        required_mse_tolerance=mse_tolerance,
    )
    training = checkpoint.get("training_config")
    validation = checkpoint.get("validation_metrics")
    model_config = checkpoint.get("model_config")
    deployment = checkpoint.get("deployment_config")
    loss_config = checkpoint.get("loss_config")
    training = training if isinstance(training, Mapping) else {}
    validation = validation if isinstance(validation, Mapping) else {}
    model_config = model_config if isinstance(model_config, Mapping) else {}
    deployment = deployment if isinstance(deployment, Mapping) else {}
    loss_config = loss_config if isinstance(loss_config, Mapping) else {}
    loss_weights = loss_config.get("weights")
    loss_weights = loss_weights if isinstance(loss_weights, Mapping) else {}

    print("v16 checkpoint inspection")
    print(f"  checkpoint: {checkpoint_path.resolve()}")
    print(f"  stage: {checkpoint.get('stage', 'not recorded')}")
    print(f"  epoch: {checkpoint.get('epoch', 'not recorded')}")
    internal_capacity = model_config.get("max_internal_knots")
    degree = model_config.get("degree", 3)
    try:
        full_capacity = int(internal_capacity) + 2 * (int(degree) + 1)
    except (TypeError, ValueError):
        full_capacity = None
    print("Selection architecture")
    print(f"  revision: {checkpoint.get('architecture_revision', 'legacy v16 threshold')}")
    print(
        "  policy: "
        + str(deployment.get(
            "one_shot_selection_policy",
            model_config.get("one_shot_selection_policy", "threshold"),
        ))
    )
    print(f"  internal candidate capacity: {internal_capacity if internal_capacity is not None else 'not recorded'}")
    print(f"  full knot-vector size at all-keep: {full_capacity if full_capacity is not None else 'not recorded'}")
    print(f"  adaptive beta: {deployment.get('adaptive_keep_threshold', model_config.get('one_shot_adaptive_threshold', False))}")
    print(
        "  simplification contract: "
        + str(checkpoint.get("simplification_contract", "not recorded"))
    )
    print(
        "  simplification curriculum mature: "
        + str(checkpoint.get("simplification_ready", False))
    )
    print("Proposal curriculum")
    print(
        "  high-K synthetic allocation: "
        + _format_rate(training.get("proposal_high_k_fraction"))
    )
    print(
        "  high-K stratum begins at internal K: "
        + _format_scalar(training.get("proposal_high_k_min_knots"))
    )
    print(
        "  ordered assignment weight: "
        + _format_scalar(loss_weights.get("proposal_knot_assignment_weight"))
    )
    print("Synthetic data contract")
    print(f"  revision: {checkpoint.get('synthetic_data_contract', 'not recorded')}")
    dataset_config = checkpoint.get("dataset_config")
    dataset_config = dataset_config if isinstance(dataset_config, Mapping) else {}
    print(f"  certified minimal source: {dataset_config.get('certified_minimal_source', False)}")
    print(
        "  source control-point range: "
        + _format_scalar(qualification.get("synthetic_min_control_points"))
        + ".."
        + _format_scalar(qualification.get("synthetic_max_control_points"))
        + " (formal: "
        + _format_scalar(
            qualification.get("formal_synthetic_min_control_points")
        )
        + ".."
        + _format_scalar(
            qualification.get("formal_synthetic_max_control_points")
        )
        + ")"
    )
    print(
        "  source internal-knot range: K="
        + _format_scalar(
            qualification.get("formal_synthetic_min_internal_knots")
        )
        + ".."
        + _format_scalar(
            qualification.get("formal_synthetic_max_internal_knots")
        )
    )
    print(
        "  knot min span: "
        + _format_scalar(qualification.get("synthetic_knot_min_span"))
        + " (formal: "
        + _format_scalar(
            qualification.get("formal_synthetic_knot_min_span")
        )
        + ")"
    )
    print(
        "  certificate RMS tolerance: "
        + _format_scalar(dataset_config.get("canonical_knot_tolerance"), scientific=True)
    )
    print("Configured targets")
    print(
        "  proposal pass: "
        + _format_rate(training.get("proposal_pass_target"))
    )
    print(
        "  deployment pass: "
        + _format_rate(training.get("deployment_pass_target"))
    )
    print(
        "  recorded MSE tolerance: "
        + _format_scalar(qualification["recorded_mse_tolerance"], scientific=True)
    )
    print("Observed validation")
    print(f"  overall dense: {_format_rate(validation.get('dense_pass_rate'))}")
    print(
        "  overall deployment: "
        + _format_rate(validation.get("deployment_pass_rate"))
    )
    print(
        "  worst-source dense: "
        + _format_rate(qualification["observed_worst_dense_pass_rate"])
    )
    print(
        "  worst-source deployment: "
        + _format_rate(qualification["observed_worst_deployment_pass_rate"])
    )
    print(
        "  qualification dense/deployment (min of worst-source and K=56): "
        + _format_rate(
            qualification.get("observed_qualification_dense_pass_rate")
        )
        + " / "
        + _format_rate(
            qualification.get("observed_qualification_deployment_pass_rate")
        )
    )
    print(
        "  certified synthetic count MAE: "
        + _format_scalar(qualification.get("synthetic_count_mae"))
    )
    print(
        "  certified synthetic knot F1 / matched MAE: "
        + _format_scalar(qualification.get("synthetic_knot_match_f1"))
        + " / "
        + _format_scalar(
            qualification.get("synthetic_knot_matched_mae"), scientific=True
        )
    )
    print(
        "  synthetic boundary: K="
        + _format_scalar(qualification.get("synthetic_boundary_knot_count"))
        + ", n="
        + _format_scalar(qualification.get("synthetic_boundary_sample_count"))
        + ", dense="
        + _format_rate(
            qualification.get("synthetic_boundary_dense_pass_rate")
        )
        + ", deployment="
        + _format_rate(
            qualification.get("synthetic_boundary_deployment_pass_rate")
        )
    )
    _print_source_metrics(validation)

    eligible = bool(qualification["formal_reporting_eligible"])
    print("Formal qualification")
    print(f"  required pass rate: {_format_rate(required_pass_rate)}")
    requested_tolerance = qualification["required_mse_tolerance"]
    print(
        "  required MSE tolerance: "
        + (
            "not constrained"
            if requested_tolerance is None
            else _format_scalar(requested_tolerance, scientific=True)
        )
    )
    print(
        "  Synthetic count MAE <= "
        + _format_scalar(qualification.get("formal_synthetic_count_mae_max"))
        + ", knot F1 >= "
        + _format_scalar(qualification.get("formal_synthetic_knot_f1_min"))
        + ", matched MAE <= "
        + _format_scalar(
            qualification.get("formal_synthetic_matched_mae_max"),
            scientific=True,
        )
    )
    print(f"  eligible: {'YES' if eligible else 'NO'}")
    if qualification["reasons"]:
        print("  reasons:")
        for reason in qualification["reasons"]:
            print(f"    - {reason}")
    else:
        print("  reasons: none")
    return eligible


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        checkpoint = _load_checkpoint(args.checkpoint)
        eligible = inspect_checkpoint(
            checkpoint,
            checkpoint_path=args.checkpoint,
            required_pass_rate=args.required_pass_rate,
            mse_tolerance=args.mse_tolerance,
        )
    except ValueError as error:
        argument_parser.error(str(error))
    return 0 if eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
