from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from spline_fitting.checkpointing import build_model_from_checkpoint  # noqa: E402
from spline_fitting.data.synthetic import SyntheticCubicBSplineDataset  # noqa: E402
from spline_fitting.evaluation.bspline_inference import (  # noqa: E402
    BSplineLeastSquaresFit,
    refit_bspline_control_points,
)
from spline_fitting.evaluation.knot_diagnostics import (  # noqa: E402
    match_internal_knots,
    warp_internal_knots_to_parameterization,
)
from spline_fitting.evaluation.timing import (  # noqa: E402
    measure_synchronized_wall_time,
)
from spline_fitting.evaluation.verified_knot_repair import (  # noqa: E402
    VerifiedKnotRepairResult,
    verified_confidence_repair,
)
from visualize_batch_comparison import (  # noqa: E402
    _dataset_config_from_checkpoint,
    _scan_counts,
    deployment_network_forward,
    deployment_geometries,
    fixed_samples_per_knot_count,
    greedy_prune_to_mse_tolerance,
    resolve_mse_tolerance,
)


T = TypeVar("T")
METHODS = ("ours", "hard")
METHOD_COLORS = {"ours": "#198a77", "hard": "#c94b4b"}


@dataclass(frozen=True)
class OursDeploymentResult:
    """Materialized learned or adaptive verified deployment for one curve."""

    output: dict[str, torch.Tensor]
    final_fit: BSplineLeastSquaresFit
    final_internal_knots: torch.Tensor
    final_parameters: torch.Tensor
    verified_repair: VerifiedKnotRepairResult | None


def ours_method_label(deployment: str) -> str:
    if deployment == "learned":
        return "Ours learned (one-shot)"
    if deployment == "verified":
        return "Ours verified (adaptive repair)"
    raise ValueError(f"unknown Ours deployment mode: {deployment}")


def deploy_ours(
    model: torch.nn.Module,
    batched_points: torch.Tensor,
    points: torch.Tensor,
    *,
    deployment: str,
    chord_parameters: torch.Tensor | None = None,
    verified_parameterization: str = "network",
    mse_tolerance: float,
    smoothness_weight: float,
    control_ridge: float,
    verified_compact: bool = True,
    verified_hard_fallback: bool = True,
    verified_residual_fallback: bool = True,
    verified_max_residual_insertions: int = 8,
    verified_residual_min_gap: float = 1e-3,
    verified_refit_device: torch.device | str = "cpu",
) -> OursDeploymentResult:
    """Run the selected Ours path without duplicating verified-repair logic."""

    if deployment not in {"learned", "verified"}:
        raise ValueError("deployment must be 'learned' or 'verified'")
    with torch.inference_mode():
        output = deployment_network_forward(model, batched_points)
        proposal_knots, deployment_knots, learned_mask = deployment_geometries(output)
        if deployment == "learned":
            fit = refit_bspline_control_points(
                output["params"][0],
                points,
                deployment_knots[learned_mask],
                degree=int(model.degree),
                smoothness_weight=smoothness_weight,
                control_ridge=control_ridge,
                interpolate_endpoints=True,
            )
            return OursDeploymentResult(
                output=output,
                final_fit=fit,
                final_internal_knots=deployment_knots[learned_mask],
                final_parameters=output["params"][0],
                verified_repair=None,
            )

        target_device = torch.device(verified_refit_device)
        repair = verified_confidence_repair(
            output["params"][0].detach().to(target_device),
            points.detach().to(target_device),
            proposal_knots.detach().to(target_device),
            deployment_knots.detach().to(target_device),
            learned_mask.detach().to(target_device),
            output["keep_probability"][0].detach().to(target_device),
            fit_tolerance_rms=math.sqrt(mse_tolerance),
            degree=int(model.degree),
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
            interpolate_endpoints=True,
            compact=verified_compact,
            hard_fallback=verified_hard_fallback,
            residual_fallback=verified_residual_fallback,
            max_residual_insertions=verified_max_residual_insertions,
            residual_min_gap=verified_residual_min_gap,
            alternate_parameters=(
                chord_parameters.detach().to(target_device)
                if chord_parameters is not None
                else None
            ),
            parameterization_policy=verified_parameterization,
        )
        return OursDeploymentResult(
            output=output,
            final_fit=repair.final_fit,
            final_internal_knots=repair.final_fit.internal_knots,
            final_parameters=repair.final_parameters,
            verified_repair=repair,
        )


def _finite_float(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, probability))


def bootstrap_interval(
    values: Sequence[float],
    *,
    statistic: str,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Return a deterministic percentile bootstrap interval over curves."""

    if not values:
        raise ValueError("values cannot be empty")
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    tensor = torch.tensor(values, dtype=torch.float64)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("bootstrap values must be finite")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randint(
        tensor.numel(),
        (replicates, tensor.numel()),
        generator=generator,
    )
    samples = tensor[indices]
    estimates = (
        samples.mean(dim=1)
        if statistic == "mean"
        else torch.quantile(samples, 0.5, dim=1)
    )
    alpha = 0.5 * (1.0 - confidence)
    return {
        "low": float(torch.quantile(estimates, alpha)),
        "high": float(torch.quantile(estimates, 1.0 - alpha)),
    }


def describe_values(
    values: Sequence[float],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    clean = [_finite_float(value, name="summary value") for value in values]
    if not clean:
        raise ValueError("summary values cannot be empty")
    return {
        "mean": float(statistics.fmean(clean)),
        "median": float(statistics.median(clean)),
        "q25": _quantile(clean, 0.25),
        "q75": _quantile(clean, 0.75),
        "p95": _quantile(clean, 0.95),
        "min": min(clean),
        "max": max(clean),
        "mean_ci95": bootstrap_interval(
            clean,
            statistic="mean",
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "median_ci95": bootstrap_interval(
            clean,
            statistic="median",
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 1,
        ),
    }


def _tolerance_key(tolerance: float) -> str:
    return f"{tolerance:.3f}"


def _summary_seed(base_seed: int, source_k: int, method_index: int, metric: int) -> int:
    return int(base_seed + 100_000 * source_k + 10_000 * method_index + metric)


def aggregate_knot_match(
    rows: Sequence[Mapping[str, object]],
    *,
    method: str,
    tolerance: float,
) -> dict[str, object]:
    key = _tolerance_key(tolerance)
    matched = sum(int(row[f"{method}_match_{key}_matched"]) for row in rows)
    predicted = sum(int(row[f"{method}_match_{key}_predicted"]) for row in rows)
    target = sum(int(row[f"{method}_match_{key}_target"]) for row in rows)
    error_sum = sum(float(row[f"{method}_match_{key}_error_sum"]) for row in rows)
    precision = matched / predicted if predicted else float(target == 0)
    recall = matched / target if target else float(predicted == 0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "matched": matched,
        "predicted": predicted,
        "target": target,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_mae": error_sum / matched if matched else None,
    }


def summarize_verified_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    source_k: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Aggregate adaptive verified-only repair and dynamic-insertion telemetry."""

    def values(suffix: str) -> list[float]:
        return [float(row.get(f"ours_verified_{suffix}", 0.0)) for row in rows]

    result: dict[str, object] = {}
    metrics = (
        ("learned_threshold_satisfied", 105),
        ("fallback_used", 110),
        ("hard_fallback_used", 120),
        ("cleanup_used", 130),
        ("residual_fallback_used", 140),
        ("parameterization_fallback_attempted", 145),
        ("parameterization_fallback_used", 147),
        ("inserted_knot_count", 150),
        ("residual_refit_count", 160),
        ("direct_refit_count", 170),
        ("fit_evaluation_count", 180),
        ("prefix_count_evaluations", 190),
    )
    for suffix, metric_seed in metrics:
        result[suffix] = describe_values(
            values(suffix),
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=_summary_seed(
                bootstrap_seed,
                source_k,
                4,
                metric_seed,
            ),
        )
    result["learned_feasible_fraction"] = float(
        result["learned_threshold_satisfied"]["mean"]
    )
    result["fast_path_fraction"] = sum(
        not bool(row.get("ours_verified_fallback_used", False))
        and not bool(row.get("ours_verified_cleanup_used", False))
        for row in rows
    ) / len(rows)
    sources: dict[str, int] = {}
    for row in rows:
        source = str(row["ours_verified_final_source"])
        sources[source] = sources.get(source, 0) + 1
    result["final_source_histogram"] = dict(sorted(sources.items()))
    return result


def summarize_method_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    method: str,
    source_k: int,
    knot_tolerances: Sequence[float],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    method_index = METHODS.index(method)

    def values(suffix: str) -> list[float]:
        return [float(row[f"{method}_{suffix}"]) for row in rows]

    count = values("knot_count")
    canonical_bias = values("count_bias_canonical")
    source_bias = values("count_bias_source")
    result: dict[str, object] = {
        "knot_count": describe_values(
            count,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=_summary_seed(bootstrap_seed, source_k, method_index, 10),
        ),
        "count_bias_canonical": describe_values(
            canonical_bias,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=_summary_seed(bootstrap_seed, source_k, method_index, 20),
        ),
        "count_mae_canonical": describe_values(
            [abs(value) for value in canonical_bias],
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=_summary_seed(bootstrap_seed, source_k, method_index, 30),
        ),
        "count_exact_canonical_rate": sum(value == 0.0 for value in canonical_bias)
        / len(canonical_bias),
        "count_bias_source": describe_values(
            source_bias,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=_summary_seed(bootstrap_seed, source_k, method_index, 40),
        ),
        "count_mae_source": describe_values(
            [abs(value) for value in source_bias],
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=_summary_seed(bootstrap_seed, source_k, method_index, 50),
        ),
        "fit_mse": describe_values(
            values("fit_mse"),
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=_summary_seed(bootstrap_seed, source_k, method_index, 60),
        ),
        "pass_rate": describe_values(
            values("threshold_satisfied"),
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=_summary_seed(bootstrap_seed, source_k, method_index, 70),
        ),
        "time_ms": describe_values(
            values("time_median_ms"),
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=_summary_seed(bootstrap_seed, source_k, method_index, 80),
        ),
        "matching_warped_to_canonical": {
            _tolerance_key(tolerance): aggregate_knot_match(
                rows,
                method=method,
                tolerance=tolerance,
            )
            for tolerance in knot_tolerances
        },
    }
    if method == "ours":
        deployment_modes = {
            str(row.get("ours_deployment_mode", "learned")) for row in rows
        }
        if len(deployment_modes) != 1:
            raise ValueError("Ours deployment mode must be consistent within a summary")
        deployment_mode = deployment_modes.pop()
        result["deployment_mode"] = deployment_mode
        result["time_scope"] = "network_forward_only"
        diagnostic_key = "ours_quality_pipeline_time_median_ms"
        if all(diagnostic_key in row for row in rows):
            result["quality_pipeline_time_ms"] = describe_values(
                [float(row[diagnostic_key]) for row in rows],
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=_summary_seed(
                    bootstrap_seed, source_k, method_index, 85
                ),
            )
        if deployment_mode == "verified":
            result["verified_adaptive"] = summarize_verified_rows(
                rows,
                source_k=source_k,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            )
    else:
        result["time_scope"] = "greedy_search_only"
    return result


def summarize_paired_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    source_k: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    count_delta = [
        float(row["ours_knot_count"]) - float(row["hard_knot_count"]) for row in rows
    ]
    mse_delta = [
        float(row["ours_fit_mse"]) - float(row["hard_fit_mse"]) for row in rows
    ]
    timing_ratio = [
        float(row["hard_time_median_ms"]) / float(row["ours_time_median_ms"])
        for row in rows
    ]
    ours_pass = [bool(row["ours_threshold_satisfied"]) for row in rows]
    hard_pass = [bool(row["hard_threshold_satisfied"]) for row in rows]
    ours_dominates = [
        ours_pass[index]
        and (
            count_delta[index] < 0.0
            or (count_delta[index] == 0.0 and mse_delta[index] <= 0.0)
        )
        for index in range(len(rows))
    ]
    hard_dominates = [
        hard_pass[index]
        and (
            count_delta[index] > 0.0
            or (count_delta[index] == 0.0 and mse_delta[index] >= 0.0)
        )
        for index in range(len(rows))
    ]
    return {
        "ours_minus_hard_knot_count": describe_values(
            count_delta,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=_summary_seed(bootstrap_seed, source_k, 2, 10),
        ),
        "ours_minus_hard_fit_mse": describe_values(
            mse_delta,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=_summary_seed(bootstrap_seed, source_k, 2, 20),
        ),
        "hard_search_over_ours_network_forward_time_ratio": describe_values(
            timing_ratio,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=_summary_seed(bootstrap_seed, source_k, 2, 30),
        ),
        # Retained so older report consumers do not fail.  New reports must use
        # the accurately named key above.
        "hard_stage_over_ours_full_time_ratio": describe_values(
            timing_ratio,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=_summary_seed(bootstrap_seed, source_k, 2, 30),
        ),
        "ours_fewer_knots_rate": sum(value < 0.0 for value in count_delta) / len(rows),
        "equal_knots_rate": sum(value == 0.0 for value in count_delta) / len(rows),
        "ours_more_knots_rate": sum(value > 0.0 for value in count_delta) / len(rows),
        "both_pass_rate": sum(
            ours_pass[index] and hard_pass[index] for index in range(len(rows))
        )
        / len(rows),
        "ours_only_pass_rate": sum(
            ours_pass[index] and not hard_pass[index] for index in range(len(rows))
        )
        / len(rows),
        "hard_only_pass_rate": sum(
            hard_pass[index] and not ours_pass[index] for index in range(len(rows))
        )
        / len(rows),
        "neither_pass_rate": sum(
            not ours_pass[index] and not hard_pass[index] for index in range(len(rows))
        )
        / len(rows),
        "ours_pareto_dominance_rate": sum(ours_dominates) / len(rows),
        "hard_pareto_dominance_rate": sum(hard_dominates) / len(rows),
    }


def summarize_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    source_k: int,
    knot_tolerances: Sequence[float],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    canonical = [float(row["canonical_knot_count"]) for row in rows]
    parameter_rmse = [float(row["parameter_rmse"]) for row in rows]
    return {
        "source_k": source_k,
        "n": len(rows),
        "canonical_knot_count": describe_values(
            canonical,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=_summary_seed(bootstrap_seed, source_k, 3, 10),
        ),
        "parameter_rmse": describe_values(
            parameter_rmse,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=_summary_seed(bootstrap_seed, source_k, 3, 20),
        ),
        "methods": {
            method: summarize_method_rows(
                rows,
                method=method,
                source_k=source_k,
                knot_tolerances=knot_tolerances,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            )
            for method in METHODS
        },
        "paired": summarize_paired_rows(
            rows,
            source_k=source_k,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        ),
    }


def _timed_once(function: Callable[[], T]) -> tuple[T, float]:
    started_at = time.perf_counter_ns()
    result = function()
    elapsed_ms = (time.perf_counter_ns() - started_at) / 1_000_000.0
    return result, elapsed_ms


def paired_timing_blocks(
    ours: Callable[[], T],
    hard: Callable[[], T],
    *,
    repeats: int,
    order_rng: random.Random,
) -> tuple[T, T, list[float], list[float], list[str]]:
    """Time paired blocks with deterministic randomized method order."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    ours_result: T | None = None
    hard_result: T | None = None
    ours_times: list[float] = []
    hard_times: list[float] = []
    orders: list[str] = []
    for _ in range(repeats):
        ours_first = bool(order_rng.getrandbits(1))
        orders.append("ours_then_hard" if ours_first else "hard_then_ours")
        if ours_first:
            ours_result, elapsed = _timed_once(ours)
            ours_times.append(elapsed)
            hard_result, elapsed = _timed_once(hard)
            hard_times.append(elapsed)
        else:
            hard_result, elapsed = _timed_once(hard)
            hard_times.append(elapsed)
            ours_result, elapsed = _timed_once(ours)
            ours_times.append(elapsed)
    assert ours_result is not None and hard_result is not None
    return ours_result, hard_result, ours_times, hard_times, orders


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_internal_knots(
    sample: Mapping[str, torch.Tensor | int],
    *,
    degree: int,
) -> torch.Tensor:
    knot_vector = sample["source_knot_vector"]
    knot_mask = sample["source_knot_mask"]
    assert isinstance(knot_vector, torch.Tensor)
    assert isinstance(knot_mask, torch.Tensor)
    vector = knot_vector[knot_mask.to(torch.bool)]
    return vector[degree + 1 : -(degree + 1)]


def _canonical_internal_knots(
    sample: Mapping[str, torch.Tensor | int],
) -> torch.Tensor:
    knots = sample["true_internal_knots"]
    mask = sample["true_internal_knot_mask"]
    assert isinstance(knots, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    return knots[mask.to(torch.bool)]


def _record_matches(
    row: dict[str, object],
    *,
    method: str,
    knots_in_predicted_domain: torch.Tensor,
    predicted_parameters: torch.Tensor,
    true_parameters: torch.Tensor,
    canonical_knots: torch.Tensor,
    tolerances: Sequence[float],
) -> None:
    warped = warp_internal_knots_to_parameterization(
        knots_in_predicted_domain,
        predicted_parameters,
        true_parameters,
    )
    row[f"{method}_warped_internal_knots"] = warped.detach().cpu().tolist()
    for tolerance in tolerances:
        key = _tolerance_key(tolerance)
        match = match_internal_knots(
            warped,
            canonical_knots,
            tolerance=tolerance,
        )
        row[f"{method}_match_{key}_matched"] = match.matched_count
        row[f"{method}_match_{key}_predicted"] = match.predicted_count
        row[f"{method}_match_{key}_target"] = match.true_count
        row[f"{method}_match_{key}_precision"] = match.precision
        row[f"{method}_match_{key}_recall"] = match.recall
        row[f"{method}_match_{key}_f1"] = match.f1
        row[f"{method}_match_{key}_matched_mae"] = (
            match.matched_mae if math.isfinite(match.matched_mae) else None
        )
        row[f"{method}_match_{key}_error_sum"] = (
            match.matched_mae * match.matched_count
            if math.isfinite(match.matched_mae)
            else 0.0
        )


def _plot_series(
    axes: Sequence[plt.Axes],
    per_k: Sequence[Mapping[str, object]],
    *,
    mse_tolerance: float,
    ours_deployment: str,
) -> None:
    x = [int(summary["source_k"]) for summary in per_k]

    canonical_mean = [
        float(summary["canonical_knot_count"]["mean"]) for summary in per_k
    ]
    canonical_low = [
        float(summary["canonical_knot_count"]["mean_ci95"]["low"]) for summary in per_k
    ]
    canonical_high = [
        float(summary["canonical_knot_count"]["mean_ci95"]["high"]) for summary in per_k
    ]
    axes[0].plot(x, x, "--", color="#777777", linewidth=1.2, label="source K")
    axes[0].plot(
        x,
        canonical_mean,
        "o-",
        color="#355c9a",
        linewidth=1.7,
        markersize=4,
        label="canonical reference K",
    )
    axes[0].fill_between(
        x,
        canonical_low,
        canonical_high,
        color="#355c9a",
        alpha=0.13,
    )

    for method in METHODS:
        color = METHOD_COLORS[method]
        label = (
            ours_method_label(ours_deployment) if method == "ours" else "Greedy hard"
        )
        time_scope = str(
            per_k[0]["methods"][method].get("time_scope", "legacy_mixed_scope")
        )
        time_label = (
            "Ours network forward only"
            if method == "ours" and time_scope == "network_forward_only"
            else "Greedy hard search"
            if method == "hard" and time_scope == "greedy_search_only"
            else label
        )
        count_mean = [
            float(summary["methods"][method]["knot_count"]["mean"]) for summary in per_k
        ]
        count_low = [
            float(summary["methods"][method]["knot_count"]["mean_ci95"]["low"])
            for summary in per_k
        ]
        count_high = [
            float(summary["methods"][method]["knot_count"]["mean_ci95"]["high"])
            for summary in per_k
        ]
        axes[0].plot(
            x,
            count_mean,
            "o-",
            color=color,
            linewidth=1.8,
            markersize=4,
            label=label,
        )
        axes[0].fill_between(x, count_low, count_high, color=color, alpha=0.13)

        mse_mean = [
            float(summary["methods"][method]["fit_mse"]["mean"]) for summary in per_k
        ]
        mse_low = [
            float(summary["methods"][method]["fit_mse"]["mean_ci95"]["low"])
            for summary in per_k
        ]
        mse_high = [
            float(summary["methods"][method]["fit_mse"]["mean_ci95"]["high"])
            for summary in per_k
        ]
        axes[1].plot(
            x,
            mse_mean,
            "o-",
            color=color,
            linewidth=1.8,
            markersize=4,
            label=label,
        )
        axes[1].fill_between(x, mse_low, mse_high, color=color, alpha=0.13)

        pass_mean = [
            100.0 * float(summary["methods"][method]["pass_rate"]["mean"])
            for summary in per_k
        ]
        pass_low = [
            100.0 * float(summary["methods"][method]["pass_rate"]["mean_ci95"]["low"])
            for summary in per_k
        ]
        pass_high = [
            100.0 * float(summary["methods"][method]["pass_rate"]["mean_ci95"]["high"])
            for summary in per_k
        ]
        axes[2].plot(
            x,
            pass_mean,
            "o-",
            color=color,
            linewidth=1.8,
            markersize=4,
            label=label,
        )
        axes[2].fill_between(x, pass_low, pass_high, color=color, alpha=0.13)

        time_median = [
            float(summary["methods"][method]["time_ms"]["median"]) for summary in per_k
        ]
        time_low = [
            float(summary["methods"][method]["time_ms"]["median_ci95"]["low"])
            for summary in per_k
        ]
        time_high = [
            float(summary["methods"][method]["time_ms"]["median_ci95"]["high"])
            for summary in per_k
        ]
        axes[3].plot(
            x,
            time_median,
            "o-",
            color=color,
            linewidth=1.8,
            markersize=4,
            label=time_label,
        )
        axes[3].fill_between(x, time_low, time_high, color=color, alpha=0.13)

    axes[0].set_title("(a) Retained knot count vs source complexity")
    axes[0].set_ylabel("internal-knot count")
    axes[0].legend(fontsize=8)

    axes[1].axhline(
        mse_tolerance,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label=(
            f"MSE threshold={mse_tolerance:.2e} (RMS={math.sqrt(mse_tolerance):.3f})"
        ),
    )
    axes[1].set_title("(b) Standard B-spline refit error")
    axes[1].set_ylabel("mean squared Euclidean error")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)

    axes[2].set_title("(c) Threshold-satisfied fraction")
    axes[2].set_ylabel("pass rate (%)")
    axes[2].set_ylim(-3.0, 103.0)
    axes[2].legend(fontsize=8)

    new_timing_scope = (
        per_k[0]["methods"]["ours"].get("time_scope")
        == "network_forward_only"
    )
    axes[3].set_title(
        "(d) Structure inference time (explicit scopes)"
        if new_timing_scope
        else "(d) User-specified asymmetric timing scopes"
    )
    axes[3].set_ylabel("median wall time per curve (ms)")
    axes[3].set_yscale("log")
    axes[3].legend(fontsize=8)

    for axis in axes:
        axis.set_xlabel("source internal-knot count K")
        axis.set_xticks(x)
        axis.grid(alpha=0.22, linewidth=0.6)


def render_summary_figure(
    per_k: Sequence[Mapping[str, object]],
    output_path: Path,
    *,
    mse_tolerance: float,
    samples_per_k: int,
    dpi: int,
    ours_deployment: str = "learned",
    verified_parameterization: str = "network",
) -> None:
    minimum_k = min(int(summary["source_k"]) for summary in per_k)
    maximum_k = max(int(summary["source_k"]) for summary in per_k)
    figure, grid = plt.subplots(2, 2, figsize=(14.5, 10.0))
    _plot_series(
        grid.ravel(),
        per_k,
        mse_tolerance=mse_tolerance,
        ours_deployment=ours_deployment,
    )
    ours_title = (
        "learned one-shot deployment"
        if ours_deployment == "learned"
        else "adaptive verified deployment"
    )
    parameterization_note = (
        "\nshared chord-length parameterization for Ours and Hard"
        if ours_deployment == "verified" and verified_parameterization == "chord"
        else ""
    )
    figure.suptitle(
        f"{ours_title} vs shared-proposal greedy hard pruning\n"
        f"source K={minimum_k}-{maximum_k}, n={samples_per_k} independent "
        "test curves per K; "
        f"bands are 95% curve-bootstrap CIs{parameterization_note}",
        fontsize=13,
    )
    new_timing_scope = (
        per_k[0]["methods"]["ours"].get("time_scope")
        == "network_forward_only"
    )
    ours_time_scope = (
        "Ours prediction time = synchronized forward_deployment only; final refit/"
        "verified repair excluded"
        if new_timing_scope
        else (
            "Ours learned time = full network forward + one final refit"
            if ours_deployment == "learned"
            else "Ours verified time = full network forward + exact adaptive repair"
        )
    )
    figure.text(
        0.5,
        0.012,
        f"{ours_time_scope}; Hard time = pruning stage only after proposal/parameter "
        "materialization. These scopes are asymmetric.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    figure.subplots_adjust(hspace=0.31, wspace=0.24, bottom=0.09, top=0.90)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def render_three_metric_figure(
    per_k: Sequence[Mapping[str, object]],
    output_path: Path,
    *,
    mse_tolerance: float,
    samples_per_k: int,
    dpi: int,
    ours_deployment: str = "learned",
    verified_parameterization: str = "network",
) -> None:
    """Render MSE, retained-K and timing from one shared stratified report."""

    x = [int(summary["source_k"]) for summary in per_k]
    minimum_k = min(x)
    maximum_k = max(x)
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 5.4))

    for method in METHODS:
        color = METHOD_COLORS[method]
        label = (
            ours_method_label(ours_deployment) if method == "ours" else "Greedy hard"
        )
        time_scope = str(
            per_k[0]["methods"][method].get("time_scope", "legacy_mixed_scope")
        )
        time_label = (
            "Ours network forward only"
            if method == "ours" and time_scope == "network_forward_only"
            else "Greedy hard search"
            if method == "hard" and time_scope == "greedy_search_only"
            else label
        )

        mse_mean = [
            float(summary["methods"][method]["fit_mse"]["mean"]) for summary in per_k
        ]
        mse_low = [
            float(summary["methods"][method]["fit_mse"]["mean_ci95"]["low"])
            for summary in per_k
        ]
        mse_high = [
            float(summary["methods"][method]["fit_mse"]["mean_ci95"]["high"])
            for summary in per_k
        ]
        axes[0].plot(
            x,
            mse_mean,
            "o-",
            color=color,
            linewidth=1.8,
            markersize=4,
            label=label,
        )
        axes[0].fill_between(x, mse_low, mse_high, color=color, alpha=0.13)

        count_mean = [
            float(summary["methods"][method]["knot_count"]["mean"]) for summary in per_k
        ]
        count_low = [
            float(summary["methods"][method]["knot_count"]["mean_ci95"]["low"])
            for summary in per_k
        ]
        count_high = [
            float(summary["methods"][method]["knot_count"]["mean_ci95"]["high"])
            for summary in per_k
        ]
        axes[1].plot(
            x,
            count_mean,
            "o-",
            color=color,
            linewidth=1.8,
            markersize=4,
            label=label,
        )
        axes[1].fill_between(x, count_low, count_high, color=color, alpha=0.13)

        time_median = [
            float(summary["methods"][method]["time_ms"]["median"]) for summary in per_k
        ]
        time_low = [
            float(summary["methods"][method]["time_ms"]["median_ci95"]["low"])
            for summary in per_k
        ]
        time_high = [
            float(summary["methods"][method]["time_ms"]["median_ci95"]["high"])
            for summary in per_k
        ]
        axes[2].plot(
            x,
            time_median,
            "o-",
            color=color,
            linewidth=1.8,
            markersize=4,
            label=time_label,
        )
        axes[2].fill_between(x, time_low, time_high, color=color, alpha=0.13)

    axes[0].axhline(
        mse_tolerance,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label=f"MSE threshold={mse_tolerance:.2e}",
    )
    axes[0].set_title("(a) Standard B-spline refit MSE")
    axes[0].set_ylabel("mean squared Euclidean error")
    axes[0].set_yscale("log")

    canonical_mean = [
        float(summary["canonical_knot_count"]["mean"]) for summary in per_k
    ]
    canonical_low = [
        float(summary["canonical_knot_count"]["mean_ci95"]["low"]) for summary in per_k
    ]
    canonical_high = [
        float(summary["canonical_knot_count"]["mean_ci95"]["high"]) for summary in per_k
    ]
    axes[1].plot(x, x, "--", color="#777777", linewidth=1.2, label="source K")
    axes[1].plot(
        x,
        canonical_mean,
        "o-",
        color="#355c9a",
        linewidth=1.7,
        markersize=4,
        label="canonical reference K",
    )
    axes[1].fill_between(
        x,
        canonical_low,
        canonical_high,
        color="#355c9a",
        alpha=0.13,
    )
    axes[1].set_title("(b) Final retained internal knots")
    axes[1].set_ylabel("internal-knot count")

    new_timing_scope = (
        per_k[0]["methods"]["ours"].get("time_scope")
        == "network_forward_only"
    )
    axes[2].set_title(
        "(c) Structure inference time (explicit scopes)"
        if new_timing_scope
        else "(c) Measured wall time (asymmetric scopes)"
    )
    axes[2].set_ylabel("median wall time per curve (ms)")
    axes[2].set_yscale("log")

    for axis in axes:
        axis.set_xlabel("initial/source internal-knot count K")
        axis.set_xticks(x)
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.legend(fontsize=8)

    ours_title = (
        "learned one-shot deployment"
        if ours_deployment == "learned"
        else "adaptive verified deployment"
    )
    parameterization_note = (
        "; shared chord-length parameterization"
        if ours_deployment == "verified" and verified_parameterization == "chord"
        else ""
    )
    figure.suptitle(
        f"{ours_title} vs shared-proposal greedy hard pruning\n"
        f"same curves and proposal per method; initial K={minimum_k}-{maximum_k}; "
        f"n={samples_per_k} curves/K; bands are 95% curve-bootstrap CIs"
        f"{parameterization_note}",
        fontsize=13,
    )
    new_timing_scope = (
        per_k[0]["methods"]["ours"].get("time_scope")
        == "network_forward_only"
    )
    ours_time_scope = (
        "Ours: synchronized forward_deployment only; final refit/verified repair excluded"
        if new_timing_scope
        else (
            "Ours: full network forward + final refit"
            if ours_deployment == "learned"
            else "Ours: full network forward + exact adaptive verification/repair"
        )
    )
    figure.text(
        0.5,
        0.015,
        f"{ours_time_scope}; Greedy hard: pruning stage only after the shared "
        "proposal/parameters are materialized. Timing scopes are asymmetric.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    figure.subplots_adjust(wspace=0.25, bottom=0.18, top=0.78)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _format_float(value: float, *, scientific: bool = False) -> str:
    return f"{value:.3e}" if scientific else f"{value:.3f}"


def _render_summary_markdown_legacy(report: Mapping[str, object]) -> str:
    overall = report["balanced_overall_summary"]
    assert isinstance(overall, Mapping)
    methods = overall["methods"]
    assert isinstance(methods, Mapping)
    tolerance = float(report["mse_tolerance"])
    requested_k = report["dataset"]["requested_source_k"]
    minimum_k = min(int(value) for value in requested_k)
    maximum_k = max(int(value) for value in requested_k)
    lines = [
        f"# K={minimum_k}–{maximum_k} 分层对比实验",
        "",
        "本实验按源内部节点数 `source K` 分层；节点数量误差以同一阈值下的 "
        "`canonical reference K` 为主要参考。canonical 是真实参数域上的贪心简化标签，"
        "不是全局最少节点证明。",
        "",
        "## K-balanced 总体结果",
        "",
        "| 方法 | mean K | 对 canonical 的 bias / MAE | mean / P95 MSE | pass rate | "
        "match F1@0.010 | median structure time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        summary = methods[method]
        assert isinstance(summary, Mapping)
        label = "Ours (v12 one-shot)" if method == "ours" else "Greedy hard"
        match = summary["matching_warped_to_canonical"]["0.010"]
        lines.append(
            f"| {label} | {float(summary['knot_count']['mean']):.3f} | "
            f"{float(summary['count_bias_canonical']['mean']):+.3f} / "
            f"{float(summary['count_mae_canonical']['mean']):.3f} | "
            f"{float(summary['fit_mse']['mean']):.3e} / "
            f"{float(summary['fit_mse']['p95']):.3e} | "
            f"{100.0 * float(summary['pass_rate']['mean']):.1f}% | "
            f"{float(match['f1']):.3f} | "
            f"{float(summary['time_ms']['median']):.2f} ms |"
        )
    paired = overall["paired"]
    assert isinstance(paired, Mapping)
    lines.extend(
        [
            "",
            f"MSE 阈值为 `{tolerance:.3e}`。逐曲线 Hard-stage/Ours-full 非对称时间比的"
            f"中位数为 `{float(paired['hard_stage_over_ours_full_time_ratio']['median']):.3f}`。",
            "",
            "## 按 source K 分层",
            "",
            "| source K | mean canonical K | ours K / hard K | ours / hard mean MSE | "
            "ours / hard pass | ours / hard median ms |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in report["per_k_summary"]:
        ours = summary["methods"]["ours"]
        hard = summary["methods"]["hard"]
        lines.append(
            f"| {int(summary['source_k'])} | "
            f"{float(summary['canonical_knot_count']['mean']):.2f} | "
            f"{float(ours['knot_count']['mean']):.2f} / "
            f"{float(hard['knot_count']['mean']):.2f} | "
            f"{float(ours['fit_mse']['mean']):.2e} / "
            f"{float(hard['fit_mse']['mean']):.2e} | "
            f"{100.0 * float(ours['pass_rate']['mean']):.0f}% / "
            f"{100.0 * float(hard['pass_rate']['mean']):.0f}% | "
            f"{float(ours['time_ms']['median']):.1f} / "
            f"{float(hard['time_ms']['median']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## 计时边界",
            "",
            "- Ours：归一化点张量已驻留 CPU 后，完整网络 forward（ParameterHead、proposal、"
            "KeepMask、relocation）加一次最终标准 B 样条 refit。",
            "- Hard：预测参数和 proposal 已物化后，仅计 greedy pruning stage；包含其内部全部"
            "标准 refit，但排除网络 forward。",
            "- 因此两者时间边界按实验要求不对称，不能称为完全同边界端到端加速比。",
            "",
            "节点匹配先按采样对应关系把预测参数域节点分段线性映射到真实参数域，再与 "
            "canonical reference 匹配；该映射仅用于诊断，不参与曲线 refit。",
            "",
        ]
    )
    return "\n".join(lines)


def render_summary_markdown(report: Mapping[str, object]) -> str:
    """Render an explicit learned/verified report with honest timing scopes."""

    overall = report["balanced_overall_summary"]
    assert isinstance(overall, Mapping)
    methods = overall["methods"]
    assert isinstance(methods, Mapping)
    ours = methods["ours"]
    hard = methods["hard"]
    assert isinstance(ours, Mapping) and isinstance(hard, Mapping)
    deployment_report = report.get("ours_deployment", {})
    assert isinstance(deployment_report, Mapping)
    ours_deployment = str(
        deployment_report.get("mode", ours.get("deployment_mode", "learned"))
    )
    ours_label = ours_method_label(ours_deployment)
    tolerance = float(report["mse_tolerance"])
    requested_k = report["dataset"]["requested_source_k"]
    minimum_k = min(int(value) for value in requested_k)
    maximum_k = max(int(value) for value in requested_k)

    lines = [
        f"# Source-K stratified comparison: K={minimum_k}-{maximum_k}",
        "",
        f"Ours deployment: **{ours_label}**. "
        + (
            "This is a pure learned one-shot path."
            if ours_deployment == "learned"
            else (
                "This is an adaptive exact-refit verification and conditional-repair "
                "path; it is not pure one-shot deployment."
            )
        ),
        "",
        f"Fit bound: MSE <= `{tolerance:.3e}` "
        f"(equivalent RMS <= `{math.sqrt(tolerance):.6f}`).",
        "Canonical K is a true-parameter greedy simplification reference, not a "
        "global-minimum certificate.",
        "",
        "## K-balanced overall results",
        "",
        "| Method | mean K | canonical bias / MAE | mean / P95 MSE | pass rate | "
        "match F1@0.010 | median time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, label in ((ours, ours_label), (hard, "Greedy hard")):
        match = method["matching_warped_to_canonical"]["0.010"]
        lines.append(
            f"| {label} | {float(method['knot_count']['mean']):.3f} | "
            f"{float(method['count_bias_canonical']['mean']):+.3f} / "
            f"{float(method['count_mae_canonical']['mean']):.3f} | "
            f"{float(method['fit_mse']['mean']):.3e} / "
            f"{float(method['fit_mse']['p95']):.3e} | "
            f"{100.0 * float(method['pass_rate']['mean']):.1f}% | "
            f"{float(match['f1']):.3f} | "
            f"{float(method['time_ms']['median']):.2f} ms |"
        )

    if ours_deployment == "verified":
        verified = ours["verified_adaptive"]
        assert isinstance(verified, Mapping)
        lines.extend(
            [
                "",
                "## Adaptive verified repair diagnostics",
                "",
                "The reported Ours time is still pure network forward only; the "
                "verified operations used to obtain the quality geometry are excluded "
                "and retained only as timing diagnostics.",
                "",
                f"- Learned-feasible: `{100.0 * float(verified['learned_feasible_fraction']):.1f}%`; "
                f"zero-cleanup fast path: `{100.0 * float(verified['fast_path_fraction']):.1f}%`; "
                f"repair/fallback: "
                f"`{100.0 * float(verified['fallback_used']['mean']):.1f}%`.",
                f"- Confidence-prefix cleanup: "
                f"`{100.0 * float(verified['cleanup_used']['mean']):.1f}%`; "
                f"hard fallback: "
                f"`{100.0 * float(verified['hard_fallback_used']['mean']):.1f}%`.",
                f"- Residual-guided dynamic insertion: "
                f"`{100.0 * float(verified['residual_fallback_used']['mean']):.1f}%`; "
                f"inserted K mean/max: "
                f"`{float(verified['inserted_knot_count']['mean']):.3f}` / "
                f"`{int(float(verified['inserted_knot_count']['max']))}`.",
                f"- Exact direct refits mean: "
                f"`{float(verified['direct_refit_count']['mean']):.3f}`; "
                f"all exact fit evaluations mean: "
                f"`{float(verified['fit_evaluation_count']['mean']):.3f}`; "
                f"residual-stage refits mean/max: "
                f"`{float(verified['residual_refit_count']['mean']):.3f}` / "
                f"`{int(float(verified['residual_refit_count']['max']))}`.",
                f"- Final sources: `{verified['final_source_histogram']}`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Per-source-K results",
            "",
            f"| source K | canonical K | {ours_label} K / Hard K | "
            "Ours / Hard mean MSE | Ours / Hard pass | forward / search median ms |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in report["per_k_summary"]:
        ours_k = summary["methods"]["ours"]
        hard_k = summary["methods"]["hard"]
        lines.append(
            f"| {int(summary['source_k'])} | "
            f"{float(summary['canonical_knot_count']['mean']):.2f} | "
            f"{float(ours_k['knot_count']['mean']):.2f} / "
            f"{float(hard_k['knot_count']['mean']):.2f} | "
            f"{float(ours_k['fit_mse']['mean']):.2e} / "
            f"{float(hard_k['fit_mse']['mean']):.2e} | "
            f"{100.0 * float(ours_k['pass_rate']['mean']):.0f}% / "
            f"{100.0 * float(hard_k['pass_rate']['mean']):.0f}% | "
            f"{float(ours_k['time_ms']['median']):.1f} / "
            f"{float(hard_k['time_ms']['median']):.1f} |"
        )

    ours_scope = report["timing_protocol"]["ours_scope"]
    hard_scope = report["timing_protocol"]["hard_scope"]
    lines.extend(
        [
            "",
            "## Timing boundary",
            "",
            f"- Ours: {ours_scope}.",
            f"- Hard: {hard_scope}.",
            "- These scopes are intentionally asymmetric; the ratio is not a "
            "same-boundary end-to-end speedup.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("CSV rows cannot be empty")
    serializable: list[dict[str, object]] = []
    for row in rows:
        serializable.append(
            {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable[0]))
        writer.writeheader()
        writer.writerows(serializable)


def _flat_per_k_rows(per_k: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary in per_k:
        row: dict[str, object] = {
            "source_k": summary["source_k"],
            "n": summary["n"],
            "canonical_k_mean": summary["canonical_knot_count"]["mean"],
            "canonical_k_median": summary["canonical_knot_count"]["median"],
            "parameter_rmse_mean": summary["parameter_rmse"]["mean"],
            "ours_deployment_mode": summary["methods"]["ours"].get(
                "deployment_mode", "learned"
            ),
        }
        row["ours_time_scope"] = summary["methods"]["ours"].get(
            "time_scope", "legacy_mixed_scope"
        )
        quality_timing = summary["methods"]["ours"].get(
            "quality_pipeline_time_ms"
        )
        if isinstance(quality_timing, Mapping):
            row["ours_quality_pipeline_diagnostic_median_ms"] = quality_timing[
                "median"
            ]
            row["ours_quality_pipeline_diagnostic_p95_ms"] = quality_timing["p95"]
        for method in METHODS:
            values = summary["methods"][method]
            match = values["matching_warped_to_canonical"]["0.010"]
            row.update(
                {
                    f"{method}_k_mean": values["knot_count"]["mean"],
                    f"{method}_count_bias_canonical": values["count_bias_canonical"][
                        "mean"
                    ],
                    f"{method}_count_mae_canonical": values["count_mae_canonical"][
                        "mean"
                    ],
                    f"{method}_count_exact_canonical_rate": values[
                        "count_exact_canonical_rate"
                    ],
                    f"{method}_mse_mean": values["fit_mse"]["mean"],
                    f"{method}_mse_median": values["fit_mse"]["median"],
                    f"{method}_mse_p95": values["fit_mse"]["p95"],
                    f"{method}_pass_rate": values["pass_rate"]["mean"],
                    f"{method}_match_precision_0p010": match["precision"],
                    f"{method}_match_recall_0p010": match["recall"],
                    f"{method}_match_f1_0p010": match["f1"],
                    f"{method}_time_median_ms": values["time_ms"]["median"],
                    f"{method}_time_p95_ms": values["time_ms"]["p95"],
                    f"{method}_time_min_ms": values["time_ms"]["min"],
                    f"{method}_time_max_ms": values["time_ms"]["max"],
                }
            )
        verified = summary["methods"]["ours"].get("verified_adaptive")
        if isinstance(verified, Mapping):
            row.update(
                {
                    "ours_verified_repair_rate": verified["fallback_used"]["mean"],
                    "ours_verified_hard_fallback_rate": verified["hard_fallback_used"][
                        "mean"
                    ],
                    "ours_verified_cleanup_rate": verified["cleanup_used"]["mean"],
                    "ours_verified_residual_insertion_rate": verified[
                        "residual_fallback_used"
                    ]["mean"],
                    "ours_verified_inserted_k_mean": verified["inserted_knot_count"][
                        "mean"
                    ],
                    "ours_verified_inserted_k_max": verified["inserted_knot_count"][
                        "max"
                    ],
                    "ours_verified_residual_refits_mean": verified[
                        "residual_refit_count"
                    ]["mean"],
                    "ours_verified_exact_fit_evaluations_mean": verified[
                        "fit_evaluation_count"
                    ]["mean"],
                    "ours_verified_final_source_histogram": verified[
                        "final_source_histogram"
                    ],
                }
            )
        row.update(
            {
                "ours_minus_hard_k_mean": summary["paired"][
                    "ours_minus_hard_knot_count"
                ]["mean"],
                "hard_stage_over_ours_full_time_ratio_median": summary["paired"][
                    "hard_stage_over_ours_full_time_ratio"
                ]["median"],
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare learned one-shot or adaptive verified deployment with "
            "shared-proposal greedy hard pruning across source-knot strata."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-knot-count", type=int, default=20)
    parser.add_argument("--min-knot-count", type=int, default=4)
    parser.add_argument("--max-knot-count", type=int, default=None)
    parser.add_argument("--scan-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--selection-seed", type=int, default=12345)
    parser.add_argument("--execution-seed", type=int, default=67890)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=24680)
    parser.add_argument("--mse-tolerance", type=float, default=None)
    parser.add_argument(
        "--ours-deployment",
        choices=("learned", "verified"),
        default="learned",
        help=(
            "learned preserves pure one-shot deployment; verified adds exact-refit "
            "verification and conditional adaptive repair."
        ),
    )
    parser.add_argument(
        "--verified-compact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Greedily clean up a feasible verified confidence prefix.",
    )
    parser.add_argument(
        "--verified-hard-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow proposal-only hard fallback if verified repair remains infeasible.",
    )
    parser.add_argument(
        "--verified-residual-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow residual-guided dynamic insertion after full-proposal failure.",
    )
    parser.add_argument(
        "--verified-max-residual-insertions",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--verified-residual-min-gap",
        type=float,
        default=None,
        help="Defaults to the checkpoint model min_knot_gap.",
    )
    parser.add_argument(
        "--verified-refit-device",
        choices=("auto", "cpu", "model"),
        default="auto",
        help="auto/cpu use CPU exact refits; model uses the network device.",
    )
    parser.add_argument(
        "--verified-parameterization",
        choices=("network", "chord-fallback", "chord"),
        default="chord-fallback",
        help=(
            "Verified exact-refit parameterization. For chord, the greedy hard "
            "baseline uses the same chord-warped proposal and chord parameters."
        ),
    )
    parser.add_argument(
        "--knot-tolerances",
        type=float,
        nargs="+",
        default=(0.005, 0.01, 0.02, 0.05),
    )
    parser.add_argument("--min-hard-knots", type=int, default=0)
    parser.add_argument("--smoothness-weight", type=float, default=1e-6)
    parser.add_argument("--control-ridge", type=float, default=0.0)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    positive_values = (
        args.samples_per_knot_count,
        args.scan_size,
        args.timing_repeats,
        args.bootstrap_replicates,
        args.torch_num_threads,
        args.dpi,
    )
    if any(value <= 0 for value in positive_values):
        parser.error("sample/scan/timing/bootstrap/thread/dpi values must be positive")
    if args.warmup_repeats < 0:
        parser.error("--warmup-repeats must be non-negative")
    if args.min_knot_count < 0 or (
        args.max_knot_count is not None and args.max_knot_count < args.min_knot_count
    ):
        parser.error("invalid knot-count range")
    if args.min_hard_knots < 0:
        parser.error("--min-hard-knots must be non-negative")
    if args.verified_max_residual_insertions < 0:
        parser.error("--verified-max-residual-insertions must be non-negative")
    if args.verified_residual_min_gap is not None and (
        not math.isfinite(args.verified_residual_min_gap)
        or args.verified_residual_min_gap < 0.0
    ):
        parser.error("--verified-residual-min-gap must be finite and non-negative")
    if args.smoothness_weight < 0.0 or args.control_ridge < 0.0:
        parser.error("solver regularization weights must be non-negative")
    if any(value < 0.0 for value in args.knot_tolerances):
        parser.error("knot tolerances must be non-negative")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    output_dir = args.output_dir
    output_paths = (
        output_dir / "stratified_comparison.json",
        output_dir / "per_sample_results.csv",
        output_dir / "per_k_summary.csv",
        output_dir / "stratified_comparison.png",
        output_dir / "stratified_three_metrics.png",
        output_dir / "summary.md",
    )
    if not args.overwrite and any(path.exists() for path in output_paths):
        parser.error(f"output files already exist in {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.torch_num_threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    try:
        mse_tolerance = resolve_mse_tolerance(checkpoint, args.mse_tolerance)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    model, model_config, _ = build_model_from_checkpoint(checkpoint)
    if model_config.get("structure_mode") != "candidate_pruning_one_shot":
        parser.error("this experiment requires a v8-v12 one-shot checkpoint")
    model.eval()
    model_device = next(model.parameters()).device
    verified_refit_device = (
        model_device if args.verified_refit_device == "model" else torch.device("cpu")
    )
    verified_residual_min_gap = (
        float(model_config.get("min_knot_gap", 1e-3))
        if args.verified_residual_min_gap is None
        else float(args.verified_residual_min_gap)
    )
    if not math.isfinite(verified_residual_min_gap) or verified_residual_min_gap < 0.0:
        parser.error(
            "resolved verified residual min gap must be finite and non-negative"
        )
    dataset_config = _dataset_config_from_checkpoint(checkpoint, model_config)

    maximum_count = args.max_knot_count
    if maximum_count is None:
        maximum_count = max(
            0,
            int(dataset_config["max_control_points"]) - model.degree - 1,
        )
    requested_counts = list(range(args.min_knot_count, maximum_count + 1))
    index_to_count = _scan_counts(
        dataset_config,
        scan_size=args.scan_size,
        dataset_seed=args.seed,
        stratify_by="source",
    )
    try:
        selected_indices = fixed_samples_per_knot_count(
            index_to_count,
            requested_counts,
            samples_per_count=args.samples_per_knot_count,
            seed=args.selection_seed,
        )
    except ValueError as error:
        parser.error(str(error))
    selected_indices_by_k = {
        str(source_k): [
            index for index in selected_indices if index_to_count[index] == source_k
        ]
        for source_k in requested_counts
    }
    execution_rng = random.Random(args.execution_seed)
    execution_indices = list(selected_indices)
    execution_rng.shuffle(execution_indices)
    order_rng = random.Random(args.execution_seed + 1)

    dataset = SyntheticCubicBSplineDataset(
        size=args.scan_size,
        seed=args.seed,
        cache_samples=False,
        **dataset_config,
    )
    print(f"Materializing {len(execution_indices)} selected test curves...", flush=True)
    materialized = [(index, dataset[index]) for index in execution_indices]

    rows: list[dict[str, object]] = []
    experiment_started = time.perf_counter()
    for sequence_index, (sample_index, sample) in enumerate(materialized, start=1):
        points = sample["points"]
        true_parameters = sample["true_params"]
        chord_parameters = sample["chord_params"]
        assert isinstance(points, torch.Tensor)
        assert isinstance(true_parameters, torch.Tensor)
        assert isinstance(chord_parameters, torch.Tensor)
        chord_parameters = chord_parameters.clamp(0.0, 1.0).clone()
        chord_parameters[0] = 0.0
        chord_parameters[-1] = 1.0
        batched_points = points.unsqueeze(0)

        with torch.inference_mode():
            materialized_output = deployment_network_forward(model, batched_points)
        proposal_knots, _, _ = deployment_geometries(materialized_output)
        network_parameters = materialized_output["params"][0]
        shared_chord_domain = (
            args.ours_deployment == "verified"
            and args.verified_parameterization == "chord"
        )
        if shared_chord_domain:
            hard_parameters = chord_parameters
            hard_proposal_knots = warp_internal_knots_to_parameterization(
                proposal_knots,
                network_parameters,
                chord_parameters,
            )
        else:
            hard_parameters = network_parameters
            hard_proposal_knots = proposal_knots

        def run_ours() -> OursDeploymentResult:
            return deploy_ours(
                model,
                batched_points,
                points,
                deployment=args.ours_deployment,
                chord_parameters=chord_parameters,
                verified_parameterization=args.verified_parameterization,
                mse_tolerance=mse_tolerance,
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
                verified_compact=args.verified_compact,
                verified_hard_fallback=args.verified_hard_fallback,
                verified_residual_fallback=args.verified_residual_fallback,
                verified_max_residual_insertions=(
                    args.verified_max_residual_insertions
                ),
                verified_residual_min_gap=verified_residual_min_gap,
                verified_refit_device=verified_refit_device,
            )

        def run_network_forward_only() -> dict[str, torch.Tensor]:
            with torch.inference_mode():
                return deployment_network_forward(model, batched_points)

        def run_hard():
            return greedy_prune_to_mse_tolerance(
                hard_parameters,
                points,
                hard_proposal_knots,
                mse_tolerance=mse_tolerance,
                min_internal_knots=args.min_hard_knots,
                degree=model.degree,
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
            )

        network_timing = measure_synchronized_wall_time(
            run_network_forward_only,
            synchronization_device=model_device,
            warmup_repeats=args.warmup_repeats,
            timing_repeats=args.timing_repeats,
        )

        # Preserve complete quality-pipeline and Hard-search timings as
        # diagnostics.  Only ``network_timing`` is used as the primary Ours
        # prediction time in summaries, tables and figures.
        for _ in range(args.warmup_repeats):
            run_ours()
            run_hard()

        (
            ours_result,
            hard_result,
            ours_quality_repeat_ms,
            hard_repeat_ms,
            method_orders,
        ) = paired_timing_blocks(
            run_ours,
            run_hard,
            repeats=args.timing_repeats,
            order_rng=order_rng,
        )
        ours_output = ours_result.output
        ours_fit = ours_result.final_fit
        ours_knots = ours_result.final_internal_knots
        ours_repair = ours_result.verified_repair
        hard_knots = hard_result.retained_internal_knots
        canonical_knots = _canonical_internal_knots(sample)
        source_knots = _source_internal_knots(sample, degree=model.degree)
        predicted_parameters = ours_output["params"][0]
        ours_final_parameters = ours_result.final_parameters

        source_k = int(source_knots.numel())
        canonical_k = int(canonical_knots.numel())
        ours_k = int(ours_knots.numel())
        hard_k = int(hard_knots.numel())
        row: dict[str, object] = {
            "sample_index": sample_index,
            "sample_seed": args.seed + sample_index,
            "execution_position": sequence_index,
            "source_knot_count": source_k,
            "canonical_knot_count": canonical_k,
            "proposal_knot_count": int(proposal_knots.numel()),
            "source_internal_knots": source_knots.detach().cpu().tolist(),
            "canonical_internal_knots": canonical_knots.detach().cpu().tolist(),
            "proposal_internal_knots": proposal_knots.detach().cpu().tolist(),
            "hard_proposal_internal_knots": (
                hard_proposal_knots.detach().cpu().tolist()
            ),
            "ours_final_parameterization": (
                ours_repair.final_parameterization
                if ours_repair is not None
                else "network_predicted"
            ),
            "hard_final_parameterization": (
                "chord_length" if shared_chord_domain else "network_predicted"
            ),
            "parameter_rmse": float(
                (predicted_parameters - true_parameters).square().mean().sqrt()
            ),
            "diagnostic_quality_hard_method_orders": method_orders,
            "method_orders": method_orders,
            "ours_deployment_mode": args.ours_deployment,
            "ours_time_scope": (
                "model.forward_deployment(points) only: ParameterHead, candidate "
                "proposal, KeepMask and survivor relocation; resident batch=1 "
                "input; excludes standard B-spline refit and verified repair"
            ),
            "ours_quality_pipeline_diagnostic_scope": (
                "complete network forward + one final standard B-spline refit"
                if args.ours_deployment == "learned"
                else (
                    "complete network forward + exact verification and conditional "
                    "adaptive repair"
                )
            ),
            "ours_knot_count": ours_k,
            "ours_count_bias_canonical": ours_k - canonical_k,
            "ours_count_bias_source": ours_k - source_k,
            "ours_fit_mse": float(ours_fit.fit_mse),
            "ours_threshold_satisfied": float(ours_fit.fit_mse) <= mse_tolerance,
            "ours_time_repeats_ms": list(network_timing.latency.repeat_ms),
            "ours_time_median_ms": network_timing.latency.p50_ms,
            "ours_network_forward_p50_ms": network_timing.latency.p50_ms,
            "ours_network_forward_p95_ms": network_timing.latency.p95_ms,
            "ours_quality_pipeline_time_repeats_ms": ours_quality_repeat_ms,
            "ours_quality_pipeline_time_median_ms": float(
                statistics.median(ours_quality_repeat_ms)
            ),
            "ours_internal_knots": ours_knots.detach().cpu().tolist(),
            "ours_verified_final_source": (
                ours_repair.final_source
                if ours_repair is not None
                else "learned_one_shot"
            ),
            "ours_verified_learned_threshold_satisfied": (
                ours_repair.learned_threshold_satisfied
                if ours_repair is not None
                else float(ours_fit.fit_mse) <= mse_tolerance
            ),
            "ours_verified_fallback_used": (
                ours_repair.fallback_used if ours_repair is not None else False
            ),
            "ours_verified_hard_fallback_used": (
                ours_repair.hard_fallback_used if ours_repair is not None else False
            ),
            "ours_verified_cleanup_used": (
                ours_repair.cleanup_used if ours_repair is not None else False
            ),
            "ours_verified_residual_fallback_used": (
                ours_repair.residual_fallback_used if ours_repair is not None else False
            ),
            "ours_verified_parameterization_fallback_attempted": (
                ours_repair.parameterization_fallback_attempted
                if ours_repair is not None
                else False
            ),
            "ours_verified_parameterization_fallback_used": (
                ours_repair.parameterization_fallback_used
                if ours_repair is not None
                else False
            ),
            "ours_verified_inserted_knot_count": (
                ours_repair.inserted_count if ours_repair is not None else 0
            ),
            "ours_verified_inserted_internal_knots": (
                ours_repair.inserted_internal_knots.detach().cpu().tolist()
                if ours_repair is not None
                else []
            ),
            "ours_verified_residual_refit_count": (
                ours_repair.residual_fallback_refit_count
                if ours_repair is not None
                else 0
            ),
            "ours_verified_direct_refit_count": (
                ours_repair.direct_refit_count if ours_repair is not None else 0
            ),
            "ours_verified_fit_evaluation_count": (
                ours_repair.fit_evaluation_count if ours_repair is not None else 0
            ),
            "ours_verified_prefix_count_evaluations": (
                len(ours_repair.prefix_counts_evaluated)
                if ours_repair is not None
                else 0
            ),
            "ours_verified_prefix_counts_evaluated": (
                list(ours_repair.prefix_counts_evaluated)
                if ours_repair is not None
                else []
            ),
            "ours_verified_retained_proposal_count": (
                int(ours_repair.retained_proposal_mask.sum().item())
                if ours_repair is not None
                else ours_k
            ),
            "ours_verified_deployment_candidate_count": (
                ours_repair.deployment_candidate_count
                if ours_repair is not None
                else int(proposal_knots.numel())
            ),
            "hard_knot_count": hard_k,
            "hard_count_bias_canonical": hard_k - canonical_k,
            "hard_count_bias_source": hard_k - source_k,
            "hard_fit_mse": float(hard_result.final_fit.fit_mse),
            "hard_threshold_satisfied": hard_result.threshold_satisfied,
            "hard_time_repeats_ms": hard_repeat_ms,
            "hard_time_median_ms": float(statistics.median(hard_repeat_ms)),
            "hard_internal_knots": hard_knots.detach().cpu().tolist(),
            "hard_accepted_deletions": hard_result.accepted_deletions,
        }
        _record_matches(
            row,
            method="ours",
            knots_in_predicted_domain=ours_knots,
            predicted_parameters=ours_final_parameters,
            true_parameters=true_parameters,
            canonical_knots=canonical_knots,
            tolerances=args.knot_tolerances,
        )
        _record_matches(
            row,
            method="hard",
            knots_in_predicted_domain=hard_knots,
            predicted_parameters=hard_parameters,
            true_parameters=true_parameters,
            canonical_knots=canonical_knots,
            tolerances=args.knot_tolerances,
        )
        rows.append(row)
        if sequence_index % 10 == 0 or sequence_index == len(materialized):
            elapsed = time.perf_counter() - experiment_started
            print(
                f"[{sequence_index}/{len(materialized)}] elapsed={elapsed:.1f}s | "
                f"last source/canonical K={source_k}/{canonical_k} | "
                f"ours-{args.ours_deployment}/hard K={ours_k}/{hard_k} | "
                f"MSE={float(ours_fit.fit_mse):.2e}/"
                f"{float(hard_result.final_fit.fit_mse):.2e}",
                flush=True,
            )

    per_k_summary: list[dict[str, object]] = []
    for source_k in requested_counts:
        group = [row for row in rows if int(row["source_knot_count"]) == source_k]
        if len(group) != args.samples_per_knot_count:
            raise RuntimeError(f"unexpected sample count for source K={source_k}")
        per_k_summary.append(
            summarize_rows(
                group,
                source_k=source_k,
                knot_tolerances=args.knot_tolerances,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed,
            )
        )
    balanced_overall = summarize_rows(
        rows,
        source_k=0,
        knot_tolerances=args.knot_tolerances,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    balanced_overall["source_k"] = "K-balanced 4-20"

    ours_summary = balanced_overall["methods"]["ours"]
    report: dict[str, object] = {
        "schema_version": 4,
        "experiment": (
            f"source_K_stratified_ours_{args.ours_deployment}_vs_"
            "shared_proposal_greedy_hard"
        ),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": _checkpoint_sha256(args.checkpoint),
            "objective_version": checkpoint.get("objective_version"),
            "epoch": checkpoint.get("epoch"),
        },
        "source_k_definition": (
            "generative source internal-knot count; cubic control points 8-24 "
            "correspond to source K=4-20"
        ),
        "count_reference": (
            "canonical reference K from true-parameter greedy simplification at "
            "the checkpoint tolerance; not a global-minimum certificate"
        ),
        "mse_tolerance": mse_tolerance,
        "rms_tolerance": math.sqrt(mse_tolerance),
        "metric_definition": "mean_i ||C(t_i)-Q_i||_2^2 in normalized coordinates",
        "square_root_applied": False,
        "figures": {
            "four_metric": "stratified_comparison.png",
            "three_metric": "stratified_three_metrics.png",
            "three_metric_panels": [
                "standard B-spline refit MSE",
                "final retained internal-knot count",
                "structure-inference wall time per curve with explicit scopes",
            ],
            "data_source": (
                "both figures use this report's identical selected curves, "
                "shared per-curve proposal, per-K summaries and timing protocol"
            ),
        },
        "parameterization_comparison_scope": (
            "shared_chord_domain_for_ours_and_hard"
            if args.ours_deployment == "verified"
            and args.verified_parameterization == "chord"
            else "network_domain_hard_with_conditional_ours_chord_rescue"
            if args.ours_deployment == "verified"
            and args.verified_parameterization == "chord-fallback"
            else "shared_network_predicted_domain"
        ),
        "ours_deployment": {
            "mode": args.ours_deployment,
            "label": ours_method_label(args.ours_deployment),
            "pure_one_shot": args.ours_deployment == "learned",
            "adaptive_verified_repair": args.ours_deployment == "verified",
            "verified_config": (
                {
                    "compact": bool(args.verified_compact),
                    "hard_fallback": bool(args.verified_hard_fallback),
                    "residual_fallback": bool(args.verified_residual_fallback),
                    "max_residual_insertions": (args.verified_max_residual_insertions),
                    "residual_min_gap": verified_residual_min_gap,
                    "refit_device_requested": args.verified_refit_device,
                    "refit_device_resolved": str(verified_refit_device),
                    "parameterization_policy": args.verified_parameterization,
                    "hard_baseline_shared_chord_domain": (
                        args.verified_parameterization == "chord"
                    ),
                }
                if args.ours_deployment == "verified"
                else None
            ),
            "balanced_adaptive_diagnostics": (
                ours_summary.get("verified_adaptive")
                if args.ours_deployment == "verified"
                else None
            ),
        },
        "dataset": {
            "config": dataset_config,
            "seed": args.seed,
            "scan_size": args.scan_size,
            "requested_source_k": requested_counts,
            "samples_per_source_k": args.samples_per_knot_count,
            "selected_indices_by_source_k": selected_indices_by_k,
            "execution_indices_randomized": execution_indices,
        },
        "timing_protocol": {
            "device": str(model_device),
            "dtype": str(next(model.parameters()).dtype),
            "batch_size": 1,
            "clock": "time.perf_counter_ns",
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "warmup_repeats_per_method_per_curve": args.warmup_repeats,
            "paired_blocks_per_curve": args.timing_repeats,
            "method_order": (
                "Ours network forward is independently synchronized; complete Ours "
                "quality-pipeline and Hard diagnostics are randomized in paired blocks"
            ),
            "execution_seed": args.execution_seed,
            "per_curve_statistic": "median of synchronized repeat durations",
            "ours_deployment_mode": args.ours_deployment,
            "ours_scope": (
                "model.forward_deployment(points) only: parameter prediction, "
                "candidate proposal, KeepMask and survivor relocation; excludes "
                "final standard B-spline refit, verification and repair"
            ),
            "ours_time_field": "ours_time_median_ms",
            "ours_time_field_semantics": "pure_network_forward_only",
            "ours_quality_pipeline_diagnostic_scope": (
                "complete model forward plus final standard B-spline refit"
                if args.ours_deployment == "learned"
                else (
                    "complete model forward plus exact verification and conditional "
                    "confidence add-back/cleanup/residual insertion/hard fallback"
                )
            ),
            "ours_quality_pipeline_diagnostic_field": (
                "ours_quality_pipeline_time_median_ms"
            ),
            "quality_geometry_may_include_verified_postprocessing": (
                args.ours_deployment == "verified"
            ),
            "reported_ours_time_excludes_quality_postprocessing": True,
            "cuda_synchronized_before_and_after_each_network_repeat": (
                model_device.type == "cuda"
            ),
            "network_inputs_resident_before_timing": True,
            "ours_verified_refit_device": (
                str(verified_refit_device)
                if args.ours_deployment == "verified"
                else None
            ),
            "hard_scope": (
                "greedy pruning stage only from the materialized shared-domain "
                "proposal (chord-warped when Ours uses chord, otherwise network "
                "predicted); includes every internal standard refit"
            ),
            "excluded": (
                "checkpoint load, dataset generation, source/canonical truth, input "
                "materialization/transfer, final Ours refit or verified repair, plotting, "
                "aggregation, and serialization"
            ),
            "asymmetric_scopes": True,
        },
        "uncertainty": {
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "confidence": 0.95,
            "unit": "independent curve",
            "interval": "percentile bootstrap",
        },
        "hardware_software": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
        },
        "knot_match": {
            "tolerances": list(args.knot_tolerances),
            "parameterization": (
                "each method's deployed-domain knots are piecewise-linearly warped "
                "through sample correspondence into the true parameter domain before "
                "matching; full-chord ablations give Ours and Hard the same chord domain"
            ),
            "role": "diagnostic only; the warp is not used for refitting",
        },
        "per_k_summary": per_k_summary,
        "balanced_overall_summary": balanced_overall,
        "samples": rows,
    }

    (
        json_path,
        sample_csv,
        per_k_csv,
        figure_path,
        three_metric_figure_path,
        markdown_path,
    ) = output_paths
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(sample_csv, rows)
    _write_csv(per_k_csv, _flat_per_k_rows(per_k_summary))
    render_summary_figure(
        per_k_summary,
        figure_path,
        mse_tolerance=mse_tolerance,
        samples_per_k=args.samples_per_knot_count,
        dpi=args.dpi,
        ours_deployment=args.ours_deployment,
        verified_parameterization=args.verified_parameterization,
    )
    render_three_metric_figure(
        per_k_summary,
        three_metric_figure_path,
        mse_tolerance=mse_tolerance,
        samples_per_k=args.samples_per_knot_count,
        dpi=args.dpi,
        ours_deployment=args.ours_deployment,
        verified_parameterization=args.verified_parameterization,
    )
    markdown_path.write_text(render_summary_markdown(report), encoding="utf-8")
    print(f"Saved JSON: {json_path}")
    print(f"Saved per-sample CSV: {sample_csv}")
    print(f"Saved per-K CSV: {per_k_csv}")
    print(f"Saved figure: {figure_path}")
    print(f"Saved three-metric figure: {three_metric_figure_path}")
    print(f"Saved summary: {markdown_path}")


if __name__ == "__main__":
    main()
