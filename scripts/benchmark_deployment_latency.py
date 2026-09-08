from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import build_model_from_checkpoint  # noqa: E402
from spline_fitting.data.synthetic import SyntheticCubicBSplineDataset  # noqa: E402
from spline_fitting.evaluation.bspline_inference import (  # noqa: E402
    BSplineLeastSquaresFit,
    refit_bspline_control_points,
)
from spline_fitting.evaluation.knot_diagnostics import (  # noqa: E402
    warp_internal_knots_to_parameterization,
)
from spline_fitting.evaluation.verified_knot_repair import (  # noqa: E402
    VerifiedKnotRepairResult,
    verified_confidence_repair,
)


T = TypeVar("T")
DEPLOYMENT_MODES = (
    "network-forward",
    "refit-network",
    "refit-chord",
    "learned-network",
    "learned-chord",
    "verified-fast",
    "verified-compact",
)


def _dataset_config(
    checkpoint: Mapping[str, object],
    model_config: Mapping[str, object],
) -> dict[str, object]:
    config = dict(checkpoint.get("dataset_config", {}))
    config.setdefault("num_points", 192)
    config.setdefault("point_dim", int(model_config.get("point_dim", 2)))
    config.setdefault("min_control_points", 8)
    config.setdefault("max_control_points", 24)
    # Label construction is outside the timed region.  Keep the checkpoint's
    # generation mode (including certified-minimal geometry) so benchmark
    # inputs follow the training distribution, but skip padded label fields.
    config.setdefault("canonical_knot_tolerance", 0.0)
    config["return_ground_truth"] = False
    for key in (
        "size",
        "seed",
        "cache_samples",
        "resample_each_epoch",
        "epoch_seed_stride",
    ):
        config.pop(key, None)
    return config


def _deployment_geometries(
    output: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    proposal = output.get("proposal_internal_knots", output["internal_knots"])
    deployment = output.get("deployment_internal_knots", output["internal_knots"])
    learned_mask = output.get(
        "learned_keep_mask",
        output["keep_probability"] >= 0.5,
    ).to(torch.bool)
    if not (
        proposal.ndim == 2 and proposal.shape == deployment.shape == learned_mask.shape
    ):
        raise ValueError("proposal/deployment/mask must share shape [B,K]")
    return proposal, deployment, learned_mask


def _canonical_chord_parameters(parameters: torch.Tensor) -> torch.Tensor:
    result = parameters.detach().clone().clamp(0.0, 1.0)
    result[0] = 0.0
    result[-1] = 1.0
    return result


def _resolve_rms_tolerance(
    checkpoint: Mapping[str, object],
    explicit: float | None,
) -> float:
    if explicit is not None:
        result = float(explicit)
    else:
        deployment = checkpoint.get("deployment_config", {})
        dataset = checkpoint.get("dataset_config", {})
        if isinstance(deployment, Mapping) and "error_tolerance" in deployment:
            result = float(deployment["error_tolerance"])
        elif isinstance(dataset, Mapping) and "canonical_knot_tolerance" in dataset:
            result = float(dataset["canonical_knot_tolerance"])
        else:
            result = 5e-3
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("fit tolerance must be finite and non-negative")
    return result


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _measure(
    function: Callable[[], T],
    *,
    synchronization_device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[T, dict[str, float | list[float]]]:
    result: T | None = None
    for _ in range(warmup):
        result = function()
    _synchronize(synchronization_device)

    durations: list[float] = []
    for _ in range(repeats):
        _synchronize(synchronization_device)
        started = time.perf_counter_ns()
        result = function()
        _synchronize(synchronization_device)
        durations.append((time.perf_counter_ns() - started) / 1_000_000.0)
    assert result is not None
    return result, {
        "p50_ms": float(statistics.median(durations)),
        "median_ms": float(statistics.median(durations)),
        "mean_ms": float(statistics.fmean(durations)),
        "p95_ms": _percentile(durations, 0.95),
        "minimum_ms": min(durations),
        "maximum_ms": max(durations),
        "repeat_ms": durations,
    }


def _fit_quality(
    fits: Sequence[BSplineLeastSquaresFit],
    *,
    rms_tolerance: float,
) -> dict[str, object]:
    mse = [float(fit.fit_mse) for fit in fits]
    counts = [int(fit.internal_knots.numel()) for fit in fits]
    bound = rms_tolerance * rms_tolerance
    return {
        "mse_mean": float(statistics.fmean(mse)),
        "mse_p95": _percentile(mse, 0.95),
        "threshold_satisfied_fraction": sum(value <= bound for value in mse) / len(mse),
        "retained_knots_mean": float(statistics.fmean(counts)),
    }


@torch.inference_mode()
def _learned_refits(
    output: Mapping[str, torch.Tensor],
    points: torch.Tensor,
    chord_parameters: torch.Tensor,
    *,
    degree: int,
    refit_device: torch.device,
    chord: bool,
    smoothness_weight: float,
) -> list[BSplineLeastSquaresFit]:
    _, deployment, learned_mask = _deployment_geometries(output)
    fits: list[BSplineLeastSquaresFit] = []
    for index in range(points.shape[0]):
        network_parameters = output["params"][index].detach().to(refit_device)
        selected_knots = (
            deployment[index, learned_mask[index]].detach().to(refit_device)
        )
        fit_parameters = network_parameters
        if chord:
            fit_parameters = chord_parameters[index].detach().to(refit_device)
            selected_knots = warp_internal_knots_to_parameterization(
                selected_knots,
                network_parameters,
                fit_parameters,
            )
        fits.append(
            refit_bspline_control_points(
                fit_parameters,
                points[index].detach().to(refit_device),
                selected_knots,
                degree=degree,
                smoothness_weight=smoothness_weight,
                interpolate_endpoints=True,
            )
        )
    return fits


@torch.inference_mode()
def _verified_refits(
    output: Mapping[str, torch.Tensor],
    points: torch.Tensor,
    chord_parameters: torch.Tensor,
    *,
    degree: int,
    refit_device: torch.device,
    rms_tolerance: float,
    parameterization: str,
    compact: bool,
    smoothness_weight: float,
    residual_min_gap: float,
) -> list[VerifiedKnotRepairResult]:
    proposal, deployment, learned_mask = _deployment_geometries(output)
    results: list[VerifiedKnotRepairResult] = []
    for index in range(points.shape[0]):
        results.append(
            verified_confidence_repair(
                output["params"][index].detach().to(refit_device),
                points[index].detach().to(refit_device),
                proposal[index].detach().to(refit_device),
                deployment[index].detach().to(refit_device),
                learned_mask[index].detach().to(refit_device),
                output["keep_probability"][index].detach().to(refit_device),
                fit_tolerance_rms=rms_tolerance,
                degree=degree,
                smoothness_weight=smoothness_weight,
                interpolate_endpoints=True,
                compact=compact,
                hard_fallback=True,
                residual_fallback=True,
                max_residual_insertions=8,
                residual_min_gap=residual_min_gap,
                alternate_parameters=chord_parameters[index].detach().to(refit_device),
                parameterization_policy=parameterization,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark synchronized deployment latency without dataset loading or "
            "file I/O. Reports both batch latency and amortized latency per curve."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--modes",
        choices=DEPLOYMENT_MODES,
        nargs="+",
        default=[
            "network-forward",
            "refit-chord",
            "learned-chord",
            "verified-fast",
        ],
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--network-path",
        choices=("full", "structure-only"),
        default="full",
        help=(
            "structure-only skips the final training surrogate solve; proposal, "
            "KeepMask and relocated deployment knots remain unchanged."
        ),
    )
    parser.add_argument(
        "--refit-device",
        choices=("auto", "cpu", "model"),
        default="auto",
        help="auto/cpu use robust CPU gelsy; model follows the network device.",
    )
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--fit-tolerance", type=float, default=None)
    parser.add_argument(
        "--verified-parameterization",
        choices=("network", "chord-fallback", "chord"),
        default="chord-fallback",
    )
    parser.add_argument("--smoothness-weight", type=float, default=1e-6)
    parser.add_argument("--latency-target-ms", type=float, default=10.0)
    parser.add_argument(
        "--include-training-lower-bound",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also time forward/backward/AdamW using only reconstruction MSE. "
            "This is a lower bound, not the full teacher-distillation step."
        ),
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("warmup must be non-negative and repeats must be positive")
    if any(size <= 0 for size in args.batch_sizes):
        parser.error("batch sizes must be positive")
    if args.torch_num_threads <= 0:
        parser.error("torch-num-threads must be positive")
    if args.latency_target_ms <= 0.0:
        parser.error("latency-target-ms must be positive")

    if args.device == "auto":
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        model_device = torch.device(args.device)
    if model_device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    refit_device = model_device if args.refit_device == "model" else torch.device("cpu")
    torch.set_num_threads(args.torch_num_threads)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, model_config, _ = build_model_from_checkpoint(checkpoint)
    if model_config.get("structure_mode") != "candidate_pruning_one_shot":
        parser.error("this benchmark requires a one-shot candidate-pruning checkpoint")
    model.to(model_device).eval()
    rms_tolerance = _resolve_rms_tolerance(checkpoint, args.fit_tolerance)
    residual_min_gap = float(model_config.get("min_knot_gap", 1e-3))

    maximum_batch = max(args.batch_sizes)
    dataset = SyntheticCubicBSplineDataset(
        size=maximum_batch,
        seed=args.seed,
        cache_samples=True,
        **_dataset_config(checkpoint, model_config),
    )
    resident_points = torch.stack([dataset[i]["points"] for i in range(maximum_batch)])
    resident_chord = torch.stack(
        [
            _canonical_chord_parameters(dataset[i]["chord_params"])
            for i in range(maximum_batch)
        ]
    )

    rows: list[dict[str, object]] = []
    for batch_size in args.batch_sizes:
        points = resident_points[:batch_size].to(model_device)
        chord_parameters = resident_chord[:batch_size]

        def run_network() -> Mapping[str, torch.Tensor]:
            if args.network_path == "structure-only":
                return model.forward_deployment(points)
            return model(points)

        with torch.inference_mode():
            materialized_output = run_network()
        _synchronize(model_device)
        for mode in args.modes:

            def deploy() -> Any:
                with torch.inference_mode():
                    if mode in {"refit-network", "refit-chord"}:
                        return _learned_refits(
                            materialized_output,
                            points,
                            chord_parameters,
                            degree=int(model.degree),
                            refit_device=refit_device,
                            chord=mode == "refit-chord",
                            smoothness_weight=args.smoothness_weight,
                        )
                    output = run_network()
                    if mode == "network-forward":
                        return output
                    if mode in {"learned-network", "learned-chord"}:
                        return _learned_refits(
                            output,
                            points,
                            chord_parameters,
                            degree=int(model.degree),
                            refit_device=refit_device,
                            chord=mode == "learned-chord",
                            smoothness_weight=args.smoothness_weight,
                        )
                    return _verified_refits(
                        output,
                        points,
                        chord_parameters,
                        degree=int(model.degree),
                        refit_device=refit_device,
                        rms_tolerance=rms_tolerance,
                        parameterization=args.verified_parameterization,
                        compact=mode == "verified-compact",
                        smoothness_weight=args.smoothness_weight,
                        residual_min_gap=residual_min_gap,
                    )

            result, timing = _measure(
                deploy,
                synchronization_device=model_device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            median_ms = float(timing["median_ms"])
            row: dict[str, object] = {
                "mode": mode,
                "batch_size": batch_size,
                **timing,
                "median_ms_per_curve": median_ms / batch_size,
                "batch_under_target": median_ms <= args.latency_target_ms,
                "batch_p95_under_target": (
                    float(timing["p95_ms"]) <= args.latency_target_ms
                ),
                "amortized_per_curve_under_target": (
                    median_ms / batch_size <= args.latency_target_ms
                ),
                "amortized_p95_per_curve_under_target": (
                    float(timing["p95_ms"]) / batch_size <= args.latency_target_ms
                ),
            }
            if mode != "network-forward":
                fits = (
                    [item.final_fit for item in result]
                    if mode.startswith("verified")
                    else result
                )
                row["quality"] = _fit_quality(
                    fits,
                    rms_tolerance=rms_tolerance,
                )
                if mode.startswith("verified"):
                    row["verified"] = {
                        "learned_feasible_fraction": sum(
                            item.learned_threshold_satisfied for item in result
                        )
                        / len(result),
                        "repair_fraction": sum(item.fallback_used for item in result)
                        / len(result),
                        "fit_evaluations_mean": statistics.fmean(
                            item.fit_evaluation_count for item in result
                        ),
                    }
            rows.append(row)
            print(
                f"{mode:18s} B={batch_size:3d}  "
                f"P50/P95={median_ms:8.3f}/{float(timing['p95_ms']):8.3f} ms  "
                f"amortized={median_ms / batch_size:7.3f}/"
                f"{float(timing['p95_ms']) / batch_size:7.3f} ms/curve  "
                f"target(batch/amortized)="
                f"{'PASS' if row['batch_p95_under_target'] else 'FAIL'}/"
                f"{'PASS' if row['amortized_p95_per_curve_under_target'] else 'FAIL'}",
                flush=True,
            )

    training_rows: list[dict[str, object]] = []
    if args.include_training_lower_bound:
        for batch_size in args.batch_sizes:
            training_model, _, _ = build_model_from_checkpoint(checkpoint)
            training_model.to(model_device).train()
            optimizer = torch.optim.AdamW(training_model.parameters(), lr=1e-5)
            points = resident_points[:batch_size].to(model_device)

            def training_step() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                output = training_model(points)
                loss = (output["reconstructed_points"] - points).square().mean()
                loss = loss + 1e-6 * output["keep_probability"].mean()
                loss.backward()
                optimizer.step()
                return loss.detach()

            _, timing = _measure(
                training_step,
                synchronization_device=model_device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            p50_ms = float(timing["p50_ms"])
            p95_ms = float(timing["p95_ms"])
            training_rows.append(
                {
                    "batch_size": batch_size,
                    **timing,
                    "p50_ms_per_curve": p50_ms / batch_size,
                    "p95_ms_per_curve": p95_ms / batch_size,
                    "batch_p50_under_target": p50_ms <= args.latency_target_ms,
                    "batch_p95_under_target": p95_ms <= args.latency_target_ms,
                    "amortized_p50_under_target": (
                        p50_ms / batch_size <= args.latency_target_ms
                    ),
                    "amortized_p95_under_target": (
                        p95_ms / batch_size <= args.latency_target_ms
                    ),
                }
            )
            print(
                f"{'train-lower-bound':18s} B={batch_size:3d}  "
                f"P50/P95={p50_ms:8.3f}/{p95_ms:8.3f} ms  "
                f"amortized={p50_ms / batch_size:7.3f}/"
                f"{p95_ms / batch_size:7.3f} ms/curve  "
                f"target(batch/amortized)="
                f"{'PASS' if p95_ms <= args.latency_target_ms else 'FAIL'}/"
                f"{'PASS' if p95_ms / batch_size <= args.latency_target_ms else 'FAIL'}",
                flush=True,
            )

    report: dict[str, object] = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "objective_version": checkpoint.get("objective_version"),
        "model_device": str(model_device),
        "network_path": args.network_path,
        "refit_device": str(refit_device),
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_device": (
            torch.cuda.get_device_name(model_device)
            if model_device.type == "cuda"
            else None
        ),
        "torch_num_threads": torch.get_num_threads(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "latency_target_ms": args.latency_target_ms,
        "fit_tolerance_rms": rms_tolerance,
        "fit_tolerance_mse": rms_tolerance * rms_tolerance,
        "timing_scope": (
            "resident normalized input -> synchronized result; deployment modes include "
            "network forward and standard B-spline refits; excludes dataset generation, "
            "file I/O, plotting, and request transport"
        ),
        "mode_scopes": {
            "network-forward": (
                "model(points), including both internal proxy solves"
                if args.network_path == "full"
                else "model.forward_deployment(points), retaining the pilot solve but "
                "skipping the final training-only surrogate solve"
            ),
            "refit-network": (
                "one standard B-spline refit per curve from materialized network-domain "
                "outputs; includes model-to-refit-device transfer"
            ),
            "refit-chord": (
                "knot warp plus one chord-domain standard refit per curve from "
                "materialized outputs; includes model-to-refit-device transfer"
            ),
            "learned-network": "network-forward plus network-domain learned refit",
            "learned-chord": "network-forward plus knot warp and chord-domain learned refit",
            "verified-fast": (
                "network-forward plus exact learned check and conditional repair; "
                "no feasible-subset compaction"
            ),
            "verified-compact": (
                "network-forward plus exact verification/repair and greedy compaction"
            ),
        },
        "latency_semantics": (
            "batch latency is request latency; median_ms_per_curve is throughput "
            "amortization and must not be reported as batch=1 latency"
        ),
        "rows": rows,
        "training_lower_bound": {
            "enabled": bool(args.include_training_lower_bound),
            "scope": (
                "model forward + reconstruction-MSE backward + AdamW step; excludes "
                "the full v13 teacher losses, data loading, host-to-device transfer, "
                "validation, and checkpointing, so it is only a lower bound"
            ),
            "rows": training_rows,
        },
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        print(f"Saved JSON report to: {args.json_output}")


if __name__ == "__main__":
    main()
