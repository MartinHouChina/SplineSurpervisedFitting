from __future__ import annotations

# Script entry points intentionally add ``src`` to sys.path before importing
# the local package so they also run from an unpacked repository.
# ruff: noqa: E402

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import build_model_from_checkpoint
from spline_fitting.data import RealWorldCurveDataset
from spline_fitting.data.point_cloud_io import (
    interpolate_parameters_by_chord,
    normalize_ordered_point_cloud,
    resample_ordered_point_cloud,
)
from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
from spline_fitting.evaluation.certified_real_world import (
    REFERENCE_CERTIFICATE_SCOPE,
    certify_reference_mse,
)
from spline_fitting.evaluation.knot_diagnostics import (
    warp_internal_knots_to_parameterization,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate pure one-shot or effect-first reference-certified deployment "
            "on prepared UJI, Natural Earth or USGS manifests. No knot ground truth "
            "is manufactured."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Prepared JSONL manifest; repeat to combine real-world sources.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Manifest split to evaluate; repeat as needed (default: test).",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Optional source_dataset filter; repeat as needed.",
    )
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--fit-tolerance", type=float, default=None)
    parser.add_argument("--smoothness-weight", type=float, default=1e-6)
    parser.add_argument("--control-ridge", type=float, default=0.0)
    parser.add_argument(
        "--deployment-mode",
        choices=("learned", "certified"),
        default="learned",
        help=(
            "learned reports the pure one-shot result; certified keeps that result "
            "and adds a slower reference-MSE repair."
        ),
    )
    parser.add_argument(
        "--certified-mse-target",
        type=float,
        default=1e-5,
        help="Normalized mean squared Euclidean reference-point target.",
    )
    parser.add_argument(
        "--certified-max-knots",
        type=int,
        default=64,
        help=(
            "Maximum repaired internal knots; zero permits the rank-safe "
            "M-degree-1 interpolation limit."
        ),
    )
    parser.add_argument("--certified-min-gap", type=float, default=1e-5)
    parser.add_argument("--certified-position-sweeps", type=int, default=1)
    parser.add_argument("--certified-position-grid-size", type=int, default=5)
    parser.add_argument("--certified-position-restarts", type=int, default=2)
    parser.add_argument("--certified-joint-refine-every", type=int, default=4)
    parser.add_argument("--certified-insertion-candidates", type=int, default=24)
    parser.add_argument("--certified-smoothness-weight", type=float, default=0.0)
    parser.add_argument(
        "--certified-interpolation-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow an interpolation-level fallback when its knot count is within "
            "the configured maximum."
        ),
    )
    parser.add_argument(
        "--certified-tolerance-polyline-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before the exact full-polyline fallback, simplify the parameterized "
            "input polyline until its measured reference-point MSE meets the target."
        ),
    )
    parser.add_argument(
        "--certified-tolerance-polyline-compact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Greedily delete redundant vertices from the tolerance polyline.",
    )
    parser.add_argument(
        "--certified-polyline-exact-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Last-resort exact C0 composite-Bezier representation of the supplied "
            "polyline. It may exceed --certified-max-knots and is not simplification."
        ),
    )
    parser.add_argument(
        "--geometry-samples",
        type=int,
        default=400,
        help="Points used for symmetric Chamfer/Hausdorff; zero disables them.",
    )
    parser.add_argument("--timing-warmup", type=int, default=5)
    parser.add_argument("--timing-repeats", type=int, default=10)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("outputs/real_world/learned_deployment.json"),
    )
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(name)


def _resolve_fit_tolerance(
    checkpoint: Mapping[str, Any],
    explicit: float | None,
) -> float:
    if explicit is not None:
        value = explicit
    else:
        deployment = checkpoint.get("deployment_config", {})
        dataset = checkpoint.get("dataset_config", {})
        if isinstance(deployment, Mapping) and "error_tolerance" in deployment:
            value = deployment["error_tolerance"]
        elif isinstance(dataset, Mapping) and "canonical_knot_tolerance" in dataset:
            value = dataset["canonical_knot_tolerance"]
        else:
            value = 5e-3
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("fit tolerance must be finite and positive")
    return result


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def _time_network_forward(
    model: torch.nn.Module,
    points: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, torch.Tensor], float]:
    forward = getattr(model, "forward_deployment", model.forward)
    for _ in range(warmup):
        forward(points)
    _synchronize(device)
    elapsed: list[float] = []
    output: dict[str, torch.Tensor] | None = None
    for _ in range(repeats):
        started = time.perf_counter()
        output = forward(points)
        _synchronize(device)
        elapsed.append(1e3 * (time.perf_counter() - started))
    assert output is not None
    return output, statistics.median(elapsed)


def _learned_mask(output: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = output.get("learned_keep_mask")
    if mask is None:
        probability = output.get("keep_probability")
        if probability is None:
            mask = output.get("knot_mask")
        else:
            mask = probability >= 0.5
    if mask is None:
        raise KeyError("checkpoint output has no learned KeepMask")
    return mask.to(torch.bool)


def _quantile(values: Iterable[float], probability: float) -> float:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        return float("nan")
    return float(torch.quantile(tensor, probability))


def _aggregate(rows: list[dict[str, Any]], threshold_mse: float) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    knot_counts = [int(row["retained_internal_knots"]) for row in rows]
    histogram: dict[str, int] = {}
    for count in knot_counts:
        histogram[str(count)] = histogram.get(str(count), 0) + 1
    result: dict[str, Any] = {
        "sample_count": len(rows),
        "normalized_reference_mse_mean": statistics.fmean(
            float(row["normalized_reference_mse"]) for row in rows
        ),
        "normalized_reference_mse_p95": _quantile(
            (float(row["normalized_reference_mse"]) for row in rows), 0.95
        ),
        "normalized_reference_rms_mean": statistics.fmean(
            float(row["normalized_reference_rms"]) for row in rows
        ),
        "normalized_reference_rms_p95": _quantile(
            (float(row["normalized_reference_rms"]) for row in rows), 0.95
        ),
        "threshold_satisfied_fraction": statistics.fmean(
            float(row["normalized_reference_mse"]) <= threshold_mse for row in rows
        ),
        "retained_internal_knots_mean": statistics.fmean(knot_counts),
        "retained_internal_knots_min": min(knot_counts),
        "retained_internal_knots_max": max(knot_counts),
        "retained_internal_knots_histogram": histogram,
        "original_unit_reference_mse_mean": statistics.fmean(
            float(row["original_unit_reference_mse"]) for row in rows
        ),
        "original_unit_reference_rms_mean": statistics.fmean(
            float(row["original_unit_reference_rms"]) for row in rows
        ),
    }
    if all(row.get("normalized_symmetric_chamfer_mse") is not None for row in rows):
        result.update(
            {
                "normalized_symmetric_chamfer_mse_mean": statistics.fmean(
                    float(row["normalized_symmetric_chamfer_mse"]) for row in rows
                ),
                "normalized_hausdorff_mean": statistics.fmean(
                    float(row["normalized_hausdorff"]) for row in rows
                ),
                "normalized_hausdorff_p95": _quantile(
                    (float(row["normalized_hausdorff"]) for row in rows), 0.95
                ),
            }
        )
    if all("learned_normalized_reference_mse" in row for row in rows):
        status_histogram: dict[str, int] = {}
        for row in rows:
            status = str(row["certified_status"])
            status_histogram[status] = status_histogram.get(status, 0) + 1
        tolerance_rows = [
            row
            for row in rows
            if bool(row["certified_tolerance_polyline_fallback_used"])
        ]
        result.update(
            {
                "learned_normalized_reference_mse_mean": statistics.fmean(
                    float(row["learned_normalized_reference_mse"]) for row in rows
                ),
                "learned_normalized_reference_mse_p95": _quantile(
                    (float(row["learned_normalized_reference_mse"]) for row in rows),
                    0.95,
                ),
                "learned_retained_internal_knots_mean": statistics.fmean(
                    int(row["learned_retained_internal_knots"]) for row in rows
                ),
                "reference_certificate_valid_fraction": statistics.fmean(
                    bool(row["reference_certificate_valid"]) for row in rows
                ),
                "certified_reference_refit_fraction": 1.0,
                "position_or_knot_count_repair_fraction": statistics.fmean(
                    str(row["certified_status"]) != "initial_reference_refit_pass"
                    for row in rows
                ),
                "added_internal_knots_mean": statistics.fmean(
                    int(row["certified_added_knot_count"]) for row in rows
                ),
                "polyline_exact_fallback_fraction": statistics.fmean(
                    bool(row["certified_polyline_exact_fallback_used"])
                    for row in rows
                ),
                "tolerance_polyline_fallback_fraction": statistics.fmean(
                    bool(row["certified_tolerance_polyline_fallback_used"])
                    for row in rows
                ),
                "tolerance_polyline_vertex_count_mean_when_used": (
                    statistics.fmean(
                        int(row["certified_tolerance_polyline_vertex_count"])
                        for row in tolerance_rows
                    )
                    if tolerance_rows
                    else None
                ),
                "tolerance_polyline_internal_knots_mean_when_used": (
                    statistics.fmean(
                        int(row["retained_internal_knots"])
                        for row in tolerance_rows
                    )
                    if tolerance_rows
                    else None
                ),
                "final_control_point_count_mean": statistics.fmean(
                    int(row["certified_final_control_point_count"]) for row in rows
                ),
                "maximum_internal_knot_multiplicity_max": max(
                    int(row["certified_maximum_internal_knot_multiplicity"])
                    for row in rows
                ),
                "certified_postprocess_time_ms_mean": statistics.fmean(
                    float(row["certified_postprocess_time_ms"]) for row in rows
                ),
                "certified_status_histogram": status_histogram,
            }
        )
    return result


def _geometry_distances(
    spline: Any,
    reference_points: torch.Tensor,
    sample_count: int,
) -> tuple[float, float]:
    reference = resample_ordered_point_cloud(reference_points, sample_count).to(
        torch.float64
    )
    parameters = torch.linspace(0.0, 1.0, sample_count, dtype=torch.float64)
    predicted = spline.evaluate(parameters)
    distances = torch.cdist(predicted, reference)
    predicted_to_reference = distances.square().amin(dim=1)
    reference_to_predicted = distances.square().amin(dim=0)
    chamfer_mse = 0.5 * (predicted_to_reference.mean() + reference_to_predicted.mean())
    hausdorff = torch.maximum(
        predicted_to_reference.max(), reference_to_predicted.max()
    ).sqrt()
    return float(chamfer_mse), float(hausdorff)


def _load_datasets(
    args: argparse.Namespace, num_points: int
) -> list[RealWorldCurveDataset]:
    splits = args.split or ["test"]
    datasets = [
        RealWorldCurveDataset(
            manifest.resolve(),
            split=splits,
            sources=args.source,
            num_points=num_points,
            normalize=True,
        )
        for manifest in args.manifest
    ]
    if not any(len(dataset) for dataset in datasets):
        raise ValueError("the selected manifests/splits contain no curves")
    return datasets


def _sample_locations(
    datasets: list[RealWorldCurveDataset],
    maximum: int,
    seed: int,
) -> list[tuple[int, int]]:
    locations = [
        (dataset_index, sample_index)
        for dataset_index, dataset in enumerate(datasets)
        for sample_index in range(len(dataset))
    ]
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(locations), generator=generator).tolist()
    if maximum:
        order = order[:maximum]
    return [locations[index] for index in order]


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max-samples must be non-negative; zero means all")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_points is not None and args.num_points < 4:
        parser.error("--num-points must be at least four")
    if args.smoothness_weight < 0.0 or args.control_ridge < 0.0:
        parser.error("refit regularization weights must be non-negative")
    if not math.isfinite(args.certified_mse_target) or args.certified_mse_target < 0.0:
        parser.error("--certified-mse-target must be finite and non-negative")
    if args.certified_max_knots < 0:
        parser.error("--certified-max-knots must be non-negative")
    if not math.isfinite(args.certified_min_gap) or args.certified_min_gap < 0.0:
        parser.error("--certified-min-gap must be finite and non-negative")
    if args.certified_position_sweeps < 0:
        parser.error("--certified-position-sweeps must be non-negative")
    if (
        args.certified_position_grid_size < 3
        or args.certified_position_grid_size % 2 == 0
    ):
        parser.error("--certified-position-grid-size must be odd and at least three")
    if args.certified_position_restarts < 1:
        parser.error("--certified-position-restarts must be positive")
    if args.certified_joint_refine_every < 1:
        parser.error("--certified-joint-refine-every must be positive")
    if args.certified_insertion_candidates < 1:
        parser.error("--certified-insertion-candidates must be positive")
    if args.certified_smoothness_weight < 0.0:
        parser.error("--certified-smoothness-weight must be non-negative")
    if args.geometry_samples not in {0} and args.geometry_samples < 4:
        parser.error("--geometry-samples must be zero or at least four")
    if args.timing_warmup < 0 or args.timing_repeats <= 0:
        parser.error("timing warmup must be non-negative and repeats positive")
    output_path = args.json_output.resolve()
    csv_path = (
        args.csv_output.resolve()
        if args.csv_output is not None
        else output_path.with_suffix(".csv")
    )
    for path in (output_path, csv_path):
        if path.exists() and not args.overwrite:
            parser.error(f"output already exists (use --overwrite): {path}")

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint root must be a dictionary")
    model, model_config, _ = build_model_from_checkpoint(checkpoint)
    if model_config.get("structure_mode") != "candidate_pruning_one_shot":
        parser.error("real-world learned evaluation requires a one-shot checkpoint")
    fit_tolerance = _resolve_fit_tolerance(checkpoint, args.fit_tolerance)
    checkpoint_dataset = checkpoint.get("dataset_config", {})
    num_points = (
        args.num_points
        if args.num_points is not None
        else int(
            checkpoint_dataset.get("num_points", 192)
            if isinstance(checkpoint_dataset, Mapping)
            else 192
        )
    )
    datasets = _load_datasets(args, num_points)
    locations = _sample_locations(datasets, args.max_samples, args.seed)
    device = _resolve_device(args.device)
    model.to(device).eval()

    rows: list[dict[str, Any]] = []
    network_batch_times: list[float] = []
    for batch_start in range(0, len(locations), args.batch_size):
        batch_locations = locations[batch_start : batch_start + args.batch_size]
        items = [datasets[d][i] for d, i in batch_locations]
        points = torch.stack([item["points"] for item in items]).to(device)
        output, batch_time_ms = _time_network_forward(
            model,
            points,
            device=device,
            warmup=args.timing_warmup if batch_start == 0 else 0,
            repeats=args.timing_repeats,
        )
        network_batch_times.append(batch_time_ms)
        masks = _learned_mask(output).detach().cpu()
        params = output["params"].detach().cpu().to(torch.float64)
        knots = output["internal_knots"].detach().cpu().to(torch.float64)
        proposal_params = (
            output.get("proposal_params", output["params"])
            .detach()
            .cpu()
            .to(torch.float64)
        )
        proposal_knots = (
            output.get("proposal_internal_knots", output["internal_knots"])
            .detach()
            .cpu()
            .to(torch.float64)
        )

        for local_index, ((dataset_index, sample_index), item) in enumerate(
            zip(batch_locations, items, strict=True)
        ):
            dataset = datasets[dataset_index]
            record = dataset.record(sample_index)
            reference_raw = dataset.load_reference_points(sample_index)
            center = item["center"].to(torch.float64)
            scale = item["scale"].to(torch.float64)
            model_points = item["points"].to(torch.float64)
            reference_normalized = (reference_raw.to(torch.float64) - center) / scale
            model_chord = item["chord_params"].to(torch.float64)
            reference_chord = normalize_ordered_point_cloud(
                reference_raw.to(torch.float64)
            )["chord_params"]
            # The model grid was originally accumulated in float32; after a
            # float64 cast its last value can remain a few float32 ulps below
            # one.  Both chord domains are mathematically [0,1], so pin their
            # endpoints before the strict interpolation-domain check.
            model_chord = model_chord.clone()
            reference_chord = reference_chord.clone()
            model_chord[0], model_chord[-1] = 0.0, 1.0
            reference_chord[0], reference_chord[-1] = 0.0, 1.0
            reference_params = interpolate_parameters_by_chord(
                model_chord,
                params[local_index],
                reference_chord,
            )
            retained = knots[local_index, masks[local_index]]
            learned_fit = refit_bspline_control_points(
                params[local_index],
                model_points,
                retained,
                degree=int(model.degree),
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
                interpolate_endpoints=True,
            )
            learned_reconstructed_reference = learned_fit.evaluate(reference_params)
            learned_distances = (
                learned_reconstructed_reference - reference_normalized
            ).norm(dim=-1)
            learned_reference_mse = float(learned_distances.square().mean())
            learned_reference_rms = learned_reference_mse**0.5

            deployed_fit = learned_fit
            deployed_distances = learned_distances
            deployed_knots = retained
            certified_fields: dict[str, Any] = {}
            if args.deployment_mode == "certified":
                repair_started = time.perf_counter()
                selected_chord_knots = warp_internal_knots_to_parameterization(
                    retained,
                    params[local_index],
                    model_chord,
                )
                proposal_parameter_row = proposal_params[local_index].clone()
                proposal_parameter_row[0], proposal_parameter_row[-1] = 0.0, 1.0
                proposal_chord_knots = warp_internal_knots_to_parameterization(
                    proposal_knots[local_index],
                    proposal_parameter_row,
                    model_chord,
                )
                certified = certify_reference_mse(
                    reference_chord,
                    reference_normalized,
                    selected_chord_knots,
                    candidate_knots=proposal_chord_knots,
                    target_mse=args.certified_mse_target,
                    maximum_internal_knots=args.certified_max_knots,
                    degree=int(model.degree),
                    smoothness_weight=args.certified_smoothness_weight,
                    control_ridge=0.0,
                    min_gap=args.certified_min_gap,
                    position_sweeps=args.certified_position_sweeps,
                    position_grid_size=args.certified_position_grid_size,
                    position_restarts=args.certified_position_restarts,
                    joint_refine_every=args.certified_joint_refine_every,
                    insertion_candidates=args.certified_insertion_candidates,
                    interpolation_fallback=args.certified_interpolation_fallback,
                    tolerance_polyline_fallback=(
                        args.certified_tolerance_polyline_fallback
                    ),
                    tolerance_polyline_compact=(
                        args.certified_tolerance_polyline_compact
                    ),
                    polyline_exact_fallback=args.certified_polyline_exact_fallback,
                )
                repair_time_ms = 1e3 * (time.perf_counter() - repair_started)
                deployed_fit = certified.final_fit
                deployed_distances = (
                    deployed_fit.reconstructed_points - reference_normalized
                ).norm(dim=-1)
                deployed_knots = deployed_fit.internal_knots
                certified_fields = {
                    "learned_retained_internal_knots": int(retained.numel()),
                    "learned_internal_knots": retained.tolist(),
                    "learned_normalized_reference_mse": learned_reference_mse,
                    "learned_normalized_reference_rms": learned_reference_rms,
                    "certified_target_mse": args.certified_mse_target,
                    "certified_status": certified.status,
                    "certified_threshold_satisfied": certified.threshold_satisfied,
                    "reference_certificate_valid": certified.certificate_valid,
                    "reference_certificate_scope": certified.certification_scope,
                    "continuous_curve_guaranteed": (
                        certified.continuous_curve_guaranteed
                    ),
                    "globally_minimal_knot_count_guaranteed": (
                        certified.globally_minimal_knot_count_guaranteed
                    ),
                    "certified_failure_reason": certified.failure_reason,
                    "certified_full_column_rank": certified.full_column_rank,
                    "certified_expected_solver_rank": certified.expected_solver_rank,
                    "certified_solver_rank": deployed_fit.solver_rank,
                    "certified_rank_safe_max_knots": (
                        certified.rank_safe_maximum_internal_knots
                    ),
                    "certified_effective_max_knots": certified.maximum_internal_knots,
                    "certified_supplied_initial_count": (
                        certified.supplied_initial_count
                    ),
                    "certified_sanitized_initial_count": (
                        certified.sanitized_initial_count
                    ),
                    "certified_added_knot_count": certified.added_knot_count,
                    "certified_position_refinement_used": (
                        certified.position_refinement_used
                    ),
                    "certified_interpolation_fallback_attempted": (
                        certified.interpolation_fallback_attempted
                    ),
                    "certified_interpolation_fallback_used": (
                        certified.interpolation_fallback_used
                    ),
                    "certified_tolerance_polyline_fallback_attempted": (
                        certified.tolerance_polyline_fallback_attempted
                    ),
                    "certified_tolerance_polyline_fallback_used": (
                        certified.tolerance_polyline_fallback_used
                    ),
                    "certified_tolerance_polyline_vertex_count": (
                        certified.tolerance_polyline_vertex_count
                    ),
                    "certified_tolerance_polyline_measured_mse": (
                        certified.tolerance_polyline_measured_mse
                    ),
                    "certified_tolerance_polyline_insertion_count": (
                        certified.tolerance_polyline_insertion_count
                    ),
                    "certified_tolerance_polyline_compaction_removed_count": (
                        certified.tolerance_polyline_compaction_removed_count
                    ),
                    "certified_tolerance_polyline_exceeded_adaptive_max_knots": (
                        certified.tolerance_polyline_fallback_exceeded_adaptive_maximum
                    ),
                    "certified_polyline_exact_fallback_attempted": (
                        certified.polyline_exact_fallback_attempted
                    ),
                    "certified_polyline_exact_fallback_used": (
                        certified.polyline_exact_fallback_used
                    ),
                    "certified_polyline_exceeded_adaptive_max_knots": (
                        certified.polyline_fallback_exceeded_adaptive_maximum
                    ),
                    "certified_final_control_point_count": (
                        certified.final_control_point_count
                    ),
                    "certified_maximum_internal_knot_multiplicity": (
                        certified.maximum_internal_knot_multiplicity
                    ),
                    "certified_refit_count": certified.refit_count,
                    "certified_insertion_candidate_evaluations": (
                        certified.insertion_candidate_evaluation_count
                    ),
                    "certified_position_refit_count": certified.position_refit_count,
                    "certified_endpoint_max_distance": certified.endpoint_max_distance,
                    "certified_postprocess_time_ms": repair_time_ms,
                }

            reference_mse = float(deployed_distances.square().mean())
            reference_rms = reference_mse**0.5
            point_p95 = float(torch.quantile(deployed_distances, 0.95))
            point_max = float(deployed_distances.max())
            if args.geometry_samples:
                chamfer_mse, hausdorff = _geometry_distances(
                    deployed_fit,
                    reference_normalized,
                    args.geometry_samples,
                )
            else:
                chamfer_mse = None
                hausdorff = None
            scale_value = float(scale)
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "source_dataset": record["source_dataset"],
                    "group_id": record["group_id"],
                    "split": record["split"],
                    "reference_point_count": int(reference_raw.shape[0]),
                    "model_point_count": int(model_points.shape[0]),
                    "deployment_mode": args.deployment_mode,
                    "retained_internal_knots": int(deployed_knots.numel()),
                    "final_internal_knots": deployed_knots.tolist(),
                    "normalized_input_refit_mse": float(learned_fit.fit_mse),
                    "normalized_reference_mse": reference_mse,
                    "normalized_reference_rms": reference_rms,
                    "normalized_point_distance_p95": point_p95,
                    "normalized_point_distance_max": point_max,
                    "normalized_symmetric_chamfer_mse": chamfer_mse,
                    "normalized_hausdorff": hausdorff,
                    "original_unit_reference_mse": reference_mse
                    * scale_value
                    * scale_value,
                    "original_unit_reference_rms": reference_rms * scale_value,
                    "network_batch_time_ms": batch_time_ms,
                    "network_amortized_time_ms": batch_time_ms / len(batch_locations),
                    **certified_fields,
                }
            )
        completed = min(batch_start + len(batch_locations), len(locations))
        print(
            f"[{completed}/{len(locations)}] network={batch_time_ms:.3f} ms/batch",
            flush=True,
        )

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["source_dataset"]), []).append(row)
    threshold_mse = (
        args.certified_mse_target
        if args.deployment_mode == "certified"
        else fit_tolerance * fit_tolerance
    )
    overall = _aggregate(rows, threshold_mse)
    if len(groups) > 1:
        # UJI is measured in millimetres while the geospatial adapters use
        # metres.  A cross-source average in original coordinates has no unit.
        overall.pop("original_unit_reference_mse_mean", None)
        overall.pop("original_unit_reference_rms_mean", None)
        overall["original_unit_metrics"] = "omitted_for_mixed_units"
    report = {
        "schema_version": (
            "real_world_reference_certified_deployment_v3"
            if args.deployment_mode == "certified"
            else "real_world_learned_deployment_v1"
        ),
        "deployment_mode": args.deployment_mode,
        "checkpoint": str(checkpoint_path),
        "objective_version": checkpoint.get("objective_version"),
        "manifests": [str(path.resolve()) for path in args.manifest],
        "splits": args.split or ["test"],
        "source_filter": args.source,
        "sampling_seed": args.seed,
        "model_point_count": num_points,
        "fit_tolerance_normalized_rms": fit_tolerance,
        "active_threshold_normalized_mse": threshold_mse,
        "error_definition": (
            "mean squared Euclidean distance on original-density ordered "
            "reference points; learned controls are fit only to the model input "
            "grid, while certified controls/search use the declared reference points"
            if args.deployment_mode == "certified"
            else "mean squared Euclidean distance on original-density ordered "
            "reference points; control points are fit only to the model input grid"
        ),
        "knot_ground_truth_available": False,
        "certified_config": (
            {
                "target_mse": args.certified_mse_target,
                "maximum_internal_knots": args.certified_max_knots,
                "zero_maximum_means_rank_safe_interpolation_limit": True,
                "minimum_knot_gap": args.certified_min_gap,
                "position_sweeps": args.certified_position_sweeps,
                "position_grid_size": args.certified_position_grid_size,
                "position_restarts": args.certified_position_restarts,
                "joint_refine_every_additions": args.certified_joint_refine_every,
                "insertion_candidates_per_round": (
                    args.certified_insertion_candidates
                ),
                "smoothness_weight": args.certified_smoothness_weight,
                "interpolation_fallback": args.certified_interpolation_fallback,
                "tolerance_polyline_fallback": (
                    args.certified_tolerance_polyline_fallback
                ),
                "tolerance_polyline_compaction": (
                    args.certified_tolerance_polyline_compact
                ),
                "tolerance_polyline_fallback_may_exceed_adaptive_maximum": True,
                "tolerance_polyline_fallback_role": (
                    "measured-MSE simplification of the supplied parameterized "
                    "polyline, represented as a C0 cubic composite"
                ),
                "polyline_exact_fallback": args.certified_polyline_exact_fallback,
                "polyline_exact_fallback_may_exceed_adaptive_maximum": True,
                "polyline_exact_fallback_role": (
                    "last-resort exact representation of the supplied input polyline; "
                    "not a simplified spline"
                ),
                "certificate_scope": REFERENCE_CERTIFICATE_SCOPE,
                "continuous_curve_guaranteed": False,
                "globally_minimal_knot_count_guaranteed": False,
            }
            if args.deployment_mode == "certified"
            else None
        ),
        "timing": {
            "scope": "network forward_deployment only; excludes loading, preprocessing, standard B-spline refit and metrics",
            "device": str(device),
            "batch_size": args.batch_size,
            "warmup": args.timing_warmup,
            "repeats_per_batch": args.timing_repeats,
            "measured_batch_count": len(network_batch_times),
            "network_total_median_batch_time_ms": sum(network_batch_times),
            "network_amortized_time_ms_per_curve": sum(network_batch_times) / len(rows),
            "certified_postprocess_total_time_ms": (
                sum(float(row["certified_postprocess_time_ms"]) for row in rows)
                if args.deployment_mode == "certified"
                else None
            ),
            "certified_postprocess_mean_time_ms_per_curve": (
                statistics.fmean(
                    float(row["certified_postprocess_time_ms"]) for row in rows
                )
                if args.deployment_mode == "certified"
                else None
            ),
            "certified_timing_scope": (
                "CPU float64 reference refit, knot insertion and joint relocation; "
                "reported separately from network-only time"
                if args.deployment_mode == "certified"
                else None
            ),
        },
        "overall": overall,
        "by_source": {
            source: _aggregate(source_rows, threshold_mse)
            for source, source_rows in sorted(groups.items())
        },
        "samples": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    overall = report["overall"]
    timing = report["timing"]
    print(f"Real-world {args.deployment_mode} deployment", flush=True)
    print(f"  samples: {overall['sample_count']}", flush=True)
    print(
        "  normalized reference MSE mean/P95: "
        f"{overall['normalized_reference_mse_mean']:.6e}/"
        f"{overall['normalized_reference_mse_p95']:.6e}",
        flush=True,
    )
    print(
        "  threshold pass / retained K: "
        f"{overall['threshold_satisfied_fraction']:.3f}/"
        f"{overall['retained_internal_knots_mean']:.3f}",
        flush=True,
    )
    print(
        "  network-only amortized time: "
        f"{timing['network_amortized_time_ms_per_curve']:.3f} ms/curve",
        flush=True,
    )
    if args.deployment_mode == "certified":
        print(
            "  reference certificate valid / repair time: "
            f"{overall['reference_certificate_valid_fraction']:.3f}/"
            f"{timing['certified_postprocess_mean_time_ms_per_curve']:.3f} ms/curve",
            flush=True,
        )
    print(f"Saved JSON: {output_path}", flush=True)
    print(f"Saved CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
