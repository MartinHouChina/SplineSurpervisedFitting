"""Deploy a v16 checkpoint once on an ordered point sequence and export its fit."""
from __future__ import annotations

# Standalone entry point from an unpacked repository.
# ruff: noqa: E402
import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.checkpointing import (
    V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
    V16_FORMAL_PASS_RATE,
    assess_v16_checkpoint,
    build_model_from_checkpoint,
)
from spline_fitting.data.point_cloud_io import (
    interpolate_parameters_by_chord,
    load_ordered_point_cloud,
    normalize_ordered_point_cloud,
    resample_ordered_point_cloud,
)
from spline_fitting.evaluation.bspline_inference import refit_bspline_control_points
from spline_fitting.evaluation.timing import (
    measure_synchronized_wall_time,
    synchronize_device,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/checkpoints/"
            "candidate_selection_v16_mse1e-4_k56.pt"
        ),
    )
    result.add_argument("--point-cloud", type=Path, required=True,
                        help="Ordered CSV/TXT/XYZ/JSON/NPY/PT coordinates; no unordered-cloud sorting")
    result.add_argument("--num-points", type=int, default=None,
                        help="Network sample count; default from checkpoint, otherwise 192")
    result.add_argument("--mse-tolerance", type=float, default=None,
                        help="Normalized mean squared Euclidean error budget, without square root")
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--output-dir", type=Path, default=Path("outputs/fits/v16"))
    result.add_argument(
        "--allow-unqualified-diagnostic", action="store_true",
        help=("Permit proposal-stage or target-not-met weights for troubleshooting; "
              "the JSON and PNG are marked DIAGNOSTIC NOT FINAL"),
    )
    result.add_argument("--overwrite", action="store_true",
                        help="Allow replacing report.json and fit.png in the output directory")
    result.add_argument("--network-warmups", type=int, default=1)
    result.add_argument("--network-repeats", type=int, default=3,
                        help="Timing-only forwards; they do not perform additional refits")
    return result


def resolve_mse_tolerance(checkpoint: dict, explicit: float | None) -> float:
    value = explicit if explicit is not None else checkpoint["model_config"].get(
        "mse_tolerance", 2.5e-5
    )
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("mse_tolerance must be finite and positive")
    return value


def original_reference_metrics(fit, parameters: torch.Tensor,
                               original_points: torch.Tensor,
                               center: torch.Tensor, scale: torch.Tensor) -> dict:
    """Evaluate original observations using their position on the source polyline.

    Network inputs were resampled uniformly along the ORIGINAL chord grid.
    Remeasuring the shorter, resampled polyline would change this mapping.
    """
    original_points = original_points.double().cpu()
    lengths = original_points.diff(dim=0).norm(dim=-1)
    original_grid = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
    original_grid = original_grid / original_grid[-1]
    parameters = parameters.double().cpu()
    source_grid = torch.linspace(0, 1, parameters.numel(), dtype=torch.float64)
    reference_parameters = interpolate_parameters_by_chord(
        source_grid, parameters, original_grid,
    )
    normalized_reference = (original_points - center) / scale
    predicted = fit.evaluate(reference_parameters)
    mse = float((predicted - normalized_reference).square().sum(-1).mean())
    return {
        "parameters": reference_parameters,
        "normalized_mse": mse,
        "source_units_mse": mse * float(scale) ** 2,
    }


def plot_fit(path: Path, *, source: torch.Tensor, resampled: torch.Tensor,
             fit, center: torch.Tensor, scale: torch.Tensor, report: dict) -> None:
    """Plot the actual refitted curve, controls, knot locations and parameter vector."""
    grid = torch.linspace(0, 1, 800, dtype=torch.float64)
    curve = (fit.evaluate(grid) * scale + center).numpy()
    controls = (fit.control_points * scale + center).numpy()
    knot_points = (fit.evaluate(fit.internal_knots) * scale + center).numpy()
    source_np, sampled_np = source.numpy(), resampled.numpy()
    dim = source.shape[1]
    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    layout = figure.add_gridspec(2, 1, height_ratios=[4.5, 1])
    axis = figure.add_subplot(layout[0], projection="3d" if dim == 3 else None)

    def columns(array):
        return [array[:, index] for index in range(dim)]

    axis.scatter(*columns(source_np), c="0.45", s=12, label="Original ordered points")
    axis.scatter(*columns(sampled_np), c="#2575bc", s=18, marker="+",
                 label="Resampled network points")
    axis.plot(*columns(curve), color="#098777", linewidth=2.0, label="v16 deployed B-spline")
    axis.plot(*columns(controls), "o--", color="#cf8a25", markersize=4,
              linewidth=1, alpha=0.8, label="Control polygon")
    if knot_points.shape[0]:
        axis.scatter(*columns(knot_points), c="#873f9e", marker="D", s=40,
                     label="Internal-knot locations on curve")
    axis.set_xlabel("x (source units)")
    axis.set_ylabel("y (source units)")
    if dim == 3:
        axis.set_zlabel("z (source units)")
        ranges = torch.cat([source, torch.as_tensor(controls)], dim=0).amax(0) - torch.cat(
            [source, torch.as_tensor(controls)], dim=0
        ).amin(0)
        axis.set_box_aspect(ranges.clamp_min(float(ranges.max()) * 0.1).numpy())
    else:
        axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, loc="best")
    axis.set_title(
        f"v16 one-shot deployment | K={report['internal_knot_count']} | "
        f"normalized MSE={report['input_normalized_mse']:.3e}\n"
        f"original-reference MSE={report['original_reference_normalized_mse']:.3e} | "
        f"network median={report['timing']['network_ms']:.2f} ms | "
        f"network + one refit={report['timing']['deployment_ms']:.2f} ms"
    )
    if report.get("diagnostic_not_final"):
        figure.text(
            0.5, 0.5, "DIAGNOSTIC NOT FINAL — UNQUALIFIED CHECKPOINT",
            ha="center", va="center", rotation=24, fontsize=27,
            color="crimson", alpha=0.22, weight="bold", zorder=100,
        )
    knot_axis = figure.add_subplot(layout[1])
    knots = fit.internal_knots.numpy()
    knot_axis.axhline(0, color="0.5", linewidth=1)
    knot_axis.scatter(knots, [0] * len(knots), marker="D", c="#873f9e", s=28)
    knot_axis.scatter([0, 1], [0, 0], marker="|", c="black", s=100)
    knot_axis.text(0, 0.1, f"0 repeated {fit.degree + 1} times", ha="left", fontsize=9)
    knot_axis.text(1, 0.1, f"1 repeated {fit.degree + 1} times", ha="right", fontsize=9)
    knot_axis.set(xlim=(-0.02, 1.02), ylim=(-0.2, 0.3), yticks=[],
                  xlabel="Full open-clamped knot vector: endpoint multiplicities + selected internal knots")
    knot_axis.grid(axis="x", alpha=0.2)
    try:
        figure.savefig(path, dpi=180)
    finally:
        plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    if not args.point_cloud.is_file():
        raise FileNotFoundError(f"point-cloud file does not exist: {args.point_cloud}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")
    report_path, image_path = args.output_dir / "report.json", args.output_dir / "fit.png"
    for target in (report_path, image_path):
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists: {target}; choose another --output-dir or use --overwrite")
    if args.network_warmups < 0 or args.network_repeats < 1:
        raise ValueError("network-warmups must be non-negative and network-repeats positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("objective_version") != V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION:
        raise ValueError("fit_v16_point_cloud.py requires a v16 checkpoint")
    tolerance = resolve_mse_tolerance(checkpoint, args.mse_tolerance)
    qualification = assess_v16_checkpoint(
        checkpoint,
        required_pass_rate=V16_FORMAL_PASS_RATE,
        required_mse_tolerance=tolerance,
    )
    diagnostic = not qualification["formal_reporting_eligible"]
    if diagnostic and not args.allow_unqualified_diagnostic:
        raise ValueError(
            "v16 checkpoint is not eligible for formal deployment: "
            + "; ".join(qualification["reasons"])
            + ". Use --allow-unqualified-diagnostic for marked troubleshooting."
        )
    model, config, _ = build_model_from_checkpoint(checkpoint)
    model = model.eval().to(device)
    count = args.num_points if args.num_points is not None else int(
        checkpoint.get("dataset_config", {}).get("num_points", 192)
    )
    if count < max(4, model.degree + 1):
        raise ValueError("num-points must be at least four and exceed the spline degree")
    source = load_ordered_point_cloud(args.point_cloud, point_dim=config.get("point_dim", 2),
                                      dtype=torch.float64)
    resampled = resample_ordered_point_cloud(source, count)
    normalized = normalize_ordered_point_cloud(resampled)
    center, scale = normalized["center"], normalized["scale"]
    host_points = normalized["points"].float()
    device_points = host_points.unsqueeze(0).to(device)

    def forward():
        with torch.inference_mode():
            return model.forward_deployment(device_points, mse_tolerance=tolerance)

    network_timing = measure_synchronized_wall_time(
        forward, synchronization_device=device, warmup_repeats=args.network_warmups,
        timing_repeats=args.network_repeats,
    )
    # Exactly one deployment forward and one final refit produce the saved result.
    synchronize_device(device)
    started = time.perf_counter_ns()
    with torch.inference_mode():
        output = model.forward_deployment(host_points.unsqueeze(0).to(device),
                                          mse_tolerance=tolerance)
    params = output["params"][0].detach().cpu().double()
    candidates = output["internal_knots"][0].detach().cpu().double()
    mask = output["learned_keep_mask"][0].detach().cpu().bool()
    selected = candidates[mask].sort().values
    fit = refit_bspline_control_points(
        params, host_points.double(), selected, degree=model.degree,
        smoothness_weight=0.0, control_ridge=0.0, interpolate_endpoints=True,
    )
    synchronize_device(device)
    deployment_ms = (time.perf_counter_ns() - started) / 1e6
    if not torch.isfinite(fit.control_points).all() or not torch.isfinite(fit.fit_mse):
        raise RuntimeError("v16 deployment produced a non-finite fit")
    reference = original_reference_metrics(fit, params, source, center, scale)
    mse = float(fit.fit_mse)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "objective_version": checkpoint["objective_version"],
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_stage": checkpoint.get("stage"),
        "checkpoint_quality": checkpoint.get("checkpoint_quality"),
        "checkpoint_qualification": qualification,
        "diagnostic_not_final": diagnostic,
        "point_cloud": str(args.point_cloud.resolve()),
        "original_point_count": int(source.shape[0]), "network_point_count": count,
        "point_dimension": int(source.shape[1]), "degree": model.degree,
        "candidate_count": candidates.numel(), "internal_knot_count": selected.numel(),
        "full_knot_vector_size": int(selected.numel() + 2 * (model.degree + 1)),
        "selection": {
            "policy": getattr(model, "one_shot_selection_policy", "threshold"),
            "adaptive_beta": float(output["adaptive_keep_threshold"][0].detach().cpu()),
            "probability_mass": float(output["one_shot_probability_mass"][0].detach().cpu()),
            "uncertainty": float(output["one_shot_selection_uncertainty"][0].detach().cpu()),
            "requested_count_score": float(output["one_shot_requested_count_score"][0].detach().cpu()),
            "safety_sigma": float(getattr(model, "one_shot_safety_sigma", 0.0)),
            "safety_knots": int(getattr(model, "one_shot_safety_knots", 0)),
            "coverage_bins": int(getattr(model, "one_shot_coverage_bins", 0)),
        },
        "mse_definition": "mean(sum((prediction - observation)^2, coordinates)); no square root",
        "mse_tolerance_normalized": tolerance,
        "mse_tolerance_source_units": tolerance * float(scale) ** 2,
        "tolerance_role": "learned network condition; measured feasibility is not guaranteed",
        "input_normalized_mse": mse, "input_source_units_mse": mse * float(scale) ** 2,
        "input_threshold_satisfied": mse <= tolerance,
        "original_reference_normalized_mse": reference["normalized_mse"],
        "original_reference_source_units_mse": reference["source_units_mse"],
        "original_reference_threshold_satisfied": reference["normalized_mse"] <= tolerance,
        "reference_parameter_mapping": "predicted parameters interpolated from uniform original-chord grid",
        "reference_role": "original observations of the same input, not an independent test curve",
        "normalization": {"center": center.tolist(), "scale": float(scale)},
        "network_input_parameters": params.tolist(),
        "original_reference_parameters": reference["parameters"].tolist(),
        "candidate_keep_mask": mask.tolist(),
        "internal_knots": selected.tolist(), "full_knot_vector": fit.knot_vector.tolist(),
        "control_points_source_units": (fit.control_points * scale + center).tolist(),
        "control_points_normalized": fit.control_points.tolist(),
        "final_refits": 1, "postprocessing_search": False,
        "timing": {
            "network_ms": network_timing.latency.p50_ms,
            "network_statistics": network_timing.latency.as_dict(),
            "network_scope": "synchronized forward with input already on the device",
            "deployment_ms": deployment_ms,
            "deployment_scope": "one normalized CPU input -> transfer -> network -> transfer -> CPU float64 refit",
            "excluded": "file loading, resampling, normalization, model loading, metrics and plotting",
            "timing_only_network_warmups": args.network_warmups,
            "timing_only_network_repeats": args.network_repeats,
            "total_network_forwards": args.network_warmups + args.network_repeats + 1,
            "total_refits_including_timing": 1,
        },
        "hardware": {"device": str(device), "platform": platform.platform(),
                     "processor": platform.processor(), "torch_version": str(torch.__version__),
                     "torch_threads": torch.get_num_threads(),
                     "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "figure": str(image_path.resolve()),
    }
    # Validate JSON finiteness before creating any artifacts.
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_fit(image_path, source=source, resampled=resampled, fit=fit,
             center=center, scale=scale, report=report)
    report_path.write_text(serialized + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> dict:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        report = run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        argument_parser.error(str(error))
    if report["diagnostic_not_final"]:
        print("DIAGNOSTIC NOT FINAL — UNQUALIFIED CHECKPOINT")
    print(f"v16 K={report['internal_knot_count']} | normalized MSE={report['input_normalized_mse']:.6e} | "
          f"pass={report['input_threshold_satisfied']}")
    print(f"Original-reference MSE={report['original_reference_normalized_mse']:.6e} | "
          f"pass={report['original_reference_threshold_satisfied']}")
    print(f"Network median={report['timing']['network_ms']:.3f} ms | "
          f"network + one refit={report['timing']['deployment_ms']:.3f} ms")
    print(f"Saved {args.output_dir / 'report.json'} and {args.output_dir / 'fit.png'}")
    return report


if __name__ == "__main__":
    main()
