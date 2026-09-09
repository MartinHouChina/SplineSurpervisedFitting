"""Visualize v16 deployments and independent baselines on held-out curves.

Each PNG uses the same normalized ordered observations for all four methods.
The plotted curve is the final endpoint-constrained, unregularized standard
B-spline fit.  Original reference observations are evaluation-only and never
participate in a refit.  ``--ours-only`` produces paper-ready single-method
figures with indexed control vertices and an explicit internal-knot strip.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_v15_datasets import (  # noqa: E402
    LABELS,
    measure_ours,
    prepare_cases,
    reference_mse,
    resolve_comparison_capacities,
)
from spline_fitting.checkpointing import (  # noqa: E402
    V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
    V16_FORMAL_PASS_RATE,
    assess_v16_checkpoint,
    build_model_from_checkpoint,
)
from spline_fitting.evaluation.published_baselines import (  # noqa: E402
    run_published_baseline,
)


# This per-curve diagnostic deliberately remains a legible 2x2 view.  The
# dataset benchmark and its three-metric figure include every published method.
METHODS = (
    "ours",
    "kang_sparse_2015_adaptation",
    "yeh_feature_cdf_2020",
    "uniform_gradient_pruning",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--checkpoint", type=Path,
        default=Path(
            "outputs/checkpoints/"
            "candidate_selection_v16_simplified_certified_k96.pt"
        ),
    )
    result.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "outputs/figures/v16_simplified_certified_k96/four_method_cases"
        ),
    )
    result.add_argument(
        "--manifest", action="append", default=[], metavar="NAME=PATH",
        help=(
            "Prepared real-data manifest; repeat for several datasets. "
            "Defaults to UJI, Natural Earth and USGS."
        ),
    )
    result.add_argument("--real-samples-per-dataset", type=int, default=2)
    result.add_argument("--selection-seed", type=int, default=20260909)
    result.add_argument("--mse-tolerance", type=float, default=2.5e-5)
    result.add_argument("--max-internal-knots", type=int, default=None)
    result.add_argument("--gradient-steps", type=int, default=12)
    result.add_argument("--paper-initial-knots", type=int, default=None)
    result.add_argument("--paper-admm-iterations", type=int, default=400)
    result.add_argument("--paper-lambda-bisections", type=int, default=8)
    result.add_argument("--paper-relocation-iterations", type=int, default=8)
    result.add_argument("--network-warmups", type=int, default=1)
    result.add_argument("--network-repeats", type=int, default=3)
    result.add_argument("--end-to-end-repeats", type=int, default=1)
    result.add_argument("--torch-num-threads", type=int, default=4)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--dpi", type=int, default=180)
    result.add_argument(
        "--ours-only", action="store_true",
        help=(
            "Draw one detailed Ours-only PNG per case instead of the compact "
            "four-method diagnostic; the JSON still records every plotted value."
        ),
    )
    result.add_argument(
        "--allow-unqualified-diagnostic", action="store_true",
        help=(
            "Permit a proposal-stage or target-not-met joint checkpoint for troubleshooting only. "
            "Every PNG and report is marked DIAGNOSTIC NOT FINAL."
        ),
    )
    result.add_argument(
        "--allow-proposal-diagnostic", action="store_true",
        dest="allow_unqualified_diagnostic", help=argparse.SUPPRESS,
    )
    result.add_argument(
        "--overwrite", action="store_true",
        help="Allow replacement of this script's report and sample PNG files.",
    )
    return result


def validate_checkpoint_for_visualization(
    checkpoint: dict, *, allow_unqualified_diagnostic: bool,
    required_mse_tolerance: float | None = None,
) -> bool:
    """Return whether results require an unqualified-checkpoint watermark."""
    objective = checkpoint.get("objective_version")
    if objective != V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION:
        raise ValueError(
            "visualize_v16_real_deployments.py requires objective_version="
            f"{V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION!r}; got {objective!r}"
        )
    qualification = assess_v16_checkpoint(
        checkpoint,
        required_pass_rate=V16_FORMAL_PASS_RATE,
        required_mse_tolerance=required_mse_tolerance,
    )
    qualified = qualification["formal_reporting_eligible"]
    if not qualified and not allow_unqualified_diagnostic:
        raise ValueError(
            "v16 checkpoint is not eligible for formal reporting: "
            + "; ".join(qualification["reasons"])
            + "; use a qualified final checkpoint, or pass "
            "--allow-unqualified-diagnostic for visibly watermarked troubleshooting"
        )
    return not qualified


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned[:100] or "curve"


def _serializable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _method_label(method: str) -> str:
    if method == "ours":
        return "Ours v16"
    return LABELS[method]


PLOT_LABELS = {
    "ours": "Ours v16",
    "kang_sparse_2015_adaptation": "Kang 2015 adaptation",
    "yeh_feature_cdf_2020": "Yeh 2020 adaptation",
    "uniform_gradient_pruning": "Uniform Kmax greedy + relocation",
}
PLOT_COLORS = {
    "ours": "#128a73",
    "kang_sparse_2015_adaptation": "#7a55a3",
    "yeh_feature_cdf_2020": "#3976b7",
    "uniform_gradient_pruning": "#cb4f4b",
}


def _run_method(method, *, model, points, case, device, args, objective_version):
    if method == "ours":
        fit, parameters, total_ms, network_ms, diagnostics = measure_ours(
            model,
            points,
            device,
            args,
            objective_version=objective_version,
        )
    else:
        result = run_published_baseline(
            method,
            points.double().cpu(),
            mse_tolerance=args.mse_tolerance,
            max_internal_knots=args.max_internal_knots,
            degree=model.degree,
            gradient_steps=args.gradient_steps,
            paper_initial_knots=args.paper_initial_knots,
            paper_admm_iterations=args.paper_admm_iterations,
            paper_lambda_bisections=args.paper_lambda_bisections,
            paper_relocation_iterations=args.paper_relocation_iterations,
        )
        fit, parameters = result.fit, result.parameters
        total_ms, network_ms, diagnostics = (
            result.elapsed_ms,
            None,
            result.diagnostics,
        )
    mse = float(fit.fit_mse)
    dense_reference_mse = reference_mse(fit, parameters, case)
    if not math.isfinite(mse) or (
        dense_reference_mse is not None and not math.isfinite(dense_reference_mse)
    ):
        raise RuntimeError("method produced a non-finite fit metric")
    return {
        "method": method,
        "label": _method_label(method),
        "fit": fit,
        "parameters": parameters,
        "mse": mse,
        "fit_pass": mse <= args.mse_tolerance,
        "reference_mse": dense_reference_mse,
        "reference_pass": (
            dense_reference_mse <= args.mse_tolerance
            if dense_reference_mse is not None else None
        ),
        "final_k": int(fit.internal_knots.numel()),
        "total_ms": float(total_ms),
        "network_ms": float(network_ms) if network_ms is not None else None,
        "diagnostics": diagnostics,
    }


def _plot_geometry(
    axis, *, case: dict, result: dict, color: str, annotate_indices: bool = False,
    ours_label: bool = False,
) -> None:
    reference = case.get("reference")
    if reference is not None:
        reference = reference.detach().cpu().double()
    samples = case["points"].detach().cpu().double()
    dimension = int(samples.shape[-1])

    def cols(values):
        return [values[:, index].numpy() for index in range(dimension)]

    if reference is not None:
        axis.plot(
            *cols(reference), color="0.72", linewidth=1.1,
            label="Original reference polyline (evaluation only)", zorder=1,
        )
    axis.scatter(*cols(samples), color="#2677b8", s=8, alpha=0.7,
                 label="Input sampled points", zorder=2)
    axis.set_xlabel("normalized x")
    axis.set_ylabel("normalized y")
    if dimension == 3:
        axis.set_zlabel("normalized z")
    else:
        axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.2)
    if result.get("status") != "ok":
        return
    fit = result["fit"]
    grid = torch.linspace(0, 1, 800, dtype=torch.float64)
    curve = fit.evaluate(grid).detach().cpu()
    controls = fit.control_points.detach().cpu()
    knots = fit.internal_knots.detach().cpu()
    knot_points = fit.evaluate(knots).detach().cpu() if knots.numel() else None
    axis.plot(*cols(curve), color=color, linewidth=2.2,
              label="Ours deployed B-spline" if ours_label else "Final B-spline",
              zorder=4)
    axis.plot(*cols(controls), "--", color="#d18c27", linewidth=0.9,
              alpha=0.85, label="Control polygon", zorder=3)
    axis.scatter(*cols(controls), color="#d18c27", marker="s", s=22,
                 edgecolors="white", linewidths=0.4,
                 label="Control vertices $P_i$", zorder=4)
    if knot_points is not None:
        axis.scatter(*cols(knot_points), color="#813f96", marker="D", s=24,
                     edgecolors="white", linewidths=0.4,
                     label="Internal knots $C(u_i)$", zorder=5)
    if not annotate_indices:
        return

    # Keep high-capacity diagnostic cases readable. Every value is serialized
    # in the JSON report even when only a sparse set is labelled on the curve.
    control_stride = max(1, math.ceil(len(controls) / 24))
    knot_stride = max(1, math.ceil(len(knots) / 18))
    for index, point in enumerate(controls):
        if index % control_stride and index != len(controls) - 1:
            continue
        if dimension == 3:
            axis.text(*point.tolist(), f"P{index}", color="#8b5a16", fontsize=7)
        else:
            axis.annotate(
                f"P{index}", xy=tuple(point.tolist()), xytext=(3, 4),
                textcoords="offset points", color="#8b5a16", fontsize=7,
            )
    if knot_points is not None:
        for index, point in enumerate(knot_points, 1):
            if (index - 1) % knot_stride and index != len(knot_points):
                continue
            if dimension == 3:
                axis.text(*point.tolist(), f"u{index}", color="#6f2b83", fontsize=7)
            else:
                axis.annotate(
                    f"u{index}", xy=tuple(point.tolist()), xytext=(3, -10),
                    textcoords="offset points", color="#6f2b83", fontsize=7,
                )


def _plot_knot_strip(axis, fit) -> None:
    """Draw the deployed internal knots and their parameter values."""
    knots = fit.internal_knots.detach().cpu().double()
    count = int(knots.numel())
    axis.axhline(0.62, color="0.48", linewidth=1.0, zorder=1)
    axis.scatter(
        [0.0, 1.0], [0.62, 0.62], marker="|", color="black", s=100,
        label=f"clamped endpoints (multiplicity {fit.degree + 1})", zorder=3,
    )
    if count:
        values = knots.numpy()
        axis.scatter(values, [0.62] * count, marker="D", color="#813f96",
                     s=30, edgecolors="white", linewidths=0.4, zorder=4)
        label_stride = max(1, math.ceil(count / 18))
        for index, value in enumerate(values, 1):
            if (index - 1) % label_stride and index != count:
                continue
            level = 0.78 if index % 2 else 0.40
            axis.annotate(
                f"$u_{{{index}}}$", xy=(value, 0.62), xytext=(value, level),
                ha="center", va="center", fontsize=7, color="#6f2b83",
                arrowprops={"arrowstyle": "-", "color": "#9b79a6", "lw": 0.5},
            )
        if count <= 32:
            entries = [
                f"u{index}={value:.4f}" for index, value in enumerate(values, 1)
            ]
            value_lines = [
                "   ".join(entries[start:start + 8])
                for start in range(0, len(entries), 8)
            ]
            axis.text(
                0.5, 0.04, "\n".join(value_lines), transform=axis.transAxes,
                ha="center", va="bottom", fontsize=7.2, family="monospace",
                color="#4f2a59",
            )
        else:
            axis.text(
                0.5, 0.06,
                f"K={count}; complete u_i values are stored in the JSON report",
                transform=axis.transAxes, ha="center", va="bottom", fontsize=8,
                color="#4f2a59",
            )
    else:
        axis.text(
            0.5, 0.08, "No internal knots (cubic Bezier fit)",
            transform=axis.transAxes, ha="center", fontsize=8, color="#4f2a59",
        )
    endpoint_multiplicity = fit.degree + 1
    axis.text(
        0.0, 0.88, f"0 x {endpoint_multiplicity}",
        transform=axis.transAxes, ha="left", fontsize=8,
    )
    axis.text(
        1.0, 0.88, f"1 x {endpoint_multiplicity}",
        transform=axis.transAxes, ha="right", fontsize=8,
    )
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([])
    axis.set_xlabel("Open-clamped parameter domain $u \\in [0,1]$")
    axis.set_title("Deployed internal-knot vector", fontsize=10)
    axis.grid(axis="x", alpha=0.2)


def plot_ours_case(
    path: Path, *, case: dict, result: dict, tolerance: float,
    diagnostic: bool, dpi: int,
) -> None:
    """Render one paper-ready Ours deployment with explicit spline structure."""
    if result.get("status") != "ok":
        raise ValueError("an Ours-only case cannot be plotted from a failed result")
    dimension = int(case["points"].shape[-1])
    figure = plt.figure(figsize=(11.5, 9.0), constrained_layout=True)
    layout = figure.add_gridspec(2, 1, height_ratios=(5.0, 1.45))
    geometry = figure.add_subplot(
        layout[0], projection="3d" if dimension == 3 else None,
    )
    _plot_geometry(
        geometry, case=case, result=result, color=PLOT_COLORS["ours"],
        annotate_indices=True, ours_label=True,
    )
    geometry.legend(loc="best", fontsize=8, ncol=2)
    reference_text = (
        f" | reference-point MSE={result['reference_mse']:.3e}"
        if result["reference_mse"] is not None else ""
    )
    geometry.set_title(
        f"Ours v16 | K={result['final_k']} | sampled-point MSE={result['mse']:.3e}"
        f"{reference_text}\n"
        f"network={result['network_ms']:.2f} ms | "
        f"network + one final refit={result['total_ms']:.2f} ms | "
        f"threshold={tolerance:.2e} ({'PASS' if result['fit_pass'] else 'FAIL'})",
        fontsize=10,
    )
    _plot_knot_strip(figure.add_subplot(layout[1]), result["fit"])
    figure.suptitle(
        f"Held-out Ours deployment: {case['dataset']} / {case['sample_id']}",
        fontsize=14,
    )
    if diagnostic:
        figure.text(
            0.5, 0.5, "DIAGNOSTIC NOT FINAL - UNQUALIFIED CHECKPOINT",
            ha="center", va="center", rotation=24, fontsize=30,
            color="crimson", alpha=0.22, weight="bold", zorder=100,
        )
    try:
        figure.savefig(path, dpi=dpi)
    finally:
        plt.close(figure)


def plot_ours_overview(
    path: Path, *, items: list[tuple[dict, dict]], tolerance: float,
    diagnostic: bool, dpi: int,
) -> None:
    """Render a compact multi-case overview while retaining spline structure."""
    if not items:
        raise ValueError("the Ours overview requires at least one case")
    columns = min(2, len(items))
    rows = math.ceil(len(items) / columns)
    figure = plt.figure(figsize=(6.2 * columns, 4.4 * rows + 0.8))
    legend_handles = legend_labels = None
    for index, (case, result) in enumerate(items, 1):
        if result.get("status") != "ok":
            raise ValueError("the Ours overview cannot include a failed result")
        dimension = int(case["points"].shape[-1])
        axis = figure.add_subplot(
            rows, columns, index, projection="3d" if dimension == 3 else None,
        )
        _plot_geometry(
            axis, case=case, result=result, color=PLOT_COLORS["ours"],
            ours_label=True,
        )
        status = "PASS" if result["fit_pass"] else "FAIL"
        axis.set_title(
            f"{case['dataset']} / {case['sample_id']}\n"
            f"K={result['final_k']} | MSE={result['mse']:.2e} | {status}",
            fontsize=9,
        )
        if legend_handles is None:
            legend_handles, legend_labels = axis.get_legend_handles_labels()
    for index in range(len(items) + 1, rows * columns + 1):
        blank = figure.add_subplot(rows, columns, index)
        blank.set_axis_off()
    figure.suptitle(
        f"Ours v16 held-out deployments | MSE threshold={tolerance:.2e}",
        fontsize=14,
    )
    figure.legend(
        legend_handles, legend_labels, loc="lower center", ncol=3,
        fontsize=8, frameon=True,
    )
    figure.tight_layout(rect=(0.0, 0.075, 1.0, 0.96))
    if diagnostic:
        figure.text(
            0.5, 0.5, "DIAGNOSTIC NOT FINAL - UNQUALIFIED CHECKPOINT",
            ha="center", va="center", rotation=24, fontsize=32,
            color="crimson", alpha=0.22, weight="bold", zorder=100,
        )
    try:
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)
def plot_case(path: Path, *, case: dict, results: list[dict],
              tolerance: float, diagnostic: bool, dpi: int) -> None:
    dimension = int(case["points"].shape[-1])
    figure = plt.figure(figsize=(14, 11), constrained_layout=True)
    for index, result in enumerate(results, 1):
        axis = figure.add_subplot(
            2, 2, index, projection="3d" if dimension == 3 else None
        )
        color = PLOT_COLORS[result["method"]]
        _plot_geometry(axis, case=case, result=result, color=color)
        if result.get("status") != "ok":
            axis.set_title(
                f"{PLOT_LABELS[result['method']]} — FAILED\n{result['error']}",
                fontsize=9,
            )
            axis.legend(loc="best", fontsize=7)
            continue
        ref_text = (
            f" | ref={result['reference_mse']:.2e}"
            if result["reference_mse"] is not None else ""
        )
        timing = (
            f"net={result['network_ms']:.2f} ms | total={result['total_ms']:.2f} ms"
            if result["network_ms"] is not None
            else f"complete={result['total_ms']:.2f} ms"
        )
        axis.set_title(
            f"{PLOT_LABELS[result['method']]}\n"
            f"K={result['final_k']} | MSE={result['mse']:.2e}{ref_text}\n"
            f"{timing}"
            , fontsize=9)
        if index == 1:
            axis.legend(loc="best", fontsize=7)
    title = (
        f"Held-out real curve: {case['dataset']} / {case['sample_id']}\n"
        f"shared MSE tolerance={tolerance:.3e}; endpoint-constrained, "
        "unregularized standard B-spline fits"
    )
    figure.suptitle(title, fontsize=15)
    if diagnostic:
        figure.text(
            0.5, 0.5, "DIAGNOSTIC NOT FINAL — UNQUALIFIED CHECKPOINT",
            ha="center", va="center", rotation=24, fontsize=32,
            color="crimson", alpha=0.22, weight="bold", zorder=100,
        )
    try:
        figure.savefig(path, dpi=dpi)
    finally:
        plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")
    if args.real_samples_per_dataset < 1:
        raise ValueError("real-samples-per-dataset must be positive")
    if not math.isfinite(args.mse_tolerance) or args.mse_tolerance <= 0:
        raise ValueError("mse-tolerance must be finite and positive")
    if args.dpi < 50:
        raise ValueError("dpi must be at least 50")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    report_path = args.output_dir / "deployment_visualizations.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"output report already exists: {report_path}; choose a new directory "
            "or pass --overwrite"
        )
    overview_path = args.output_dir / "ours_cases_overview.png"
    if args.ours_only and overview_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"output overview already exists: {overview_path}; use --overwrite"
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    diagnostic = validate_checkpoint_for_visualization(
        checkpoint,
        allow_unqualified_diagnostic=args.allow_unqualified_diagnostic,
        required_mse_tolerance=args.mse_tolerance,
    )
    qualification = assess_v16_checkpoint(
        checkpoint,
        required_pass_rate=V16_FORMAL_PASS_RATE,
        required_mse_tolerance=args.mse_tolerance,
    )
    model, model_config, _ = build_model_from_checkpoint(checkpoint)
    if model_config.get("structure_mode") != "candidate_pruning_one_shot":
        raise ValueError("checkpoint must expose one-shot candidate pruning")
    capacities = resolve_comparison_capacities(args, checkpoint)
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    model.to(device).eval()

    # Reuse the benchmark's group-balanced held-out test selection and dense
    # reference parameter mapping. Synthetic cases are intentionally disabled.
    args.skip_synthetic = True
    args.skip_real = False
    args.seed = 0
    args.scan_size = 1
    args.min_knot_count = 0
    args.max_knot_count = 0
    args.samples_per_knot_count = 1
    cases, provenance = prepare_cases(args, checkpoint, model_config)
    if not cases:
        raise ValueError("selected manifests contain no held-out test curves")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plotted_methods = ("ours",) if args.ours_only else METHODS
    records = []
    overview_items: list[tuple[dict, dict]] = []
    for index, case in enumerate(cases, 1):
        print(
            f"[{index}/{len(cases)}] {case['dataset']} / {case['sample_id']}",
            flush=True,
        )
        results = []
        for method in plotted_methods:
            try:
                result = _run_method(
                    method,
                    model=model,
                    points=case["points"],
                    case=case,
                    device=device,
                    args=args,
                    objective_version=checkpoint["objective_version"],
                )
                result["status"] = "ok"
                result["error"] = None
            except (RuntimeError, ValueError) as error:
                result = {
                    "method": method,
                    "label": _method_label(method),
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "fit": None,
                    "parameters": None,
                    "mse": None,
                    "fit_pass": False,
                    "reference_mse": None,
                    "reference_pass": False,
                    "final_k": None,
                    "total_ms": None,
                    "network_ms": None,
                    "diagnostics": {},
                }
            results.append(result)
            if result["status"] == "ok":
                print(
                    f"  {result['label']}: MSE={result['mse']:.3e}, "
                    f"K={result['final_k']}, time={result['total_ms']:.2f} ms",
                    flush=True,
                )
            else:
                print(f"  {result['label']}: {result['error']}", flush=True)
        prefix = "ours__" if args.ours_only else ""
        filename = f"{prefix}{_slug(case['dataset'])}__{_slug(case['sample_id'])}.png"
        image_path = args.output_dir / filename
        if image_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"output image already exists: {image_path}; use --overwrite"
            )
        if args.ours_only:
            plot_ours_case(
                image_path, case=case, result=results[0],
                tolerance=args.mse_tolerance, diagnostic=diagnostic, dpi=args.dpi,
            )
            overview_items.append((case, results[0]))
        else:
            plot_case(
                image_path,
                case=case,
                results=results,
                tolerance=args.mse_tolerance,
                diagnostic=diagnostic,
                dpi=args.dpi,
            )
        records.append({
            "dataset": case["dataset"],
            "sample_id": case["sample_id"],
            "group_id": case["group_id"],
            "image": str(image_path.resolve()),
            "network_input_points_normalized": case["points"],
            "original_reference_points_normalized": case["reference"],
            "methods": [({
                key: value for key, value in result.items()
                if key not in ("fit", "parameters")
            } | ({
                "parameters": result["parameters"],
                "internal_knots": result["fit"].internal_knots,
                "internal_knot_curve_points_normalized": (
                    result["fit"].evaluate(result["fit"].internal_knots)
                    if result["fit"].internal_knots.numel()
                    else []
                ),
                "full_knot_vector": result["fit"].knot_vector,
                "control_points_normalized": result["fit"].control_points,
            } if result["status"] == "ok" else {
                "parameters": None,
                "internal_knots": None,
                "internal_knot_curve_points_normalized": None,
                "full_knot_vector": None,
                "control_points_normalized": None,
            })) for result in results],
        })
        partial = {
            "checkpoint": str(args.checkpoint.resolve()),
            "objective_version": checkpoint["objective_version"],
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_stage": checkpoint.get("stage"),
            "checkpoint_quality": checkpoint.get("checkpoint_quality"),
            "checkpoint_qualification": qualification,
            "diagnostic_not_final": diagnostic,
            "mse_tolerance": args.mse_tolerance,
            "mse_definition": (
                "mean(sum((prediction-observation)^2, coordinates)); no square root"
            ),
            "visualization_mode": "ours_only" if args.ours_only else "four_method",
            "overview_image": None,
            "method_order": list(plotted_methods),
            "knot_capacities": capacities,
            "configuration": {
                key: _serializable(value) for key, value in vars(args).items()
            },
            "hardware": {
                "requested_device": args.device,
                "resolved_device": str(device),
                "gpu": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda" else None
                ),
                "processor": platform.processor(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "torch_threads": torch.get_num_threads(),
            },
            "dataset_provenance": provenance,
            "timing_note": (
                "Ours reports device-resident network median and normalized-input "
                "end-to-end median separately; numerical baselines report their "
                "complete algorithm time. Plotting and metrics are excluded."
            ),
            "records": records,
        }
        report_path.write_text(
            json.dumps(_serializable(partial), ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if args.ours_only:
        plot_ours_overview(
            overview_path, items=overview_items, tolerance=args.mse_tolerance,
            diagnostic=diagnostic, dpi=args.dpi,
        )
        partial["overview_image"] = str(overview_path.resolve())
        report_path.write_text(
            json.dumps(_serializable(partial), ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return _serializable(partial)


def main(argv: list[str] | None = None) -> dict:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        report = run(args)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        argument_parser.error(str(error))
    png_count = len(report["records"]) + int(report["overview_image"] is not None)
    print(
        f"Saved {png_count} PNG files and "
        f"{args.output_dir / 'deployment_visualizations.json'}",
        flush=True,
    )
    return report


if __name__ == "__main__":
    main()
