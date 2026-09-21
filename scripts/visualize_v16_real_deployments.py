"""Visualize v16 deployments and independent baselines on held-out curves.

Each PNG uses the same normalized ordered observations for all selected methods.
The plotted curve and errors come from each method's returned final fit;
no display-only refit or repair is applied. Original reference observations are evaluation-only and never
participate in a refit.  ``--ours-only`` produces paper-ready single-method
figures with indexed control vertices and an explicit internal-knot strip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import textwrap
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
    SAFEGUARDED_PUBLISHED_METHODS,
    measure_ours,
    measure_numerical_baseline,
    native_baseline_audit,
    published_baseline_protocol,
    prepare_cases,
    fit_error_metrics,
    resolve_comparison_capacities,
    sha256_file,
)
from spline_fitting.checkpointing import (  # noqa: E402
    V16_COUNTERFACTUAL_SUBSET_OBJECTIVE_VERSION,
    V16_FORMAL_PASS_RATE,
    assess_v16_checkpoint,
    build_model_from_checkpoint,
)
from plot_v16_method_comparison import PUBLISHED_METHODS  # noqa: E402


# Keep the legacy four-method default; --method-set published selects the six
# methods used by the overnight benchmark and its four-metric figure.
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
            "Defaults to UJI, Natural Earth, USGS and procedural IndustrialOffset (not measured)."
        ),
    )
    result.add_argument("--data-root", type=Path,
                        help="Complete external data tree for the four default sources; missing sources fail explicitly")
    result.add_argument("--real-samples-per-dataset", type=int, default=2)
    result.add_argument("--selection-seed", type=int, default=20260909)
    result.add_argument("--mse-tolerance", type=float, default=2.5e-5)
    result.add_argument("--max-internal-knots", type=int, default=None)
    result.add_argument(
        "--allow-unequal-capacity", action="store_true",
        help="Allow unequal network/numerical knot budgets only as a visibly marked diagnostic ablation; irrelevant to --ours-only",
    )
    result.add_argument("--gradient-steps", type=int, default=12)
    result.add_argument("--paper-initial-knots", type=int, default=None)
    result.add_argument("--paper-admm-iterations", type=int, default=400)
    result.add_argument("--paper-lambda-bisections", type=int, default=8)
    result.add_argument("--paper-relocation-iterations", type=int, default=8)
    result.add_argument("--method-set", choices=("legacy", "published"), default="legacy")
    result.add_argument(
        "--published-feasibility-safeguard", action=argparse.BooleanOptionalAction,
        default=False,
        help="Historical opt-in threshold-safe Dung/Kang/Luo repairs; disabled by default and every repair refit is timed",
    )
    result.add_argument("--force-diagnostic", action="store_true",
                        help="Mark reduced-budget case studies as diagnostic")
    for option, kind, default in (
        ("park-shape-weight", float, 0.8), ("liang-dense-knots", int, None),
        ("liang-initial-knots", int, 4), ("liang-curvature-weight", float, 0.5),
        ("liang-feature-samples", int, 1025), ("dung-max-error", float, None),
        ("dung-scan-intervals", int, 10), ("dung-optimization-iterations", int, 10),
        ("luo-eta", float, 0.5), ("luo-de-population", int, 10),
        ("luo-de-iterations", int, 50), ("luo-seed", int, 2022),
    ):
        result.add_argument("--" + option, type=kind, default=default)
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
            "multi-method diagnostic; the JSON still records every plotted value."
        ),
    )
    result.add_argument(
        "--allow-unqualified-diagnostic", action="store_true",
        help=(
            "Permit a proposal-stage or target-not-met joint checkpoint for troubleshooting only. "
            "Qualification remains recorded in the report; PNGs have no watermark."
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
    """Return whether the report must record unqualified-checkpoint status."""
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
            "--allow-unqualified-diagnostic for troubleshooting recorded in the report"
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


def _method_label(method: str, *, safeguard_enabled: bool = False) -> str:
    if method == "ours":
        return "Ours v16"
    suffix = " [threshold-safe adaptation]" if safeguard_enabled and method in SAFEGUARDED_PUBLISHED_METHODS else ""
    return LABELS[method] + suffix


PLOT_LABELS = {
    "ours": "Ours v16",
    "kang_sparse_2015_adaptation": "Kang 2015 adaptation",
    "yeh_feature_cdf_2020": "Yeh 2020 adaptation",
    "uniform_gradient_pruning": "Uniform Kmax greedy + relocation",
    "park_dominant_point_2007_adaptation": "Park & Lee 2007 adaptation",
    "liang_feature_iki_2017_adaptation": "Liang et al. 2017 adaptation",
    "dung_direct_knot_2017_adaptation": "Dung & Tjahjowidodo 2017 adaptation",
    "luo_linf_de_2022_adaptation": "Luo et al. 2022 adaptation",
}
PLOT_COLORS = {
    "ours": "#128a73",
    "kang_sparse_2015_adaptation": "#7a55a3",
    "yeh_feature_cdf_2020": "#3976b7",
    "uniform_gradient_pruning": "#cb4f4b",
    "park_dominant_point_2007_adaptation": "#d89423",
    "liang_feature_iki_2017_adaptation": "#536eb5",
    "dung_direct_knot_2017_adaptation": "#a45d91",
    "luo_linf_de_2022_adaptation": "#8b6b4f",
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
        fit, parameters, total_ms, network_ms, diagnostics = measure_numerical_baseline(
            method, points.double().cpu(), args, degree=model.degree,
        )
    errors = fit_error_metrics(fit, parameters, case)
    mse = errors["mse"]
    dense_reference_mse = errors["reference_mse"]
    if any(value is not None and not math.isfinite(value) for value in errors.values()):
        raise RuntimeError("method produced a non-finite fit metric")
    return {
        "method": method,
        "label": _method_label(method, safeguard_enabled=getattr(args, "published_feasibility_safeguard", False)),
        "published_feasibility_safeguard": getattr(args, "published_feasibility_safeguard", False),
        "fit": fit,
        "parameters": parameters,
        **errors,
        "fit_pass": mse <= args.mse_tolerance,
        "reference_pass": (
            dense_reference_mse <= args.mse_tolerance
            if dense_reference_mse is not None else None
        ),
        "final_k": int(fit.internal_knots.numel()),
        "total_ms": float(total_ms),
        "network_ms": float(network_ms) if network_ms is not None else None,
        "diagnostics": diagnostics,
    }


def _peak_error_text(result: dict, *, reference: bool = False) -> str:
    """Never reconstruct a missing pointwise maximum from a curve mean."""
    key = "reference_max_squared_error" if reference else "max_squared_error"
    value = result.get(key)
    label = "ref max SE" if reference else "max SE"
    return f"{label}={float(value):.2e}" if value is not None else f"{label}=N/A"


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
        plot_case(path, case=case, results=[result], tolerance=tolerance,
                  diagnostic=diagnostic, dpi=dpi)
        return
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
        f"{_peak_error_text(result)} | {_peak_error_text(result, reference=True)} "
        "(maximum point squared error)\n"
        f"network={result['network_ms']:.2f} ms | "
        f"network + one final refit={result['total_ms']:.2f} ms | "
        f"threshold={tolerance:.2e} ({'PASS' if result['fit_pass'] else 'FAIL'})",
        fontsize=10,
    )
    _plot_knot_strip(figure.add_subplot(layout[1]), result["fit"])
    figure.suptitle(
        f"Held-out Ours deployment: {case.get('dataset_label', case['dataset'])} / {case['sample_id']}",
        fontsize=14,
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
        dimension = int(case["points"].shape[-1])
        axis = figure.add_subplot(
            rows, columns, index, projection="3d" if dimension == 3 else None,
        )
        _plot_geometry(
            axis, case=case, result=result, color=PLOT_COLORS["ours"],
            ours_label=True,
        )
        status = "PASS" if result.get("fit_pass") else "FAIL"
        detail = (f"K={result['final_k']} | MSE={result['mse']:.2e} | {status}\n"
                  f"{_peak_error_text(result)} (point squared error)"
                  if result.get("status") == "ok" else f"FAILED: {result.get('error', 'no fit')}")
        axis.set_title(
            f"{case.get('dataset_label', case['dataset'])} / {case['sample_id']}\n{detail}",
            fontsize=9,
        )
        handles, labels = axis.get_legend_handles_labels()
        if legend_handles is None or len(handles) > len(legend_handles):
            legend_handles, legend_labels = handles, labels
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
    try:
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)
def plot_case(path: Path, *, case: dict, results: list[dict],
              tolerance: float, diagnostic: bool, dpi: int,
              capacity_note: str | None = None) -> None:
    dimension = int(case["points"].shape[-1])
    columns = 3 if len(results) > 4 else min(2, len(results))
    rows = math.ceil(len(results) / columns)
    figure = plt.figure(figsize=(6.4 * columns, 5.1 * rows + 0.5), constrained_layout=True)
    has_fit_legend = False
    for index, result in enumerate(results, 1):
        axis = figure.add_subplot(
            rows, columns, index, projection="3d" if dimension == 3 else None
        )
        color = PLOT_COLORS[result["method"]]
        label = PLOT_LABELS[result["method"]]
        if result.get("published_feasibility_safeguard") and result["method"] in SAFEGUARDED_PUBLISHED_METHODS:
            label += " [threshold-safe adaptation]"
        _plot_geometry(axis, case=case, result=result, color=color)
        if result.get("status") != "ok":
            axis.set_title(
                f"{label} — FAILED\n{result['error']}",
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
        native = result.get("diagnostics", {})
        native_mse = native.get("comparison_feasibility_native_mse")
        native_text = (
            f"\nnative K={native['comparison_feasibility_native_k']} | MSE={native_mse:.2e} | "
            f"extra refits={native['comparison_feasibility_refit_count']}"
            if native_mse is not None else ""
        )
        axis.set_title(
            f"{label}\n"
            f"K={result['final_k']} | MSE={result['mse']:.2e}{ref_text}\n"
            f"{_peak_error_text(result)} | {_peak_error_text(result, reference=True)}\n"
            f"{timing}{native_text}"
            , fontsize=9)
        if not has_fit_legend:
            axis.legend(loc="best", fontsize=7)
            has_fit_legend = True
    title = (
        f"Held-out external curve: {case.get('dataset_label', case['dataset'])} / {case['sample_id']}\n"
        f"shared MSE tolerance={tolerance:.3e}; curves and errors from each method's returned fit\n"
        "max SE = maximum observed-point squared Euclidean error (no square root; not the MSE pass criterion)"
    )
    if any(result.get("published_feasibility_safeguard") and result["method"] in SAFEGUARDED_PUBLISHED_METHODS for result in results):
        title += "\nThreshold-safe adaptations: repair refits included in complete time; not paper-original algorithms."
    if capacity_note:
        title += "\n" + capacity_note
    title = "\n".join(textwrap.fill(line, width=56 * columns) for line in title.splitlines())
    figure.suptitle(title, fontsize=15)
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
    if min(args.network_repeats, args.end_to_end_repeats, args.torch_num_threads) < 1:
        raise ValueError("timing repeats and torch-num-threads must be positive")
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
    diagnostic = diagnostic or args.force_diagnostic
    qualification = assess_v16_checkpoint(
        checkpoint,
        required_pass_rate=V16_FORMAL_PASS_RATE,
        required_mse_tolerance=args.mse_tolerance,
    )
    model, model_config, _ = build_model_from_checkpoint(checkpoint)
    if model_config.get("structure_mode") != "candidate_pruning_one_shot":
        raise ValueError("checkpoint must expose one-shot candidate pruning")
    capacities = resolve_comparison_capacities(args, checkpoint)
    capacity_note = None
    unequal_capacity_ablation = not args.ours_only and not capacities["equal_initial_capacity"]
    if unequal_capacity_ablation:
        if not args.allow_unequal_capacity:
            raise ValueError(
                "v16 multi-method case comparisons require the same internal candidate cap "
                "for Ours and configurable numerical methods; change all caps to the checkpoint Kc "
                "or add --allow-unequal-capacity for a visibly marked diagnostic ablation"
            )
        diagnostic = True
        capacity_note = (
            "UNEQUAL-CAPACITY ABLATION: "
            f"Ours Kc={capacities['network_candidates']}; "
            f"numerical cap={capacities['greedy_initial_and_yeh_max']}; "
            f"Kang initial={capacities['kang_dense_initial']}; "
            f"Liang initial={capacities['liang_dense_initial']}"
        )
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
    plotted_methods = (("ours",) if args.ours_only else
                       PUBLISHED_METHODS if args.method_set == "published" else METHODS)
    records = []
    code_paths = sorted(set((ROOT / "src").rglob("*.py")) | {
        Path(__file__), ROOT / "scripts/benchmark_v15_datasets.py", ROOT / "scripts/overnight_datasets.py",
    })
    comparison_provenance = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "published_baseline_protocol": published_baseline_protocol(args),
        "code_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths},
        "configuration": {key: _serializable(value) for key, value in vars(args).items()},
        "dataset_provenance": provenance,
        "sample_content_sha256": [
            hashlib.sha256(case["points"].numpy().tobytes() + (
                case["reference"].numpy().tobytes() if case["reference"] is not None else b""
            )).hexdigest() for case in cases
        ],
    }
    comparison_fingerprint = hashlib.sha256(
        json.dumps(comparison_provenance, sort_keys=True).encode()
    ).hexdigest()
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
                    "label": _method_label(method, safeguard_enabled=getattr(args, "published_feasibility_safeguard", False)),
                    "published_feasibility_safeguard": getattr(args, "published_feasibility_safeguard", False),
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "fit": None,
                    "parameters": None,
                    "mse": None,
                    "max_squared_error": None,
                    "fit_pass": False,
                    "reference_mse": None,
                    "reference_max_squared_error": None,
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
                    f"max SE={result['max_squared_error']:.3e}, "
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
                capacity_note=capacity_note,
            )
        records.append({
            "dataset": case["dataset"],
            "dataset_label": case.get("dataset_label", case["dataset"]),
            "source_kind": case.get("source_kind", "external_geometry_unspecified"),
            "source_note": case.get("source_note", ""),
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
            "comparison_provenance": comparison_provenance,
            "comparison_fingerprint": comparison_fingerprint,
            "published_baseline_protocol": comparison_provenance["published_baseline_protocol"],
            "checkpoint": str(args.checkpoint.resolve()),
            "objective_version": checkpoint["objective_version"],
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_stage": checkpoint.get("stage"),
            "checkpoint_quality": checkpoint.get("checkpoint_quality"),
            "checkpoint_qualification": qualification,
            "diagnostic_not_final": diagnostic,
            "watermark_rendered": False,
            "unequal_capacity_ablation": unequal_capacity_ablation,
            "capacity_comparison_note": capacity_note,
            "mse_tolerance": args.mse_tolerance,
            "mse_definition": (
                "mean(sum((prediction-observation)^2, coordinates)); no square root"
            ),
            "visualization_mode": ("ours_only" if args.ours_only else
                                   "six_method" if args.method_set == "published" else "four_method"),
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
                "complete algorithm time including every safeguard refit. Plotting and metrics are excluded."
            ),
            "max_squared_error_definition": (
                "max(sum((prediction-observation)^2, coordinates)) over observed points; "
                "normalized coordinates, no square root; not maximum curve MSE, "
                "not a continuous/Hausdorff bound, and not the MSE pass criterion. "
                "reference_max_squared_error uses the same original-reference parameter "
                "mapping as reference_mse."
            ),
            "native_baseline_summary": native_baseline_audit([
                dict(method, dataset=record["dataset"], sample_id=record["sample_id"])
                for record in records for method in record["methods"]
            ]),
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
