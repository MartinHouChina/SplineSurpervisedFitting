from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

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
from spline_fitting.evaluation.gradient_knot_pruning import (  # noqa: E402
    chord_length_parameters,
    gradient_knot_pruning_baseline,
)
from spline_fitting.evaluation.feature_cdf_knot_placement import (  # noqa: E402
    fit_feature_cdf_to_tolerance,
)
from spline_fitting.evaluation.sparse_knot_paper import (  # noqa: E402
    fit_sparse_knots_paper,
)
from spline_fitting.evaluation.timing import (  # noqa: E402
    measure_synchronized_wall_time,
)
from visualize_batch_comparison import (  # noqa: E402
    ComparisonPanel,
    _dataset_config_from_checkpoint,
    deployment_geometries,
    fixed_samples_per_knot_count,
    render_comparison_figure,
    resolve_mse_tolerance,
    run_verified_visualization_deployment,
    source_knot_count_from_seed,
    source_panel,
)


MSE_DEFINITION = "mean_i ||C(t_i) - Q_i||_2^2 (no square root)"
METHOD_ORDER = (
    "ours",
    "kang_sparse_2015_adaptation",
    "yeh_feature_cdf_2020",
    "uniform_gradient_pruning",
)
METHOD_LABELS = {
    "ours": "Ours",
    "kang_sparse_2015_adaptation": "Kang 2015 no-repair adaptation",
    "yeh_feature_cdf_2020": "Yeh et al. 2020 feature-CDF",
    "uniform_gradient_pruning": "Uniform Kmax greedy + relocation",
}
METHOD_COLORS = {
    "ours": "#198a77",
    "kang_sparse_2015_adaptation": "#7251b5",
    "yeh_feature_cdf_2020": "#2878b5",
    "uniform_gradient_pruning": "#c94b4b",
}


@dataclass(frozen=True)
class MethodMeasurement:
    method: str
    mse: float
    final_internal_knot_count: int
    elapsed_ms: float
    timing_scope: str
    parameterization: str
    internal_knots: tuple[float, ...]
    threshold_satisfied: bool
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SampleComparison:
    sample_index: int
    source_internal_knot_count: int
    canonical_internal_knot_count: int
    source_mse: float
    mse_tolerance: float
    methods: tuple[MethodMeasurement, ...]
    figure: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one-shot knot deployment, an auditable no-repair vector "
            "adaptation of Kang et al. (CAD 2015), Yeh et al. derivative-feature knot "
            "placement (CAD 2020), and a fully non-network uniform-Kmax deletion "
            "baseline. MSE is mean squared Euclidean error."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-knot-count", type=int, default=1)
    parser.add_argument("--min-knot-count", type=int, default=4)
    parser.add_argument("--max-knot-count", type=int, default=20)
    parser.add_argument("--scan-size", type=int, default=4096)
    parser.add_argument("--dataset-seed", type=int, default=20000)
    parser.add_argument("--selection-seed", type=int, default=12345)
    parser.add_argument(
        "--mse-tolerance",
        type=float,
        default=None,
        help="Defaults to the squared historical RMS tolerance in the checkpoint.",
    )
    parser.add_argument(
        "--ours-mode",
        choices=("learned", "verified"),
        default="learned",
        help=(
            "verified may repair/refit the learned selection for quality, but its "
            "displayed latency remains pure network-forward time."
        ),
    )
    parser.add_argument(
        "--verified-parameterization",
        choices=("network", "chord-fallback", "chord"),
        default="chord-fallback",
    )
    parser.add_argument(
        "--verified-compact",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--verified-hard-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--verified-residual-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--verified-max-residual-insertions", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use torch.compile for the network timing path. Compilation/setup is "
            "excluded by warmup and is not part of the reported latency."
        ),
    )
    parser.add_argument("--network-warmups", type=int, default=5)
    parser.add_argument("--network-repeats", type=int, default=20)
    parser.add_argument(
        "--max-internal-knots",
        type=int,
        default=20,
        help="Uniform Kmax used only by the non-network gradient baseline.",
    )
    parser.add_argument("--min-internal-knots", type=int, default=0)
    parser.add_argument("--gradient-steps", type=int, default=12)
    parser.add_argument("--gradient-learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--gradient-optimizer", choices=("adam", "lbfgs"), default="adam"
    )
    parser.add_argument("--gradient-min-gap", type=float, default=1e-4)
    parser.add_argument(
        "--paper-initial-knots",
        type=int,
        default=40,
        help="Dense uniform first-stage knot count for the Kang adaptation.",
    )
    parser.add_argument(
        "--paper-jump-threshold",
        type=float,
        default=1e-7,
        help=(
            "Absolute derivative-jump activity threshold; combined with the "
            "relative threshold by taking their maximum."
        ),
    )
    parser.add_argument(
        "--paper-relative-jump-threshold",
        type=float,
        default=1e-3,
        help=(
            "Numerical-zero floor relative to the largest jump. This is needed "
            "because the ADMM adaptation does not produce CVX-exact zeros."
        ),
    )
    parser.add_argument("--paper-admm-rho", type=float, default=1e4)
    parser.add_argument("--paper-admm-iterations", type=int, default=400)
    parser.add_argument("--paper-admm-tolerance", type=float, default=1e-6)
    parser.add_argument("--paper-lambda-bisections", type=int, default=8)
    parser.add_argument("--paper-relocation-iterations", type=int, default=8)
    parser.add_argument(
        "--feature-density-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply Yeh et al.'s local knot-density limiter.",
    )
    parser.add_argument("--smoothness-weight", type=float, default=1e-6)
    parser.add_argument("--control-ridge", type=float, default=0.0)
    parser.add_argument(
        "--sample-figures",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also write one Original/Ours/Kang/Yeh/Gradient PNG per sample.",
    )
    parser.add_argument("--dense-points", type=int, default=500)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def _finite_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {args.checkpoint}")
    for name in (
        "samples_per_knot_count",
        "scan_size",
        "network_repeats",
        "max_internal_knots",
        "paper_initial_knots",
        "paper_admm_iterations",
        "paper_lambda_bisections",
        "paper_relocation_iterations",
        "dense_points",
        "dpi",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.network_warmups < 0 or args.gradient_steps < 0:
        raise ValueError("network warmups and gradient steps must be non-negative")
    if args.min_knot_count < 0 or args.max_knot_count < args.min_knot_count:
        raise ValueError("invalid inclusive source-knot range")
    if not 0 <= args.min_internal_knots <= args.max_internal_knots:
        raise ValueError("min internal knots must lie in [0, max internal knots]")
    if args.verified_max_residual_insertions < 0:
        raise ValueError("verified residual insertions must be non-negative")
    for name in (
        "gradient_min_gap",
        "smoothness_weight",
        "control_ridge",
        "paper_jump_threshold",
        "paper_relative_jump_threshold",
    ):
        _finite_nonnegative(float(getattr(args, name)), name)
    if args.gradient_learning_rate <= 0.0:
        raise ValueError("gradient learning rate must be positive")
    if args.paper_admm_rho <= 0.0 or args.paper_admm_tolerance <= 0.0:
        raise ValueError("paper ADMM rho/tolerance must be positive")


def _float_knots(knots: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(value) for value in knots.detach().cpu().flatten())


def _panel_from_fit(
    fit: BSplineLeastSquaresFit,
    *,
    dense_points: int,
    title: str,
    curve_label: str,
    timing_text: str,
    color: str,
) -> ComparisonPanel:
    dense_parameters = torch.linspace(
        0.0,
        1.0,
        dense_points,
        device=fit.control_points.device,
        dtype=fit.control_points.dtype,
    )
    return ComparisonPanel(
        title=title,
        curve_label=curve_label,
        dense_curve=fit.evaluate(dense_parameters).detach().cpu().to(torch.float32),
        reconstructed_points=(
            fit.reconstructed_points.detach().cpu().to(torch.float32)
        ),
        control_points=fit.control_points.detach().cpu().to(torch.float32),
        internal_knots=fit.internal_knots.detach().cpu().to(torch.float32),
        knot_vector=fit.knot_vector.detach().cpu().to(torch.float32),
        mse=float(fit.fit_mse),
        timing_text=timing_text,
        curve_color=color,
    )


def _float32_panel(panel: ComparisonPanel) -> ComparisonPanel:
    return ComparisonPanel(
        title=panel.title,
        curve_label=panel.curve_label,
        dense_curve=panel.dense_curve.to(torch.float32),
        reconstructed_points=panel.reconstructed_points.to(torch.float32),
        control_points=panel.control_points.to(torch.float32),
        internal_knots=panel.internal_knots.to(torch.float32),
        knot_vector=panel.knot_vector.to(torch.float32),
        mse=panel.mse,
        timing_text=panel.timing_text,
        curve_color=panel.curve_color,
    )


def aggregate_by_source_count(
    comparisons: Sequence[SampleComparison],
) -> dict[int, dict[str, object]]:
    """Return deterministic per-source-K averages used by JSON and plots."""

    grouped: dict[int, list[SampleComparison]] = {}
    for comparison in comparisons:
        grouped.setdefault(comparison.source_internal_knot_count, []).append(comparison)
    aggregate: dict[int, dict[str, object]] = {}
    for source_count in sorted(grouped):
        samples = grouped[source_count]
        method_rows: dict[str, dict[str, float]] = {}
        for method in METHOD_ORDER:
            values = [
                next(item for item in sample.methods if item.method == method)
                for sample in samples
            ]
            canonical_errors = [
                value.final_internal_knot_count
                - sample.canonical_internal_knot_count
                for sample, value in zip(samples, values, strict=True)
            ]
            source_errors = [
                value.final_internal_knot_count
                - sample.source_internal_knot_count
                for sample, value in zip(samples, values, strict=True)
            ]
            method_rows[method] = {
                "mse_mean": statistics.fmean(value.mse for value in values),
                "mse_stdev": (
                    statistics.stdev(value.mse for value in values)
                    if len(values) > 1
                    else 0.0
                ),
                "final_internal_knot_count_mean": statistics.fmean(
                    value.final_internal_knot_count for value in values
                ),
                "elapsed_ms_mean": statistics.fmean(
                    value.elapsed_ms for value in values
                ),
                "pass_rate": statistics.fmean(
                    float(value.threshold_satisfied) for value in values
                ),
                "canonical_count_bias": statistics.fmean(canonical_errors),
                "canonical_count_mae": statistics.fmean(
                    abs(value) for value in canonical_errors
                ),
                "source_count_bias": statistics.fmean(source_errors),
                "source_count_mae": statistics.fmean(
                    abs(value) for value in source_errors
                ),
            }
        aggregate[source_count] = {
            "sample_count": len(samples),
            "canonical_internal_knot_count_mean": statistics.fmean(
                sample.canonical_internal_knot_count for sample in samples
            ),
            "methods": method_rows,
        }
    return aggregate


def _overall_summary(
    comparisons: Sequence[SampleComparison],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for method in METHOD_ORDER:
        values = [
            next(item for item in sample.methods if item.method == method)
            for sample in comparisons
        ]
        canonical_errors = [
            value.final_internal_knot_count
            - sample.canonical_internal_knot_count
            for sample, value in zip(comparisons, values, strict=True)
        ]
        source_errors = [
            value.final_internal_knot_count - sample.source_internal_knot_count
            for sample, value in zip(comparisons, values, strict=True)
        ]
        summary[method] = {
            "mse_mean": statistics.fmean(value.mse for value in values),
            "mse_median": statistics.median(value.mse for value in values),
            "final_internal_knot_count_mean": statistics.fmean(
                value.final_internal_knot_count for value in values
            ),
            "elapsed_ms_mean": statistics.fmean(value.elapsed_ms for value in values),
            "elapsed_ms_median": statistics.median(
                value.elapsed_ms for value in values
            ),
            "threshold_pass_rate": statistics.fmean(
                float(value.threshold_satisfied) for value in values
            ),
            "canonical_count_bias": statistics.fmean(canonical_errors),
            "canonical_count_mae": statistics.fmean(
                abs(value) for value in canonical_errors
            ),
            "source_count_bias": statistics.fmean(source_errors),
            "source_count_mae": statistics.fmean(
                abs(value) for value in source_errors
            ),
        }
    return summary


def render_aggregate_figure(
    comparisons: Sequence[SampleComparison],
    output_path: Path,
    *,
    mse_tolerance: float,
    dpi: int,
) -> None:
    aggregate = aggregate_by_source_count(comparisons)
    source_counts = sorted(aggregate)
    figure, grid = plt.subplots(2, 2, figsize=(15.5, 11.0))
    axes = grid.ravel()

    for method in METHOD_ORDER:
        rows = [aggregate[count]["methods"][method] for count in source_counts]
        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
        mse = [float(row["mse_mean"]) for row in rows]
        stdev = [float(row["mse_stdev"]) for row in rows]
        axes[0].plot(source_counts, mse, "o-", color=color, label=label)
        if any(value > 0.0 for value in stdev):
            lower = [max(value - spread, 1e-16) for value, spread in zip(mse, stdev)]
            upper = [value + spread for value, spread in zip(mse, stdev)]
            axes[0].fill_between(source_counts, lower, upper, color=color, alpha=0.12)
        axes[1].plot(
            source_counts,
            [float(row["final_internal_knot_count_mean"]) for row in rows],
            "o-",
            color=color,
            label=label,
        )
        axes[2].plot(
            source_counts,
            [100.0 * float(row["pass_rate"]) for row in rows],
            "o-",
            color=color,
            label=label,
        )
        axes[3].plot(
            source_counts,
            [max(float(row["elapsed_ms_mean"]), 1e-9) for row in rows],
            "o-",
            color=color,
            label=label,
        )

    axes[0].axhline(
        mse_tolerance,
        color="#555555",
        linestyle="--",
        linewidth=1.1,
        label=f"MSE threshold={mse_tolerance:.2e}",
    )
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Standard B-spline refit error")
    axes[0].set_ylabel("mean squared Euclidean error")
    axes[1].plot(
        source_counts,
        source_counts,
        linestyle="--",
        color="#777777",
        linewidth=1.0,
        label="source K",
    )
    axes[1].plot(
        source_counts,
        [
            float(aggregate[count]["canonical_internal_knot_count_mean"])
            for count in source_counts
        ],
        "o-",
        color="#355c9a",
        linewidth=1.2,
        label="canonical reference K",
    )
    axes[1].set_title("(b) Final internal-knot count")
    axes[1].set_ylabel("internal-knot count")
    axes[2].set_ylim(-2.0, 102.0)
    axes[2].set_title("(c) MSE-threshold satisfied fraction")
    axes[2].set_ylabel("pass rate (%)")
    axes[3].set_yscale("log")
    axes[3].set_title("(d) Inference/search wall time")
    axes[3].set_ylabel("milliseconds per curve")

    for axis in axes:
        axis.set_xlabel("source internal-knot count K")
        axis.grid(alpha=0.22, linewidth=0.7)
        axis.legend(fontsize=8)
        axis.set_xticks(source_counts)
    figure.suptitle(
        "Ours vs published knot-placement baselines and uniform-Kmax pruning",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        "MSE has no square root. Ours time is synchronized batch-1 network "
        "forward only (inputs already on device; refit/verification excluded). "
        "Kang, Yeh, and gradient-baseline times include their complete numerical search.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    figure.subplots_adjust(
        bottom=0.12,
        top=0.90,
        wspace=0.24,
        hspace=0.30,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _write_outputs(
    output_dir: Path,
    comparisons: Sequence[SampleComparison],
    *,
    metadata: Mapping[str, object],
) -> tuple[Path, Path]:
    json_path = output_dir / "four_method_comparison.json"
    csv_path = output_dir / "four_method_comparison.csv"
    aggregate = aggregate_by_source_count(comparisons)
    report = {
        "schema_version": 1,
        "metadata": dict(metadata),
        "mse_definition": MSE_DEFINITION,
        "timing_warning": (
            "Ours reports pure network-forward latency only; its final B-spline "
            "refit and optional verified repair are excluded. Kang, Yeh, and "
            "the uniform-gradient baseline report complete numerical-method time."
        ),
        "method_definitions": {
            "ours": (
                "One-shot learned keep mask and survivor relocation; quality may "
                "optionally use verified postprocessing."
            ),
            "kang_sparse_2015_adaptation": (
                "Vector group-L1 ADMM adaptation of Kang et al. (2015), followed "
                "by active-cluster midpoint/bisection relocation and one exact "
                "refit. Repository-specific knot add-back and endpoint fallback "
                "are disabled so failure remains visible."
            ),
            "yeh_feature_cdf_2020": (
                "Yeh et al. (2020) high-order finite-difference feature-CDF knot "
                "placement. A disclosed ascending cardinality scan replaces the "
                "paper's dataset-specific target-error regression."
            ),
            "uniform_gradient_pruning": (
                "No network. Uniform Kmax initialization, full-set relocation, "
                "greedy one-knot deletion, differentiable variable-projection "
                "relocation, and exact refit checks."
            ),
        },
        "overall": _overall_summary(comparisons),
        "by_source_internal_knot_count": aggregate,
        "samples": [asdict(comparison) for comparison in comparisons],
    }
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    fieldnames = (
        "sample_index",
        "source_internal_knot_count",
        "canonical_internal_knot_count",
        "mse_tolerance",
        "method",
        "mse",
        "final_internal_knot_count",
        "canonical_count_error",
        "source_count_error",
        "threshold_satisfied",
        "elapsed_ms",
        "timing_scope",
        "parameterization",
        "internal_knots",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for comparison in comparisons:
            for method in comparison.methods:
                writer.writerow(
                    {
                        "sample_index": comparison.sample_index,
                        "source_internal_knot_count": (
                            comparison.source_internal_knot_count
                        ),
                        "canonical_internal_knot_count": (
                            comparison.canonical_internal_knot_count
                        ),
                        "mse_tolerance": comparison.mse_tolerance,
                        "method": method.method,
                        "mse": method.mse,
                        "final_internal_knot_count": (
                            method.final_internal_knot_count
                        ),
                        "canonical_count_error": (
                            method.final_internal_knot_count
                            - comparison.canonical_internal_knot_count
                        ),
                        "source_count_error": (
                            method.final_internal_knot_count
                            - comparison.source_internal_knot_count
                        ),
                        "threshold_satisfied": method.threshold_satisfied,
                        "elapsed_ms": method.elapsed_ms,
                        "timing_scope": method.timing_scope,
                        "parameterization": method.parameterization,
                        "internal_knots": json.dumps(method.internal_knots),
                    }
                )
    return json_path, csv_path


def _select_indices(
    dataset_config: Mapping[str, object],
    *,
    scan_size: int,
    dataset_seed: int,
    selection_seed: int,
    minimum_count: int,
    maximum_count: int,
    samples_per_count: int,
) -> list[int]:
    index_to_count = {
        index: source_knot_count_from_seed(
            dataset_config,
            dataset_seed=dataset_seed,
            sample_index=index,
        )
        for index in range(scan_size)
    }
    return fixed_samples_per_knot_count(
        index_to_count,
        range(minimum_count, maximum_count + 1),
        samples_per_count=samples_per_count,
        seed=selection_seed,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        device = _resolve_device(args.device)
    except ValueError as error:
        parser.error(str(error))

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    try:
        mse_tolerance = resolve_mse_tolerance(checkpoint, args.mse_tolerance)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    model, model_config, _ = build_model_from_checkpoint(checkpoint)
    if model_config.get("structure_mode") != "candidate_pruning_one_shot":
        parser.error("comparison requires a one-shot candidate-pruning checkpoint")
    degree = int(model.degree)
    quality_model = model.eval().to(device)
    network_forward_callable = getattr(quality_model, "forward_deployment", None)
    if not callable(network_forward_callable):
        network_forward_callable = quality_model
    if args.compile_model:
        if not hasattr(torch, "compile"):
            parser.error("this PyTorch build does not provide torch.compile")
        network_forward_callable = torch.compile(network_forward_callable)

    dataset_config = _dataset_config_from_checkpoint(checkpoint, model_config)
    if int(dataset_config["point_dim"]) != 2:
        parser.error("qualitative comparison currently supports 2D checkpoints")
    try:
        selected_indices = _select_indices(
            dataset_config,
            scan_size=args.scan_size,
            dataset_seed=args.dataset_seed,
            selection_seed=args.selection_seed,
            minimum_count=args.min_knot_count,
            maximum_count=args.max_knot_count,
            samples_per_count=args.samples_per_knot_count,
        )
    except ValueError as error:
        parser.error(str(error))

    output_dir: Path = args.output_dir
    expected_outputs = (
        output_dir / "four_method_comparison.json",
        output_dir / "four_method_comparison.csv",
        output_dir / "four_method_comparison.png",
    )
    if not args.overwrite and any(path.exists() for path in expected_outputs):
        parser.error(f"comparison output already exists in {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "samples"
    if args.sample_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    dataset = SyntheticCubicBSplineDataset(
        size=max(selected_indices) + 1,
        seed=args.dataset_seed,
        cache_samples=False,
        **dataset_config,
    )
    deployment_config = checkpoint.get("deployment_config", {})
    if not isinstance(deployment_config, Mapping):
        deployment_config = {}
    verified_min_knots = int(deployment_config.get("min_internal_knots", 0))
    verified_residual_min_gap = float(model_config.get("min_knot_gap", 1e-3))

    comparisons: list[SampleComparison] = []
    total = len(selected_indices)
    for ordinal, sample_index in enumerate(selected_indices, start=1):
        sample = dataset[sample_index]
        points = sample["points"]
        chord = sample["chord_params"]
        canonical_mask = sample["true_internal_knot_mask"]
        assert isinstance(points, torch.Tensor)
        assert isinstance(chord, torch.Tensor)
        assert isinstance(canonical_mask, torch.Tensor)
        source_count = int(sample["source_internal_knot_count"])
        canonical_count = int(canonical_mask.sum().item())
        model_points = points.unsqueeze(0).to(device)

        def network_forward() -> dict[str, torch.Tensor]:
            with torch.inference_mode():
                return network_forward_callable(model_points)

        network_call = measure_synchronized_wall_time(
            network_forward,
            synchronization_device=device,
            warmup_repeats=args.network_warmups,
            timing_repeats=args.network_repeats,
        )
        output = network_call.result
        _, deployment_positions, learned_mask = deployment_geometries(output)
        parameters = output["params"][0].detach().cpu().to(torch.float64)
        # Joint survivor relocation can temporarily swap the order of two
        # retained slots.  A knot vector is a set of selected positions with a
        # non-decreasing deployment order, so canonicalize only the retained
        # subset at the standard-B-spline boundary.  This preserves both the
        # selected set and its cardinality.
        learned_knots = torch.sort(
            deployment_positions[learned_mask].detach().cpu().to(torch.float64)
        ).values
        numeric_points = points.detach().cpu().to(torch.float64)
        numeric_chord = chord_length_parameters(numeric_points)

        quality_contains_postprocessing = args.ours_mode == "verified"
        ours_parameterization = "network_predicted"
        ours_diagnostics: dict[str, object]
        if args.ours_mode == "verified":
            verified = run_verified_visualization_deployment(
                quality_model,
                model_points,
                points,
                chord,
                mse_tolerance=mse_tolerance,
                min_internal_knots=verified_min_knots,
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
                compact=args.verified_compact,
                hard_fallback=args.verified_hard_fallback,
                residual_fallback=args.verified_residual_fallback,
                max_residual_insertions=args.verified_max_residual_insertions,
                residual_min_gap=verified_residual_min_gap,
                parameterization_policy=args.verified_parameterization,
                refit_device="cpu",
            ).repair
            ours_native_fit = verified.final_fit
            ours_report_parameters = verified.final_parameters.detach().to(
                device="cpu", dtype=torch.float64
            )
            ours_parameterization = verified.final_parameterization
            ours_diagnostics = {
                "quality_mode": "verified",
                "quality_contains_postprocessing": True,
                "repair_path": verified.final_source,
                "initial_learned_count": int(learned_mask.sum().item()),
            }
        else:
            ours_native_fit = refit_bspline_control_points(
                parameters,
                numeric_points,
                learned_knots,
                degree=degree,
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
                interpolate_endpoints=True,
            )
            ours_report_parameters = parameters
            ours_diagnostics = {
                "quality_mode": "learned",
                "quality_contains_postprocessing": False,
            }
        # Every reported method uses the same final endpoint-constrained
        # standard B-spline refit. Selection/repair may use its native solver
        # settings, which remain available in diagnostics.
        ours_fit = refit_bspline_control_points(
            ours_report_parameters,
            numeric_points,
            ours_native_fit.internal_knots.detach().to(
                device="cpu", dtype=torch.float64
            ),
            degree=degree,
            smoothness_weight=0.0,
            control_ridge=0.0,
            interpolate_endpoints=True,
        )
        ours_diagnostics.update(
            {
                "native_final_fit_mse": float(ours_native_fit.fit_mse),
                "reported_refit": (
                    "standard B-spline least squares with exact endpoint "
                    "interpolation"
                ),
            }
        )
        ours_measurement = MethodMeasurement(
            method="ours",
            mse=float(ours_fit.fit_mse),
            final_internal_knot_count=int(ours_fit.internal_knots.numel()),
            elapsed_ms=network_call.latency.p50_ms,
            timing_scope=(
                "synchronized batch-1 network forward only; input preloaded on "
                "device; final refit and verified repair excluded"
            ),
            parameterization=ours_parameterization,
            internal_knots=_float_knots(ours_fit.internal_knots),
            threshold_satisfied=float(ours_fit.fit_mse) <= mse_tolerance,
            diagnostics={
                **ours_diagnostics,
                "network_used": True,
                "network_latency": network_call.latency.as_dict(),
                "compiled": bool(args.compile_model),
                "device": str(device),
            },
        )

        paper_result = fit_sparse_knots_paper(
            numeric_chord,
            numeric_points,
            degree=degree,
            initial_internal_knot_count=args.paper_initial_knots,
            data_tolerance=mse_tolerance,
            jump_threshold=args.paper_jump_threshold,
            relative_jump_threshold=args.paper_relative_jump_threshold,
            admm_rho=args.paper_admm_rho,
            admm_max_iterations=args.paper_admm_iterations,
            admm_tolerance=args.paper_admm_tolerance,
            lambda_bisection_iterations=args.paper_lambda_bisections,
            relocation_max_iterations=args.paper_relocation_iterations,
            feasibility_repair=False,
        )
        paper_postprocess_started = time.perf_counter()
        paper_native_endpoint_fit = refit_bspline_control_points(
            numeric_chord,
            numeric_points,
            paper_result.knots,
            degree=degree,
            smoothness_weight=0.0,
            control_ridge=0.0,
            interpolate_endpoints=True,
        )
        paper_fit = paper_native_endpoint_fit
        paper_elapsed_ms = (
            paper_result.elapsed_seconds
            + time.perf_counter()
            - paper_postprocess_started
        ) * 1000.0
        paper_measurement = MethodMeasurement(
            method="kang_sparse_2015_adaptation",
            mse=float(paper_fit.fit_mse),
            final_internal_knot_count=int(paper_fit.internal_knots.numel()),
            elapsed_ms=paper_elapsed_ms,
            timing_scope=(
                "complete sparse ADMM selection + relocation + one common "
                "endpoint-constrained final refit; no feasibility add-back/fallback"
            ),
            parameterization="chord_length",
            internal_knots=_float_knots(paper_fit.internal_knots),
            threshold_satisfied=float(paper_fit.fit_mse) <= mse_tolerance,
            diagnostics={
                "initial_internal_knot_count": int(
                    paper_result.initial_internal_knots.numel()
                ),
                "dense_initial_fit_mse": float(paper_result.dense_initial_fit_mse),
                "dense_initial_threshold_satisfied": (
                    paper_result.dense_initial_threshold_satisfied
                ),
                "network_used": False,
                "active_internal_knot_count": paper_result.active_count,
                "cluster_sizes": paper_result.cluster_sizes,
                "sparse_stage_mse": float(paper_result.sparse_fit_mse),
                "sparse_stage_threshold_satisfied": (
                    paper_result.threshold_satisfied
                ),
                "sparse_iterations": paper_result.sparse_iterations,
                "sparse_solver_seconds": paper_result.sparse_solver_seconds,
                "relocation_seconds": paper_result.relocation_seconds,
                "non_paper_feasibility_repair_enabled": False,
                "local_refit_count": paper_result.local_refit_count,
                "method_note": paper_result.method,
                "absolute_jump_threshold": paper_result.jump_threshold,
                "relative_jump_threshold": paper_result.relative_jump_threshold,
                "effective_jump_threshold": paper_result.effective_jump_threshold,
                "paper_feasibility_repair_used": False,
                "paper_feasibility_repair_added_count": 0,
                "native_final_fit_mse_without_endpoint_constraint": float(
                    paper_result.final_fit.fit_mse
                ),
                "native_endpoint_constrained_mse": float(
                    paper_native_endpoint_fit.fit_mse
                ),
                "endpoint_constraint_fallback": "disabled",
                "reported_refit": (
                    "unregularized standard B-spline least squares with exact "
                    "endpoint interpolation"
                ),
            },
        )

        feature_result = fit_feature_cdf_to_tolerance(
            numeric_chord,
            numeric_points,
            degree=degree,
            mse_tolerance=mse_tolerance,
            max_internal_knots=args.max_internal_knots,
            min_internal_knots=args.min_internal_knots,
            density_limit=args.feature_density_limit,
            interpolate_endpoints=True,
        )
        feature_fit = feature_result.final_fit
        feature_measurement = MethodMeasurement(
            method="yeh_feature_cdf_2020",
            mse=float(feature_fit.fit_mse),
            final_internal_knot_count=int(feature_fit.internal_knots.numel()),
            elapsed_ms=feature_result.elapsed_seconds * 1000.0,
            timing_scope=(
                "complete derivative-feature construction + ascending cardinality "
                "scan + endpoint-constrained standard B-spline refits"
            ),
            parameterization="chord_length",
            internal_knots=_float_knots(feature_fit.internal_knots),
            threshold_satisfied=feature_result.threshold_satisfied,
            diagnostics={
                "network_used": False,
                "paper_placement_rule": "equal high-order derivative-feature mass",
                "comparison_wrapper": (
                    "ascending cardinality scan replaces the paper's dataset-specific "
                    "target-error regression"
                ),
                "derivative_order": feature_result.derivative_order,
                "density_limit": bool(args.feature_density_limit),
                "scanned_counts": feature_result.scanned_counts,
                "scanned_mse": feature_result.scanned_mse,
                "exact_refit_count": feature_result.refit_count,
                "reported_refit": (
                    "unregularized standard B-spline least squares with exact "
                    "endpoint interpolation"
                ),
            },
        )

        gradient_result = gradient_knot_pruning_baseline(
            numeric_points,
            max_internal_knots=args.max_internal_knots,
            mse_tolerance=mse_tolerance,
            min_internal_knots=args.min_internal_knots,
            degree=degree,
            min_gap=args.gradient_min_gap,
            optimization_steps=args.gradient_steps,
            learning_rate=args.gradient_learning_rate,
            optimizer=args.gradient_optimizer,
            smoothness_weight=0.0,
            control_ridge=0.0,
            interpolate_endpoints=True,
        )
        gradient_fit = refit_bspline_control_points(
            gradient_result.parameters,
            numeric_points,
            gradient_result.knots,
            degree=degree,
            smoothness_weight=0.0,
            control_ridge=0.0,
            interpolate_endpoints=True,
        )
        gradient_measurement = MethodMeasurement(
            method="uniform_gradient_pruning",
            mse=float(gradient_fit.fit_mse),
            final_internal_knot_count=gradient_result.K,
            elapsed_ms=gradient_result.total_search_time * 1000.0,
            timing_scope=(
                "complete uniform-Kmax initialization + all deletion scoring + "
                "gradient relocation + exact refit validation"
            ),
            parameterization="chord_length",
            internal_knots=_float_knots(gradient_result.knots),
            threshold_satisfied=float(gradient_fit.fit_mse) <= mse_tolerance,
            diagnostics={
                "initial_internal_knot_count": args.max_internal_knots,
                "network_used": False,
                "initial_location_update_accepted": (
                    gradient_result.initial_location_update_accepted
                ),
                "uniform_initial_fit_mse": float(
                    gradient_result.initial_fit.fit_mse
                ),
                "relocated_initial_fit_mse": float(
                    gradient_result.initial_relocated_fit.fit_mse
                ),
                "accepted_deletions": gradient_result.accepted_deletions,
                "gradient_steps": gradient_result.gradient_steps,
                "variable_projection_evaluations": (
                    gradient_result.evaluation_count
                ),
                "exact_refit_count": gradient_result.refit_count,
                "native_final_fit_mse": float(gradient_result.final_fit.fit_mse),
                "reported_refit": (
                    "unregularized standard B-spline least squares with exact "
                    "endpoint interpolation"
                ),
            },
        )

        source_dense = torch.linspace(0.0, 1.0, args.dense_points)
        source = _float32_panel(source_panel(sample, source_dense, degree=degree))
        figure_name: str | None = None
        if args.sample_figures:
            figure_name = (
                f"sample_{sample_index:05d}_sourceK{source_count:02d}.png"
            )
            panels = (
                source,
                _panel_from_fit(
                    ours_fit,
                    dense_points=args.dense_points,
                    title=(
                        "(b) Ours: verified quality, network time only"
                        if quality_contains_postprocessing
                        else "(b) Ours: learned one-shot deployment"
                    ),
                    curve_label="Ours standard B-spline refit",
                    timing_text=(
                        f"network-only p50={network_call.latency.p50_ms:.3f} ms, "
                        f"p95={network_call.latency.p95_ms:.3f} ms"
                    ),
                    color=METHOD_COLORS["ours"],
                ),
                _panel_from_fit(
                    paper_fit,
                    dense_points=args.dense_points,
                    title="(c) Kang 2015 no-repair adaptation",
                    curve_label="sparse selection + relocated refit; no add-back",
                    timing_text=f"complete algorithm={paper_measurement.elapsed_ms:.2f} ms",
                    color=METHOD_COLORS["kang_sparse_2015_adaptation"],
                ),
                _panel_from_fit(
                    feature_fit,
                    dense_points=args.dense_points,
                    title="(d) Yeh 2020 derivative-feature placement",
                    curve_label="feature-CDF + cardinality scan",
                    timing_text=(
                        f"complete algorithm={feature_measurement.elapsed_ms:.2f} ms"
                    ),
                    color=METHOD_COLORS["yeh_feature_cdf_2020"],
                ),
                _panel_from_fit(
                    gradient_fit,
                    dense_points=args.dense_points,
                    title="(e) Uniform Kmax delete + gradient relocation",
                    curve_label="network-free numerical baseline",
                    timing_text=(
                        f"complete algorithm={gradient_measurement.elapsed_ms:.2f} ms"
                    ),
                    color=METHOD_COLORS["uniform_gradient_pruning"],
                ),
            )
            render_comparison_figure(
                panels,
                points,
                figures_dir / figure_name,
                degree=degree,
                sample_index=sample_index,
                source_count=source_count,
                canonical_count=canonical_count,
                mse_tolerance=mse_tolerance,
                dpi=args.dpi,
                parameterization_note=(
                    "Ours uses its reported deployment parameterization; Kang, "
                    "Yeh, and the non-network baseline use chord-length parameters."
                ),
                timing_scope_note=(
                    "Ours displays synchronized pure network-forward latency only; "
                    "its refit/verified repair is excluded. Numerical methods show "
                    "complete algorithm time. Kang has no hidden repair/fallback."
                ),
            )

        comparison = SampleComparison(
            sample_index=sample_index,
            source_internal_knot_count=source_count,
            canonical_internal_knot_count=canonical_count,
            source_mse=source.mse,
            mse_tolerance=mse_tolerance,
            methods=(
                ours_measurement,
                paper_measurement,
                feature_measurement,
                gradient_measurement,
            ),
            figure=(f"samples/{figure_name}" if figure_name is not None else None),
        )
        comparisons.append(comparison)
        print(
            f"[{ordinal:02d}/{total:02d}] sample={sample_index} sourceK={source_count} "
            f"| Ours K/MSE/net={ours_measurement.final_internal_knot_count}/"
            f"{ours_measurement.mse:.3e}/{ours_measurement.elapsed_ms:.3f}ms "
            f"| Kang={paper_measurement.final_internal_knot_count}/"
            f"{paper_measurement.mse:.3e}/{paper_measurement.elapsed_ms:.1f}ms "
            f"| Yeh={feature_measurement.final_internal_knot_count}/"
            f"{feature_measurement.mse:.3e}/{feature_measurement.elapsed_ms:.1f}ms "
            f"| UniformGrad={gradient_measurement.final_internal_knot_count}/"
            f"{gradient_measurement.mse:.3e}/{gradient_measurement.elapsed_ms:.1f}ms",
            flush=True,
        )

    aggregate_path = output_dir / "four_method_comparison.png"
    render_aggregate_figure(
        comparisons,
        aggregate_path,
        mse_tolerance=mse_tolerance,
        dpi=args.dpi,
    )
    json_path, csv_path = _write_outputs(
        output_dir,
        comparisons,
        metadata={
            "checkpoint": str(args.checkpoint),
            "ours_mode": args.ours_mode,
            "source_knot_range": [args.min_knot_count, args.max_knot_count],
            "samples_per_source_knot_count": args.samples_per_knot_count,
            "selected_sample_indices": selected_indices,
            "dataset_seed": args.dataset_seed,
            "selection_seed": args.selection_seed,
            "network_device": str(device),
            "network_batch_size": 1,
            "network_warmups": args.network_warmups,
            "network_repeats": args.network_repeats,
            "network_compiled": bool(args.compile_model),
            "gradient_baseline_uniform_Kmax": args.max_internal_knots,
            "paper_initial_uniform_knot_count": args.paper_initial_knots,
            "paper_absolute_jump_threshold": args.paper_jump_threshold,
            "paper_relative_jump_threshold": args.paper_relative_jump_threshold,
            "paper_non_paper_feasibility_repair_enabled": False,
            "paper_endpoint_fallback_enabled": False,
            "feature_cdf_max_internal_knots": args.max_internal_knots,
            "feature_cdf_density_limit": bool(args.feature_density_limit),
            "mse_tolerance": mse_tolerance,
            "timing_scopes_are_asymmetric": True,
            "reported_refit_policy": (
                "all four methods are re-evaluated by unregularized standard "
                "B-spline least squares with exact endpoint interpolation"
            ),
        },
    )
    print(f"Saved aggregate PNG: {aggregate_path}")
    print(f"Saved long-form CSV: {csv_path}")
    print(f"Saved JSON report: {json_path}")
    if args.sample_figures:
        print(f"Saved {len(comparisons)} qualitative PNGs under: {figures_dir}")


if __name__ == "__main__":
    main()
