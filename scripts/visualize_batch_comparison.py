from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import textwrap
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

from spline_fitting.checkpointing import build_model_from_checkpoint
from spline_fitting.data.synthetic import (
    SyntheticCubicBSplineDataset,
    bspline_basis_matrix,
)
from spline_fitting.evaluation.bspline_inference import (
    BSplineLeastSquaresFit,
    refit_bspline_control_points,
)
from spline_fitting.spline.bspline_deletion_teacher import (
    single_knot_deletion_mse_batch,
)


T = TypeVar("T")
MSE_DEFINITION = "mean_i ||C(t_i) - Q_i||_2^2"


@dataclass(frozen=True)
class MSEPruningStep:
    iteration: int
    removed_index: int
    removed_knot: float
    accepted: bool
    refit_mse: float


@dataclass(frozen=True)
class MSEPruningResult:
    initial_internal_knots: torch.Tensor
    retained_internal_knots: torch.Tensor
    retained_mask: torch.Tensor
    final_fit: BSplineLeastSquaresFit
    steps: tuple[MSEPruningStep, ...]
    mse_trajectory: tuple[float, ...]
    mse_tolerance: float

    @property
    def initial_count(self) -> int:
        return int(self.initial_internal_knots.numel())

    @property
    def final_count(self) -> int:
        return int(self.retained_internal_knots.numel())

    @property
    def accepted_deletions(self) -> int:
        return sum(step.accepted for step in self.steps)

    @property
    def threshold_satisfied(self) -> bool:
        return float(self.final_fit.fit_mse) <= self.mse_tolerance


@dataclass(frozen=True)
class ComparisonPanel:
    title: str
    curve_label: str
    dense_curve: torch.Tensor
    reconstructed_points: torch.Tensor
    control_points: torch.Tensor
    internal_knots: torch.Tensor
    knot_vector: torch.Tensor
    mse: float
    timing_text: str
    curve_color: str

    @property
    def knot_count(self) -> int:
        return int(self.internal_knots.numel())


def mean_squared_euclidean_error(
    predicted: torch.Tensor,
    observed: torch.Tensor,
) -> torch.Tensor:
    """Return pointwise squared Euclidean error averaged over samples."""

    if predicted.shape != observed.shape or predicted.ndim != 2:
        raise ValueError("predicted and observed must share shape [N,D]")
    if not predicted.is_floating_point() or not observed.is_floating_point():
        raise ValueError("predicted and observed must be floating-point tensors")
    return (predicted - observed).square().sum(dim=-1).mean()


def resolve_mse_tolerance(
    checkpoint: Mapping[str, object],
    explicit_mse_tolerance: float | None,
) -> float:
    """Resolve an MSE threshold, squaring historical checkpoint RMS metadata."""

    if explicit_mse_tolerance is not None:
        tolerance = float(explicit_mse_tolerance)
    else:
        deployment = checkpoint.get("deployment_config", {})
        dataset = checkpoint.get("dataset_config", {})
        if isinstance(deployment, Mapping) and "error_tolerance" in deployment:
            historical_rms = float(deployment["error_tolerance"])
        elif isinstance(dataset, Mapping) and "canonical_knot_tolerance" in dataset:
            historical_rms = float(dataset["canonical_knot_tolerance"])
        else:
            historical_rms = 5e-3
        tolerance = historical_rms * historical_rms
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("MSE tolerance must be finite and non-negative")
    return tolerance


def stratified_random_indices(
    index_to_count: Mapping[int, int],
    num_samples: int,
    *,
    seed: int,
    allowed_counts: Sequence[int] | None = None,
) -> list[int]:
    """Choose unique indices by randomized round-robin over knot-count strata."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    grouped: dict[int, list[int]] = {}
    for raw_index, raw_count in index_to_count.items():
        index = int(raw_index)
        count = int(raw_count)
        grouped.setdefault(count, []).append(index)
    if not grouped:
        raise ValueError("index_to_count cannot be empty")

    if allowed_counts is None:
        requested_counts = sorted(grouped)
    else:
        requested_counts = list(dict.fromkeys(int(value) for value in allowed_counts))
        missing = [count for count in requested_counts if count not in grouped]
        if missing:
            raise ValueError(
                f"requested knot counts are absent from scan pool: {missing}"
            )
    available_total = sum(len(grouped[count]) for count in requested_counts)
    if num_samples > available_total:
        raise ValueError(
            f"requested {num_samples} samples but only {available_total} are available"
        )

    rng = random.Random(int(seed))
    buckets = {count: list(grouped[count]) for count in requested_counts}
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[int] = []
    while len(selected) < num_samples:
        active_counts = [count for count in requested_counts if buckets[count]]
        if not active_counts:
            raise RuntimeError("stratified sampling exhausted unexpectedly")
        rng.shuffle(active_counts)
        for count in active_counts:
            selected.append(buckets[count].pop())
            if len(selected) == num_samples:
                break
    return selected


def timed_call(function: Callable[[], T], *, repeats: int) -> tuple[T, float]:
    """Return the final result and median CPU wall time in milliseconds."""

    if repeats <= 0:
        raise ValueError("timing repeats must be positive")
    result: T | None = None
    durations: list[float] = []
    for _ in range(repeats):
        started_at = time.perf_counter()
        result = function()
        durations.append(1e3 * (time.perf_counter() - started_at))
    assert result is not None
    return result, float(statistics.median(durations))


@torch.no_grad()
def greedy_prune_to_mse_tolerance(
    parameters: torch.Tensor,
    points: torch.Tensor,
    candidate_internal_knots: torch.Tensor,
    *,
    mse_tolerance: float,
    min_internal_knots: int = 0,
    degree: int = 3,
    smoothness_weight: float = 1e-6,
    control_ridge: float = 0.0,
) -> MSEPruningResult:
    """Traditional greedy one-knot deletion using MSE for ranking and stopping."""

    if not math.isfinite(mse_tolerance) or mse_tolerance < 0.0:
        raise ValueError("mse_tolerance must be finite and non-negative")
    if min_internal_knots < 0:
        raise ValueError("min_internal_knots must be non-negative")
    if min_internal_knots > candidate_internal_knots.numel():
        raise ValueError("min_internal_knots cannot exceed candidate count")

    initial_knots = candidate_internal_knots.detach().clone()
    retained = initial_knots.clone()
    retained_indices = torch.arange(retained.numel(), device=retained.device)
    current_fit = refit_bspline_control_points(
        parameters,
        points,
        retained,
        degree=degree,
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        interpolate_endpoints=True,
    )
    trajectory = [float(current_fit.fit_mse)]
    steps: list[MSEPruningStep] = []

    while retained.numel() > min_internal_knots:
        deletion_mse = single_knot_deletion_mse_batch(
            parameters.unsqueeze(0),
            points.unsqueeze(0),
            retained.unsqueeze(0),
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
            interpolate_endpoints=True,
        )[0]
        best_index = int(torch.argmin(deletion_mse).item())
        candidate_knots = torch.cat([retained[:best_index], retained[best_index + 1 :]])
        candidate_fit = refit_bspline_control_points(
            parameters,
            points,
            candidate_knots,
            degree=degree,
            smoothness_weight=smoothness_weight,
            control_ridge=control_ridge,
            interpolate_endpoints=True,
        )
        candidate_mse = float(candidate_fit.fit_mse)
        accepted = candidate_mse <= mse_tolerance
        steps.append(
            MSEPruningStep(
                iteration=len(steps) + 1,
                removed_index=best_index,
                removed_knot=float(retained[best_index]),
                accepted=accepted,
                refit_mse=candidate_mse,
            )
        )
        if not accepted:
            break
        retained = candidate_knots
        retained_indices = torch.cat(
            [retained_indices[:best_index], retained_indices[best_index + 1 :]]
        )
        current_fit = candidate_fit
        trajectory.append(candidate_mse)

    retained_mask = torch.zeros_like(initial_knots, dtype=torch.bool)
    if retained_indices.numel():
        retained_mask[retained_indices] = True
    return MSEPruningResult(
        initial_internal_knots=initial_knots,
        retained_internal_knots=retained,
        retained_mask=retained_mask,
        final_fit=current_fit,
        steps=tuple(steps),
        mse_trajectory=tuple(trajectory),
        mse_tolerance=float(mse_tolerance),
    )


def deployment_geometries(
    output: Mapping[str, torch.Tensor],
    sample_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return proposal knots, deployment knots and authoritative learned mask."""

    if "internal_knots" not in output or "keep_probability" not in output:
        raise KeyError("one-shot output is missing internal_knots/keep_probability")
    proposal = output.get("proposal_internal_knots", output["internal_knots"])
    deployment = output.get("deployment_internal_knots", output["internal_knots"])
    learned_mask = output.get(
        "learned_keep_mask",
        output["keep_probability"] >= 0.5,
    )
    if not (
        proposal.shape == deployment.shape == learned_mask.shape and proposal.ndim == 2
    ):
        raise ValueError("proposal/deployment/mask tensors must share shape [B,K]")
    learned_mask = learned_mask.to(torch.bool)
    return (
        proposal[sample_index],
        deployment[sample_index],
        learned_mask[sample_index],
    )


def compact_internal_knots(internal_knots: torch.Tensor) -> str:
    values = ", ".join(f"{float(value):.3f}" for value in internal_knots)
    return f"u=[{values}]" if values else "u=[]"


def fit_panel(
    fit: BSplineLeastSquaresFit,
    dense_parameters: torch.Tensor,
    *,
    title: str,
    curve_label: str,
    timing_text: str,
    curve_color: str,
) -> ComparisonPanel:
    return ComparisonPanel(
        title=title,
        curve_label=curve_label,
        dense_curve=fit.evaluate(dense_parameters).detach().cpu(),
        reconstructed_points=fit.reconstructed_points.detach().cpu(),
        control_points=fit.control_points.detach().cpu(),
        internal_knots=fit.internal_knots.detach().cpu(),
        knot_vector=fit.knot_vector.detach().cpu(),
        mse=float(fit.fit_mse),
        timing_text=timing_text,
        curve_color=curve_color,
    )


def source_panel(
    sample: Mapping[str, torch.Tensor | int],
    dense_parameters: torch.Tensor,
    *,
    degree: int,
) -> ComparisonPanel:
    control_mask = sample["source_control_mask"]
    knot_mask = sample["source_knot_mask"]
    assert isinstance(control_mask, torch.Tensor)
    assert isinstance(knot_mask, torch.Tensor)
    controls = sample["source_control_points"]
    knot_vector = sample["source_knot_vector"]
    true_parameters = sample["true_params"]
    points = sample["points"]
    assert isinstance(controls, torch.Tensor)
    assert isinstance(knot_vector, torch.Tensor)
    assert isinstance(true_parameters, torch.Tensor)
    assert isinstance(points, torch.Tensor)
    controls = controls[control_mask.to(torch.bool)]
    knot_vector = knot_vector[knot_mask.to(torch.bool)]
    internal_knots = knot_vector[degree + 1 : -(degree + 1)]
    dense_basis = bspline_basis_matrix(
        dense_parameters,
        knot_vector,
        degree,
        num_control_points=controls.shape[0],
    )
    observed_basis = bspline_basis_matrix(
        true_parameters,
        knot_vector,
        degree,
        num_control_points=controls.shape[0],
    )
    reconstructed = observed_basis @ controls
    return ComparisonPanel(
        title="(a) Original source curve and observations",
        curve_label="source B-spline",
        dense_curve=(dense_basis @ controls).detach().cpu(),
        reconstructed_points=reconstructed.detach().cpu(),
        control_points=controls.detach().cpu(),
        internal_knots=internal_knots.detach().cpu(),
        knot_vector=knot_vector.detach().cpu(),
        mse=float(mean_squared_euclidean_error(reconstructed, points)),
        timing_text="source parameters",
        curve_color="#355c9a",
    )


def render_comparison_figure(
    panels: Sequence[ComparisonPanel],
    observed_points: torch.Tensor,
    output_path: Path,
    *,
    degree: int,
    sample_index: int,
    source_count: int,
    canonical_count: int,
    mse_tolerance: float,
    dpi: int,
) -> None:
    """Render one paper-ready four-panel comparison PNG."""

    if len(panels) != 4:
        raise ValueError("exactly four comparison panels are required")
    observed = observed_points.detach().cpu()
    if observed.ndim != 2 or observed.shape[1] != 2:
        raise ValueError("batch comparison plotting supports 2D points only")

    figure, axes = plt.subplots(2, 2, figsize=(16, 12))
    all_geometry = [observed]
    for panel in panels:
        all_geometry.extend([panel.dense_curve, panel.control_points])

    for axis, panel in zip(axes.ravel(), panels, strict=True):
        axis.scatter(
            observed[:, 0],
            observed[:, 1],
            s=12,
            color="#202020",
            alpha=0.65,
            label="observed samples",
            zorder=3,
        )
        axis.plot(
            panel.dense_curve[:, 0],
            panel.dense_curve[:, 1],
            color=panel.curve_color,
            linewidth=2.0,
            label=panel.curve_label,
            zorder=4,
        )
        axis.plot(
            panel.control_points[:, 0],
            panel.control_points[:, 1],
            "o--",
            color="#e08b2c",
            markersize=4.2,
            linewidth=1.0,
            alpha=0.75,
            label="control polygon",
            zorder=2,
        )
        if panel.internal_knots.numel():
            knot_basis = bspline_basis_matrix(
                panel.internal_knots,
                panel.knot_vector,
                degree,
                num_control_points=panel.control_points.shape[0],
            )
            knot_locations = knot_basis @ panel.control_points
            axis.scatter(
                knot_locations[:, 0],
                knot_locations[:, 1],
                marker="D",
                s=34,
                facecolors="none",
                edgecolors="#6a3d9a",
                linewidths=1.2,
                label="internal knots C(u)",
                zorder=5,
            )
        axis.set_title(
            f"{panel.title}\nK={panel.knot_count} | MSE={panel.mse:.5e} | "
            f"{panel.timing_text}",
            fontsize=10.5,
        )
        axis.text(
            0.5,
            -0.15,
            textwrap.fill(compact_internal_knots(panel.internal_knots), width=88),
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
        )
        axis.set_xlabel("normalized x")
        axis.set_ylabel("normalized y")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.18, linewidth=0.6)
        axis.legend(fontsize=7.4, loc="best")

    stacked = torch.cat(all_geometry, dim=0)
    minima = stacked.amin(dim=0)
    maxima = stacked.amax(dim=0)
    span = (maxima - minima).clamp_min(1e-3)
    padding = 0.07 * span
    for axis in axes.ravel():
        axis.set_xlim(float(minima[0] - padding[0]), float(maxima[0] + padding[0]))
        axis.set_ylim(float(minima[1] - padding[1]), float(maxima[1] + padding[1]))

    figure.suptitle(
        f"Sample {sample_index} | source K={source_count} | canonical K="
        f"{canonical_count} | hard-pruning MSE tolerance={mse_tolerance:.3e}",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.012,
        "MSE = mean_i ||C(t_i)-Q_i||_2^2 in normalized coordinates (no square "
        "root). Source uses true parameters; fitted methods use predicted parameters.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    figure.subplots_adjust(hspace=0.40, wspace=0.23, bottom=0.09, top=0.93)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _dataset_config_from_checkpoint(
    checkpoint: Mapping[str, object],
    model_config: Mapping[str, object],
) -> dict[str, object]:
    config = dict(checkpoint.get("dataset_config", {}))
    config.setdefault("num_points", 192)
    config.setdefault("point_dim", int(model_config.get("point_dim", 2)))
    config.setdefault("min_control_points", 8)
    config.setdefault("max_control_points", 24)
    config.setdefault("canonical_knot_tolerance", 5e-3)
    config["return_ground_truth"] = True
    for key in (
        "size",
        "seed",
        "cache_samples",
        "resample_each_epoch",
        "epoch_seed_stride",
    ):
        config.pop(key, None)
    return config


def _scan_counts(
    dataset_config: Mapping[str, object],
    *,
    scan_size: int,
    dataset_seed: int,
    stratify_by: str,
) -> dict[int, int]:
    scan_config = dict(dataset_config)
    if stratify_by == "source":
        # Source K is independent of canonical pruning.  Disabling that
        # expensive step makes scanning hundreds of deterministic samples fast.
        scan_config["canonical_knot_tolerance"] = 0.0
    dataset = SyntheticCubicBSplineDataset(
        size=scan_size,
        seed=dataset_seed,
        cache_samples=False,
        **scan_config,
    )
    index_to_count: dict[int, int] = {}
    report_interval = max(scan_size // 10, 1)
    for index in range(scan_size):
        sample = dataset[index]
        if stratify_by == "source":
            count = int(sample["source_internal_knot_count"])
        else:
            mask = sample["true_internal_knot_mask"]
            assert isinstance(mask, torch.Tensor)
            count = int(mask.sum().item())
        index_to_count[index] = count
        if (index + 1) % report_interval == 0 or index + 1 == scan_size:
            print(f"  scanning counts: {index + 1}/{scan_size}", flush=True)
    return index_to_count


def _write_manifests(
    output_dir: Path,
    report: dict[str, object],
    flat_rows: list[dict[str, object]],
) -> None:
    json_path = output_dir / "comparison_manifest.json"
    csv_path = output_dir / "comparison_manifest.csv"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    if not flat_rows:
        raise ValueError("manifest rows cannot be empty")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-render source/all-proposal/v11-LearnedKeep/traditional-hard "
            "B-spline comparisons using mean squared Euclidean error."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-figures", type=int, default=8)
    parser.add_argument("--scan-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--selection-seed", type=int, default=12345)
    parser.add_argument(
        "--stratify-by",
        choices=("source", "canonical"),
        default="source",
        help="Knot-count label used for random stratified sampling.",
    )
    parser.add_argument(
        "--knot-counts",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of source/canonical internal-knot counts.",
    )
    parser.add_argument(
        "--mse-tolerance",
        type=float,
        default=None,
        help=(
            "Hard-pruning MSE bound. By default the historical checkpoint RMS "
            "tolerance is squared."
        ),
    )
    parser.add_argument("--min-hard-knots", type=int, default=0)
    parser.add_argument("--smoothness-weight", type=float, default=1e-6)
    parser.add_argument("--control-ridge", type=float, default=0.0)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--dense-points", type=int, default=500)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.num_figures <= 0 or args.scan_size <= 0:
        parser.error("--num-figures and --scan-size must be positive")
    if args.num_figures > args.scan_size:
        parser.error("--num-figures cannot exceed --scan-size")
    if args.timing_repeats <= 0 or args.dense_points < 2 or args.dpi <= 0:
        parser.error("timing repeats/dense points/dpi must be positive")
    if args.smoothness_weight < 0.0 or args.control_ridge < 0.0:
        parser.error("solver regularization weights must be non-negative")
    if args.min_hard_knots < 0:
        parser.error("--min-hard-knots must be non-negative")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    try:
        mse_tolerance = resolve_mse_tolerance(checkpoint, args.mse_tolerance)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    model, model_config, _ = build_model_from_checkpoint(checkpoint)
    if model_config.get("structure_mode") != "candidate_pruning_one_shot":
        parser.error("batch comparison requires a v8-v11 one-shot pruning checkpoint")
    model.eval()
    dataset_config = _dataset_config_from_checkpoint(checkpoint, model_config)
    if int(dataset_config["point_dim"]) != 2:
        parser.error("batch comparison PNGs currently require a 2D checkpoint")

    index_to_count = _scan_counts(
        dataset_config,
        scan_size=args.scan_size,
        dataset_seed=args.seed,
        stratify_by=args.stratify_by,
    )
    try:
        selected_indices = stratified_random_indices(
            index_to_count,
            args.num_figures,
            seed=args.selection_seed,
            allowed_counts=args.knot_counts,
        )
    except ValueError as error:
        parser.error(str(error))

    output_dir = args.output_dir
    manifest_paths = (
        output_dir / "comparison_manifest.json",
        output_dir / "comparison_manifest.csv",
    )
    if not args.overwrite and any(path.exists() for path in manifest_paths):
        parser.error(f"output manifest already exists in {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SyntheticCubicBSplineDataset(
        size=args.scan_size,
        seed=args.seed,
        cache_samples=False,
        **dataset_config,
    )
    dense_parameters = torch.linspace(0.0, 1.0, args.dense_points)

    # Warm the network once so the first reported sample does not include lazy
    # backend initialization.  Timings remain batch=1 CPU wall times.
    warmup_sample = dataset[selected_indices[0]]
    warmup_points = warmup_sample["points"]
    assert isinstance(warmup_points, torch.Tensor)
    with torch.no_grad():
        model(warmup_points.unsqueeze(0))

    detailed_rows: list[dict[str, object]] = []
    flat_rows: list[dict[str, object]] = []
    for figure_index, sample_index in enumerate(selected_indices, start=1):
        sample = dataset[sample_index]
        points = sample["points"]
        assert isinstance(points, torch.Tensor)
        batched_points = points.unsqueeze(0)

        def run_network() -> dict[str, torch.Tensor]:
            with torch.no_grad():
                return model(batched_points)

        output, network_ms = timed_call(
            run_network,
            repeats=args.timing_repeats,
        )
        proposal_knots, deployment_knots, learned_mask = deployment_geometries(output)
        parameters = output["params"][0]
        learned_knots = deployment_knots[learned_mask]

        all_fit, all_refit_ms = timed_call(
            lambda: refit_bspline_control_points(
                parameters,
                points,
                proposal_knots,
                degree=model.degree,
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
                interpolate_endpoints=True,
            ),
            repeats=args.timing_repeats,
        )
        learned_fit, learned_refit_ms = timed_call(
            lambda: refit_bspline_control_points(
                parameters,
                points,
                learned_knots,
                degree=model.degree,
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
                interpolate_endpoints=True,
            ),
            repeats=args.timing_repeats,
        )
        hard_result, hard_prune_ms = timed_call(
            lambda: greedy_prune_to_mse_tolerance(
                parameters,
                points,
                proposal_knots,
                mse_tolerance=mse_tolerance,
                min_internal_knots=args.min_hard_knots,
                degree=model.degree,
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
            ),
            repeats=args.timing_repeats,
        )

        source_count = int(sample["source_internal_knot_count"])
        canonical_mask = sample["true_internal_knot_mask"]
        assert isinstance(canonical_mask, torch.Tensor)
        canonical_count = int(canonical_mask.sum().item())
        panels = (
            source_panel(sample, dense_parameters, degree=model.degree),
            fit_panel(
                all_fit,
                dense_parameters,
                title="(b) Redundant all-proposal refit",
                curve_label="all-proposal B-spline",
                timing_text=(
                    f"net={network_ms:.2f} ms, refit={all_refit_ms:.2f} ms, "
                    f"total={network_ms + all_refit_ms:.2f} ms"
                ),
                curve_color="#6f6f6f",
            ),
            fit_panel(
                learned_fit,
                dense_parameters,
                title="(c) Ours: one-shot LearnedKeep + position update",
                curve_label="learned deployment B-spline",
                timing_text=(
                    f"net={network_ms:.2f} ms, refit={learned_refit_ms:.2f} ms, "
                    f"total={network_ms + learned_refit_ms:.2f} ms"
                ),
                curve_color="#198a77",
            ),
            fit_panel(
                hard_result.final_fit,
                dense_parameters,
                title="(d) Traditional greedy hard pruning from same proposal",
                curve_label="hard-pruned B-spline",
                timing_text=(
                    f"net={network_ms:.2f} ms, prune={hard_prune_ms:.2f} ms, "
                    f"total={network_ms + hard_prune_ms:.2f} ms"
                ),
                curve_color="#c94b4b",
            ),
        )
        filename = (
            f"comparison_{figure_index:03d}_sourceK{source_count:02d}_"
            f"canonicalK{canonical_count:02d}_sample{sample_index:05d}.png"
        )
        output_path = output_dir / filename
        if output_path.exists() and not args.overwrite:
            parser.error(f"figure already exists: {output_path}; use --overwrite")
        render_comparison_figure(
            panels,
            points,
            output_path,
            degree=model.degree,
            sample_index=sample_index,
            source_count=source_count,
            canonical_count=canonical_count,
            mse_tolerance=mse_tolerance,
            dpi=args.dpi,
        )

        source_knots = panels[0].internal_knots.tolist()
        proposal_values = proposal_knots.detach().cpu().tolist()
        learned_values = learned_knots.detach().cpu().tolist()
        hard_values = hard_result.retained_internal_knots.detach().cpu().tolist()
        detailed_row: dict[str, object] = {
            "figure": filename,
            "sample_index": sample_index,
            "sample_seed": args.seed + sample_index,
            "stratification_count": index_to_count[sample_index],
            "source_knot_count": source_count,
            "canonical_knot_count": canonical_count,
            "proposal_knot_count": int(proposal_knots.numel()),
            "learned_knot_count": int(learned_knots.numel()),
            "hard_knot_count": hard_result.final_count,
            "source_mse": panels[0].mse,
            "proposal_mse": float(all_fit.fit_mse),
            "learned_mse": float(learned_fit.fit_mse),
            "hard_mse": float(hard_result.final_fit.fit_mse),
            "hard_threshold_satisfied": hard_result.threshold_satisfied,
            "hard_accepted_deletions": hard_result.accepted_deletions,
            "network_forward_ms": network_ms,
            "proposal_refit_ms": all_refit_ms,
            "proposal_total_ms": network_ms + all_refit_ms,
            "learned_refit_ms": learned_refit_ms,
            "learned_total_ms": network_ms + learned_refit_ms,
            "hard_prune_ms": hard_prune_ms,
            "hard_total_ms": network_ms + hard_prune_ms,
            "source_internal_knots": source_knots,
            "proposal_internal_knots": proposal_values,
            "learned_internal_knots": learned_values,
            "hard_internal_knots": hard_values,
            "hard_mse_trajectory": list(hard_result.mse_trajectory),
        }
        detailed_rows.append(detailed_row)
        flat_rows.append(
            {
                **{
                    key: value
                    for key, value in detailed_row.items()
                    if not isinstance(value, list)
                },
                "source_internal_knots": json.dumps(source_knots),
                "proposal_internal_knots": json.dumps(proposal_values),
                "learned_internal_knots": json.dumps(learned_values),
                "hard_internal_knots": json.dumps(hard_values),
                "hard_mse_trajectory": json.dumps(list(hard_result.mse_trajectory)),
            }
        )
        print(
            f"[{figure_index}/{len(selected_indices)}] sample={sample_index} "
            f"K source/proposal/learned/hard={source_count}/"
            f"{proposal_knots.numel()}/{learned_knots.numel()}/"
            f"{hard_result.final_count} | MSE learned/hard="
            f"{float(learned_fit.fit_mse):.5e}/{float(hard_result.final_fit.fit_mse):.5e}",
            flush=True,
        )

    report: dict[str, object] = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_objective_version": checkpoint.get("objective_version"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "metric": "mean_squared_euclidean_error",
        "metric_definition": MSE_DEFINITION,
        "square_root_applied": False,
        "coordinate_domain": "normalized",
        "mse_tolerance": mse_tolerance,
        "dataset_seed": args.seed,
        "selection_seed": args.selection_seed,
        "scan_size": args.scan_size,
        "stratify_by": args.stratify_by,
        "requested_knot_counts": args.knot_counts,
        "selected_indices": selected_indices,
        "timing": {
            "device": "cpu",
            "batch_size": 1,
            "repeats": args.timing_repeats,
            "statistic": "median_wall_time_ms",
            "network_time_is_shared": True,
        },
        "methods": {
            "source": "original generated B-spline evaluated at true parameters",
            "proposal": "all immutable proposal knots + one standard refit",
            "learned": (
                "one network forward + authoritative LearnedKeep mask + v11 "
                "deployment positions + one standard refit"
            ),
            "hard": (
                "same immutable proposal + exhaustive greedy one-knot deletion "
                "with MSE stopping + standard refits"
            ),
        },
        "samples": detailed_rows,
    }
    _write_manifests(output_dir, report, flat_rows)
    print(f"Saved {len(detailed_rows)} PNG figures to: {output_dir}")
    print(f"Saved manifest: {output_dir / 'comparison_manifest.json'}")
    print(f"Saved table: {output_dir / 'comparison_manifest.csv'}")


if __name__ == "__main__":
    main()
