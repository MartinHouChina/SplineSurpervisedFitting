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

from spline_fitting.checkpointing import build_model_from_checkpoint  # noqa: E402
from spline_fitting.data.synthetic import (  # noqa: E402
    SyntheticCubicBSplineDataset,
    bspline_basis_matrix,
)
from spline_fitting.evaluation.bspline_inference import (  # noqa: E402
    BSplineLeastSquaresFit,
    refit_bspline_control_points,
)
from spline_fitting.evaluation.knot_diagnostics import (  # noqa: E402
    warp_internal_knots_to_parameterization,
)
from spline_fitting.evaluation.timing import (  # noqa: E402
    measure_synchronized_wall_time,
)
from spline_fitting.evaluation.verified_knot_repair import (  # noqa: E402
    VerifiedKnotRepairResult,
    verified_confidence_repair,
)
from spline_fitting.spline.bspline_deletion_teacher import (  # noqa: E402
    single_knot_deletion_mse_batch,
)


T = TypeVar("T")
MSE_DEFINITION = "mean_i ||C(t_i) - Q_i||_2^2"


def deployment_network_forward(
    model: torch.nn.Module,
    points: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Materialize one-shot structure without the training-only final proxy fit."""

    forward_deployment = getattr(model, "forward_deployment", None)
    if callable(forward_deployment):
        return forward_deployment(points)
    return model(points)


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


@dataclass(frozen=True)
class VerifiedVisualizationResult:
    """Network output and exact adaptive deployment used by panel (c)."""

    output: dict[str, torch.Tensor]
    repair: VerifiedKnotRepairResult


@dataclass(frozen=True)
class HardTimingReproductionDiagnostics:
    """Difference between authoritative Hard geometry and a timed E2E replay."""

    count_equal: bool
    retained_mask_equal: bool
    retained_knots_allclose: bool
    retained_knots_max_abs_difference: float | None
    mse_abs_difference: float
    threshold_satisfied_equal: bool

    @property
    def structurally_consistent(self) -> bool:
        return (
            self.count_equal
            and self.retained_mask_equal
            and self.retained_knots_allclose
        )


def compare_hard_timing_reproduction(
    authoritative: MSEPruningResult,
    timed_replay: MSEPruningResult,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> HardTimingReproductionDiagnostics:
    """Diagnose an independently timed replay without changing plot geometry.

    Re-running the network solely to measure an end-to-end boundary can produce
    tiny floating-point changes, and an unstable legacy checkpoint may even
    cross a discrete deletion boundary.  Neither case should abort a long PNG
    batch.  The first, already materialized ``authoritative`` result remains
    the plotted Hard baseline; this helper only records replay differences.
    """

    if rtol < 0.0 or atol < 0.0:
        raise ValueError("Hard replay comparison tolerances must be non-negative")
    count_equal = authoritative.final_count == timed_replay.final_count
    retained_mask_equal = (
        authoritative.retained_mask.shape == timed_replay.retained_mask.shape
        and torch.equal(authoritative.retained_mask, timed_replay.retained_mask)
    )
    if count_equal:
        authoritative_knots = authoritative.retained_internal_knots
        replay_knots = timed_replay.retained_internal_knots
        retained_knots_allclose = bool(
            torch.allclose(
                authoritative_knots,
                replay_knots,
                rtol=rtol,
                atol=atol,
            )
        )
        retained_knots_max_abs_difference = (
            0.0
            if authoritative_knots.numel() == 0
            else float((authoritative_knots - replay_knots).abs().max())
        )
    else:
        retained_knots_allclose = False
        retained_knots_max_abs_difference = None
    mse_abs_difference = abs(
        float(authoritative.final_fit.fit_mse) - float(timed_replay.final_fit.fit_mse)
    )
    return HardTimingReproductionDiagnostics(
        count_equal=count_equal,
        retained_mask_equal=retained_mask_equal,
        retained_knots_allclose=retained_knots_allclose,
        retained_knots_max_abs_difference=retained_knots_max_abs_difference,
        mse_abs_difference=mse_abs_difference,
        threshold_satisfied_equal=(
            authoritative.threshold_satisfied == timed_replay.threshold_satisfied
        ),
    )


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


def resolve_verified_refit_device(
    requested: str,
    *,
    model_device: torch.device,
) -> torch.device:
    """Resolve the exact small-system refit device used by verified mode."""

    if requested not in {"auto", "cpu", "model"}:
        raise ValueError("verified refit device must be one of auto, cpu, model")
    return torch.device(model_device) if requested == "model" else torch.device("cpu")


def canonical_chord_parameters(chord_parameters: torch.Tensor) -> torch.Tensor:
    """Clamp floating-point endpoint drift while preserving chord plateaus."""

    if chord_parameters.ndim != 1 or not chord_parameters.is_floating_point():
        raise ValueError("chord parameters must be floating-point with shape [M]")
    if not torch.isfinite(chord_parameters).all():
        raise ValueError("chord parameters must be finite")
    result = chord_parameters.detach().clone().clamp(0.0, 1.0)
    result[0] = 0.0
    result[-1] = 1.0
    if torch.any(result[1:] < result[:-1]):
        raise ValueError("chord parameters must be non-decreasing")
    return result


def comparison_parameterization_geometry(
    parameters: torch.Tensor,
    proposal_knots: torch.Tensor,
    chord_parameters: torch.Tensor,
    *,
    ours_deployment: str,
    verified_parameterization: str,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return the shared Redundant/Greedy geometry for the four-panel figure.

    Full-chord verified deployment is a deliberate controlled comparison: all
    learned-proposal identities are mapped to the chord-length domain before
    either the redundant refit or greedy deletion is run.  The conditional
    chord fallback remains an Ours-only guard and therefore leaves the two
    reference panels in the original network parameterization.
    """

    if ours_deployment not in {"learned", "verified"}:
        raise ValueError("ours deployment must be learned or verified")
    if verified_parameterization not in {"network", "chord-fallback", "chord"}:
        raise ValueError("invalid verified parameterization policy")
    shared_chord_domain = (
        ours_deployment == "verified" and verified_parameterization == "chord"
    )
    if not shared_chord_domain:
        return parameters, proposal_knots, False
    return (
        chord_parameters,
        warp_internal_knots_to_parameterization(
            proposal_knots,
            parameters,
            chord_parameters,
        ),
        True,
    )


@torch.no_grad()
def run_verified_visualization_deployment(
    model: torch.nn.Module,
    batched_points: torch.Tensor,
    points: torch.Tensor,
    chord_parameters: torch.Tensor,
    *,
    mse_tolerance: float,
    min_internal_knots: int,
    smoothness_weight: float,
    control_ridge: float,
    compact: bool,
    hard_fallback: bool,
    residual_fallback: bool,
    max_residual_insertions: int,
    residual_min_gap: float,
    parameterization_policy: str,
    refit_device: torch.device | str,
) -> VerifiedVisualizationResult:
    """Run one network forward plus the exact adaptive verified deployment."""

    if not math.isfinite(mse_tolerance) or mse_tolerance < 0.0:
        raise ValueError("MSE tolerance must be finite and non-negative")
    output = deployment_network_forward(model, batched_points)
    proposal, deployment, learned_mask = deployment_geometries(output)
    target_device = torch.device(refit_device)
    repair = verified_confidence_repair(
        output["params"][0].detach().to(target_device),
        points.detach().to(target_device),
        proposal.detach().to(target_device),
        deployment.detach().to(target_device),
        learned_mask.detach().to(target_device),
        output["keep_probability"][0].detach().to(target_device),
        fit_tolerance_rms=math.sqrt(mse_tolerance),
        min_internal_knots=min_internal_knots,
        degree=int(model.degree),
        smoothness_weight=smoothness_weight,
        control_ridge=control_ridge,
        interpolate_endpoints=True,
        compact=compact,
        hard_fallback=hard_fallback,
        residual_fallback=residual_fallback,
        max_residual_insertions=max_residual_insertions,
        residual_min_gap=residual_min_gap,
        alternate_parameters=chord_parameters.detach().to(target_device),
        parameterization_policy=parameterization_policy,
    )
    return VerifiedVisualizationResult(output=output, repair=repair)


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


def fixed_samples_per_knot_count(
    index_to_count: Mapping[int, int],
    requested_counts: Sequence[int],
    *,
    samples_per_count: int,
    seed: int,
) -> list[int]:
    """Draw exactly ``samples_per_count`` deterministic samples per K stratum."""

    if samples_per_count <= 0:
        raise ValueError("samples_per_count must be positive")
    counts = list(dict.fromkeys(int(value) for value in requested_counts))
    if not counts:
        raise ValueError("requested_counts cannot be empty")

    grouped: dict[int, list[int]] = {count: [] for count in counts}
    for raw_index, raw_count in index_to_count.items():
        count = int(raw_count)
        if count in grouped:
            grouped[count].append(int(raw_index))

    missing = [count for count in counts if len(grouped[count]) < samples_per_count]
    if missing:
        availability = {count: len(grouped[count]) for count in missing}
        raise ValueError(
            "insufficient samples in requested knot-count strata: "
            f"{availability}; increase --scan-size"
        )

    rng = random.Random(int(seed))
    selected: list[int] = []
    for count in counts:
        bucket = grouped[count]
        rng.shuffle(bucket)
        selected.extend(bucket[:samples_per_count])
    return selected


def validate_explicit_sample_indices(values: Sequence[int]) -> list[int]:
    """Validate exact deterministic sample IDs for single or batch rendering."""

    indices = [int(value) for value in values]
    if not indices:
        raise ValueError("sample indices cannot be empty")
    if any(index < 0 for index in indices):
        raise ValueError("sample indices must be non-negative")
    if len(indices) != len(set(indices)):
        raise ValueError("sample indices must be unique")
    return indices


def source_knot_count_from_seed(
    dataset_config: Mapping[str, object],
    *,
    dataset_seed: int,
    sample_index: int,
    degree: int = 3,
) -> int:
    """Recover source K without materializing and canonicalizing a curve.

    Both legacy and certified generators draw the source control count as the
    first value from the per-index generator.  Keeping this lightweight scan
    path matters for certified datasets, whose canonical tolerance cannot be
    set to zero merely to accelerate source-K stratification.
    """

    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    minimum = int(dataset_config["min_control_points"])
    maximum = int(dataset_config["max_control_points"])
    if minimum < degree + 1 or maximum < minimum:
        raise ValueError("invalid control-point range in dataset config")
    generator = torch.Generator().manual_seed(int(dataset_seed) + sample_index)
    control_count = int(
        torch.randint(minimum, maximum + 1, (1,), generator=generator).item()
    )
    return control_count - degree - 1


def timed_call(
    function: Callable[[], T],
    *,
    repeats: int,
    warmup_repeats: int = 0,
) -> tuple[T, float]:
    """Return the final result and median CPU wall time in milliseconds."""

    if repeats <= 0:
        raise ValueError("timing repeats must be positive")
    if warmup_repeats < 0:
        raise ValueError("timing warmup repeats must be non-negative")
    for _ in range(warmup_repeats):
        function()
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
        timing_text="reference (not timed)",
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
    parameterization_note: str = "Fitted methods use predicted parameters.",
    timing_scope_note: str = (
        "Timing scopes are asymmetric: Ours is end-to-end; Hard is pruning-only."
    ),
) -> None:
    """Render one paper-ready comparison PNG with four to six panels."""

    if not 4 <= len(panels) <= 6:
        raise ValueError("four to six comparison panels are required")
    observed = observed_points.detach().cpu()
    if observed.ndim != 2 or observed.shape[1] != 2:
        raise ValueError("batch comparison plotting supports 2D points only")

    columns = 2 if len(panels) == 4 else 3
    figure, axes = plt.subplots(2, columns, figsize=(8 * columns, 12))
    all_geometry = [observed]
    for panel in panels:
        all_geometry.extend([panel.dense_curve, panel.control_points])

    flat_axes = axes.ravel()
    for axis, panel in zip(flat_axes, panels, strict=False):
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
            f"{panel.title}\nK={panel.knot_count} | MSE={panel.mse:.5e}\n"
            f"{panel.timing_text}",
            fontsize=10.5,
        )
        axis.text(
            0.5,
            -0.25,
            textwrap.fill(compact_internal_knots(panel.internal_knots), width=88),
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
        )
        axis.set_xlabel("normalized x", labelpad=2)
        axis.set_ylabel("normalized y")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.18, linewidth=0.6)
        axis.legend(fontsize=7.4, loc="best")

    stacked = torch.cat(all_geometry, dim=0)
    minima = stacked.amin(dim=0)
    maxima = stacked.amax(dim=0)
    span = (maxima - minima).clamp_min(1e-3)
    padding = 0.07 * span
    for axis in flat_axes[: len(panels)]:
        axis.set_xlim(float(minima[0] - padding[0]), float(maxima[0] + padding[0]))
        axis.set_ylim(float(minima[1] - padding[1]), float(maxima[1] + padding[1]))
    for axis in flat_axes[len(panels) :]:
        axis.set_visible(False)

    figure.suptitle(
        f"Sample {sample_index} | source K={source_count} | canonical K="
        f"{canonical_count} | deployment MSE tolerance={mse_tolerance:.3e}",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.012,
        "MSE = mean_i ||C(t_i)-Q_i||_2^2 in normalized coordinates (no square "
        f"root). Source uses true parameters; {parameterization_note} "
        "Internal u is printed per panel; full clamped U is in the manifest. "
        f"{timing_scope_note}",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    figure.subplots_adjust(hspace=0.54, wspace=0.23, bottom=0.12, top=0.91)
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
    if stratify_by == "source":
        return {
            index: source_knot_count_from_seed(
                dataset_config,
                dataset_seed=dataset_seed,
                sample_index=index,
            )
            for index in range(scan_size)
        }
    scan_config = dict(dataset_config)
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
            "Batch-render Original/Redundant/Ours/Greedy B-spline comparisons. "
            "Ours can be the historical LearnedKeep result or adaptive verified "
            "deployment; errors are mean squared Euclidean distances."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--num-figures",
        type=int,
        default=8,
        help="Balanced sample count; ignored with --samples-per-knot-count.",
    )
    parser.add_argument("--scan-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--selection-seed", type=int, default=12345)
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Render these exact deterministic sample IDs. One value produces "
            "one four-panel PNG; multiple values produce an explicit batch. "
            "This bypasses stratified random selection."
        ),
    )
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
        "--samples-per-knot-count",
        type=int,
        default=None,
        help=(
            "Draw exactly this many examples for every requested K. When "
            "--knot-counts is omitted, use the inclusive min/max K range."
        ),
    )
    parser.add_argument(
        "--min-knot-count",
        type=int,
        default=None,
        help="Inclusive lower K for fixed-per-stratum mode.",
    )
    parser.add_argument(
        "--max-knot-count",
        type=int,
        default=None,
        help="Inclusive upper K for fixed-per-stratum mode; defaults to dataset max.",
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
    parser.add_argument(
        "--ours-deployment",
        choices=("learned", "verified"),
        default="learned",
        help=(
            "Panel (c): learned preserves the historical pure one-shot result; "
            "verified adds exact checking and conditional adaptive repair."
        ),
    )
    parser.add_argument(
        "--verified-compact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Greedily compact every feasible verified subset.",
    )
    parser.add_argument(
        "--verified-hard-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow proposal-only hard fallback after verified repair failure.",
    )
    parser.add_argument(
        "--verified-residual-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow residual-guided insertion after full-proposal failure.",
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
        "--verified-parameterization",
        choices=("network", "chord-fallback", "chord"),
        default="chord-fallback",
        help=(
            "Exact-refit parameterization. Full chord mode also maps Redundant "
            "and Greedy to the same chord domain for a fair four-panel comparison."
        ),
    )
    parser.add_argument(
        "--verified-refit-device",
        choices=("auto", "cpu", "model"),
        default="auto",
        help="auto/cpu use CPU exact refits; model follows the network device.",
    )
    parser.add_argument("--min-hard-knots", type=int, default=0)
    parser.add_argument("--smoothness-weight", type=float, default=1e-6)
    parser.add_argument("--control-ridge", type=float, default=0.0)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument(
        "--timing-warmups",
        type=int,
        default=1,
        help="Unmeasured warmup calls for every independently timed pipeline.",
    )
    parser.add_argument(
        "--timing-scope",
        choices=("historical-asymmetric", "end-to-end"),
        default="historical-asymmetric",
        help=(
            "Time printed in panel (d). historical-asymmetric preserves the "
            "old pruning-only Hard number; end-to-end includes the same network "
            "proposal/parameter materialization boundary used by Ours. Both "
            "component values are saved in the manifest."
        ),
    )
    parser.add_argument("--dense-points", type=int, default=500)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.num_figures <= 0 or args.scan_size <= 0:
        parser.error("--num-figures and --scan-size must be positive")
    if (
        args.sample_indices is None
        and args.samples_per_knot_count is None
        and args.num_figures > args.scan_size
    ):
        parser.error("--num-figures cannot exceed --scan-size")
    if args.sample_indices is not None and any(
        value is not None
        for value in (
            args.knot_counts,
            args.samples_per_knot_count,
            args.min_knot_count,
            args.max_knot_count,
        )
    ):
        parser.error(
            "--sample-indices cannot be combined with knot-count sampling options"
        )
    if args.samples_per_knot_count is not None and args.samples_per_knot_count <= 0:
        parser.error("--samples-per-knot-count must be positive")
    if args.samples_per_knot_count is None and (
        args.min_knot_count is not None or args.max_knot_count is not None
    ):
        parser.error(
            "--min-knot-count/--max-knot-count require --samples-per-knot-count"
        )
    if args.knot_counts is not None and (
        args.min_knot_count is not None or args.max_knot_count is not None
    ):
        parser.error("use either --knot-counts or the min/max K range, not both")
    if (
        args.timing_repeats <= 0
        or args.timing_warmups < 0
        or args.dense_points < 2
        or args.dpi <= 0
    ):
        parser.error(
            "timing repeats/dense points/dpi must be positive and timing "
            "warmups must be non-negative"
        )
    if args.smoothness_weight < 0.0 or args.control_ridge < 0.0:
        parser.error("solver regularization weights must be non-negative")
    if args.min_hard_knots < 0:
        parser.error("--min-hard-knots must be non-negative")
    if args.verified_max_residual_insertions < 0:
        parser.error("--verified-max-residual-insertions must be non-negative")
    if args.verified_residual_min_gap is not None and (
        not math.isfinite(args.verified_residual_min_gap)
        or args.verified_residual_min_gap < 0.0
    ):
        parser.error("--verified-residual-min-gap must be finite and non-negative")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    try:
        mse_tolerance = resolve_mse_tolerance(checkpoint, args.mse_tolerance)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    model, model_config, _ = build_model_from_checkpoint(checkpoint)
    if model_config.get("structure_mode") != "candidate_pruning_one_shot":
        parser.error("batch comparison requires a v8-v12 one-shot pruning checkpoint")
    model.eval()
    model_device = next(model.parameters()).device
    verified_refit_device = resolve_verified_refit_device(
        args.verified_refit_device,
        model_device=model_device,
    )
    verified_residual_min_gap = (
        float(model_config.get("min_knot_gap", 1e-3))
        if args.verified_residual_min_gap is None
        else float(args.verified_residual_min_gap)
    )
    raw_deployment_config = checkpoint.get("deployment_config", {})
    deployment_config = (
        raw_deployment_config if isinstance(raw_deployment_config, Mapping) else {}
    )
    verified_min_internal_knots = int(deployment_config.get("min_internal_knots", 0))
    dataset_config = _dataset_config_from_checkpoint(checkpoint, model_config)
    if int(dataset_config["point_dim"]) != 2:
        parser.error("batch comparison PNGs currently require a 2D checkpoint")

    requested_counts = args.knot_counts
    if args.sample_indices is not None:
        try:
            selected_indices = validate_explicit_sample_indices(args.sample_indices)
        except ValueError as error:
            parser.error(str(error))
        index_to_count: dict[int, int] = {}
        effective_dataset_size = max(args.scan_size, max(selected_indices) + 1)
    else:
        index_to_count = _scan_counts(
            dataset_config,
            scan_size=args.scan_size,
            dataset_seed=args.seed,
            stratify_by=args.stratify_by,
        )
        effective_dataset_size = args.scan_size
        if args.samples_per_knot_count is not None and requested_counts is None:
            discovered_counts = sorted(set(index_to_count.values()))
            if args.stratify_by == "source":
                default_min_count = max(
                    0,
                    int(dataset_config["min_control_points"]) - model.degree - 1,
                )
                default_max_count = max(
                    0,
                    int(dataset_config["max_control_points"]) - model.degree - 1,
                )
            else:
                default_min_count = discovered_counts[0]
                default_max_count = discovered_counts[-1]
            minimum_count = (
                default_min_count
                if args.min_knot_count is None
                else args.min_knot_count
            )
            maximum_count = (
                default_max_count
                if args.max_knot_count is None
                else args.max_knot_count
            )
            if minimum_count < 0 or maximum_count < minimum_count:
                parser.error("invalid inclusive knot-count range")
            requested_counts = list(range(minimum_count, maximum_count + 1))

        try:
            if args.samples_per_knot_count is None:
                selected_indices = stratified_random_indices(
                    index_to_count,
                    args.num_figures,
                    seed=args.selection_seed,
                    allowed_counts=requested_counts,
                )
            else:
                assert requested_counts is not None
                selected_indices = fixed_samples_per_knot_count(
                    index_to_count,
                    requested_counts,
                    samples_per_count=args.samples_per_knot_count,
                    seed=args.selection_seed,
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
        size=effective_dataset_size,
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
        deployment_network_forward(model, warmup_points.unsqueeze(0))

    detailed_rows: list[dict[str, object]] = []
    flat_rows: list[dict[str, object]] = []
    for figure_index, sample_index in enumerate(selected_indices, start=1):
        sample = dataset[sample_index]
        points = sample["points"]
        chord_parameters = sample["chord_params"]
        assert isinstance(points, torch.Tensor)
        assert isinstance(chord_parameters, torch.Tensor)
        chord_parameters = canonical_chord_parameters(chord_parameters)
        batched_points = points.unsqueeze(0)

        def run_network() -> dict[str, torch.Tensor]:
            with torch.no_grad():
                return deployment_network_forward(model, batched_points)

        network_timing = measure_synchronized_wall_time(
            run_network,
            warmup_repeats=args.timing_warmups,
            timing_repeats=args.timing_repeats,
            synchronization_device=model_device,
        )
        output = network_timing.result
        network_ms = network_timing.latency.p50_ms
        proposal_knots, deployment_knots, learned_mask = deployment_geometries(output)
        parameters = output["params"][0]
        learned_knots = deployment_knots[learned_mask]
        (
            comparison_parameters,
            comparison_proposal_knots,
            shared_chord_domain,
        ) = comparison_parameterization_geometry(
            parameters,
            proposal_knots,
            chord_parameters,
            ours_deployment=args.ours_deployment,
            verified_parameterization=args.verified_parameterization,
        )

        def run_learned_deployment() -> tuple[
            dict[str, torch.Tensor],
            BSplineLeastSquaresFit,
        ]:
            deployment_output = run_network()
            _, deployed_positions, deployed_mask = deployment_geometries(
                deployment_output
            )
            deployment_fit = refit_bspline_control_points(
                deployment_output["params"][0],
                points,
                torch.sort(deployed_positions[deployed_mask]).values,
                degree=model.degree,
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
                interpolate_endpoints=True,
            )
            return deployment_output, deployment_fit

        (_, learned_fit), learned_deployment_ms = timed_call(
            run_learned_deployment,
            repeats=args.timing_repeats,
            warmup_repeats=args.timing_warmups,
        )

        verified_result: VerifiedKnotRepairResult | None = None
        if args.ours_deployment == "verified":

            def run_verified_deployment() -> VerifiedVisualizationResult:
                return run_verified_visualization_deployment(
                    model,
                    batched_points,
                    points,
                    chord_parameters,
                    mse_tolerance=mse_tolerance,
                    min_internal_knots=verified_min_internal_knots,
                    smoothness_weight=args.smoothness_weight,
                    control_ridge=args.control_ridge,
                    compact=args.verified_compact,
                    hard_fallback=args.verified_hard_fallback,
                    residual_fallback=args.verified_residual_fallback,
                    max_residual_insertions=(args.verified_max_residual_insertions),
                    residual_min_gap=verified_residual_min_gap,
                    parameterization_policy=args.verified_parameterization,
                    refit_device=verified_refit_device,
                )

            verified_visualization, ours_deployment_ms = timed_call(
                run_verified_deployment,
                repeats=args.timing_repeats,
                warmup_repeats=args.timing_warmups,
            )
            verified_result = verified_visualization.repair
            ours_fit = verified_result.final_fit
            ours_parameters = verified_result.final_parameters
            ours_parameterization = verified_result.final_parameterization
        else:
            ours_deployment_ms = learned_deployment_ms
            ours_fit = learned_fit
            ours_parameters = parameters
            ours_parameterization = "network_predicted"
        ours_knots = ours_fit.internal_knots

        all_fit, all_refit_ms = timed_call(
            lambda: refit_bspline_control_points(
                comparison_parameters,
                points,
                comparison_proposal_knots,
                degree=model.degree,
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
                interpolate_endpoints=True,
            ),
            repeats=args.timing_repeats,
            warmup_repeats=args.timing_warmups,
        )
        _, learned_refit_ms = timed_call(
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
            warmup_repeats=args.timing_warmups,
        )
        hard_result, hard_prune_ms = timed_call(
            lambda: greedy_prune_to_mse_tolerance(
                comparison_parameters,
                points,
                comparison_proposal_knots,
                mse_tolerance=mse_tolerance,
                min_internal_knots=args.min_hard_knots,
                degree=model.degree,
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
            ),
            repeats=args.timing_repeats,
            warmup_repeats=args.timing_warmups,
        )

        def run_hard_end_to_end() -> MSEPruningResult:
            hard_output = deployment_network_forward(model, batched_points)
            hard_proposal, _, _ = deployment_geometries(hard_output)
            hard_parameters, hard_proposal, _ = comparison_parameterization_geometry(
                hard_output["params"][0],
                hard_proposal,
                chord_parameters,
                ours_deployment=args.ours_deployment,
                verified_parameterization=args.verified_parameterization,
            )
            return greedy_prune_to_mse_tolerance(
                hard_parameters,
                points,
                hard_proposal,
                mse_tolerance=mse_tolerance,
                min_internal_knots=args.min_hard_knots,
                degree=model.degree,
                smoothness_weight=args.smoothness_weight,
                control_ridge=args.control_ridge,
            )

        hard_end_to_end_result, hard_end_to_end_ms = timed_call(
            run_hard_end_to_end,
            repeats=args.timing_repeats,
            warmup_repeats=args.timing_warmups,
        )
        hard_timing_reproduction = compare_hard_timing_reproduction(
            hard_result,
            hard_end_to_end_result,
        )
        if not hard_timing_reproduction.structurally_consistent:
            print(
                "  warning: Hard E2E timing replay differs from authoritative "
                f"geometry for sample {sample_index}: count_equal="
                f"{hard_timing_reproduction.count_equal}, mask_equal="
                f"{hard_timing_reproduction.retained_mask_equal}, knots_allclose="
                f"{hard_timing_reproduction.retained_knots_allclose}, max|du|="
                f"{hard_timing_reproduction.retained_knots_max_abs_difference}, "
                f"|dMSE|={hard_timing_reproduction.mse_abs_difference:.3e}. "
                "The plotted/metric Hard result remains the authoritative first run.",
                flush=True,
            )

        source_count = int(sample["source_internal_knot_count"])
        canonical_mask = sample["true_internal_knot_mask"]
        assert isinstance(canonical_mask, torch.Tensor)
        canonical_count = int(canonical_mask.sum().item())
        if sample_index in index_to_count:
            stratification_count = index_to_count[sample_index]
        elif args.stratify_by == "source":
            stratification_count = source_count
        else:
            stratification_count = canonical_count
        redundant_domain_label = (
            "shared chord-domain" if shared_chord_domain else "network-domain"
        )
        ours_panel_title = (
            "(c) Ours: adaptive verified selection + exact refit"
            if args.ours_deployment == "verified"
            else "(c) Ours: one-shot LearnedKeep + position update"
        )
        ours_curve_label = (
            "verified deployment B-spline"
            if args.ours_deployment == "verified"
            else "learned deployment B-spline"
        )
        ours_timing_text = f"network-forward-only={network_ms:.2f} ms (batch=1 CPU)"
        if args.timing_scope == "end-to-end":
            hard_timing_text = f"end-to-end={hard_end_to_end_ms:.2f} ms"
            timing_scope_note = (
                "Ours reports pure network forward only; Hard reports end-to-end "
                "network materialization plus greedy search. Final Ours refit/repair "
                "is excluded from the displayed prediction time."
            )
        else:
            hard_timing_text = f"hard-pruning-stage={hard_prune_ms:.2f} ms"
            timing_scope_note = (
                "Ours reports pure network forward only; Hard reports pruning-only. "
                "Final Ours refit/verified repair is excluded from prediction time."
            )
        panels = (
            source_panel(sample, dense_parameters, degree=model.degree),
            fit_panel(
                all_fit,
                dense_parameters,
                title=(
                    "(b) Network-proposed redundant knots + refit "
                    f"[{redundant_domain_label}]"
                ),
                curve_label="all-proposal B-spline",
                timing_text=f"refit-only={all_refit_ms:.2f} ms",
                curve_color="#6f6f6f",
            ),
            fit_panel(
                ours_fit,
                dense_parameters,
                title=ours_panel_title,
                curve_label=ours_curve_label,
                timing_text=ours_timing_text,
                curve_color="#198a77",
            ),
            fit_panel(
                hard_result.final_fit,
                dense_parameters,
                title=(
                    "(d) Traditional greedy hard pruning from same "
                    f"{redundant_domain_label} proposal"
                ),
                curve_label="hard-pruned B-spline",
                timing_text=hard_timing_text,
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
            parameterization_note=(
                "Redundant, Ours verified, and Greedy share chord-length parameters."
                if shared_chord_domain
                else (
                    "Redundant and Greedy use predicted parameters; Ours verified "
                    f"finished in {ours_parameterization}."
                    if args.ours_deployment == "verified"
                    else "Fitted methods use predicted parameters."
                )
            ),
            timing_scope_note=timing_scope_note,
        )

        source_knots = panels[0].internal_knots.tolist()
        proposal_values = comparison_proposal_knots.detach().cpu().tolist()
        network_proposal_values = proposal_knots.detach().cpu().tolist()
        learned_values = learned_knots.detach().cpu().tolist()
        ours_values = ours_knots.detach().cpu().tolist()
        hard_values = hard_result.retained_internal_knots.detach().cpu().tolist()
        source_knot_vector = panels[0].knot_vector.tolist()
        proposal_knot_vector = all_fit.knot_vector.detach().cpu().tolist()
        learned_knot_vector = learned_fit.knot_vector.detach().cpu().tolist()
        ours_knot_vector = ours_fit.knot_vector.detach().cpu().tolist()
        hard_knot_vector = hard_result.final_fit.knot_vector.detach().cpu().tolist()
        detailed_row: dict[str, object] = {
            "figure": filename,
            "sample_index": sample_index,
            "sample_seed": args.seed + sample_index,
            "stratification_count": stratification_count,
            "source_knot_count": source_count,
            "canonical_knot_count": canonical_count,
            "proposal_knot_count": int(proposal_knots.numel()),
            "learned_knot_count": int(learned_knots.numel()),
            "ours_deployment_mode": args.ours_deployment,
            "ours_knot_count": int(ours_knots.numel()),
            "hard_knot_count": hard_result.final_count,
            "source_mse": panels[0].mse,
            "proposal_mse": float(all_fit.fit_mse),
            "learned_mse": float(learned_fit.fit_mse),
            "ours_mse": float(ours_fit.fit_mse),
            "ours_threshold_satisfied": (float(ours_fit.fit_mse) <= mse_tolerance),
            "hard_mse": float(hard_result.final_fit.fit_mse),
            "hard_threshold_satisfied": hard_result.threshold_satisfied,
            "hard_accepted_deletions": hard_result.accepted_deletions,
            "ours_network_forward_ms": network_ms,
            "ours_network_forward_p50_ms": network_timing.latency.p50_ms,
            "ours_network_forward_p95_ms": network_timing.latency.p95_ms,
            "ours_network_forward_repeats_ms": list(
                network_timing.latency.repeat_ms
            ),
            # Compatibility alias.  Unlike the deployment diagnostics below,
            # this is the only Ours time displayed in figures and summaries.
            "network_forward_diagnostic_ms": network_ms,
            "proposal_refit_diagnostic_ms": all_refit_ms,
            "learned_refit_diagnostic_ms": learned_refit_ms,
            "learned_deployment_total_ms": learned_deployment_ms,
            "ours_deployment_total_ms": ours_deployment_ms,
            "ours_prediction_time_scope": (
                "model.forward_deployment(points) only: parameter prediction, "
                "candidate proposal, KeepMask selection and survivor relocation; "
                "resident batch=1 input; excludes final refit and verified repair"
            ),
            "ours_quality_pipeline_diagnostic_scope": (
                "network forward + exact learned-subset verification + conditional "
                "confidence add-back/compaction/chord rescue/residual insertion/"
                "hard fallback"
                if args.ours_deployment == "verified"
                else (
                    "network forward + proposal generation + LearnedKeep selection + "
                    "survivor relocation + one final standard B-spline refit"
                )
            ),
            "ours_timing_scope": "pure_network_forward_only",
            "hard_pruning_stage_ms": hard_prune_ms,
            "hard_end_to_end_ms": hard_end_to_end_ms,
            "hard_reported_ms": (
                hard_end_to_end_ms
                if args.timing_scope == "end-to-end"
                else hard_prune_ms
            ),
            "hard_reported_timing_scope": args.timing_scope,
            "hard_timing_replay_count": hard_end_to_end_result.final_count,
            "hard_timing_replay_mse": float(hard_end_to_end_result.final_fit.fit_mse),
            "hard_timing_replay_count_equal": hard_timing_reproduction.count_equal,
            "hard_timing_replay_mask_equal": (
                hard_timing_reproduction.retained_mask_equal
            ),
            "hard_timing_replay_knots_allclose": (
                hard_timing_reproduction.retained_knots_allclose
            ),
            "hard_timing_replay_knots_max_abs_difference": (
                hard_timing_reproduction.retained_knots_max_abs_difference
            ),
            "hard_timing_replay_mse_abs_difference": (
                hard_timing_reproduction.mse_abs_difference
            ),
            "hard_timing_replay_threshold_satisfied_equal": (
                hard_timing_reproduction.threshold_satisfied_equal
            ),
            "hard_timing_replay_structurally_consistent": (
                hard_timing_reproduction.structurally_consistent
            ),
            "comparison_parameterization": (
                "shared_chord_length"
                if shared_chord_domain
                else "network_predicted_with_optional_ours_chord_fallback"
                if args.ours_deployment == "verified"
                and args.verified_parameterization == "chord-fallback"
                else "shared_network_predicted"
            ),
            "ours_final_parameterization": ours_parameterization,
            "source_internal_knots": source_knots,
            "proposal_internal_knots": proposal_values,
            "network_proposal_internal_knots": network_proposal_values,
            "learned_internal_knots": learned_values,
            "ours_internal_knots": ours_values,
            "hard_internal_knots": hard_values,
            "source_full_knot_vector": source_knot_vector,
            "proposal_full_knot_vector": proposal_knot_vector,
            "learned_full_knot_vector": learned_knot_vector,
            "ours_full_knot_vector": ours_knot_vector,
            "hard_full_knot_vector": hard_knot_vector,
            "ours_refit_parameters": ours_parameters.detach().cpu().tolist(),
            "hard_mse_trajectory": list(hard_result.mse_trajectory),
            "verified_final_source": (
                verified_result.final_source if verified_result is not None else None
            ),
            "verified_learned_threshold_satisfied": (
                verified_result.learned_threshold_satisfied
                if verified_result is not None
                else None
            ),
            "verified_fallback_used": (
                verified_result.fallback_used if verified_result is not None else None
            ),
            "verified_cleanup_used": (
                verified_result.cleanup_used if verified_result is not None else None
            ),
            "verified_parameterization_fallback_used": (
                verified_result.parameterization_fallback_used
                if verified_result is not None
                else None
            ),
            "verified_residual_fallback_used": (
                verified_result.residual_fallback_used
                if verified_result is not None
                else None
            ),
            "verified_inserted_knot_count": (
                verified_result.inserted_count if verified_result is not None else None
            ),
            "verified_direct_refit_count": (
                verified_result.direct_refit_count
                if verified_result is not None
                else None
            ),
            "verified_fit_evaluation_count": (
                verified_result.fit_evaluation_count
                if verified_result is not None
                else None
            ),
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
                "network_proposal_internal_knots": json.dumps(network_proposal_values),
                "learned_internal_knots": json.dumps(learned_values),
                "ours_internal_knots": json.dumps(ours_values),
                "hard_internal_knots": json.dumps(hard_values),
                "source_full_knot_vector": json.dumps(source_knot_vector),
                "proposal_full_knot_vector": json.dumps(proposal_knot_vector),
                "learned_full_knot_vector": json.dumps(learned_knot_vector),
                "ours_full_knot_vector": json.dumps(ours_knot_vector),
                "hard_full_knot_vector": json.dumps(hard_knot_vector),
                "ours_refit_parameters": json.dumps(
                    ours_parameters.detach().cpu().tolist()
                ),
                "hard_mse_trajectory": json.dumps(list(hard_result.mse_trajectory)),
            }
        )
        print(
            f"[{figure_index}/{len(selected_indices)}] sample={sample_index} "
            f"K source/proposal/ours/hard={source_count}/"
            f"{proposal_knots.numel()}/{ours_knots.numel()}/"
            f"{hard_result.final_count} | MSE ours/hard="
            f"{float(ours_fit.fit_mse):.5e}/{float(hard_result.final_fit.fit_mse):.5e}",
            flush=True,
        )

    report: dict[str, object] = {
        "schema_version": 4,
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
        "effective_dataset_size": effective_dataset_size,
        "selection_mode": (
            "explicit_sample_indices"
            if args.sample_indices is not None
            else "stratified_random"
            if args.samples_per_knot_count is None
            else "fixed_samples_per_knot_count"
        ),
        "explicit_sample_indices": args.sample_indices,
        "stratify_by": args.stratify_by,
        "requested_knot_counts": requested_counts,
        "samples_per_knot_count": args.samples_per_knot_count,
        "selected_indices": selected_indices,
        "ours_deployment": args.ours_deployment,
        "verified_deployment": {
            "enabled": args.ours_deployment == "verified",
            "compact": args.verified_compact,
            "hard_fallback": args.verified_hard_fallback,
            "residual_fallback": args.verified_residual_fallback,
            "max_residual_insertions": args.verified_max_residual_insertions,
            "residual_min_gap": verified_residual_min_gap,
            "parameterization_policy": args.verified_parameterization,
            "refit_device": str(verified_refit_device),
            "min_internal_knots": verified_min_internal_knots,
        },
        "parameterization_comparison": (
            "Redundant, Ours verified, and Greedy use the same mapped chord-length "
            "proposal and chord-length sample parameters."
            if args.ours_deployment == "verified"
            and args.verified_parameterization == "chord"
            else (
                "Redundant and Greedy use network-predicted parameters; Ours may "
                "conditionally finish in chord-length parameters, recorded per sample."
                if args.ours_deployment == "verified"
                and args.verified_parameterization == "chord-fallback"
                else "All fitted comparison methods use network-predicted parameters."
            )
        ),
        "timing": {
            "device": str(model_device),
            "input_state": (
                f"normalized batch=1 point tensor resident on {model_device}"
            ),
            "batch_size": 1,
            "repeats": args.timing_repeats,
            "warmup_repeats": args.timing_warmups,
            "statistic": "median_wall_time_ms",
            "clock": "time.perf_counter_ns",
            "torch_num_threads": torch.get_num_threads(),
            "ours_reported_prediction_scope": (
                "model.forward_deployment(points) only; includes ParameterHead, "
                "candidate proposal, KeepMask and survivor relocation; excludes "
                "standard B-spline refit, verification and repair"
            ),
            "ours_reported_prediction_device": str(model_device),
            "cuda_synchronized_before_and_after_each_repeat": (
                model_device.type == "cuda"
            ),
            "network_inputs_resident_before_timing": True,
            "learned_deployment_scope": (
                "one end-to-end timed call: ParameterHead + redundant proposal "
                "generation + LearnedKeep selection + survivor relocation + one "
                "final standard B-spline refit"
            ),
            "ours_deployment_scope": (
                "one end-to-end timed call: network forward + exact learned-subset "
                "verification + conditional confidence repair, compaction, "
                "parameterization rescue, residual insertion, and hard fallback"
                if args.ours_deployment == "verified"
                else "same as learned_deployment_scope"
            ),
            "ours_verified_refit_device": str(verified_refit_device),
            "component_times_are_independent_medians": True,
            "learned_and_verified_total_times_are_diagnostic_only": True,
            "hard_pruning_scope": (
                "greedy hard-pruning call only, starting from already materialized "
                "proposal knots and parameters; includes every refit used by pruning"
            ),
            "hard_end_to_end_scope": (
                "network parameter/proposal materialization + optional shared chord "
                "mapping + greedy hard pruning and all of its refits"
            ),
            "hard_end_to_end_replay_is_timing_only": True,
            "hard_geometry_authority": (
                "the independently materialized hard_pruning_stage result; replay "
                "differences are diagnostics and never abort or replace the plot"
            ),
            "reported_hard_scope": args.timing_scope,
            "hard_pruning_stage_excludes_network_and_proposal_generation": True,
            "reported_hard_excludes_network_and_proposal_generation": (
                args.timing_scope == "historical-asymmetric"
            ),
            "excluded_from_both": "dataset loading, count scanning, plotting, file I/O",
            "comparison_is_asymmetric_by_user_request": (
                args.timing_scope == "historical-asymmetric"
            ),
        },
        "methods": {
            "source": "original generated B-spline evaluated at true parameters",
            "proposal": "all immutable proposal knots + one standard refit",
            "learned": (
                "one network forward + authoritative LearnedKeep mask + joint "
                "deployment positions + one standard refit"
            ),
            "ours": (
                "adaptive verified deployment: one network forward, exact threshold "
                "verification, and only the enabled conditional repairs"
                if args.ours_deployment == "verified"
                else "historical one-shot LearnedKeep deployment"
            ),
            "hard": (
                "already materialized immutable proposal + exhaustive greedy "
                "one-knot deletion with MSE stopping + its standard refits"
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
