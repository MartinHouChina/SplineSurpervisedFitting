from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.synthetic import evaluate_bspline_curve  # noqa: E402
from spline_fitting.evaluation.knot_diagnostics import (  # noqa: E402
    build_open_knot_vector,
)
from spline_fitting.evaluation.sparse_knot_paper import (  # noqa: E402
    SparseKnotPaperResult,
    fit_sparse_knots_paper,
)


REPRODUCTION_NOTE = (
    "Numerical reproduction of the sparse-knot workflow in Kang et al. "
    "(Computer-Aided Design 58, 2015). It uses this repository's ADMM "
    "approximation and active-cluster relocation, not the paper's CVX solve; "
    "matching the paper's exact knot vector or reported numerical values is "
    "therefore not claimed. Both cases here are scalar (1D), as in the paper."
)


@dataclass(frozen=True)
class ReproductionCase:
    name: str
    title: str
    parameters: torch.Tensor
    observations: torch.Tensor
    original_values: torch.Tensor
    initial_internal_knots: int
    epsilon: float
    jump_threshold: float
    admm_max_iterations: int
    lambda_bisection_iterations: int
    relocation_max_iterations: int
    cluster_gap_factor: float
    paper_setting: str
    source_internal_knots: torch.Tensor | None = None
    paper_reference_final_knot_count: int | None = None
    paper_reference_mse: float | None = None


def _chebyshev_t10(*, quick: bool) -> ReproductionCase:
    count = 81 if quick else 401
    initial_knots = 15 if quick else 25
    parameters = torch.linspace(0.0, 1.0, count, dtype=torch.float64)
    domain = 2.0 * parameters - 1.0
    values = torch.cos(10.0 * torch.acos(domain.clamp(-1.0, 1.0))).unsqueeze(-1)
    return ReproductionCase(
        name="chebyshev_t10",
        title="Chebyshev polynomial $T_{10}$",
        parameters=parameters,
        observations=values,
        original_values=values,
        initial_internal_knots=initial_knots,
        epsilon=0.003,
        # The ADMM solution is numerically, rather than exactly, sparse.  This
        # absolute threshold isolates its large derivative-jump coefficients.
        jump_threshold=5_000.0,
        admm_max_iterations=150 if quick else 600,
        lambda_bisection_iterations=3 if quick else 8,
        relocation_max_iterations=2 if quick else 8,
        # Keep distinct uniform-grid active sites separate for this oscillatory
        # example.  The paper also skips its second stage for T10.
        cluster_gap_factor=0.75,
        paper_setting=(
            "quick smoke profile derived from the paper's T10 experiment"
            if quick
            else "paper T10 setting: N=401, 25 uniform initial knots, epsilon=0.003"
        ),
        paper_reference_final_knot_count=(None if quick else 14),
        paper_reference_mse=(None if quick else 3.4745e-5),
    )


def _known_cubic_bspline(*, quick: bool) -> ReproductionCase:
    count = 81 if quick else 201
    parameters = torch.linspace(0.0, 1.0, count, dtype=torch.float64)
    source_knots = torch.tensor([0.16, 0.31, 0.52, 0.68, 0.84], dtype=torch.float64)
    source_controls = torch.tensor(
        [0.0, 0.9, -0.4, 1.1, -0.7, 0.8, -0.2, 0.65, 0.1],
        dtype=torch.float64,
    ).unsqueeze(-1)
    values = evaluate_bspline_curve(
        parameters,
        source_controls,
        build_open_knot_vector(source_knots, degree=3),
        degree=3,
    )
    return ReproductionCase(
        name="known_cubic_bspline",
        title="Known cubic B-spline function",
        parameters=parameters,
        observations=values,
        original_values=values,
        initial_internal_knots=17 if quick else 41,
        epsilon=1e-4 if quick else 2e-5,
        jump_threshold=1_000.0 if quick else 1.0,
        admm_max_iterations=150 if quick else 600,
        lambda_bisection_iterations=3 if quick else 8,
        relocation_max_iterations=2 if quick else 8,
        cluster_gap_factor=1.25,
        paper_setting=(
            "quick smoke profile of a known-knot cubic B-spline"
            if quick
            else "paper-style known-knot cubic B-spline validation case"
        ),
        source_internal_knots=source_knots,
    )


def build_cases(*, quick: bool) -> tuple[ReproductionCase, ...]:
    """Return the two deterministic scalar validation cases."""
    return (_chebyshev_t10(quick=quick), _known_cubic_bspline(quick=quick))


def _to_float_list(values: torch.Tensor) -> list[float]:
    return [float(value) for value in values.detach().cpu().flatten()]


def _case_report(
    case: ReproductionCase,
    result: SparseKnotPaperResult,
    *,
    quick: bool,
) -> dict[str, Any]:
    source_knots = (
        None
        if case.source_internal_knots is None
        else _to_float_list(case.source_internal_knots)
    )
    return {
        "schema_version": 1,
        "case": case.name,
        "title": case.title,
        "profile": "quick" if quick else "full",
        "reproduction_scope": REPRODUCTION_NOTE,
        "paper_setting": case.paper_setting,
        "paper_reference": (
            None
            if case.paper_reference_final_knot_count is None
            else {
                "reported_final_internal_knot_count": (
                    case.paper_reference_final_knot_count
                ),
                "reported_mse": case.paper_reference_mse,
                "final_internal_knot_count_difference": (
                    int(result.knots.numel())
                    - case.paper_reference_final_knot_count
                ),
                "final_mse_difference": (
                    float(result.fit_mse) - float(case.paper_reference_mse)
                ),
            }
        ),
        "dimension": 1,
        "degree": result.degree,
        "sample_count": int(case.parameters.numel()),
        "epsilon_mean_squared_error": case.epsilon,
        "initial_internal_knot_count": int(case.initial_internal_knots),
        "source_internal_knots": source_knots,
        "method": result.method,
        "relocation_method": result.relocation_method,
        "jump_threshold": result.jump_threshold,
        "relative_jump_threshold": result.relative_jump_threshold,
        "effective_jump_threshold": result.effective_jump_threshold,
        "sparse_stage": {
            "mse": float(result.sparse_fit_mse),
            "epsilon_satisfied": bool(result.threshold_satisfied),
            "active_knot_count": result.active_count,
            "active_internal_knots": _to_float_list(result.active_internal_knots),
            "jump_norms": _to_float_list(result.jump_norms),
            "admm_iterations_total": result.sparse_iterations,
            "lambda_bisection_iterations_configured": (
                result.lambda_bisection_iterations
            ),
        },
        "final_standard_bspline_refit": {
            "mse": float(result.fit_mse),
            "epsilon_satisfied": result.final_threshold_satisfied,
            "internal_knot_count": int(result.knots.numel()),
            "internal_knots": _to_float_list(result.knots),
            "pre_repair_relocated_internal_knots": _to_float_list(
                result.relocated_internal_knots
            ),
            "local_refit_count": result.local_refit_count,
            "final_refit_count": result.final_refit_count,
            "feasibility_repair": {
                "used": result.repair_used,
                "method": result.repair_method,
                "added_internal_knots": _to_float_list(result.repair_added_knots),
                "refit_count": result.repair_refit_count,
                "scope": "post-relocation safeguard; not part of Kang et al.",
            },
        },
        "elapsed_seconds": result.elapsed_seconds,
        "timing_breakdown_seconds": {
            "sparse_solver": result.sparse_solver_seconds,
            "relocation": result.relocation_seconds,
            "non_paper_feasibility_repair": result.repair_seconds,
            "total_adapted_pipeline": result.elapsed_seconds,
        },
    }


def _draw_knot_markers(
    axes: plt.Axes,
    knots: torch.Tensor,
    *,
    color: str,
    label: str,
) -> None:
    for index, knot in enumerate(_to_float_list(knots)):
        axes.axvline(
            knot,
            color=color,
            alpha=0.45,
            linewidth=1.0,
            linestyle="--",
            label=label if index == 0 else None,
        )


def _plot_case(
    case: ReproductionCase,
    result: SparseKnotPaperResult,
    output_path: Path,
    *,
    dpi: int,
) -> None:
    parameters = case.parameters.detach().cpu().numpy()
    observations = case.observations[:, 0].detach().cpu().numpy()
    original = case.original_values[:, 0].detach().cpu().numpy()
    sparse = result.sparse_reconstructed_points[:, 0].detach().cpu().numpy()
    final = result.final_fit.reconstructed_points[:, 0].detach().cpu().numpy()

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), sharex=True, sharey=True)
    figure.suptitle(case.title, fontsize=14)
    common = (
        ("Original function", original, "#202020", 2.0),
        ("Observed samples", observations, "#7f8c8d", 0.0),
    )
    for axis in axes:
        axis.plot(
            parameters,
            common[0][1],
            color=common[0][2],
            linewidth=common[0][3],
            label=common[0][0],
        )
        marker_stride = max(1, len(parameters) // 50)
        axis.scatter(
            parameters[::marker_stride],
            common[1][1][::marker_stride],
            color=common[1][2],
            s=10,
            alpha=0.55,
            zorder=2,
            label=common[1][0],
        )
        axis.grid(alpha=0.22)
        axis.set_xlabel("parameter $t$")
    axes[0].set_ylabel("scalar value")

    axes[0].plot(
        parameters,
        sparse,
        color="#d95f02",
        linewidth=1.7,
        label=f"Sparse stage (MSE={float(result.sparse_fit_mse):.3e})",
    )
    _draw_knot_markers(
        axes[0],
        result.active_internal_knots,
        color="#d95f02",
        label=f"Active knots ({result.active_count})",
    )
    axes[0].set_title("Sparse derivative-jump stage")

    axes[1].plot(
        parameters,
        final,
        color="#1b9e77",
        linewidth=1.7,
        label=f"Final refit (MSE={float(result.fit_mse):.3e})",
    )
    _draw_knot_markers(
        axes[1],
        result.knots,
        color="#1b9e77",
        label=f"Final knots ({int(result.knots.numel())})",
    )
    if case.source_internal_knots is not None:
        _draw_knot_markers(
            axes[1],
            case.source_internal_knots,
            color="#7570b3",
            label=f"Known source knots ({int(case.source_internal_knots.numel())})",
        )
    axes[1].set_title("Relocation + standard B-spline refit")

    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        # Avoid repeating the first knot marker when multiple vertical lines exist.
        unique = dict(zip(labels, handles, strict=False))
        axis.legend(unique.values(), unique.keys(), fontsize=8, loc="best")
    figure.text(
        0.5,
        0.015,
        "ADMM numerical reproduction; not an exact reproduction of the paper's CVX values.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def run_reproduction(
    output_dir: Path,
    *,
    quick: bool = False,
    case_name: str = "all",
    dpi: int = 220,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Run deterministic cases and write one JSON and PNG for every case."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    cases = build_cases(quick=quick)
    if case_name != "all":
        cases = tuple(case for case in cases if case.name == case_name)
        if not cases:
            raise ValueError(f"unknown case: {case_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [
        destination
        for case in cases
        for destination in (
            output_dir / f"{case.name}.json",
            output_dir / f"{case.name}.png",
        )
    ]
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"output already exists ({names}); use --overwrite")

    reports: list[dict[str, Any]] = []
    for case in cases:
        result = fit_sparse_knots_paper(
            case.parameters,
            case.observations,
            degree=3,
            initial_internal_knot_count=case.initial_internal_knots,
            data_tolerance=case.epsilon,
            jump_threshold=case.jump_threshold,
            admm_max_iterations=case.admm_max_iterations,
            lambda_bisection_iterations=case.lambda_bisection_iterations,
            relocation_max_iterations=case.relocation_max_iterations,
            cluster_gap_factor=case.cluster_gap_factor,
        )
        report = _case_report(case, result, quick=quick)
        json_path = output_dir / f"{case.name}.json"
        png_path = output_dir / f"{case.name}.png"
        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _plot_case(case, result, png_path, dpi=dpi)
        reports.append(report)
        print(
            f"{case.name}: active/final={result.active_count}/"
            f"{result.knots.numel()}, sparse/final MSE="
            f"{float(result.sparse_fit_mse):.6e}/{float(result.fit_mse):.6e}"
        )
        print(f"  JSON: {json_path}")
        print(f"  PNG:  {png_path}")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one paper-configured 1D numerical reproduction and one "
            "deterministic workflow validation of Kang et al.'s sparse-knot "
            "method, then save JSON diagnostics plus PNGs."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper/kang_sparse_reproduction"),
    )
    parser.add_argument(
        "--case",
        choices=("all", "chebyshev_t10", "known_cubic_bspline"),
        default="all",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller deterministic cases and fewer solver iterations for a smoke run.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not math.isfinite(float(args.dpi)) or args.dpi <= 0:
        parser.error("--dpi must be positive")
    try:
        run_reproduction(
            args.output_dir,
            quick=args.quick,
            case_name=args.case,
            dpi=args.dpi,
            overwrite=args.overwrite,
        )
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
