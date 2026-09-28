"""Render the verified historical K32 model and its measured postprocessing.

This is a code-native architectural schematic, not an experimental curve.
Unlike the newer coupled models, this selected checkpoint has no repeated
proposal/subset geometry blocks. Numerical insertion and deletion are explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "outputs/checkpoints/overnight_stable_k32_3090_r1.pt"
DEFAULT_REPORT = ROOT / "outputs/comparisons/historical_best_dual_error_postprune_fresh_20260923/comparison.json"
DEFAULT_OUTPUT = ROOT / "outputs/figures/postprune_paper_20260923"
INK, MUTED = "#142C48", "#647285"
COLORS = ["#325D89", "#267DB0", "#B57B28", "#8261A0", "#378062"]
FILLS = ["#F3F6FB", "#F1F8FC", "#FCF8EF", "#F7F3FA", "#F1F8F4"]
OPTIONAL_DEFAULTS = {
    "subset_geometry_mode": "legacy", "subset_geometry_residual_scale": 1.0,
    "proposal_refinement_layers": 0, "selection_refinement_layers": 0,
    "survivor_refinement_layers": 0, "coupled_proposal_steps": 0,
    "coupled_subset_steps": 0, "parameter_chord_blend": 0.0,
}
REQUIRED = {
    "degree": 3, "point_dim": 2, "hidden_dim": 128, "encoder_layers": 3,
    "max_internal_knots": 32, "attention_heads": 4, "selector_layers": 2,
    "one_shot_selection_policy": "mass_topk", "one_shot_adaptive_threshold": True,
    "parameter_trust_enabled": True, "one_shot_safety_sigma": 0.0,
    "one_shot_safety_knots": 0, "one_shot_coverage_bins": 4,
    "min_selected_knots": 4,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(checkpoint_path: Path, report_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recorded = checkpoint["model_config"]
    effective = {**OPTIONAL_DEFAULTS, **recorded}
    for name, expected in {**REQUIRED, **OPTIONAL_DEFAULTS}.items():
        if effective.get(name) != expected:
            raise ValueError(f"This schematic requires {name}={expected!r}, got {effective.get(name)!r}")
    protocol = report["metadata"]
    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint_hash != protocol["checkpoint_sha256"]:
        raise ValueError("The framework checkpoint is not the benchmark checkpoint")
    for name, expected in {"max_internal_knots": 32, "model_capacity": 32,
                           "num_points": 192, "mse_tolerance": 5e-5,
                           "max_squared_error_tolerance": 5e-4}.items():
        if protocol.get(name) != expected:
            raise ValueError(f"Unexpected benchmark {name}={protocol.get(name)!r}")
    pruning = protocol.get("ours_post_pruning", {})
    if not pruning.get("enabled") or pruning.get("stage") != "after_common_insertion":
        raise ValueError("This schematic requires Ours pruning after common insertion")
    if pruning.get("min_internal_knots") != 0 or pruning.get("max_deletions") != 32:
        raise ValueError("This schematic requires the measured zero-minimum, 32-deletion budget")
    if protocol.get("endpoint_convention", {}).get("ours") is not True:
        raise ValueError("This schematic requires endpoint-constrained Ours refits")
    return {
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "objective_version": checkpoint.get("objective_version"),
        "report": str(report_path.resolve()), "report_sha256": sha256(report_path),
        "model_config_recorded": recorded, "model_config_effective": effective,
        "historical_default_fields": [key for key in OPTIONAL_DEFAULTS if key not in recorded],
        "deployment_config_recorded": checkpoint.get("deployment_config", {}),
        "model_source_sha256": sha256(ROOT / "src/spline_fitting/models/v16_network.py"),
        "schematic_source_sha256": sha256(Path(__file__)),
        "architecture": {
            "encoder": "3 Conv1d + GroupNorm + GELU blocks; coordinates and chord derivatives; local features and global max pool",
            "parameterization": "chord-residual ParameterHead; curve-level learned trust blend toward strict chord reference",
            "candidate_queries": 33, "candidate_capacity": 32,
            "candidate_local_gaussian_bandwidth": 0.08,
            "candidate_geometry": "bounded residuals about ordered uniform anchors; adjacent interval-token fusion",
            "selector": "2 self-/point-cross-attention blocks; tolerance and coverage embeddings; adaptive beta; probability-mass TopK",
            "selector_count_rule": "ceil(sum sigmoid(centered_importance - beta)), clamped to [4,32]; four spatial-bin anchors when affordable",
            "decoder": "one selected-memory decoder updates monotone point parameters and transports/relocates surviving knots; legacy geometry projection",
            "decoder_iterations": 1,
            "optional_repeated_refinement_blocks": 0,
            "final_control_points": "endpoint-constrained unregularized CPU float64 standard cubic B-spline least-squares refit",
        },
        "postprocessing": {
            "common_insertion": protocol["repair_protocol"],
            "ours_pruning": pruning,
            "mse_tolerance": protocol["mse_tolerance"],
            "max_squared_error_tolerance": protocol["max_squared_error_tolerance"],
            "max_internal_knots": protocol["max_internal_knots"],
            "fixed_during_numerical_stages": "point parameters and all surviving knot coordinates; refit controls only",
            "failure_policy": "At capacity or without an admissible deletion, retain measured current fit and record success/failure; no guarantee all inputs pass",
        },
        "scope": "deployment only; no training-teacher claim in this diagram",
        "illustrative_geometry": "All curve and token glyphs are schematic, not measured model predictions",
    }


def _text(ax, x, y, text, size=11, color=INK, weight="normal", ha="center"):
    return ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va="center", linespacing=1.42, zorder=5)


def _box(ax, x, y, w, h, color, fill="white", lw=1.1):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                          edgecolor=color, facecolor=fill, linewidth=lw, zorder=1)
    ax.add_patch(patch)


def _arrow(ax, start, end, color=INK, lw=1.5):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                color=color, linewidth=lw, zorder=4))


def _knots(ax, x, y, w, color, kept=None):
    u = np.linspace(x + .13, x + w - .13, 10)
    ax.plot([x, x + w], [y, y], color="#9DA7B1", lw=.85, zorder=2)
    if kept is None:
        kept = np.ones(10, dtype=bool)
    ax.scatter(u[~kept], np.full((~kept).sum(), y), s=21, marker="D", c="#D8DDE4", zorder=3)
    ax.scatter(u[kept], np.full(kept.sum(), y), s=31, marker="D", c=color,
               edgecolors="white", linewidths=.4, zorder=3)


def make_figure(metadata: dict, output_dir: Path, dpi: int = 300) -> dict:
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(20.7, 9.8))
    ax.set(xlim=(0, 21.35), ylim=(0, 10.0))
    ax.axis("off")
    fig.subplots_adjust(left=.015, right=.985, bottom=.018, top=.99)
    _text(ax, 10.68, 9.58, "Learning-guided B-spline knot selection and redundancy reduction",
          size=22, weight="bold")
    _text(ax, 8.18, 8.96, "Learned prediction", size=13.5, color=COLORS[0], weight="bold")
    _text(ax, 18.69, 8.96, "Numerical refinement", size=13.5, color=COLORS[4], weight="bold")
    ax.plot([.35, 16.00], [8.75, 8.75], color="#D0DCE8", lw=2)
    ax.plot([16.38, 21.0], [8.75, 8.75], color="#D0E4D8", lw=2)
    starts, widths = [.32, 4.32, 8.32, 12.32, 16.38], [3.63, 3.63, 3.63, 3.63, 4.62]
    titles = ["Geometry & parameters", "Candidate proposal", "Adaptive selection",
              "Subset-conditioned update", "Error-checked simplification"]
    for i, (x, w) in enumerate(zip(starts, widths)):
        _box(ax, x, 1.12, w, 7.40, COLORS[i], FILLS[i])
        _text(ax, x + .34, 8.12, str(i + 1), size=17, color=COLORS[i], weight="bold")
        _text(ax, x + w / 2 + .10, 8.12, titles[i], size=12.2 if i != 3 else 11.2,
              color=COLORS[i], weight="bold")

    # 1: actual encoder features and the chord-residual/trust parameter head.
    x, w = starts[0], widths[0]
    _text(ax, x + w/2, 7.53, r"Ordered samples  $Q\in\mathbb{R}^{192\times2}$", size=12)
    s = np.linspace(0, 1, 65)
    ys = .23 * np.sin(2*np.pi*s) + .09 * np.sin(4*np.pi*s)
    ax.plot(x + .45 + 2.7*s, 6.97 + ys, color=COLORS[0], lw=1.7)
    ax.scatter(x + .45 + 2.7*s[::6], 6.97 + ys[::6], s=16, c=COLORS[0], zorder=4)
    _arrow(ax, (x+w/2, 6.56), (x+w/2, 6.31), COLORS[0])
    _box(ax, x+.22, 5.38, w-.44, .9, COLORS[0])
    _text(ax, x+w/2, 5.99, "Geometry encoder", size=13, weight="bold")
    _text(ax, x+w/2, 5.62, "Coordinates + chord derivatives\n3 convolution blocks · width 128", size=10.1)
    _arrow(ax, (x+w/2, 5.36), (x+w/2, 4.99), COLORS[0])
    _box(ax, x+.22, 3.75, w-.44, 1.2, COLORS[0])
    _text(ax, x+w/2, 4.62, "Parameter head", size=13, weight="bold")
    _text(ax, x+w/2, 4.12, "Chord-residual gaps\nLearned curve-level trust", size=11)
    _arrow(ax, (x+w/2, 3.73), (x+w/2, 3.36), COLORS[0])
    _text(ax, x+w/2, 3.09, r"$0=t_0<\cdots<t_{191}=1$", size=14)
    _text(ax, x+w/2, 2.48, r"Point memory: $F+\mathrm{PE}(t)$", size=12)
    _text(ax, x+w/2, 1.75, "Local + global geometry features", size=10.4, color=MUTED)

    # 2: 33 interval queries produce 32 candidate boundaries (not 32 independent queries).
    x, w = starts[1], widths[1]
    _text(ax, x+w/2, 7.51, "33 learned interval queries\n+ positional anchors", size=12)
    _arrow(ax, (x+w/2, 6.98), (x+w/2, 6.66), COLORS[1])
    _box(ax, x+.22, 5.40, w-.44, 1.24, COLORS[1])
    _text(ax, x+w/2, 6.28, "Local cross-attention", size=13, weight="bold")
    _text(ax, x+w/2, 5.80, "Gaussian parameter-space bias\nKeys / values: point memory", size=10.5)
    _arrow(ax, (x+w/2, 5.38), (x+w/2, 4.98), COLORS[1])
    _box(ax, x+.22, 3.85, w-.44, 1.1, COLORS[1])
    _text(ax, x+w/2, 4.62, "Ordered anchor offsets", size=12.7, weight="bold")
    _text(ax, x+w/2, 4.14, "Adjacent interval-token fusion\nOne token per candidate", size=10.5)
    _knots(ax, x+.33, 3.22, w-.66, COLORS[1])
    _text(ax, x+w/2, 2.60, r"$U_{\rm prop}$: 32 internal candidates", size=11.7)
    _text(ax, x+w/2, 1.75, "Fixed capacity · ordered by construction", size=10.1, color=MUTED)

    # 3: beta affects probabilities; deployed mask is mass-TopK with spatial anchors.
    x, w = starts[2], widths[2]
    _text(ax, x+w/2, 7.51, "Candidate tokens + spacing\n+ tolerance embedding", size=11.7)
    _arrow(ax, (x+w/2, 6.98), (x+w/2, 6.66), COLORS[2])
    _box(ax, x+.22, 5.40, w-.44, 1.24, COLORS[2])
    _text(ax, x+w/2, 6.28, "Interactive selector", size=13, weight="bold")
    _text(ax, x+w/2, 5.80, "2 self-/cross-attention blocks\nCurve-adaptive threshold", size=10.7)
    _text(ax, x+w/2, 4.94, r"$p_j=\sigma(s_j-\bar{s}-\beta)$", size=13.4)
    _arrow(ax, (x+w/2, 4.64), (x+w/2, 4.38), COLORS[2])
    _box(ax, x+.22, 3.16, w-.44, 1.19, COLORS[2])
    _text(ax, x+w/2, 4.02, "Probability-mass TopK", size=12.4, weight="bold")
    _text(ax, x+w/2, 3.56, "4 spatial-bin anchors\nPredicted count within [4, 32]", size=10.7)
    keep = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0], dtype=bool)
    _knots(ax, x+.33, 2.55, w-.66, COLORS[2], keep)
    _text(ax, x+w/2, 1.75, r"Binary subset $m$ + selected memory", size=10.7, color=MUTED)

    # 4: both outputs update in the actual legacy selected-memory decoder.
    x, w = starts[3], widths[3]
    _text(ax, x+w/2, 7.51, "Survivors + neighboring gaps\n+ compact rank / count", size=11.3)
    _arrow(ax, (x+w/2, 6.98), (x+w/2, 6.66), COLORS[3])
    _box(ax, x+.22, 5.40, w-.44, 1.24, COLORS[3])
    _text(ax, x+w/2, 6.28, "Selected-only attention", size=12.5, weight="bold")
    _text(ax, x+w/2, 5.80, "Point queries + survivor queries\nKeys / values exclude discarded nodes", size=9.9)
    _arrow(ax, (x+w/2, 5.38), (x+w/2, 4.97), COLORS[3])
    _box(ax, x+.22, 3.57, w-.44, 1.37, COLORS[3])
    _text(ax, x+w/2, 4.64, "Joint geometric decoding", size=12.1, weight="bold")
    _text(ax, x+w/2, 4.07, "Monotone parameter-gap update\nKnot transport + relocation\nOrdered interval projection", size=10.4)
    _text(ax, x+w/2, 2.89, r"$t_{\rm net},\ U_{\rm net}[m]$", size=15)
    _text(ax, x+w/2, 2.32, "One subset-conditioned decoder", size=10.4)
    _text(ax, x+w/2, 1.75, "Point features also condition the update", size=9.7, color=MUTED)

    # 5: these stages are numerical, counted in deployment runtime, and iterative.
    x, w = starts[4], widths[4]
    _box(ax, x+.25, 6.81, w-.50, .77, COLORS[4])
    _text(ax, x+w/2, 7.34, "Standard cubic B-spline fit", size=12.8, weight="bold")
    _text(ax, x+w/2, 7.03, "Endpoint-constrained least squares", size=10.6)
    _arrow(ax, (x+w/2, 6.79), (x+w/2, 6.51), COLORS[4])
    _box(ax, x+.25, 5.40, w-.50, 1.08, COLORS[4])
    _text(ax, x+w/2, 6.19, "Residual-guided insertion", size=12.5, weight="bold")
    _text(ax, x+w/2, 5.72, "If either bound fails: insert + refit\nStop at both bounds or 32 internal knots", size=10.4)
    _arrow(ax, (x+w/2, 5.38), (x+w/2, 5.10), COLORS[4])
    _box(ax, x+.25, 3.86, w-.50, 1.21, COLORS[4])
    _text(ax, x+w/2, 4.75, "Redundancy deletion", size=12.7, weight="bold")
    _text(ax, x+w/2, 4.25, "Test each single-knot deletion + refit\nAccept a dual-feasible trial; repeat", size=10.7)
    _text(ax, x+w/2, 3.41, r"MSE $\leq 5\!\times\!10^{-5}$   and   MaxSE $\leq 5\!\times\!10^{-4}$", size=10.9)
    _arrow(ax, (x+w/2, 3.12), (x+w/2, 2.89), COLORS[4])
    _box(ax, x+.25, 1.63, w-.50, 1.22, COLORS[4], "#E5F1E9")
    _text(ax, x+w/2, 2.48, r"Final curve $C(t)$, controls $P$, knots $U$", size=12, weight="bold")
    _text(ax, x+w/2, 1.99, "Fixed point parameters and surviving knots\nControls are refitted after each accepted edit", size=10)

    # Explicit data-flow links between the modules, avoiding spurious extra loops.
    for i in range(4):
        _arrow(ax, (starts[i]+widths[i]+.03, 5.97), (starts[i+1]-.025, 5.97), COLORS[i])
    _text(ax, 10.68, .57, "Ordered points  →  candidates  →  learned subset + geometry  →  error-checked B-spline simplification",
          size=13.3, color=INK)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "framework.png"
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    metadata = {**metadata, "figure": str(png.resolve()), "dpi": dpi,
                "pixel_size": [round(20.7*dpi), round(9.8*dpi)]}
    (output_dir / "framework_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    if args.dpi < 50:
        parser.error("--dpi must be at least 50")
    metadata = load_contract(args.checkpoint, args.report)
    make_figure(metadata, args.output_dir, args.dpi)
    print(f"Saved {args.output_dir / 'framework.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
