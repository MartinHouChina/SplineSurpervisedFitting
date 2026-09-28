"""Draw the checkpoint-verified M16/M32 coupled selection architecture.

This is an architectural schematic, not a rendering of a model prediction.
PNG and SVG share the same Matplotlib source.  Only metadata/configuration is
read from checkpoints, using torch.load(..., weights_only=True).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs/downloads/server_models_20260922_215418"
DEFAULT_CHECKPOINTS = [
    MODEL_DIR / "paper_coupled_clean_3090_r2_m16.pt",
    MODEL_DIR / "paper_coupled_clean_3090_r3_m32.pt",
]
INK = "#12233C"
MUTED = "#506176"
COLORS = ["#244B79", "#1466AD", "#BC6B13", "#7346A1", "#187445"]
FILLS = ["#F3F7FC", "#F0F7FF", "#FFF9ED", "#F8F3FC", "#F1F9F3"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_configs(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint["model_config"]
        required = {
            "degree": 3,
            "one_shot_selection_policy": "mass_topk",
            "one_shot_adaptive_threshold": True,
            "subset_geometry_mode": "anchored",
            "parameter_trust_enabled": True,
            "parameter_chord_blend": 0.0,
            "one_shot_safety_sigma": 0.0,
            "one_shot_safety_knots": 0,
        }
        for key, expected in required.items():
            if config.get(key) != expected:
                raise ValueError(f"Diagram does not describe {path.name}: {key}={config.get(key)!r}")
        records.append({
            "path": str(path.resolve()), "sha256": _sha256(path),
            "epoch": checkpoint.get("epoch"),
            "stage": checkpoint.get("stage"),
            "model_config": config,
            "deployment_config": checkpoint.get("deployment_config", {}),
            "training_config": checkpoint.get("training_config", {}),
        })
    comparable = {key: value for key, value in records[0]["model_config"].items()
                  if key != "max_internal_knots"}
    for record in records[1:]:
        current = {key: value for key, value in record["model_config"].items()
                   if key != "max_internal_knots"}
        if current != comparable:
            raise ValueError("Checkpoints differ in more than candidate capacity; draw separate diagrams")
    return records


def label(ax, x, y, text, *, size=12, color=INK, weight="normal", ha="center", va="center"):
    return ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, linespacing=1.4, zorder=5)


def box(ax, x, y, w, h, *, edge=INK, face="white", dash=False, lw=1.2):
    patch = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.015,rounding_size=0.10",
                          linewidth=lw, edgecolor=edge, facecolor=face,
                          linestyle=(0, (5, 3)) if dash else "solid", zorder=1)
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color=INK, dashed=False, style="-|>", lw=1.8):
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=16,
                            color=color, linewidth=lw,
                            linestyle=(0, (4, 3)) if dashed else "solid", zorder=4)
    ax.add_patch(patch)
    return patch


def token_strip(ax, x, y, w, *, color, kept=None, diamonds=False):
    positions = np.linspace(x + .13, x + w - .13, 10)
    if diamonds:
        ax.plot([x, x + w], [y, y], color=MUTED, lw=.9, zorder=2)
    for index, px in enumerate(positions):
        active = kept is None or kept[index]
        face = color if active else "#DFE4EA"
        if diamonds:
            ax.plot(px, y, "D", ms=6.3, color=face, markeredgecolor="white", mew=.45, zorder=3)
        else:
            ax.add_patch(Rectangle((px - .085, y - .09), .17, .18,
                                   facecolor=face, edgecolor="white", linewidth=.7, zorder=3))


def make_figure(records: list[dict], output_dir: Path, dpi: int) -> None:
    c = records[0]["model_config"]
    capacities = sorted({r["model_config"]["max_internal_knots"] for r in records})
    capacity_text = " / ".join(str(k) for k in capacities)
    hidden = c["hidden_dim"]
    prop_layers = c["proposal_refinement_layers"]
    sel_layers = c["selection_refinement_layers"]
    subset_layers = c["survivor_refinement_layers"]
    prop_steps = c["coupled_proposal_steps"]
    subset_steps = c["coupled_subset_steps"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "mathtext.fontset": "dejavusans", "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(20.0, 10.3))
    fig.subplots_adjust(left=.018, right=.982, top=.98, bottom=.018)
    ax.set(xlim=(0, 20), ylim=(0, 10.1))
    ax.axis("off")
    label(ax, 10, 9.72, "Coupled candidate selection for adaptive B-spline fitting", size=24, weight="bold")
    label(ax, 10, 9.25,
          f"Ordered samples  →  dense candidates  →  one adaptive mask  →  coupled refinement  →  one final refit",
          size=13.5, color=MUTED)
    panel_w = 3.70
    panel_x = [.06, 4.10, 8.14, 12.18, 16.22]
    titles = ["Geometry + parameters", "Candidate proposal", "Adaptive selection", "Survivor refinement", "B-spline output"]
    for number, (x, title, color, fill) in enumerate(zip(panel_x, titles, COLORS, FILLS), start=1):
        box(ax, x, 2.51, panel_w, 6.23, edge=color, face=fill, lw=1.45)
        label(ax, x + .28, 8.39, str(number), size=18, color=color, weight="bold")
        label(ax, x + 2.02, 8.39, title, size=15, color=color, weight="bold")
    # First column: raw ordered geometry and the actual chord-residual head.
    x = panel_x[0]
    label(ax, x + panel_w / 2, 7.91, r"Ordered points $Q\in\mathbb{R}^{N\times 2}$", size=13)
    sample_t = np.linspace(0, 1, 19)
    sample_y = .24 * np.sin(2.3 * np.pi * sample_t) + .11 * np.cos(5 * np.pi * sample_t)
    ax.plot(x + .40 + sample_t * 2.90, 7.28 + sample_y, "o-", ms=4.3, lw=1.0, color=COLORS[0])
    arrow(ax, (x + 1.85, 6.89), (x + 1.85, 6.58), color=COLORS[0])
    box(ax, x + .20, 5.56, 3.30, 1.03, edge=COLORS[0])
    label(ax, x + 1.85, 6.26, "GeometryEncoder", size=15, weight="bold", color=COLORS[0])
    label(ax, x + 1.85, 5.84, f"Coordinates + chord derivatives\n{c['encoder_layers']} convolution blocks; width {hidden}", size=11.1)
    label(ax, x + 1.85, 5.24, r"Point features $F$ + pooled feature $g$", size=11.5)
    arrow(ax, (x + 1.85, 4.98), (x + 1.85, 4.70), color=COLORS[0])
    box(ax, x + .20, 3.48, 3.30, 1.21, edge=COLORS[0])
    label(ax, x + 1.85, 4.40, "ParameterHead", size=15, color=COLORS[0], weight="bold")
    label(ax, x + 1.85, 3.92, "Chord reference + learned residual\nCurve-level learned trust", size=11.8)
    label(ax, x + 1.85, 2.94, r"$0=t_0<t_1<\cdots<t_{N-1}=1$", size=13.5)
    # Second column: initial local queries, feature refinement, numerical coupling.
    x = panel_x[1]
    box(ax, x + .20, 6.69, 3.30, 1.23, edge=COLORS[1])
    label(ax, x + 1.85, 7.62, "CandidateKnotHead", size=15, color=COLORS[1], weight="bold")
    label(ax, x + 1.85, 7.15, "Local Gaussian cross-attention\nK/V: point features + PosEnc(t)", size=11.8)
    label(ax, x + 1.85, 6.41, r"$K_c+1$ interval queries $\rightarrow K_c$ knots", size=12)
    token_strip(ax, x + .36, 6.00, 2.98, color=COLORS[1], diamonds=True)
    label(ax, x + 1.85, 5.60, "Ordered candidates + node tokens", size=11.6)
    arrow(ax, (x + 1.85, 5.34), (x + 1.85, 5.07), color=COLORS[1])
    box(ax, x + .20, 3.70, 3.30, 1.38, edge=COLORS[1])
    label(ax, x + 1.85, 4.77, f"Local refinement × {prop_layers}", size=13, weight="bold", color=COLORS[1])
    label(ax, x + 1.85, 4.20, f"Coupled t / U updates × {prop_steps}\nParameter warp + bounded knot shift", size=11.7)
    label(ax, x + 1.85, 3.16, rf"$K_c={capacity_text}$; point–node interaction", size=12)
    # Selection: score / learned beta / mass count, not independent p>.5 gating.
    x = panel_x[2]
    box(ax, x + .20, 6.62, 3.30, 1.30, edge=COLORS[2])
    label(ax, x + 1.85, 7.59, "Interactive selector", size=15, weight="bold", color=COLORS[2])
    label(ax, x + 1.85, 7.07,
          f"Attention blocks × {c['selector_layers']}\nLocal refinement × {sel_layers}", size=12)
    label(ax, x + 1.85, 6.21, r"$p_j=\sigma(s_j-\bar{s}-\beta)$", size=17, color=COLORS[2])
    label(ax, x + 1.85, 5.80, r"Curve-conditioned threshold $\beta$", size=11.6)
    box(ax, x + .20, 4.25, 3.30, 1.23, edge=COLORS[2])
    label(ax, x + 1.85, 5.18, "Probability-mass Top-K", size=13.5, weight="bold", color=COLORS[2])
    label(ax, x + 1.85, 4.70,
          f"Minimum {c['min_selected_knots']} knots; {c['one_shot_coverage_bins']} coverage bins\nRank once; build one discrete mask", size=11.7)
    kept = [True, False, True, True, False, False, True, False, False, True]
    token_strip(ax, x + .36, 3.74, 2.98, color=COLORS[2], kept=kept)
    label(ax, x + 1.85, 3.15, r"KeepMask $m\in\{0,1\}^{K_c}$", size=14)
    # Survivor column: no discarded K/V, actual t/U updates repeated.
    x = panel_x[3]
    box(ax, x + .20, 6.36, 3.30, 1.56, edge=COLORS[3])
    label(ax, x + 1.85, 7.60, "Selected-node decoder", size=14.5, color=COLORS[3], weight="bold")
    label(ax, x + 1.85, 7.02,
          "K/V: retained nodes only\nNeighbor gaps + rank + selected count", size=11.4)
    label(ax, x + 1.85, 6.57, f"Survivor feature refinement × {subset_layers}", size=11.8)
    arrow(ax, (x + 1.85, 6.14), (x + 1.85, 5.86), color=COLORS[3])
    box(ax, x + .20, 4.37, 3.30, 1.50, edge=COLORS[3])
    label(ax, x + 1.85, 5.54, f"Coupled t / U updates × {subset_steps}", size=13.1, color=COLORS[3], weight="bold")
    label(ax, x + 1.85, 4.94,
          "Refresh positions and point features\nMonotone parameter update\nAnchored survivor relocation", size=11.6)
    token_strip(ax, x + .36, 3.86, 2.98, color=COLORS[3], kept=kept, diamonds=True)
    label(ax, x + 1.85, 3.25, r"Updated $t^*, U^*$; same mask $m$", size=12.4)
    # Standard fit and output, not an error-search loop.
    x = panel_x[4]
    box(ax, x + .20, 6.36, 3.30, 1.56, edge=COLORS[4])
    label(ax, x + 1.85, 7.60, "Standard cubic B-spline", size=14.2, color=COLORS[4], weight="bold")
    label(ax, x + 1.85, 7.05, r"$P^*=\arg\min_P\|B(t^*,U^*[m])P-Q\|_F^2$", size=12.2)
    label(ax, x + 1.85, 6.58, "Endpoint-constrained refit × 1", size=12)
    arrow(ax, (x + 1.85, 6.14), (x + 1.85, 5.89), color=COLORS[4])
    box(ax, x + .20, 3.61, 3.30, 2.29, edge=COLORS[4])
    tt = np.linspace(0, 1, 80)
    yy = .26 * np.sin(2 * np.pi * tt)
    ax.plot(x + .43 + tt * 2.83, 5.28 + yy, color=COLORS[4], lw=2.2, zorder=3)
    ix = np.array([0, 14, 30, 49, 63, 79])
    ax.plot(x + .43 + tt[ix] * 2.83, 5.28 + yy[ix], "o", ms=3.8, color=COLORS[4], zorder=4)
    label(ax, x + 1.85, 4.56, "Fitted curve + control vertices", size=12)
    label(ax, x + 1.85, 4.01, "Internal knots + measured errors", size=12)
    label(ax, x + 1.85, 3.03, r"One network forward; no greedy search", size=11.6)
    # Main left-to-right data flow at a consistent height.
    for index in range(4):
        arrow(ax, (panel_x[index] + panel_w + .015, 5.95),
              (panel_x[index + 1] - .015, 5.95), color=COLORS[index], lw=2.2)
    # Explicit point-feature bypass to the selected decoder.
    # It is explained within each attention box instead of drawing a crossing
    # long edge: selected-node Q/K/V roles are not confused with the main stream.
    # Training-only source and sinks.  The teacher is not called "offline":
    # current joint training does on-the-fly counterfactual teacher evaluations.
    box(ax, .65, .52, 18.70, 1.26, edge=MUTED, face="#F7F8FA", dash=True)
    label(ax, 1.02, 1.37, "TRAINING ONLY", size=12.5, weight="bold", color=MUTED, ha="left")
    label(ax, 4.86, 1.36, "Synthetic labels", size=13, weight="bold", color=COLORS[0])
    label(ax, 10.04, 1.36, "Counterfactual / compact-subset teacher", size=13, weight="bold", color=COLORS[2])
    label(ax, 16.70, 1.36, "Joint supervision", size=13, weight="bold", color=COLORS[3])
    label(ax, 4.86, .88, "True parameters + knot positions", size=11.4)
    label(ax, 10.04, .88, "Try masks / geometry; evaluate actual B-spline fits", size=11.4)
    label(ax, 16.70, .88, "Fit / peak error + keep / count / geometry", size=11.1)
    arrow(ax, (5.95, 2.49), (5.95, 1.81), color=MUTED, dashed=True, lw=1.3)
    arrow(ax, (9.99, 1.81), (9.99, 2.49), color=MUTED, dashed=True, lw=1.3)
    arrow(ax, (14.03, 1.81), (14.03, 2.49), color=MUTED, dashed=True, lw=1.3)
    label(ax, 10, .19,
          "Fixed-depth learned refinement; discrete selection is made once. Feasibility is measured after the final fit.",
          size=11.0, color=MUTED)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"framework.{suffix}", dpi=dpi)
    plt.close(fig)
    metadata = {
        "kind": "checkpoint_verified_architecture_schematic",
        "script": str(Path(__file__).resolve()),
        "checkpoints": records,
        "data_flow_source": "src/spline_fitting/models/v16_network.py",
        "notes": [
            "Curve and token glyphs are schematic, not experimental observations.",
            "Teacher evaluations occur only in training; they are not asserted to be precomputed offline.",
            "One mask means a single discrete subset decision, not a single attention layer.",
            "The error tolerance is an optimization target; the diagram does not claim guaranteed feasibility.",
            "M16/M32 are separate checkpoints, not a per-test-sample oracle model chooser.",
        ],
    }
    (output_dir / "framework.metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/figures/local_models_paper_20260922")
    parser.add_argument("--dpi", type=int, default=280)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("dpi must be >= 72")
    records = read_configs(args.checkpoints)
    make_figure(records, args.output_dir, args.dpi)
    print(f"Saved {args.output_dir / 'framework.png'}")
    print(f"Saved {args.output_dir / 'framework.svg'}")


if __name__ == "__main__":
    main()
