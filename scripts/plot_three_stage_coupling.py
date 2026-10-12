"""Draw the data interfaces and dependencies of the three learned stages."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]


def box(ax, xy, width, height, title, body, edge, face, *, title_size=11, body_size=9):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.5, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height - .045, title, ha="center", va="top",
            fontsize=title_size, fontweight="semibold", color=edge)
    ax.text(x + .025, y + height - .105, body, ha="left", va="top",
            fontsize=body_size, color="#202020", linespacing=1.35)
    return patch


def arrow(ax, start, end, label=None, *, color="#39434c", curve=0.0, label_offset=(0, .018)):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.35,
        color=color, connectionstyle=f"arc3,rad={curve}", shrinkA=3, shrinkB=3,
    )
    ax.add_patch(patch)
    if label:
        x = (start[0] + end[0]) / 2 + label_offset[0]
        y = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(x, y, label, ha="center", va="bottom", fontsize=8.2, color=color)


def draw(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(16.2, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(.5, .965, "Coupled proposal, selection, and survivor-conditioned relocation",
            ha="center", va="top", fontsize=18, fontweight="semibold", color="#16233a")
    ax.text(.5, .925,
            "Each stage resolves one part of the constrained objective and passes an explicit state to the next stage",
            ha="center", va="top", fontsize=10.5, color="#4b5663")

    blue, blue_fill = "#1764a5", "#edf5fc"
    orange, orange_fill = "#b56608", "#fff5e5"
    purple, purple_fill = "#6842a8", "#f4effb"
    green, green_fill = "#237541", "#eef8f1"

    # Inputs and Stage I.
    box(ax, (.025, .585), .13, .25, "Ordered samples",
        r"$\mathbf{Q}\in\mathbb{R}^{M\times d}$" "\n\nOrdered traversal\nis preserved.",
        "#3f4a54", "#f5f6f7", title_size=10.5)
    box(ax, (.18, .54), .245, .335, "Stage I  |  feasible search space",
        r"Input: $\mathbf{Q}$ and chord reference" "\n"
        r"Encode: $[\mathbf{q},\partial_s\mathbf{q},\partial_s^2\mathbf{q}]$" "\n"
        r"ParameterHead $\rightarrow\mathbf{t}^{p}$" "\n"
        r"Local cross-attention $\rightarrow\mathbf{U}^{p},\mathbf{H}^{p}$" "\n\n"
        "Role: recall-first ordered proposal; do not\nirreversibly miss high-residual regions.",
        blue, blue_fill)
    arrow(ax, (.155, .71), (.18, .71))

    # Stage II.
    box(ax, (.455, .54), .235, .335, "Stage II  |  one subset decision",
        r"Input: $(\mathbf{F},\mathbf{H}^{p},\mathbf{U}^{p},\mathbf{t}^{p},\epsilon)$" "\n"
        "Pairwise candidate interaction\n"
        r"Keep probabilities: $p_j=\sigma(\ell_j)$" "\n"
        "Adaptive mass-TopK + coverage anchors\n"
        r"Output: $\mathbf{m}$ and $K=\sum_jm_j$" "\n\n"
        "Role: allocate one curve-level capacity, not\nindependent candidate classifications.",
        orange, orange_fill)
    arrow(ax, (.425, .70), (.455, .70))

    # Stage III and solver.
    box(ax, (.72, .54), .255, .335, "Stage III  |  adapt after deletion",
        r"Input: $\mathbf{m}$, survivors, and $\mathbf{F}$" "\n"
        r"Memory: $\{\mathbf{H}_j^p:m_j=1\}$ only" "\n"
        "Selected-neighbour gaps + relative rank\n"
        r"Survivor $\leftrightarrow$ point attention" "\n"
        r"Monotone update: $\mathbf{t}^p\mapsto\mathbf{t}'$" "\n"
        r"Bounded relocation: $\mathbf{U}^p\mapsto\mathbf{U}'$" "\n\n"
        "Role: compensate for changed supports without\nreintroducing rejected candidate tokens.",
        purple, purple_fill)
    arrow(ax, (.69, .70), (.72, .70))

    # Explicit state strips make the interfaces visible.
    box(ax, (.18, .33), .245, .15, "Interface $S_1$",
        r"$\{\mathbf{F},\mathbf{t}^p,\mathbf{U}^p,\mathbf{H}^p,\epsilon\}$" "\n"
        "Point memory + ordered candidate state",
        blue, "#ffffff", title_size=9.4, body_size=8.2)
    box(ax, (.455, .33), .235, .15, "Interface $S_2$",
        r"One KeepMask $\mathbf{m}$, count $K$, and survivors" "\n"
        "The same mask conditions update and extraction.",
        orange, "#ffffff", title_size=9.4, body_size=8.2)
    box(ax, (.72, .33), .255, .15, "Interface $S_3$",
        r"$\mathbf{t}'$ and variable-length $\mathbf{U}'[\mathbf{m}]$" "\n"
        "Strict order is retained in one parameter frame.",
        purple, "#ffffff", title_size=9.4, body_size=8.2)

    arrow(ax, (.302, .54), (.302, .48), color=blue)
    arrow(ax, (.572, .54), (.572, .48), color=orange)
    arrow(ax, (.847, .54), (.847, .48), color=purple)

    box(ax, (.30, .075), .40, .225, "Explicit CAD solve",
        r"Open-clamped cubic basis $\mathbf{A}(\mathbf{t}',\mathbf{U}')$" "\n"
        r"Endpoint-constrained least squares: $\mathbf{P}^*=\arg\min_{\mathbf{P}}\|\mathbf{A}\mathbf{P}-\mathbf{Q}\|_2^2$" "\n"
        r"Output: control polygon $\mathbf{P}^*$ and standard B-spline curve",
        green, green_fill, title_size=10.5, body_size=8.7)
    arrow(ax, (.847, .33), (.685, .30), r"$\mathbf{t}',\mathbf{U}'[\mathbf{m}]$", color=purple, curve=.08,
          label_offset=(.025, .0))
    arrow(ax, (.09, .585), (.30, .19), color="#3f4a54", curve=.22)

    ax.text(.5, .025,
            r"Deployment feasibility is checked after the explicit solve:  MSE $\leq\epsilon_\mu$  and  MaxSE $\leq\epsilon_\infty$.  "
            "Training-only teacher supervision is not part of the forward path.",
            ha="center", va="center", fontsize=9.2, color="#3f4a54")

    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix("." + suffix), dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    source = Path(__file__)
    metadata = {
        "schema_version": 1,
        "figure": "three_stage_coupling",
        "source": str(source.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "claim_scope": "Learned forward path followed by one explicit least-squares solve; training-only teacher and numerical deployment checks are shown outside the learned stages.",
        "outputs": [output.with_suffix("." + suffix).name for suffix in ("png", "pdf", "svg")],
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper/overnight/figures/three_stage_coupling",
    )
    args = parser.parse_args(argv)
    draw(args.output)
    print(args.output.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
