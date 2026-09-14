"""Re-render saved six-method deployment geometry as vector PDF and HD PNG."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline


COLORS = {
    "ours": "#1665ad",
    "park_dominant_point_2007_adaptation": "#c44e52",
    "liang_feature_iki_2017_adaptation": "#55a868",
    "dung_direct_knot_2017_adaptation": "#8172b3",
    "kang_sparse_2015_adaptation": "#cc8a21",
    "luo_linf_de_2022_adaptation": "#4c4c4c",
}


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")[:110]


def evaluate(method: dict) -> np.ndarray:
    knots = np.asarray(method["full_knot_vector"], dtype=float)
    controls = np.asarray(method["control_points_normalized"], dtype=float)
    degree = len(knots) - len(controls) - 1
    grid = np.linspace(0.0, 1.0, 1200)
    return BSpline(knots, controls, degree, extrapolate=False)(grid)


def render(record: dict, target: Path, diagnostic: bool, tolerance: float, dpi: int):
    samples = np.asarray(record["network_input_points_normalized"], dtype=float)
    reference = np.asarray(record["original_reference_points_normalized"], dtype=float)
    figure, axes = plt.subplots(3, 2, figsize=(20, 18), constrained_layout=True)
    axes = axes.ravel()
    legend = None
    for axis, method in zip(axes, record["methods"], strict=True):
        axis.plot(reference[:, 0], reference[:, 1], color="#a8adb5", linewidth=2.0,
                  label="Original reference curve", zorder=1)
        axis.scatter(samples[:, 0], samples[:, 1], s=22, color="#2b7bbb", alpha=0.78,
                     edgecolors="white", linewidths=0.35, label="192 input samples", zorder=3)
        if method["status"] == "ok":
            curve = evaluate(method)
            controls = np.asarray(method["control_points_normalized"], dtype=float)
            knot_points = np.asarray(method["internal_knot_curve_points_normalized"], dtype=float)
            color = COLORS.get(method["method"], "#1f77b4")
            axis.plot(curve[:, 0], curve[:, 1], color=color, linewidth=3.4,
                      label="Final B-spline", zorder=5)
            axis.plot(controls[:, 0], controls[:, 1], "--", color="#e18727", linewidth=2.0,
                      label="Control polygon", zorder=4)
            axis.scatter(controls[:, 0], controls[:, 1], s=48, marker="s", color="#e18727",
                         edgecolors="white", linewidths=0.7, label="Control vertices", zorder=6)
            if knot_points.size:
                axis.scatter(knot_points[:, 0], knot_points[:, 1], s=58, marker="D",
                             color="#873c9b", edgecolors="white", linewidths=0.7,
                             label="Internal-knot locations", zorder=7)
            status = "PASS" if method["fit_pass"] else "FAIL"
            network = (
                f" | network={method['network_ms']:.2f} ms"
                if method.get("network_ms") is not None else ""
            )
            ref = (
                f" | ref MSE={method['reference_mse']:.2e}"
                if method.get("reference_mse") is not None else ""
            )
            axis.set_title(
                f"{method['label']} — {status}\n"
                f"input MSE={method['mse']:.2e}{ref} | K={method['final_k']}\n"
                f"full method={method['total_ms']:.2f} ms{network}",
                fontsize=13.5, weight="semibold", pad=9,
            )
        else:
            axis.set_title(f"{method['label']} — FAILED\n{method['error']}", fontsize=13)
        axis.set_aspect("equal", adjustable="datalim")
        axis.tick_params(labelsize=10)
        axis.grid(alpha=0.22, linewidth=0.8)
        axis.set_xlabel("normalized x", fontsize=11)
        axis.set_ylabel("normalized y", fontsize=11)
        if legend is None:
            legend = axis.get_legend_handles_labels()
    figure.suptitle(
        f"Six-method fit comparison — {record['dataset']} / {record['sample_id']}\n"
        f"shared MSE tolerance = {tolerance:.1e}",
        fontsize=20, weight="bold",
    )
    if legend:
        figure.legend(*legend, loc="lower center", ncol=5, fontsize=11, frameon=True)
    if diagnostic:
        figure.text(
            0.5, 0.5, "DIAGNOSTIC NOT FINAL — UNQUALIFIED CHECKPOINT",
            ha="center", va="center", rotation=24, fontsize=38,
            color="crimson", alpha=0.16, weight="bold", zorder=100,
        )
    figure.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(target.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(report["records"], 1):
        name = f"six_methods_hd__{slug(record['dataset'])}__{slug(record['sample_id'])}"
        print(f"[{index}/{len(report['records'])}] {name}", flush=True)
        render(
            record,
            args.output_dir / name,
            bool(report.get("diagnostic_not_final")),
            float(report["mse_tolerance"]),
            args.dpi,
        )


if __name__ == "__main__":
    main()
