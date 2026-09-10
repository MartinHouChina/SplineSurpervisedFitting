"""Evaluate trained v16 versus independent numerical baselines on paired data.

This is an explicit v16 entry point; v15 weights are not relabeled or converted.
The implementation and MSE/timing protocol are shared with the v15 benchmark.
"""
from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_v15_datasets import main as shared_main
from spline_fitting.checkpointing import V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION


def _has_option(arguments: list[str], option: str) -> bool:
    return any(
        value == option or value.startswith(option + "=")
        for value in arguments
    )


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not _has_option(arguments, "--max-knot-count"):
        arguments.extend(("--max-knot-count", "56"))
    return shared_main(
        arguments,
        expected_objective=V16_SUPERVISED_SUBSET_OBJECTIVE_VERSION,
        default_checkpoint=Path(
            "outputs/checkpoints/"
            "candidate_selection_v16_mse1e-4_k56_supervised.pt"
        ),
        default_output_dir=Path(
            "outputs/comparisons/v16_mse1e-4_k56_supervised"
        ),
    )


if __name__ == "__main__":
    main()
