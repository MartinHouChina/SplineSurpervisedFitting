"""Create paper-ready Ours-only v16 fitting examples on held-out real curves.

This thin entry point deliberately reuses the case selection, model deployment,
qualification checks, timing, final refit and JSON export implemented by
``visualize_v16_real_deployments.py``.  It only supplies Ours-only defaults.
"""
from __future__ import annotations

import sys
from pathlib import Path

from visualize_v16_real_deployments import main as shared_main


DEFAULT_CHECKPOINT = Path(
    "outputs/checkpoints/candidate_selection_v16_mse1e-4_k56.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/figures/candidate_selection_v16_mse1e-4_k56/ours_cases"
)


def _has_option(arguments: list[str], option: str) -> bool:
    return any(value == option or value.startswith(option + "=") for value in arguments)


def with_ours_defaults(argv: list[str]) -> list[str]:
    """Add only missing defaults so every shared CLI option remains overridable."""
    arguments = list(argv)
    if not _has_option(arguments, "--ours-only"):
        arguments.append("--ours-only")
    if not _has_option(arguments, "--checkpoint"):
        arguments.extend(("--checkpoint", str(DEFAULT_CHECKPOINT)))
    if not _has_option(arguments, "--output-dir"):
        arguments.extend(("--output-dir", str(DEFAULT_OUTPUT_DIR)))
    return arguments


def main(argv: list[str] | None = None) -> dict:
    return shared_main(with_ours_defaults(list(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    main()
