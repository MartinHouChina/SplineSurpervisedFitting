"""Render the five requested paper deliverables from saved final comparison data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "outputs/comparisons/historical_best_dual_error_postprune_fresh_20260923/comparison.json"


def commands(report, checkpoint, output_dir, dpi=300, seed=20260922):
    common = ["--report", str(report), "--output-dir", str(output_dir), "--dpi", str(dpi)]
    return [
        [sys.executable, str(ROOT / "scripts/plot_postprune_framework.py"), *common, "--checkpoint", str(checkpoint)],
        [sys.executable, str(ROOT / "scripts/plot_postprune_case_panels.py"), *common, "--seed", str(seed)],
        [sys.executable, str(ROOT / "scripts/export_postprune_metrics.py"), *common],
        [sys.executable, str(ROOT / "scripts/export_postprune_parameters.py"), *common, "--checkpoint", str(checkpoint)],
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--checkpoint", type=Path, help="Optional local path override; contents must match the selected checkpoint hash")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args(argv)
    report_path = args.report.resolve()
    if not report_path.is_file():
        parser.error(f"Comparison report does not exist: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected_path = Path(report["metadata"]["checkpoint"])
    # Windows-recorded absolute paths are not absolute under Linux. Locate only
    # the same checkpoint basename, then verify its hash before any rendering.
    basename = str(selected_path).replace("\\", "/").rsplit("/", 1)[-1]
    checkpoint = args.checkpoint or (selected_path if selected_path.is_file() else ROOT / "outputs/checkpoints" / basename)
    if not checkpoint.is_file():
        parser.error("Selected checkpoint missing locally; supply --checkpoint with the matching file")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report["metadata"]["checkpoint_sha256"]:
        parser.error("Checkpoint contents differ from the measured model")
    if args.dpi < 72:
        parser.error("dpi must be >= 72")
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output directory must be new or empty; original figures will not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for command in commands(report_path, checkpoint.resolve(), output, args.dpi, args.seed):
        print("Rendering: " + Path(command[1]).name, flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    print(f"Completed framework, six-method cases, Ours showcase, metrics and parameters: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
