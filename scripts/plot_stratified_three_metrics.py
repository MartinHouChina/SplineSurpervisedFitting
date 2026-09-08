from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_knot_count_strata import render_three_metric_figure  # noqa: E402


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"report field {name!r} must be an object")
    return value


def render_report(
    report: Mapping[str, object],
    output_path: Path,
    *,
    dpi: int,
) -> None:
    per_k = report.get("per_k_summary")
    if not isinstance(per_k, Sequence) or isinstance(per_k, (str, bytes)) or not per_k:
        raise ValueError("report field 'per_k_summary' must be a non-empty array")
    if not all(isinstance(item, Mapping) for item in per_k):
        raise ValueError("every per_k_summary item must be an object")

    dataset = _mapping(report.get("dataset"), name="dataset")
    ours = _mapping(report.get("ours_deployment"), name="ours_deployment")
    mode = str(ours.get("mode", "learned"))
    if mode not in {"learned", "verified"}:
        raise ValueError("ours_deployment.mode must be learned or verified")
    verified_config = ours.get("verified_config", {})
    if not isinstance(verified_config, Mapping):
        verified_config = {}

    render_three_metric_figure(
        per_k,
        output_path,
        mse_tolerance=float(report["mse_tolerance"]),
        samples_per_k=int(dataset["samples_per_source_k"]),
        dpi=dpi,
        ours_deployment=mode,
        verified_parameterization=str(
            verified_config.get("parameterization_policy", "network")
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render MSE/final-knot-count/time panels from an existing stratified "
            "comparison report without resampling or rerunning either method."
        )
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.report.is_file():
        parser.error(f"report does not exist: {args.report}")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    output = args.output or args.report.with_name("stratified_three_metrics.png")
    if output.exists() and not args.overwrite:
        parser.error(f"output already exists: {output}; use --overwrite")

    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if not isinstance(report, Mapping):
            raise ValueError("report root must be an object")
        render_report(report, output, dpi=args.dpi)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(f"Saved three-metric figure: {output}")


if __name__ == "__main__":
    main()
