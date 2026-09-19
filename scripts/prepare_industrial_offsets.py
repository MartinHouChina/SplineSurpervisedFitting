from __future__ import annotations

# Run directly from a source checkout without installing the package.
# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.industrial_offsets import (
    INDUSTRIAL_OFFSET_PREPROCESSING_VERSION,
    INDUSTRIAL_PROFILE_FAMILIES,
    prepare_industrial_offset_curves,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate grouped engineering-profile offset curves in the shared "
            "manifest format. This is a CAD-driven semi-synthetic external "
            "geometry benchmark, not a measured real-world dataset."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "industrial_offsets" / "v1",
    )
    parser.add_argument(
        "--family",
        action="append",
        choices=INDUSTRIAL_PROFILE_FAMILIES,
        default=[],
        help="Profile family to include; repeat it. The default includes all families.",
    )
    parser.add_argument("--variants-per-family", type=int, default=12)
    parser.add_argument(
        "--offset-fraction",
        action="append",
        type=float,
        default=[],
        help=(
            "Signed offset divided by half of the profile's maximum extent; "
            "repeat it. Defaults to ±0.02, ±0.04 and ±0.08."
        ),
    )
    parser.add_argument("--source-points", type=int, default=384)
    parser.add_argument("--reference-points", type=int, default=768)
    parser.add_argument("--miter-limit", type=float, default=4.0)
    parser.add_argument("--generation-seed", type=int, default=20260914)
    parser.add_argument("--split-seed", type=int, default=20260914)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    families = tuple(args.family) if args.family else INDUSTRIAL_PROFILE_FAMILIES
    offsets = (
        tuple(args.offset_fraction)
        if args.offset_fraction
        else (-0.08, -0.04, -0.02, 0.02, 0.04, 0.08)
    )
    result = prepare_industrial_offset_curves(
        args.output_dir,
        families=families,
        variants_per_family=args.variants_per_family,
        offset_fractions=offsets,
        source_points=args.source_points,
        reference_points=args.reference_points,
        miter_limit=args.miter_limit,
        generation_seed=args.generation_seed,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        overwrite=args.overwrite,
    )
    print("Industrial offset-curve preparation complete")
    print(f"  preprocessing: {INDUSTRIAL_OFFSET_PREPROCESSING_VERSION}")
    print(f"  manifest: {result.manifest_path}")
    print(f"  samples/groups: {result.sample_count}/{result.group_count}")
    print(f"  splits: {result.split_counts}")
    print(
        "  rejected self-intersection/degenerate/topology-change: "
        f"{result.rejected_self_intersection}/"
        f"{result.rejected_degenerate}/"
        f"{result.rejected_topology_change}"
    )
    print("  role: validation/test only; B-spline knot labels are unavailable")


if __name__ == "__main__":
    main()
