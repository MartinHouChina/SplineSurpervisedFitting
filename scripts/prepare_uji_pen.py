from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.uji_pen import (  # noqa: E402
    UJI_DATA_FILENAME,
    UJI_DOWNLOAD_URL,
    download_uji_pen_v2,
    prepare_uji_pen_v2,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download/parse UJI Pen Characters v2 into continuous single-stroke "
            "curves and a writer-disjoint JSONL manifest."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        help=("Existing ujipenchars2.txt. Defaults to <raw-dir>/ujipenchars2.txt."),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the official UCI ZIP when the source file is absent.",
    )
    parser.add_argument("--download-url", default=UJI_DOWNLOAD_URL)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/uji_pen_v2"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/uji_pen_v2"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/uji_pen_v2.jsonl"),
    )
    parser.add_argument(
        "--split-summary",
        type=Path,
        default=Path("data/splits/uji_pen_v2_split.json"),
    )
    parser.add_argument(
        "--validation-writers",
        type=int,
        default=8,
        help="Writers held out from the 40 official training writers (default: 8).",
    )
    parser.add_argument("--split-seed", type=int, default=20260904)
    parser.add_argument(
        "--minimum-unique-points",
        type=int,
        default=4,
        help="Skip degenerate strokes shorter than this after duplicate removal.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Do not require the official 11640-sample, 60-writer counts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace matching processed files and manifests; unrelated files remain.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.validation_writers < 0:
        parser.error("--validation-writers must be non-negative")
    if args.minimum_unique_points < 2:
        parser.error("--minimum-unique-points must be at least two")

    source = args.source or (args.raw_dir / UJI_DATA_FILENAME)
    if not source.is_file():
        if not args.download:
            parser.error(
                f"source file does not exist: {source}; pass --download or --source"
            )
        print(f"Downloading official UJI archive from {args.download_url} ...")
        source = download_uji_pen_v2(
            args.raw_dir,
            url=args.download_url,
            overwrite=args.overwrite,
        )

    print(f"Parsing UJI strokes from {source} ...")
    result = prepare_uji_pen_v2(
        source,
        args.output_dir,
        args.manifest,
        args.split_summary,
        validation_writer_count=args.validation_writers,
        split_seed=args.split_seed,
        minimum_unique_points=args.minimum_unique_points,
        strict_official_counts=not args.allow_incomplete,
        overwrite=args.overwrite,
    )
    print("UJI Pen Characters v2 preparation complete")
    print(f"  source character samples: {result.source_sample_count}")
    print(f"  retained continuous strokes: {result.curve_count}")
    print(f"  skipped short/degenerate strokes: {result.skipped_stroke_count}")
    print(f"  curves by split: {result.split_curve_counts}")
    print(f"  writers by split: {result.split_writer_counts}")
    print(f"  manifest: {result.manifest_path}")
    print(f"  split summary: {result.split_summary_path}")
    print("  knot labels: absent (polyline vertices are not treated as knots)")


if __name__ == "__main__":
    main()
