from __future__ import annotations

# Script entry points add ``src`` to sys.path so they run from a source checkout.
# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.geospatial_curves import (
    NATURAL_EARTH_LAYERS,
    NATURAL_EARTH_VERSION,
    download_cached_file,
    load_line_features,
    natural_earth_geojson_url,
    prepare_geospatial_curves,
    sha256_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download or read Natural Earth line data and create a grouped, "
            "manifest-backed real-world curve dataset."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Existing GeoJSON/Shapefile/ZIP; omit to download pinned GeoJSON.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "natural_earth",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resolution", choices=("10m", "50m", "110m"), default="10m")
    parser.add_argument(
        "--layer",
        choices=tuple(sorted(NATURAL_EARTH_LAYERS)),
        default="coastline",
    )
    parser.add_argument("--version", default=NATURAL_EARTH_VERSION)
    parser.add_argument("--url", default=None, help="Override the pinned download URL.")
    parser.add_argument("--expected-sha256", default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reference-points", type=int, default=768)
    parser.add_argument("--min-source-vertices", type=int, default=8)
    parser.add_argument("--max-source-vertices", type=int, default=2048)
    parser.add_argument("--min-length-m", type=float, default=1000.0)
    parser.add_argument("--tile-degrees", type=float, default=5.0)
    parser.add_argument("--split-seed", type=int, default=20260904)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    default_url = natural_earth_geojson_url(
        resolution=args.resolution,
        layer=args.layer,
        version=args.version,
    )
    source_url = args.url or default_url
    if args.input is None:
        filename = f"ne_{args.resolution}_{args.layer}.geojson"
        source_path = args.raw_dir / args.version / filename
        print(f"Natural Earth source: {source_url}", flush=True)
        source_path, source_sha256 = download_cached_file(
            source_url,
            source_path,
            expected_sha256=args.expected_sha256,
            force=args.force_download,
        )
    else:
        source_path = args.input
        source_sha256 = sha256_path(source_path)
        if args.expected_sha256 is not None:
            expected = args.expected_sha256.lower().strip()
            if source_sha256 != expected:
                raise ValueError(
                    f"SHA-256 mismatch for {source_path}: expected {expected}, "
                    f"got {source_sha256}"
                )
        source_url = str(source_path.resolve())

    output_dir = args.output_dir or (
        ROOT
        / "data"
        / "processed"
        / "natural_earth"
        / f"{args.version}_{args.resolution}_{args.layer}"
    )
    features = load_line_features(source_path)
    print(
        f"Read {len(features)} LineString parts; preparing local metric curves...",
        flush=True,
    )
    result = prepare_geospatial_curves(
        features,
        output_dir,
        source_dataset=f"natural_earth_{args.resolution}_{args.layer}",
        source_version=args.version,
        source_url=source_url,
        source_sha256=source_sha256,
        source_crs="EPSG:4326",
        reference_points=args.reference_points,
        min_source_vertices=args.min_source_vertices,
        max_source_vertices=args.max_source_vertices,
        min_length_m=args.min_length_m,
        tile_degrees=args.tile_degrees,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
    )
    print("Natural Earth preparation complete")
    print(f"  manifest: {result.manifest_path}")
    print(f"  samples: {result.sample_count}")
    print(f"  geographic groups: {result.group_count}")
    print(f"  splits: {result.split_counts}")
    print(
        "  rejected short/degenerate/duplicate: "
        f"{result.skipped_short}/{result.skipped_degenerate}/{result.duplicate_count}"
    )
    print("  knot labels: unavailable (geometry-only external evaluation)")


if __name__ == "__main__":
    main()
