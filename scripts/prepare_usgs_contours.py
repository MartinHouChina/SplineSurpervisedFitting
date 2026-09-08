from __future__ import annotations

# Script entry points add ``src`` to sys.path so they run from a source checkout.
# ruff: noqa: E402

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.data.geospatial_curves import (
    USGS_CONTOURS_FEATURE_SERVER,
    fetch_usgs_contours_region,
    load_line_features,
    prepare_geospatial_curves,
    sha256_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download bounded USGS 1:24,000 contour queries or read cached vector "
            "files, then build a leakage-safe real-world curve manifest."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Existing ArcGIS JSON/GeoJSON/Shapefile/ZIP (repeatable).",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        action="append",
        default=[],
        help="Small WGS84 query extent (repeatable).",
    )
    parser.add_argument(
        "--region-id",
        action="append",
        default=[],
        help="Identifier paired by order with --bbox.",
    )
    parser.add_argument(
        "--bbox-file",
        type=Path,
        default=None,
        help="JSON list (or {'regions': [...]}) of {region_id,bbox} objects.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "usgs_contours",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "usgs_contours" / "large_scale",
    )
    parser.add_argument("--service-url", default=USGS_CONTOURS_FEATURE_SERVER)
    parser.add_argument("--layer-id", type=int, default=5)
    parser.add_argument("--max-features-per-region", type=int, default=2000)
    parser.add_argument("--request-page-size", type=int, default=1000)
    parser.add_argument("--selection-seed", type=int, default=20260904)
    parser.add_argument("--max-bbox-span-degrees", type=float, default=1.0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reference-points", type=int, default=768)
    parser.add_argument("--min-source-vertices", type=int, default=8)
    parser.add_argument("--max-source-vertices", type=int, default=2048)
    parser.add_argument("--min-length-m", type=float, default=100.0)
    parser.add_argument("--tile-degrees", type=float, default=0.25)
    # Seed 2 gives non-empty train/val/test groups for the bundled three-region
    # example while remaining a deterministic, performance-independent split.
    parser.add_argument("--split-seed", type=int, default=2)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


def load_region_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    if args.bbox_file is not None:
        raw = json.loads(args.bbox_file.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("regions")
        if not isinstance(raw, list):
            raise ValueError(
                "bbox file must contain a list or {'regions': [...]} object"
            )
        for index, item in enumerate(raw):
            if not isinstance(item, dict) or "bbox" not in item:
                raise ValueError(f"invalid bbox-file region at index {index}")
            regions.append(
                {
                    "region_id": str(item.get("region_id", f"region_{index:03d}")),
                    "bbox": [float(value) for value in item["bbox"]],
                }
            )

    if args.region_id and len(args.region_id) != len(args.bbox):
        raise ValueError(
            "repeat --region-id exactly once for every --bbox, or omit all"
        )
    for index, bbox in enumerate(args.bbox):
        region_id = args.region_id[index] if args.region_id else f"bbox_{index:03d}"
        regions.append({"region_id": str(region_id), "bbox": list(bbox)})
    for region in regions:
        bbox = region["bbox"]
        if len(bbox) != 4:
            raise ValueError(
                f"region {region['region_id']} bbox must contain four values"
            )
        if (
            float(bbox[2]) - float(bbox[0]) > args.max_bbox_span_degrees
            or float(bbox[3]) - float(bbox[1]) > args.max_bbox_span_degrees
        ):
            raise ValueError(
                f"region {region['region_id']} exceeds --max-bbox-span-degrees; "
                "split it into smaller tiles to avoid an unbounded service query"
            )
    return regions


def main() -> None:
    args = build_parser().parse_args()
    regions = load_region_specs(args)
    if not args.input and not regions:
        raise ValueError(
            "provide at least one --input, --bbox, or --bbox-file; the script never "
            "downloads the entire national contour service implicitly"
        )

    sources: list[tuple[Path, str, str]] = []
    for source in args.input:
        sources.append((source, sha256_path(source), str(source.resolve())))

    for region in regions:
        region_id = str(region["region_id"])
        safe_region_id = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in region_id
        ).strip("_")
        if not safe_region_id:
            safe_region_id = "region"
        destination = args.raw_dir / f"layer{args.layer_id}_{safe_region_id}.json"
        print(f"USGS query {region_id}: bbox={region['bbox']}", flush=True)
        path, digest = fetch_usgs_contours_region(
            bbox=region["bbox"],
            destination=destination,
            region_id=region_id,
            layer_id=args.layer_id,
            service_url=args.service_url,
            max_features=args.max_features_per_region,
            request_page_size=args.request_page_size,
            selection_seed=args.selection_seed,
            force=args.force_download,
        )
        sources.append(
            (path, digest, f"{args.service_url.rstrip('/')}/{args.layer_id}")
        )

    all_features = []
    source_descriptors = []
    for source_index, (path, digest, url) in enumerate(sources):
        features = load_line_features(path)
        source_descriptors.append(
            {"path": str(path.resolve()), "sha256": digest, "url": url}
        )
        for feature in features:
            properties = dict(feature.properties)
            properties["preparation_source_index"] = source_index
            properties["preparation_source_sha256"] = digest
            all_features.append(replace(feature, properties=properties))
        print(f"Read {len(features)} line parts from {path}", flush=True)

    if len(source_descriptors) == 1:
        combined_hash = str(source_descriptors[0]["sha256"])
    else:
        combined_hash = hashlib.sha256(
            "\n".join(
                sorted(descriptor["sha256"] for descriptor in source_descriptors)
            ).encode("ascii")
        ).hexdigest()
    result = prepare_geospatial_curves(
        all_features,
        args.output_dir,
        source_dataset="usgs_tnm_contours_large_scale",
        source_version=f"FeatureServer-layer-{args.layer_id}-cached-snapshot",
        source_url=f"{args.service_url.rstrip('/')}/{args.layer_id}",
        source_sha256=combined_hash,
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
    provenance_path = args.output_dir / "source_snapshots.json"
    if provenance_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"source provenance already exists: {provenance_path}; pass --overwrite"
        )
    temporary_provenance = provenance_path.with_suffix(provenance_path.suffix + ".tmp")
    temporary_provenance.write_text(
        json.dumps(source_descriptors, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary_provenance.replace(provenance_path)
    print("USGS contour preparation complete")
    print(f"  manifest: {result.manifest_path}")
    print(f"  source snapshots: {provenance_path}")
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
