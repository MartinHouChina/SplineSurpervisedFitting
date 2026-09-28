"""Shared external-source catalog; old snapshots are not independent datasets."""
from pathlib import Path


MANIFEST_RELATIVE_PATHS = {
    "UJI": "splits/uji_pen_v2.jsonl",
    "NaturalEarth": "processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl",
    "USGS": "processed/usgs_contours/large_scale/manifest.jsonl",
    "IndustrialOffset": "processed/industrial_offsets/v1/manifest.jsonl",
}
VALIDATION_SOURCE_NAMES = ("UJI", "NaturalEarth", "USGS")


def default_manifests(data_root):
    return {name: Path(data_root) / relative for name, relative in MANIFEST_RELATIVE_PATHS.items()}


def parse_manifests(values, *, defaults=None):
    """Add explicit independent sources; reject duplicate names and paths."""
    manifests = dict(defaults or {})
    paths = {path.resolve() for path in manifests.values()}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError("--manifest must be NAME=PATH")
        if name in manifests or name == "Synthetic":
            raise ValueError(f"Duplicate or reserved dataset name: {name}")
        path = Path(raw_path)
        if path.resolve() in paths:
            raise ValueError(f"Duplicate manifest path under another dataset name: {path}")
        manifests[name] = path
        paths.add(path.resolve())
    return manifests


def source_description(name, records):
    procedural = name == "IndustrialOffset" or any(
        record.get("metadata", {}).get("benchmark_kind") == "cad_driven_semi_synthetic_offset_curve"
        or record.get("source_dataset") == "industrial_model_offset_curves"
        for record in records
    )
    return {
        "source_kind": ("procedural_cad_offset" if procedural else
                        "observed_or_derived_real_world" if name in VALIDATION_SOURCE_NAMES else
                        "external_geometry_unspecified"),
        "source_note": ("Procedurally generated CAD-style offset curves; not measured industrial data; "
                        "no ground-truth B-spline knots." if procedural else
                        "External held-out observations; no ground-truth B-spline knots."),
        "dataset_label": f"{name} (procedural CAD; not measured)" if procedural else name,
    }


def validate_source_records(name, records, *, seen_sources=None):
    """Reject split leakage, duplicate source snapshots, and offset-family leakage."""
    if not records:
        raise ValueError(f"Empty external manifest: {name}")
    sources = {record["source_dataset"] for record in records}
    if seen_sources is not None:
        overlap = sources.intersection(seen_sources)
        if overlap:
            raise ValueError(f"Duplicate source_dataset in multiple manifests: {sorted(overlap)}")
        seen_sources.update(sources)
    groups, parents = {}, {}
    for record in records:
        split = record["split"]
        group = record["group_id"]
        if group in groups and groups[group] != split:
            raise ValueError(f"Group leakage between splits in {name}: {group}")
        groups[group] = split
        metadata = record.get("metadata", {})
        if source_description(name, [record])["source_kind"] == "procedural_cad_offset":
            parent = metadata.get("source_geometry_sha256")
            if not parent and "profile_family" in metadata and "profile_variant" in metadata:
                parent = (metadata["profile_family"], metadata["profile_variant"], metadata.get("generation_seed"))
            if not parent:
                raise ValueError(f"Industrial offset missing parent-profile provenance: {record['sample_id']}")
            if parent in parents and parents[parent] != split:
                raise ValueError(f"Industrial offset parent-profile leakage between splits: {parent}")
            parents[parent] = split
    if not any(record["split"] == "test" for record in records):
        raise ValueError(f"No test curves in {name}")
