from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .real_world import write_curve_manifest


EARTH_RADIUS_M = 6_371_008.8
GEOSPATIAL_PREPROCESSING_VERSION = "geospatial_curves_v1"
USGS_CONTOURS_FEATURE_SERVER = (
    "https://cartowfs.nationalmap.gov/arcgis/rest/services/contours/FeatureServer"
)
NATURAL_EARTH_VERSION = "v5.1.2"
NATURAL_EARTH_LAYERS = frozenset(
    {
        "coastline",
        "rivers_lake_centerlines",
        "geographic_lines",
    }
)


@dataclass(frozen=True)
class LineFeature:
    """One ordered polyline part read from a vector source."""

    feature_id: str
    part_index: int
    coordinates: np.ndarray
    properties: dict[str, Any]


@dataclass(frozen=True)
class GeospatialPreparationResult:
    """Paths and counts produced by :func:`prepare_geospatial_curves`."""

    manifest_path: Path
    metadata_path: Path
    sample_count: int
    split_counts: dict[str, int]
    group_count: int
    skipped_short: int
    skipped_degenerate: int
    duplicate_count: int


def natural_earth_geojson_url(
    *,
    resolution: str = "10m",
    layer: str = "coastline",
    version: str = NATURAL_EARTH_VERSION,
) -> str:
    """Return the version-pinned Natural Earth GeoJSON URL."""

    if resolution not in {"10m", "50m", "110m"}:
        raise ValueError("Natural Earth resolution must be 10m, 50m or 110m")
    if layer not in NATURAL_EARTH_LAYERS:
        supported = ", ".join(sorted(NATURAL_EARTH_LAYERS))
        raise ValueError(f"unsupported Natural Earth layer; choose one of {supported}")
    if not re.fullmatch(r"v[0-9]+(?:\.[0-9]+){1,2}", version):
        raise ValueError("Natural Earth version must look like v5.1.2")
    filename = f"ne_{resolution}_{layer}.geojson"
    return (
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
        f"{version}/geojson/{filename}"
    )


def sha256_path(path: str | Path) -> str:
    """Hash one file or a directory tree without loading it all into memory."""

    source = Path(path)
    digest = hashlib.sha256()
    if source.is_file():
        with source.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()
    if source.is_dir():
        files = sorted(item for item in source.rglob("*") if item.is_file())
        if not files:
            raise ValueError(f"cannot hash empty directory: {source}")
        for item in files:
            digest.update(item.relative_to(source).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with item.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    digest.update(block)
            digest.update(b"\0")
        return digest.hexdigest()
    raise FileNotFoundError(f"vector source does not exist: {source}")


def download_cached_file(
    url: str,
    destination: str | Path,
    *,
    expected_sha256: str | None = None,
    force: bool = False,
    timeout: float = 120.0,
) -> tuple[Path, str]:
    """Download a source atomically, or reuse the immutable local cache."""

    target = Path(destination)
    if target.is_file() and not force:
        digest = sha256_path(target)
        _check_expected_hash(digest, expected_sha256, source=target)
        return target, digest
    if target.exists() and not target.is_file():
        raise ValueError(f"download destination is not a file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "SelfSurpervisedSplineFitting/real-world-data "
                "(research dataset preparation)"
            )
        },
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=target.name + ".",
            suffix=".part",
            dir=target.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                while block := response.read(1024 * 1024):
                    output.write(block)
        digest = sha256_path(temporary)
        _check_expected_hash(digest, expected_sha256, source=url)
        temporary.replace(target)
        return target, digest
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _check_expected_hash(
    actual: str,
    expected: str | None,
    *,
    source: str | Path,
) -> None:
    if expected is None:
        return
    normalized = expected.lower().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError("expected SHA-256 must contain exactly 64 hexadecimal digits")
    if actual != normalized:
        raise ValueError(
            f"SHA-256 mismatch for {source}: expected {normalized}, got {actual}"
        )


def load_line_features(path: str | Path) -> list[LineFeature]:
    """Read LineString parts from GeoJSON, ArcGIS JSON, Shapefile or GeoPackage.

    GeoJSON and ArcGIS JSON use only the Python standard library. ``.shp`` and
    zipped Shapefiles require the optional ``pyshp`` package. GeoPackage and
    FileGDB inputs require the optional ``geopandas`` stack.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"vector source does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix in {".json", ".geojson"}:
        payload = json.loads(source.read_text(encoding="utf-8"))
        return line_features_from_json(payload)
    if suffix == ".zip":
        return _load_lines_from_zip(source)
    if suffix == ".shp":
        return _load_lines_from_shapefile(source)
    if suffix in {".gpkg", ".gdb"} or source.is_dir():
        return _load_lines_with_geopandas(source)
    raise ValueError(
        "unsupported vector source; use .geojson, .json, .shp, .zip, .gpkg or .gdb"
    )


def line_features_from_json(payload: Mapping[str, Any]) -> list[LineFeature]:
    """Extract ordered line parts from GeoJSON or an ArcGIS FeatureSet."""

    if not isinstance(payload, Mapping):
        raise ValueError("vector JSON root must be an object")
    raw_features: Sequence[Any]
    root_type = str(payload.get("type", ""))
    if root_type == "FeatureCollection" or "features" in payload:
        value = payload.get("features")
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("vector JSON 'features' must be an array")
        raw_features = value
    elif root_type in {
        "Feature",
        "LineString",
        "MultiLineString",
        "GeometryCollection",
    }:
        raw_features = [payload]
    else:
        raise ValueError("JSON is neither GeoJSON nor an ArcGIS FeatureSet")

    results: list[LineFeature] = []
    for feature_index, raw in enumerate(raw_features):
        if not isinstance(raw, Mapping):
            continue
        properties_raw = raw.get("properties", raw.get("attributes", {}))
        properties = (
            {str(key): _json_safe(value) for key, value in properties_raw.items()}
            if isinstance(properties_raw, Mapping)
            else {}
        )
        geometry = raw.get("geometry", raw)
        if not isinstance(geometry, Mapping):
            continue
        feature_id = _feature_identifier(raw, properties, feature_index)
        for part_index, coordinates in enumerate(_geometry_line_parts(geometry)):
            values = np.asarray(coordinates, dtype=np.float64)
            if values.ndim != 2 or values.shape[1] < 2:
                continue
            results.append(
                LineFeature(
                    feature_id=feature_id,
                    part_index=part_index,
                    coordinates=np.ascontiguousarray(values[:, :2]),
                    properties=dict(properties),
                )
            )
    return results


def _geometry_line_parts(geometry: Mapping[str, Any]) -> Iterable[Sequence[Any]]:
    if "paths" in geometry:
        paths = geometry.get("paths", [])
        if isinstance(paths, Sequence):
            yield from paths
        return
    geometry_type = str(geometry.get("type", ""))
    coordinates = geometry.get("coordinates")
    if geometry_type == "LineString" and isinstance(coordinates, Sequence):
        yield coordinates
    elif geometry_type == "MultiLineString" and isinstance(coordinates, Sequence):
        yield from coordinates
    elif geometry_type == "GeometryCollection":
        children = geometry.get("geometries", [])
        if isinstance(children, Sequence):
            for child in children:
                if isinstance(child, Mapping):
                    yield from _geometry_line_parts(child)


def _feature_identifier(
    feature: Mapping[str, Any],
    properties: Mapping[str, Any],
    feature_index: int,
) -> str:
    if feature.get("id") not in {None, ""}:
        return str(feature["id"])
    lowered = {str(key).lower(): value for key, value in properties.items()}
    for key in (
        "permanent_identifier",
        "globalid",
        "objectid",
        "object_id",
        "fid",
        "id",
        "ne_id",
    ):
        if lowered.get(key) not in {None, ""}:
            return str(lowered[key])
    return f"feature_{feature_index:08d}"


def _load_lines_from_zip(path: Path) -> list[LineFeature]:
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if not name.endswith("/") and "__MACOSX" not in name
        )
        json_names = [
            name for name in names if Path(name).suffix.lower() in {".geojson", ".json"}
        ]
        if json_names:
            payload = json.loads(archive.read(json_names[0]).decode("utf-8"))
            return line_features_from_json(payload)

        shape_names = [name for name in names if Path(name).suffix.lower() == ".shp"]
        if not shape_names:
            raise ValueError(f"ZIP contains neither GeoJSON nor a Shapefile: {path}")
        stem = str(Path(shape_names[0]).with_suffix(""))
        members = [name for name in names if str(Path(name).with_suffix("")) == stem]
        with tempfile.TemporaryDirectory(prefix="spline_geo_") as directory:
            root = Path(directory)
            for name in members:
                destination = root / Path(name).name
                destination.write_bytes(archive.read(name))
            return _load_lines_from_shapefile(root / Path(shape_names[0]).name)


def _load_lines_from_shapefile(path: Path) -> list[LineFeature]:
    try:
        import shapefile  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "reading Shapefiles requires the optional dependency 'pyshp'; "
            "install it or provide GeoJSON"
        ) from error

    reader = shapefile.Reader(str(path))
    features: list[LineFeature] = []
    field_names = [field[0] for field in reader.fields[1:]]
    for feature_index, record in enumerate(reader.iterShapeRecords()):
        properties = {
            name: _json_safe(value)
            for name, value in zip(field_names, list(record.record), strict=True)
        }
        feature_id = _feature_identifier({}, properties, feature_index)
        geometry = record.shape.__geo_interface__
        for part_index, coordinates in enumerate(_geometry_line_parts(geometry)):
            values = np.asarray(coordinates, dtype=np.float64)
            if values.ndim == 2 and values.shape[1] >= 2:
                features.append(
                    LineFeature(
                        feature_id=feature_id,
                        part_index=part_index,
                        coordinates=np.ascontiguousarray(values[:, :2]),
                        properties=properties,
                    )
                )
    return features


def _load_lines_with_geopandas(path: Path) -> list[LineFeature]:
    try:
        import geopandas as gpd  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "reading GeoPackage/FileGDB sources requires the optional 'geopandas' stack"
        ) from error

    frame = gpd.read_file(path)
    features: list[LineFeature] = []
    for feature_index, (_, row) in enumerate(frame.iterrows()):
        geometry_value = row.get("geometry")
        if geometry_value is None:
            continue
        properties = {
            str(key): _json_safe(value)
            for key, value in row.items()
            if key != "geometry"
        }
        feature_id = _feature_identifier({}, properties, feature_index)
        for part_index, coordinates in enumerate(
            _geometry_line_parts(geometry_value.__geo_interface__)
        ):
            values = np.asarray(coordinates, dtype=np.float64)
            if values.ndim == 2 and values.shape[1] >= 2:
                features.append(
                    LineFeature(
                        feature_id=feature_id,
                        part_index=part_index,
                        coordinates=np.ascontiguousarray(values[:, :2]),
                        properties=properties,
                    )
                )
    return features


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def remove_consecutive_duplicate_coordinates(
    coordinates: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Remove invalid and consecutive duplicate 2D vertices."""

    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("coordinates must have shape [M, 2]")
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) <= 1:
        return np.ascontiguousarray(values)
    steps = np.linalg.norm(np.diff(values, axis=0), axis=1)
    keep = np.concatenate(([True], steps > tolerance))
    return np.ascontiguousarray(values[keep])


def canonicalize_polyline_direction(coordinates: np.ndarray) -> np.ndarray:
    """Choose a deterministic orientation, including for closed rings."""

    values = remove_consecutive_duplicate_coordinates(coordinates)
    if len(values) < 2:
        return values
    closed = np.linalg.norm(values[0] - values[-1]) <= 1e-12
    if closed and len(values) > 3:
        body = values[:-1]
        rounded = np.round(body, decimals=12)
        start = min(range(len(body)), key=lambda index: tuple(rounded[index]))
        forward = np.concatenate((body[start:], body[:start]), axis=0)
        reverse_body = body[::-1]
        reverse_start = len(body) - 1 - start
        reverse = np.concatenate(
            (reverse_body[reverse_start:], reverse_body[:reverse_start]), axis=0
        )
        if reverse.tobytes() < forward.tobytes():
            forward = reverse
        return np.ascontiguousarray(np.concatenate((forward, forward[:1]), axis=0))
    first = tuple(np.round(values[0], decimals=12))
    last = tuple(np.round(values[-1], decimals=12))
    return np.ascontiguousarray(values[::-1] if last < first else values)


def split_polyline_by_vertices(
    coordinates: np.ndarray,
    *,
    max_vertices: int,
    min_vertices: int,
) -> list[np.ndarray]:
    """Split long lines deterministically with one shared boundary vertex."""

    if min_vertices < 4:
        raise ValueError("min_vertices must be at least four")
    if max_vertices < min_vertices:
        raise ValueError("max_vertices must be at least min_vertices")
    values = np.asarray(coordinates, dtype=np.float64)
    if len(values) < min_vertices:
        return []
    if len(values) <= max_vertices:
        return [np.ascontiguousarray(values)]

    windows: list[np.ndarray] = []
    start = 0
    while start < len(values) - 1:
        end = min(start + max_vertices, len(values))
        remaining = len(values) - (end - 1)
        if 1 < remaining < min_vertices:
            end = len(values)
        window = values[start:end]
        if len(window) >= min_vertices:
            windows.append(np.ascontiguousarray(window))
        if end == len(values):
            break
        start = end - 1
    return windows


def project_lonlat_local_azimuthal(
    coordinates: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Project one lon/lat line to a local spherical azimuthal-equidistant CRS."""

    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("coordinates must have shape [M, 2]")
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite lon/lat coordinates are required")
    if np.any(np.abs(values[:, 1]) > 90.0):
        raise ValueError("latitude must lie in [-90, 90]")

    longitude = np.deg2rad(values[:, 0])
    latitude = np.deg2rad(values[:, 1])
    cartesian = np.stack(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ),
        axis=1,
    )
    center = cartesian.mean(axis=0)
    norm = float(np.linalg.norm(center))
    if norm <= 1e-12:
        longitude0 = float(np.unwrap(longitude).mean())
        latitude0 = float(latitude.mean())
    else:
        center /= norm
        longitude0 = math.atan2(float(center[1]), float(center[0]))
        latitude0 = math.asin(float(np.clip(center[2], -1.0, 1.0)))

    delta_longitude = (longitude - longitude0 + math.pi) % (2.0 * math.pi) - math.pi
    sin_latitude0 = math.sin(latitude0)
    cos_latitude0 = math.cos(latitude0)
    sin_latitude = np.sin(latitude)
    cos_latitude = np.cos(latitude)
    cosine_c = np.clip(
        sin_latitude0 * sin_latitude
        + cos_latitude0 * cos_latitude * np.cos(delta_longitude),
        -1.0,
        1.0,
    )
    c = np.arccos(cosine_c)
    sine_c = np.sin(c)
    if np.any((math.pi - c) < 1e-7):
        raise ValueError("a line reaches the antipode of its local projection center")
    k = np.ones_like(c)
    nonzero = c > 1e-12
    k[nonzero] = c[nonzero] / sine_c[nonzero]
    x = EARTH_RADIUS_M * k * cos_latitude * np.sin(delta_longitude)
    y = (
        EARTH_RADIUS_M
        * k
        * (
            cos_latitude0 * sin_latitude
            - sin_latitude0 * cos_latitude * np.cos(delta_longitude)
        )
    )
    points = np.ascontiguousarray(np.stack((x, y), axis=1))
    metadata: dict[str, float | str] = {
        "name": "local_azimuthal_equidistant_spherical",
        "source_crs": "EPSG:4326",
        "center_longitude_deg": math.degrees(longitude0),
        "center_latitude_deg": math.degrees(latitude0),
        "earth_radius_m": EARTH_RADIUS_M,
        "unit": "meter",
    }
    return points, metadata


def resample_polyline(coordinates: np.ndarray, num_points: int) -> np.ndarray:
    """Sample a polyline uniformly in chord length, preserving both endpoints."""

    if num_points < 4:
        raise ValueError("num_points must be at least four")
    values = remove_consecutive_duplicate_coordinates(coordinates, tolerance=1e-9)
    if len(values) < 2:
        raise ValueError("polyline has zero length")
    lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    total = float(lengths.sum())
    if not math.isfinite(total) or total <= 1e-8:
        raise ValueError("polyline has zero length")
    source = np.concatenate(([0.0], np.cumsum(lengths))) / total
    target = np.linspace(0.0, 1.0, num_points, dtype=np.float64)
    result = np.stack(
        [np.interp(target, source, values[:, dimension]) for dimension in range(2)],
        axis=1,
    )
    result[0] = values[0]
    result[-1] = values[-1]
    return np.ascontiguousarray(result)


def spatial_tile_group(
    coordinates: np.ndarray,
    *,
    source_dataset: str,
    tile_degrees: float,
) -> str:
    """Assign a line to a deterministic lon/lat tile using its spherical center."""

    if not 0.0 < tile_degrees <= 90.0:
        raise ValueError("tile_degrees must lie in (0, 90]")
    values = np.asarray(coordinates, dtype=np.float64)
    unwrapped_longitude = np.rad2deg(np.unwrap(np.deg2rad(values[:, 0])))
    longitude = ((float(unwrapped_longitude.mean()) + 180.0) % 360.0) - 180.0
    latitude = float(np.clip(values[:, 1].mean(), -90.0, 90.0 - 1e-12))
    longitude_index = int(math.floor((longitude + 180.0) / tile_degrees))
    latitude_index = int(math.floor((latitude + 90.0) / tile_degrees))
    longitude_start = -180.0 + longitude_index * tile_degrees
    latitude_start = -90.0 + latitude_index * tile_degrees
    return (
        f"{_slug(source_dataset)}:tile:"
        f"lon{longitude_start:+08.3f}:lat{latitude_start:+07.3f}:"
        f"size{tile_degrees:g}deg"
    )


def deterministic_group_split(
    group_id: str,
    *,
    seed: int = 20260904,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> str:
    """Hash a whole spatial group into train/val/test without random leakage."""

    if not 0.0 <= train_fraction <= 1.0:
        raise ValueError("train_fraction must lie in [0, 1]")
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError("val_fraction must lie in [0, 1]")
    if train_fraction + val_fraction > 1.0:
        raise ValueError("train_fraction + val_fraction must not exceed one")
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], byteorder="big") / float(1 << 64)
    if unit < train_fraction:
        return "train"
    if unit < train_fraction + val_fraction:
        return "val"
    return "test"


def canonical_geometry_hash(coordinates: np.ndarray) -> str:
    """Hash a line independently of its forward/reverse orientation."""

    values = np.round(np.asarray(coordinates, dtype=np.float64), decimals=9)
    forward = values.astype("<f8", copy=False).tobytes()
    reverse = values[::-1].astype("<f8", copy=False).tobytes()
    return hashlib.sha256(min(forward, reverse)).hexdigest()


def prepare_geospatial_curves(
    features: Iterable[LineFeature],
    output_dir: str | Path,
    *,
    source_dataset: str,
    source_version: str,
    source_url: str,
    source_sha256: str,
    source_crs: str = "EPSG:4326",
    reference_points: int = 768,
    min_source_vertices: int = 8,
    max_source_vertices: int = 2048,
    min_length_m: float = 100.0,
    tile_degrees: float = 5.0,
    split_seed: int = 20260904,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    max_samples: int | None = None,
    overwrite: bool = False,
) -> GeospatialPreparationResult:
    """Project, resample and write unlabeled geographic curves plus manifest."""

    if source_crs.upper() not in {"EPSG:4326", "OGC:CRS84", "CRS84"}:
        raise ValueError(
            "the dependency-free preparation path expects longitude/latitude "
            "coordinates in EPSG:4326/CRS84; reproject other sources first"
        )
    if reference_points < 4:
        raise ValueError("reference_points must be at least four")
    if min_length_m < 0.0 or not math.isfinite(min_length_m):
        raise ValueError("min_length_m must be finite and non-negative")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")

    root = Path(output_dir)
    manifest_path = root / "manifest.jsonl"
    metadata_path = root / "dataset_metadata.json"
    if not overwrite and (manifest_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"prepared dataset already exists in {root}; pass overwrite=True to replace it"
        )
    curves_dir = root / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    geometry_hashes: set[str] = set()
    skipped_short = 0
    skipped_degenerate = 0
    duplicate_count = 0
    sorted_features = sorted(
        features,
        key=lambda item: (str(item.feature_id), int(item.part_index)),
    )
    stop = False
    for feature in sorted_features:
        lonlat = canonicalize_polyline_direction(feature.coordinates)
        if len(lonlat) < min_source_vertices:
            skipped_short += 1
            continue
        feature_hash = canonical_geometry_hash(lonlat)
        if feature_hash in geometry_hashes:
            duplicate_count += 1
            continue
        geometry_hashes.add(feature_hash)
        windows = split_polyline_by_vertices(
            lonlat,
            max_vertices=max_source_vertices,
            min_vertices=min_source_vertices,
        )
        for window_index, window in enumerate(windows):
            try:
                projected, projection = project_lonlat_local_azimuthal(window)
                projected = remove_consecutive_duplicate_coordinates(
                    projected,
                    tolerance=1e-6,
                )
                length_m = float(
                    np.linalg.norm(np.diff(projected, axis=0), axis=1).sum()
                )
                if len(projected) < 4 or length_m < min_length_m:
                    skipped_degenerate += 1
                    continue
                reference = resample_polyline(projected, reference_points).astype(
                    np.float32,
                    copy=False,
                )
            except ValueError:
                skipped_degenerate += 1
                continue

            window_hash = canonical_geometry_hash(window)
            sample_id = (
                f"{_slug(source_dataset)}_{feature_hash[:12]}_"
                f"w{window_index:03d}_{window_hash[:8]}"
            )
            points_path = curves_dir / f"{sample_id}.npy"
            if points_path.exists() and not overwrite:
                raise FileExistsError(f"prepared curve already exists: {points_path}")
            np.save(points_path, reference, allow_pickle=False)
            group_id = spatial_tile_group(
                window,
                source_dataset=source_dataset,
                tile_degrees=tile_degrees,
            )
            split = deterministic_group_split(
                group_id,
                seed=split_seed,
                train_fraction=train_fraction,
                val_fraction=val_fraction,
            )
            center = reference.astype(np.float64).mean(axis=0)
            scale = float(
                np.linalg.norm(reference.astype(np.float64) - center, axis=1).max()
            )
            relative_path = points_path.relative_to(root).as_posix()
            records.append(
                {
                    "sample_id": sample_id,
                    "source_dataset": source_dataset,
                    "group_id": group_id,
                    "split": split,
                    "points_path": relative_path,
                    "num_points": int(reference.shape[0]),
                    "point_dim": 2,
                    "has_knot_labels": False,
                    "metadata": {
                        "source_feature_id": str(feature.feature_id),
                        "source_part_index": int(feature.part_index),
                        "source_window_index": window_index,
                        "source_geometry_sha256": window_hash,
                        "source_vertex_count": int(len(window)),
                        "reference_sampling": "uniform_projected_chord",
                        "reference_point_count": int(reference.shape[0]),
                        "projected_length_m": length_m,
                        "stored_coordinate_space": "local_metric",
                        "projection": projection,
                        "reference_normalization_transform": {
                            "center_m": center.tolist(),
                            "scale_m": scale,
                            "definition": "subtract mean, divide by maximum radial norm",
                            "note": (
                                "RealWorldCurveDataset recomputes this transform after "
                                "network-input resampling."
                            ),
                        },
                        "source_properties": feature.properties,
                        "source_version": source_version,
                        "source_url": source_url,
                        "source_sha256": source_sha256,
                        "preprocessing_version": GEOSPATIAL_PREPROCESSING_VERSION,
                        "split_seed": split_seed,
                        "group_strategy": f"centroid spatial tile ({tile_degrees:g} degrees)",
                        "knot_labels": "unavailable",
                    },
                }
            )
            if max_samples is not None and len(records) >= max_samples:
                stop = True
                break
        if stop:
            break

    if not records:
        raise ValueError("no usable line samples were produced from the vector source")
    records.sort(key=lambda item: str(item["sample_id"]))
    write_curve_manifest(records, manifest_path, overwrite=overwrite)
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "val", "test")
    }
    groups = {str(record["group_id"]) for record in records}
    summary = {
        "schema_version": 1,
        "preprocessing_version": GEOSPATIAL_PREPROCESSING_VERSION,
        "source_dataset": source_dataset,
        "source_version": source_version,
        "source_url": source_url,
        "source_sha256": source_sha256,
        "source_crs": source_crs,
        "stored_coordinate_space": "per-sample local metric",
        "projection": "spherical local azimuthal equidistant",
        "reference_points": reference_points,
        "min_source_vertices": min_source_vertices,
        "max_source_vertices": max_source_vertices,
        "min_length_m": min_length_m,
        "group_strategy": f"centroid spatial tile ({tile_degrees:g} degrees)",
        "split_seed": split_seed,
        "split_fractions": {
            "train": train_fraction,
            "val": val_fraction,
            "test": 1.0 - train_fraction - val_fraction,
        },
        "sample_count": len(records),
        "group_count": len(groups),
        "split_counts": split_counts,
        "skipped_short": skipped_short,
        "skipped_degenerate": skipped_degenerate,
        "duplicate_count": duplicate_count,
        "has_knot_labels": False,
        "evaluation_note": (
            "Do not report knot precision/recall. Report dense-reference geometry "
            "error, threshold pass rate, retained knot count and latency."
        ),
    }
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(metadata_path)
    return GeospatialPreparationResult(
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        sample_count=len(records),
        split_counts=split_counts,
        group_count=len(groups),
        skipped_short=skipped_short,
        skipped_degenerate=skipped_degenerate,
        duplicate_count=duplicate_count,
    )


def fetch_usgs_contours_region(
    *,
    bbox: Sequence[float],
    destination: str | Path,
    region_id: str,
    layer_id: int = 5,
    service_url: str = USGS_CONTOURS_FEATURE_SERVER,
    max_features: int = 2000,
    request_page_size: int = 1000,
    selection_seed: int = 20260904,
    expected_sha256: str | None = None,
    force: bool = False,
    timeout: float = 120.0,
    request_json: Callable[[str, Mapping[str, str], float], Mapping[str, Any]]
    | None = None,
) -> tuple[Path, str]:
    """Cache a deterministic bbox query from the official USGS contour service."""

    bounds = _validate_bbox(bbox)
    if layer_id < 0:
        raise ValueError("layer_id must be non-negative")
    if max_features < 1 or request_page_size < 1:
        raise ValueError("max_features and request_page_size must be positive")
    target = Path(destination)
    if target.is_file() and not force:
        cached = json.loads(target.read_text(encoding="utf-8"))
        _validate_usgs_cache_identity(
            cached,
            bbox=bounds,
            region_id=region_id,
            layer_id=layer_id,
            service_url=service_url,
            max_features=max_features,
            selection_seed=selection_seed,
        )
        digest = sha256_path(target)
        _check_expected_hash(digest, expected_sha256, source=target)
        return target, digest
    target.parent.mkdir(parents=True, exist_ok=True)
    requester = request_json or _arcgis_post_json
    layer_url = f"{service_url.rstrip('/')}/{layer_id}"
    metadata = dict(requester(layer_url, {"f": "json"}, timeout))
    _raise_arcgis_error(metadata, context="USGS layer metadata")
    object_id_field = str(metadata.get("objectIdField", "objectid"))
    query_url = layer_url + "/query"
    spatial_query = {
        "where": "1=1",
        "geometry": ",".join(f"{value:.10g}" for value in bounds),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "f": "json",
    }
    identifiers_payload = dict(
        requester(
            query_url,
            {**spatial_query, "returnIdsOnly": "true", "returnGeometry": "false"},
            timeout,
        )
    )
    _raise_arcgis_error(identifiers_payload, context="USGS contour ID query")
    raw_identifiers = identifiers_payload.get("objectIds", [])
    if not isinstance(raw_identifiers, Sequence):
        raise RuntimeError("USGS contour ID query did not return an objectIds array")
    identifiers = sorted({int(value) for value in raw_identifiers})
    if len(identifiers) > max_features:
        identifiers = sorted(
            identifiers,
            key=lambda value: hashlib.sha256(
                f"{selection_seed}:{region_id}:{value}".encode("utf-8")
            ).digest(),
        )[:max_features]
        identifiers.sort()

    features: list[Mapping[str, Any]] = []
    for start in range(0, len(identifiers), request_page_size):
        page = identifiers[start : start + request_page_size]
        payload = dict(
            requester(
                query_url,
                {
                    "objectIds": ",".join(str(value) for value in page),
                    "outFields": "*",
                    "returnGeometry": "true",
                    "returnZ": "false",
                    "returnM": "false",
                    "outSR": "4326",
                    "geometryPrecision": "9",
                    "f": "json",
                },
                timeout,
            )
        )
        _raise_arcgis_error(payload, context="USGS contour feature query")
        page_features = payload.get("features", [])
        if not isinstance(page_features, Sequence):
            raise RuntimeError("USGS contour feature query returned invalid features")
        features.extend(item for item in page_features if isinstance(item, Mapping))

    def feature_order(item: Mapping[str, Any]) -> tuple[int, str]:
        attributes = item.get("attributes", {})
        if isinstance(attributes, Mapping):
            lowered = {str(key).lower(): value for key, value in attributes.items()}
            value = lowered.get(object_id_field.lower(), lowered.get("objectid"))
            try:
                return int(value), str(value)
            except (TypeError, ValueError):
                pass
        return (2**63 - 1, json.dumps(item, sort_keys=True, default=str))

    features.sort(key=feature_order)
    cache = {
        "schema": "usgs_contours_bbox_cache_v1",
        "source_service": service_url,
        "layer_id": layer_id,
        "region_id": region_id,
        "bbox_wgs84": list(bounds),
        "selection_seed": selection_seed,
        "max_features": max_features,
        "available_feature_count": len(raw_identifiers),
        "selected_feature_count": len(identifiers),
        "object_id_field": object_id_field,
        "selected_object_ids": identifiers,
        "layer_metadata": metadata,
        "spatialReference": {"wkid": 4326},
        "features": features,
    }
    encoded = (
        json.dumps(cache, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    _check_expected_hash(digest, expected_sha256, source=layer_url)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(target)
    return target, digest


def _arcgis_post_json(
    url: str,
    parameters: Mapping[str, str],
    timeout: float,
) -> Mapping[str, Any]:
    body = urllib.parse.urlencode(parameters).encode("ascii")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": (
                "SelfSurpervisedSplineFitting/USGS-contour-preparation (research use)"
            ),
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"ArcGIS endpoint returned a non-object response: {url}")
    return payload


def _raise_arcgis_error(payload: Mapping[str, Any], *, context: str) -> None:
    error = payload.get("error")
    if error is not None:
        raise RuntimeError(f"{context} failed: {error}")


def _validate_usgs_cache_identity(
    payload: Any,
    *,
    bbox: Sequence[float],
    region_id: str,
    layer_id: int,
    service_url: str,
    max_features: int,
    selection_seed: int,
) -> None:
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "usgs_contours_bbox_cache_v1"
    ):
        raise ValueError(
            "existing USGS cache is not a recognized bbox snapshot; use a different "
            "destination or pass force=True"
        )
    identity = {
        "bbox_wgs84": list(bbox),
        "region_id": region_id,
        "layer_id": layer_id,
        "source_service": service_url,
        "max_features": max_features,
        "selection_seed": selection_seed,
    }
    mismatched = [key for key, value in identity.items() if payload.get(key) != value]
    if mismatched:
        names = ", ".join(mismatched)
        raise ValueError(
            f"existing USGS cache was created for different settings ({names}); "
            "use a different destination or pass force=True"
        )


def _validate_bbox(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox must contain west, south, east, north")
    west, south, east, north = (float(value) for value in bbox)
    if not all(math.isfinite(value) for value in (west, south, east, north)):
        raise ValueError("bbox values must be finite")
    if not -180.0 <= west < east <= 180.0:
        raise ValueError("bbox longitude must satisfy -180 <= west < east <= 180")
    if not -90.0 <= south < north <= 90.0:
        raise ValueError("bbox latitude must satisfy -90 <= south < north <= 90")
    return west, south, east, north


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "curve"
