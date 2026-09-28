from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geospatial_curves import (
    canonical_geometry_hash,
    deterministic_group_split,
    remove_consecutive_duplicate_coordinates,
    resample_polyline,
)
from .real_world import write_curve_manifest


INDUSTRIAL_OFFSET_DATASET_NAME = "industrial_model_offset_curves"
INDUSTRIAL_OFFSET_PREPROCESSING_VERSION = "industrial-offset-curves-v1"
INDUSTRIAL_PROFILE_FAMILIES = (
    "elliptic_bore",
    "rounded_plate",
    "capsule_slot",
    "naca_airfoil",
    "radial_cam",
    "lobed_rotor",
    "keyway_bore",
)


class OffsetCurveError(ValueError):
    """An offset was rejected because it is not one usable simple contour."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class IndustrialOffsetPreparationResult:
    manifest_path: Path
    metadata_path: Path
    sample_count: int
    split_counts: dict[str, int]
    group_count: int
    rejected_self_intersection: int
    rejected_degenerate: int
    rejected_topology_change: int


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _signed_area(body: np.ndarray) -> float:
    following = np.roll(body, -1, axis=0)
    return 0.5 * float(
        np.sum(body[:, 0] * following[:, 1] - body[:, 1] * following[:, 0])
    )


def _canonical_closed_contour(coordinates: np.ndarray) -> np.ndarray:
    values = remove_consecutive_duplicate_coordinates(
        np.asarray(coordinates, dtype=np.float64),
        tolerance=1e-10,
    )
    if len(values) > 1 and np.linalg.norm(values[0] - values[-1]) <= 1e-9:
        values = values[:-1]
    if len(values) < 4:
        raise OffsetCurveError(
            "degenerate", "a closed contour needs at least four vertices"
        )
    scale = max(float(np.ptp(values, axis=0).max()), 1.0)
    edges = np.roll(values, -1, axis=0) - values
    if float(np.linalg.norm(edges, axis=1).min()) <= scale * 1e-10:
        raise OffsetCurveError("degenerate", "contour contains a zero-length edge")
    area = _signed_area(values)
    if not math.isfinite(area) or abs(area) <= scale * scale * 1e-10:
        raise OffsetCurveError("degenerate", "contour has negligible signed area")
    if area < 0.0:
        values = values[::-1]

    # A stable start point makes files and hashes independent of the generator phase.
    rounded = np.round(values, decimals=10)
    start = min(range(len(values)), key=lambda index: tuple(rounded[index]))
    values = np.concatenate((values[start:], values[:start]), axis=0)
    return np.ascontiguousarray(np.concatenate((values, values[:1]), axis=0))


def closed_polyline_self_intersects(
    coordinates: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> bool:
    """Return whether a closed polyline has a non-adjacent segment intersection."""

    values = np.asarray(coordinates, dtype=np.float64)
    if len(values) > 1 and np.linalg.norm(values[0] - values[-1]) <= tolerance:
        values = values[:-1]
    if len(values) < 4:
        return True
    starts = values
    ends = np.roll(values, -1, axis=0)
    minimum = np.minimum(starts, ends) - tolerance
    maximum = np.maximum(starts, ends) + tolerance
    count = len(values)

    for first_index in range(count):
        candidate_indices = np.arange(first_index + 2, count)
        if first_index == 0:
            candidate_indices = candidate_indices[candidate_indices != count - 1]
        if candidate_indices.size == 0:
            continue
        overlap = np.all(
            (maximum[first_index] >= minimum[candidate_indices])
            & (maximum[candidate_indices] >= minimum[first_index]),
            axis=1,
        )
        for second_index in candidate_indices[overlap]:
            if _segments_intersect(
                starts[first_index],
                ends[first_index],
                starts[int(second_index)],
                ends[int(second_index)],
                tolerance=tolerance,
            ):
                return True
    return False


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    *,
    tolerance: float,
) -> bool:
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    scale = max(
        float(np.linalg.norm(first_direction)),
        float(np.linalg.norm(second_direction)),
        1.0,
    )
    epsilon = tolerance * scale
    o1 = _cross_2d(first_direction, second_start - first_start)
    o2 = _cross_2d(first_direction, second_end - first_start)
    o3 = _cross_2d(second_direction, first_start - second_start)
    o4 = _cross_2d(second_direction, first_end - second_start)
    if ((o1 > epsilon and o2 < -epsilon) or (o1 < -epsilon and o2 > epsilon)) and (
        (o3 > epsilon and o4 < -epsilon) or (o3 < -epsilon and o4 > epsilon)
    ):
        return True

    def on_segment(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> bool:
        return bool(
            np.all(point >= np.minimum(start, end) - epsilon)
            and np.all(point <= np.maximum(start, end) + epsilon)
        )

    return bool(
        (abs(o1) <= epsilon and on_segment(first_start, first_end, second_start))
        or (abs(o2) <= epsilon and on_segment(first_start, first_end, second_end))
        or (abs(o3) <= epsilon and on_segment(second_start, second_end, first_start))
        or (abs(o4) <= epsilon and on_segment(second_start, second_end, first_end))
    )


def offset_closed_polyline(
    coordinates: np.ndarray,
    distance: float,
    *,
    miter_limit: float = 4.0,
) -> np.ndarray:
    """Offset one simple CCW polygon with miter joins and safe bevel fallback.

    Positive distances are outward and negative distances are inward.  Acute
    corners whose miter exceeds ``miter_limit * abs(distance)`` are bevelled.
    An offset that self-intersects, collapses, or reverses orientation is
    rejected rather than silently choosing one loop from an ambiguous result.
    """

    if not math.isfinite(distance) or distance == 0.0:
        raise ValueError("offset distance must be finite and non-zero")
    if not math.isfinite(miter_limit) or miter_limit < 1.0:
        raise ValueError("miter_limit must be finite and at least one")
    source = _canonical_closed_contour(coordinates)
    if closed_polyline_self_intersects(source):
        raise OffsetCurveError("self_intersection", "source contour self-intersects")
    body = source[:-1]
    scale = max(float(np.ptp(body, axis=0).max()), 1.0)
    edges = np.roll(body, -1, axis=0) - body
    edge_lengths = np.linalg.norm(edges, axis=1)
    if float(edge_lengths.min()) <= scale * 1e-10:
        raise OffsetCurveError("degenerate", "source contour has a zero-length edge")
    directions = edges / edge_lengths[:, None]
    outward_normals = np.stack((directions[:, 1], -directions[:, 0]), axis=1)

    result: list[np.ndarray] = []
    absolute_distance = abs(distance)
    for index, vertex in enumerate(body):
        previous = (index - 1) % len(body)
        previous_shift = vertex + distance * outward_normals[previous]
        next_shift = vertex + distance * outward_normals[index]
        denominator = _cross_2d(directions[previous], directions[index])
        if abs(denominator) <= 1e-12:
            candidate = 0.5 * (previous_shift + next_shift)
        else:
            parameter = (
                _cross_2d(
                    next_shift - previous_shift,
                    directions[index],
                )
                / denominator
            )
            candidate = previous_shift + parameter * directions[previous]
        miter_length = float(np.linalg.norm(candidate - vertex))
        if not np.isfinite(candidate).all():
            raise OffsetCurveError(
                "degenerate", "offset produced non-finite coordinates"
            )
        if miter_length > miter_limit * absolute_distance:
            result.extend((previous_shift, next_shift))
        else:
            result.append(candidate)

    raw_offset = np.asarray(result, dtype=np.float64)
    minimum_boundary_distance = _minimum_distance_to_closed_polyline(
        raw_offset,
        body,
    )
    if minimum_boundary_distance < absolute_distance * (1.0 - 1e-6):
        raise OffsetCurveError(
            "topology_change",
            "offset crossed the source medial axis or another boundary branch",
        )
    offset = _canonical_closed_contour(raw_offset)
    output_body = offset[:-1]
    source_area = abs(_signed_area(body))
    output_area = _signed_area(output_body)
    if output_area <= source_area * 1e-8:
        raise OffsetCurveError(
            "topology_change",
            "offset collapsed or reversed contour orientation",
        )
    if closed_polyline_self_intersects(offset, tolerance=scale * 1e-10):
        raise OffsetCurveError("self_intersection", "offset contour self-intersects")
    return offset


def _minimum_distance_to_closed_polyline(
    query: np.ndarray,
    source_body: np.ndarray,
) -> float:
    segment_start = source_body
    segment_vector = np.roll(source_body, -1, axis=0) - source_body
    squared_length = np.sum(segment_vector * segment_vector, axis=1)
    minimum = math.inf
    for point in query:
        relative = point[None, :] - segment_start
        fraction = np.sum(relative * segment_vector, axis=1) / squared_length
        fraction = np.clip(fraction, 0.0, 1.0)
        projection = segment_start + fraction[:, None] * segment_vector
        minimum = min(minimum, float(np.linalg.norm(point - projection, axis=1).min()))
    return minimum


def generate_industrial_profile(
    family: str,
    variant: int,
    *,
    source_points: int = 384,
    seed: int = 20260914,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate a deterministic engineering-style planar source contour in mm."""

    if family not in INDUSTRIAL_PROFILE_FAMILIES:
        raise ValueError(f"unsupported industrial profile family: {family}")
    if variant < 0:
        raise ValueError("variant must be non-negative")
    if source_points < 64:
        raise ValueError("source_points must be at least 64")
    family_index = INDUSTRIAL_PROFILE_FAMILIES.index(family)
    rng = np.random.default_rng(seed + 104_729 * family_index + 7_919 * variant)

    if family == "elliptic_bore":
        semi_major = float(rng.uniform(55.0, 95.0))
        aspect = float(rng.uniform(0.45, 0.9))
        theta = np.linspace(0.0, 2.0 * math.pi, source_points, endpoint=False)
        points = np.stack(
            (semi_major * np.cos(theta), semi_major * aspect * np.sin(theta)), axis=1
        )
        parameters = {"semi_major_mm": semi_major, "aspect_ratio": aspect}
    elif family == "rounded_plate":
        half_width = float(rng.uniform(55.0, 100.0))
        half_height = float(rng.uniform(30.0, 65.0))
        exponent = float(rng.uniform(3.0, 6.0))
        theta = np.linspace(0.0, 2.0 * math.pi, source_points, endpoint=False)
        cosine, sine = np.cos(theta), np.sin(theta)
        power = 2.0 / exponent
        points = np.stack(
            (
                half_width * np.sign(cosine) * np.abs(cosine) ** power,
                half_height * np.sign(sine) * np.abs(sine) ** power,
            ),
            axis=1,
        )
        parameters = {
            "half_width_mm": half_width,
            "half_height_mm": half_height,
            "superellipse_exponent": exponent,
        }
    elif family == "capsule_slot":
        radius = float(rng.uniform(22.0, 42.0))
        half_straight = float(rng.uniform(25.0, 75.0))
        half = max(source_points // 2, 32)
        right_angles = np.linspace(-math.pi / 2.0, math.pi / 2.0, half, endpoint=False)
        left_angles = np.linspace(
            math.pi / 2.0, 3.0 * math.pi / 2.0, half, endpoint=False
        )
        right = np.stack(
            (
                half_straight + radius * np.cos(right_angles),
                radius * np.sin(right_angles),
            ),
            axis=1,
        )
        left = np.stack(
            (
                -half_straight + radius * np.cos(left_angles),
                radius * np.sin(left_angles),
            ),
            axis=1,
        )
        points = np.concatenate((right, left), axis=0)
        parameters = {"radius_mm": radius, "half_straight_mm": half_straight}
    elif family == "naca_airfoil":
        chord = float(rng.uniform(100.0, 180.0))
        camber = float(rng.uniform(0.0, 0.05))
        camber_position = float(rng.uniform(0.3, 0.6))
        thickness = float(rng.uniform(0.09, 0.18))
        half = max(source_points // 2, 32)
        beta = np.linspace(0.0, math.pi, half + 1)
        x = 0.5 * (1.0 - np.cos(beta))
        yt = (
            5.0
            * thickness
            * (
                0.2969 * np.sqrt(x)
                - 0.1260 * x
                - 0.3516 * x**2
                + 0.2843 * x**3
                - 0.1036 * x**4
            )
        )
        yc = np.where(
            x < camber_position,
            camber / camber_position**2 * (2.0 * camber_position * x - x**2),
            camber
            / (1.0 - camber_position) ** 2
            * ((1.0 - 2.0 * camber_position) + 2.0 * camber_position * x - x**2),
        )
        slope = np.where(
            x < camber_position,
            2.0 * camber / camber_position**2 * (camber_position - x),
            2.0 * camber / (1.0 - camber_position) ** 2 * (camber_position - x),
        )
        angle = np.arctan(slope)
        upper = np.stack((x - yt * np.sin(angle), yc + yt * np.cos(angle)), axis=1)
        lower = np.stack((x + yt * np.sin(angle), yc - yt * np.cos(angle)), axis=1)
        points = chord * np.concatenate((upper[::-1], lower[1:-1]), axis=0)
        parameters = {
            "chord_mm": chord,
            "maximum_camber_fraction": camber,
            "camber_position_fraction": camber_position,
            "thickness_fraction": thickness,
        }
    elif family in {"radial_cam", "lobed_rotor"}:
        base_radius = float(rng.uniform(42.0, 75.0))
        theta = np.linspace(0.0, 2.0 * math.pi, source_points, endpoint=False)
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        if family == "radial_cam":
            primary = float(rng.uniform(0.08, 0.18))
            secondary = float(rng.uniform(0.02, 0.07))
            radius = base_radius * (
                1.0
                + primary * np.cos(theta - phase)
                + secondary * np.cos(2.0 * theta + 0.5 * phase)
            )
            parameters = {
                "base_radius_mm": base_radius,
                "primary_eccentricity": primary,
                "secondary_eccentricity": secondary,
                "phase_rad": phase,
            }
        else:
            lobes = int(rng.integers(5, 13))
            amplitude = float(rng.uniform(0.055, 0.13))
            radius = base_radius * (
                1.0
                + amplitude * np.cos(lobes * theta + phase)
                + 0.02 * np.cos(2.0 * theta - phase)
            )
            parameters = {
                "base_radius_mm": base_radius,
                "lobe_count": lobes,
                "lobe_amplitude": amplitude,
                "phase_rad": phase,
            }
        points = np.stack((radius * np.cos(theta), radius * np.sin(theta)), axis=1)
    else:  # keyway_bore
        radius = float(rng.uniform(45.0, 80.0))
        shoulder_angle = float(rng.uniform(math.radians(13.0), math.radians(23.0)))
        slot_half_width = float(rng.uniform(0.12, 0.22) * radius)
        slot_floor = float(rng.uniform(0.58, 0.78) * radius)
        right_angle = math.pi / 2.0 - shoulder_angle
        left_angle = math.pi / 2.0 + shoulder_angle
        first_count = max(source_points // 4, 16)
        second_count = max(source_points - first_count, 48)
        first_arc = np.stack(
            (
                radius
                * np.cos(np.linspace(0.0, right_angle, first_count, endpoint=False)),
                radius
                * np.sin(np.linspace(0.0, right_angle, first_count, endpoint=False)),
            ),
            axis=1,
        )
        shoulder_y = radius * math.sin(right_angle)
        notch = np.asarray(
            [
                [radius * math.cos(right_angle), shoulder_y],
                [slot_half_width, shoulder_y],
                [slot_half_width, slot_floor],
                [-slot_half_width, slot_floor],
                [-slot_half_width, shoulder_y],
                [radius * math.cos(left_angle), shoulder_y],
            ],
            dtype=np.float64,
        )
        remaining_angles = np.linspace(
            left_angle, 2.0 * math.pi, second_count, endpoint=False
        )
        remaining_arc = np.stack(
            (radius * np.cos(remaining_angles), radius * np.sin(remaining_angles)),
            axis=1,
        )
        points = np.concatenate((first_arc, notch, remaining_arc), axis=0)
        parameters = {
            "radius_mm": radius,
            "keyway_half_width_mm": slot_half_width,
            "keyway_floor_y_mm": slot_floor,
            "shoulder_angle_deg": math.degrees(shoulder_angle),
        }

    contour = _canonical_closed_contour(points)
    if closed_polyline_self_intersects(contour):
        raise RuntimeError(
            f"internal generator error: {family} profile self-intersects"
        )
    return contour, parameters


def prepare_industrial_offset_curves(
    output_dir: str | Path,
    *,
    families: Sequence[str] = INDUSTRIAL_PROFILE_FAMILIES,
    variants_per_family: int = 12,
    offset_fractions: Sequence[float] = (-0.08, -0.04, -0.02, 0.02, 0.04, 0.08),
    source_points: int = 384,
    reference_points: int = 768,
    miter_limit: float = 4.0,
    generation_seed: int = 20260914,
    split_seed: int = 20260914,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    overwrite: bool = False,
) -> IndustrialOffsetPreparationResult:
    """Create a grouped manifest of procedural industrial contour offsets.

    This is an external *CAD-driven semi-synthetic* geometry benchmark.  It has
    exact source/offset provenance but deliberately has no B-spline knot labels.
    All offsets of one base profile share a group and therefore a data split.
    """

    selected_families = tuple(dict.fromkeys(str(item) for item in families))
    unsupported = set(selected_families).difference(INDUSTRIAL_PROFILE_FAMILIES)
    if not selected_families or unsupported:
        raise ValueError(
            f"unsupported or empty profile families: {sorted(unsupported)}"
        )
    if variants_per_family < 1:
        raise ValueError("variants_per_family must be positive")
    if reference_points < 4:
        raise ValueError("reference_points must be at least four")
    normalized_offsets = tuple(float(value) for value in offset_fractions)
    if not normalized_offsets or any(
        not math.isfinite(value) or value == 0.0 for value in normalized_offsets
    ):
        raise ValueError("offset fractions must be finite, non-zero, and non-empty")
    if len(set(normalized_offsets)) != len(normalized_offsets):
        raise ValueError("offset fractions must be unique")
    if any(abs(value) >= 0.5 for value in normalized_offsets):
        raise ValueError("absolute offset fractions must be below 0.5")

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
    rejection_counts: Counter[str] = Counter()
    family_acceptance: Counter[str] = Counter()
    for family in selected_families:
        for variant in range(variants_per_family):
            source, profile_parameters = generate_industrial_profile(
                family,
                variant,
                source_points=source_points,
                seed=generation_seed,
            )
            source_body = source[:-1]
            characteristic_scale = 0.5 * float(np.ptp(source_body, axis=0).max())
            source_hash = canonical_geometry_hash(source)
            group_id = (
                f"{INDUSTRIAL_OFFSET_DATASET_NAME}:"
                f"{family}:variant-{variant:03d}:seed-{generation_seed}"
            )
            split = deterministic_group_split(
                group_id,
                seed=split_seed,
                train_fraction=train_fraction,
                val_fraction=val_fraction,
            )
            for offset_index, fraction in enumerate(normalized_offsets):
                distance_mm = fraction * characteristic_scale
                try:
                    offset = offset_closed_polyline(
                        source,
                        distance_mm,
                        miter_limit=miter_limit,
                    )
                    reference = resample_polyline(offset, reference_points).astype(
                        np.float32,
                        copy=False,
                    )
                    if not np.isfinite(reference).all():
                        raise OffsetCurveError(
                            "degenerate", "resampling produced non-finite points"
                        )
                except OffsetCurveError as error:
                    rejection_counts[error.reason] += 1
                    continue
                except ValueError:
                    rejection_counts["degenerate"] += 1
                    continue

                offset_token = (
                    f"m{abs(fraction):.4f}" if fraction < 0.0 else f"p{fraction:.4f}"
                )
                offset_token = offset_token.replace(".", "p")
                sample_id = (
                    f"industrial_offset_{family}_v{variant:03d}_"
                    f"o{offset_index:02d}_{offset_token}"
                )
                points_path = curves_dir / f"{sample_id}.npy"
                if points_path.exists() and not overwrite:
                    raise FileExistsError(
                        f"prepared curve already exists: {points_path}"
                    )
                np.save(points_path, reference, allow_pickle=False)
                center = reference.astype(np.float64).mean(axis=0)
                scale = float(
                    np.linalg.norm(reference.astype(np.float64) - center, axis=1).max()
                )
                offset_hash = canonical_geometry_hash(reference)
                records.append(
                    {
                        "sample_id": sample_id,
                        "source_dataset": INDUSTRIAL_OFFSET_DATASET_NAME,
                        "group_id": group_id,
                        "split": split,
                        "points_path": points_path.relative_to(root).as_posix(),
                        "num_points": int(reference.shape[0]),
                        "point_dim": 2,
                        "has_knot_labels": False,
                        "metadata": {
                            "benchmark_kind": "cad_driven_semi_synthetic_offset_curve",
                            "profile_family": family,
                            "profile_variant": variant,
                            "profile_parameters": profile_parameters,
                            "source_geometry_sha256": source_hash,
                            "offset_geometry_sha256": offset_hash,
                            "offset_fraction_of_half_max_extent": fraction,
                            "offset_distance_mm": distance_mm,
                            "offset_side": "outward" if fraction > 0.0 else "inward",
                            "offset_join": "miter_with_bevel_limit",
                            "miter_limit": miter_limit,
                            "topology_policy": (
                                "reject self-intersection, collapse, orientation reversal, "
                                "and non-finite geometry"
                            ),
                            "reference_sampling": "uniform_offset_polyline_chord",
                            "reference_point_count": int(reference.shape[0]),
                            "stored_coordinate_space": "millimetre",
                            "reference_normalization_transform": {
                                "center_mm": center.tolist(),
                                "scale_mm": scale,
                                "definition": "subtract mean, divide by maximum radial norm",
                            },
                            "preprocessing_version": INDUSTRIAL_OFFSET_PREPROCESSING_VERSION,
                            "generation_seed": generation_seed,
                            "split_seed": split_seed,
                            "group_strategy": "all offsets of one base profile stay together",
                            "knot_labels": "unavailable",
                        },
                    }
                )
                family_acceptance[family] += 1

    if not records:
        raise ValueError("no valid industrial offset curves were produced")
    records.sort(key=lambda item: str(item["sample_id"]))
    write_curve_manifest(records, manifest_path, overwrite=overwrite)
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "val", "test")
    }
    groups = {str(record["group_id"]) for record in records}
    summary = {
        "schema_version": 1,
        "preprocessing_version": INDUSTRIAL_OFFSET_PREPROCESSING_VERSION,
        "source_dataset": INDUSTRIAL_OFFSET_DATASET_NAME,
        "dataset_class": "CAD-driven semi-synthetic external geometry benchmark",
        "profile_families": list(selected_families),
        "variants_per_family": variants_per_family,
        "offset_fractions": list(normalized_offsets),
        "source_points": source_points,
        "reference_points": reference_points,
        "miter_limit": miter_limit,
        "generation_seed": generation_seed,
        "split_seed": split_seed,
        "split_fractions": {
            "train": train_fraction,
            "val": val_fraction,
            "test": 1.0 - train_fraction - val_fraction,
        },
        "sample_count": len(records),
        "group_count": len(groups),
        "split_counts": split_counts,
        "accepted_by_family": dict(sorted(family_acceptance.items())),
        "rejected": dict(sorted(rejection_counts.items())),
        "has_knot_labels": False,
        "training_role": "external validation/test only; never synthetic-label training",
        "evaluation_note": (
            "Report dense-reference MSE, threshold pass rate, retained knot count, "
            "and latency. Do not report knot precision/recall."
        ),
        "scope_note": (
            "Procedural engineering contours are not measured CAD data and must be "
            "reported as CAD-driven semi-synthetic offsets."
        ),
    }
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(metadata_path)
    return IndustrialOffsetPreparationResult(
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        sample_count=len(records),
        split_counts=split_counts,
        group_count=len(groups),
        rejected_self_intersection=rejection_counts["self_intersection"],
        rejected_degenerate=rejection_counts["degenerate"],
        rejected_topology_change=rejection_counts["topology_change"],
    )
