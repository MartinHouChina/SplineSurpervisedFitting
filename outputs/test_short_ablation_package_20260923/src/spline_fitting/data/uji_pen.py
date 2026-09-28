from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

import numpy as np

from .real_world import write_curve_manifest


UJI_DATASET_NAME = "uji_pen_characters_v2"
UJI_DATASET_VERSION = "UCI dataset 177"
UJI_PREPROCESSING_VERSION = "uji-stroke-curves-v1"
UJI_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/177/"
    "uji%2Bpen%2Bcharacters%2Bversion%2B2.zip"
)
UJI_DATA_FILENAME = "ujipenchars2.txt"
UJI_EXPECTED_SAMPLE_COUNT = 11_640
UJI_EXPECTED_WRITER_COUNT = 60
UJI_EXPECTED_TRAIN_WRITERS = 40
UJI_EXPECTED_TEST_WRITERS = 20
UJI_UNITS_PER_MILLIMETRE = {"UJI": 100.0, "UPV": 152.0}

_HEADER_PATTERN = re.compile(
    r"^WORD\s+(?P<character>.+?)\s+"
    r"(?P<session>(?P<partition>trn|tst)_(?P<site>UJI|UPV)_"
    r"(?P<short_writer>W\d+)-(?P<repetition>\d+))\s*$"
)


@dataclass(frozen=True)
class UJIPenSample:
    source_index: int
    character: str
    session_id: str
    official_partition: str
    site: str
    writer_id: str
    short_writer_id: str
    repetition: str
    strokes: tuple[np.ndarray, ...]
    source_line: int


@dataclass(frozen=True)
class UJIPreparationResult:
    manifest_path: Path
    split_summary_path: Path
    source_sample_count: int
    curve_count: int
    skipped_stroke_count: int
    split_curve_counts: dict[str, int]
    split_writer_counts: dict[str, int]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_uji_pen_v2(
    raw_dir: str | Path,
    *,
    url: str = UJI_DOWNLOAD_URL,
    overwrite: bool = False,
) -> Path:
    """Download and extract the two official UCI files with the standard library."""

    destination = Path(raw_dir)
    destination.mkdir(parents=True, exist_ok=True)
    data_path = destination / UJI_DATA_FILENAME
    names_path = destination / "uji2.names"
    if data_path.is_file() and not overwrite:
        return data_path

    archive_path = destination / "uji_pen_characters_v2.zip"
    temporary_archive = archive_path.with_suffix(".zip.part")
    request = urllib.request.Request(url, headers={"User-Agent": "spline-fitting/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            with temporary_archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        temporary_archive.replace(archive_path)
    finally:
        if temporary_archive.exists():
            temporary_archive.unlink()

    with zipfile.ZipFile(archive_path) as archive:
        members_by_name = {Path(name).name: name for name in archive.namelist()}
        for filename, output_path in (
            (UJI_DATA_FILENAME, data_path),
            ("uji2.names", names_path),
        ):
            member = members_by_name.get(filename)
            if member is None:
                raise ValueError(f"official archive is missing {filename}")
            temporary_output = output_path.with_suffix(output_path.suffix + ".part")
            with archive.open(member) as source, temporary_output.open("wb") as output:
                shutil.copyfileobj(source, output)
            temporary_output.replace(output_path)
    return data_path


def _data_lines(handle: TextIO) -> Iterator[tuple[int, str]]:
    for line_number, line in enumerate(handle, start=1):
        stripped = line.strip()
        if stripped and not stripped.startswith("//"):
            yield line_number, stripped


def parse_uji_pen_v2(path: str | Path) -> Iterator[UJIPenSample]:
    """Parse the official UTF-8 ``ujipenchars2.txt`` stream.

    Strokes remain separate. Joining across a pen-up event would create an
    artificial curve segment for which the dataset recorded no trajectory.
    """

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"UJI source file does not exist: {source_path}")

    with source_path.open("r", encoding="utf-8-sig") as handle:
        lines = iter(_data_lines(handle))
        source_index = 0
        while True:
            try:
                header_line, header = next(lines)
            except StopIteration:
                return
            match = _HEADER_PATTERN.fullmatch(header)
            if match is None:
                raise ValueError(
                    f"invalid UJI WORD header on line {header_line}: {header!r}"
                )

            try:
                count_line, count_text = next(lines)
            except StopIteration as error:
                raise ValueError(
                    f"missing NUMSTROKES after line {header_line}"
                ) from error
            count_tokens = count_text.split()
            if len(count_tokens) != 2 or count_tokens[0] != "NUMSTROKES":
                raise ValueError(
                    f"invalid NUMSTROKES line {count_line}: {count_text!r}"
                )
            try:
                stroke_count = int(count_tokens[1])
            except ValueError as error:
                raise ValueError(
                    f"invalid stroke count on line {count_line}"
                ) from error
            if stroke_count <= 0:
                raise ValueError(f"stroke count must be positive on line {count_line}")

            strokes: list[np.ndarray] = []
            for _ in range(stroke_count):
                try:
                    points_line, points_text = next(lines)
                except StopIteration as error:
                    raise ValueError(
                        f"sample on line {header_line} ended before all strokes"
                    ) from error
                tokens = points_text.split()
                if len(tokens) < 3 or tokens[0] != "POINTS" or tokens[2] != "#":
                    raise ValueError(
                        f"invalid POINTS record on line {points_line}: {points_text!r}"
                    )
                try:
                    point_count = int(tokens[1])
                except ValueError as error:
                    raise ValueError(
                        f"invalid point count on line {points_line}"
                    ) from error
                coordinates = tokens[3:]
                if point_count <= 0 or len(coordinates) != 2 * point_count:
                    raise ValueError(
                        f"POINTS line {points_line} declares {point_count} points but "
                        f"contains {len(coordinates) // 2} coordinate pairs"
                    )
                try:
                    values = np.asarray(coordinates, dtype=np.int32).reshape(
                        point_count, 2
                    )
                except ValueError as error:
                    raise ValueError(
                        f"non-integer coordinate on line {points_line}"
                    ) from error
                strokes.append(values)

            session_id = match.group("session")
            writer_id = session_id.rsplit("-", maxsplit=1)[0]
            yield UJIPenSample(
                source_index=source_index,
                character=match.group("character"),
                session_id=session_id,
                official_partition=match.group("partition"),
                site=match.group("site"),
                writer_id=writer_id,
                short_writer_id=match.group("short_writer"),
                repetition=match.group("repetition"),
                strokes=tuple(strokes),
                source_line=header_line,
            )
            source_index += 1


def remove_consecutive_duplicate_points(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("UJI stroke points must have shape [M, 2]")
    if points.shape[0] <= 1:
        return points.copy()
    keep = np.ones(points.shape[0], dtype=np.bool_)
    keep[1:] = np.any(points[1:] != points[:-1], axis=1)
    return points[keep]


def select_validation_writers(
    official_train_writers: set[str],
    *,
    count: int,
    seed: int,
) -> set[str]:
    """Select a stable writer-level validation subset without RNG-version coupling."""

    if count < 0 or count >= len(official_train_writers):
        raise ValueError(
            "validation writer count must be non-negative and smaller than the "
            "official training writer count"
        )
    ranked = sorted(
        official_train_writers,
        key=lambda writer: hashlib.sha256(f"{seed}:{writer}".encode()).hexdigest(),
    )
    return set(ranked[:count])


def _write_json(path: Path, payload: object, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_uji_pen_v2(
    source: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    split_summary_path: str | Path,
    *,
    validation_writer_count: int = 8,
    split_seed: int = 20260904,
    minimum_unique_points: int = 4,
    strict_official_counts: bool = True,
    overwrite: bool = False,
) -> UJIPreparationResult:
    """Convert UJI v2 into metric, single-stroke curves and a group-safe manifest."""

    if minimum_unique_points < 2:
        raise ValueError("minimum_unique_points must be at least two")
    source_path = Path(source)
    destination = Path(output_dir)
    curves_dir = destination / "curves"
    manifest = Path(manifest_path)
    split_summary = Path(split_summary_path)
    existing_curves = (
        next(curves_dir.glob("*.npy"), None) if curves_dir.is_dir() else None
    )
    if existing_curves is not None and not overwrite:
        raise FileExistsError(
            f"processed UJI curves already exist in {curves_dir}; use overwrite=True"
        )
    if manifest.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists: {manifest}")
    if split_summary.exists() and not overwrite:
        raise FileExistsError(f"split summary already exists: {split_summary}")

    samples = list(parse_uji_pen_v2(source_path))
    official_train_writers = {
        sample.writer_id for sample in samples if sample.official_partition == "trn"
    }
    official_test_writers = {
        sample.writer_id for sample in samples if sample.official_partition == "tst"
    }
    all_writers = official_train_writers | official_test_writers
    if official_train_writers & official_test_writers:
        raise ValueError("UJI writer appears in both official trn and tst partitions")
    if strict_official_counts:
        observed = (
            len(samples),
            len(all_writers),
            len(official_train_writers),
            len(official_test_writers),
        )
        expected = (
            UJI_EXPECTED_SAMPLE_COUNT,
            UJI_EXPECTED_WRITER_COUNT,
            UJI_EXPECTED_TRAIN_WRITERS,
            UJI_EXPECTED_TEST_WRITERS,
        )
        if observed != expected:
            raise ValueError(
                "incomplete or unexpected UJI v2 source: "
                f"samples/writers/trn/tst={observed}, expected {expected}"
            )

    validation_writers = select_validation_writers(
        official_train_writers,
        count=validation_writer_count,
        seed=split_seed,
    )
    train_writers = official_train_writers - validation_writers
    test_writers = official_test_writers
    writer_split = {
        **{writer: "train" for writer in train_writers},
        **{writer: "val" for writer in validation_writers},
        **{writer: "test" for writer in test_writers},
    }

    source_sha256 = sha256_file(source_path)
    curves_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    split_curve_counts = {"train": 0, "val": 0, "test": 0}
    for sample in samples:
        split = writer_split[sample.writer_id]
        scale = UJI_UNITS_PER_MILLIMETRE[sample.site]
        for stroke_index, raw_stroke in enumerate(sample.strokes):
            deduplicated = remove_consecutive_duplicate_points(raw_stroke)
            if deduplicated.shape[0] < minimum_unique_points:
                skipped.append(
                    {
                        "source_index": sample.source_index,
                        "session_id": sample.session_id,
                        "stroke_index": stroke_index,
                        "points_after_deduplication": int(deduplicated.shape[0]),
                        "reason": "too_few_unique_ordered_points",
                    }
                )
                continue
            metric_points = deduplicated.astype(np.float32) / np.float32(scale)
            curve_id = f"uji2_{sample.source_index:05d}_stroke_{stroke_index:02d}"
            output_path = curves_dir / f"{curve_id}.npy"
            temporary_path = output_path.with_suffix(".tmp.npy")
            np.save(temporary_path, metric_points, allow_pickle=False)
            temporary_path.replace(output_path)
            relative_points_path = Path(
                os.path.relpath(output_path.resolve(), manifest.resolve().parent)
            ).as_posix()
            records.append(
                {
                    "sample_id": curve_id,
                    "source_dataset": UJI_DATASET_NAME,
                    "group_id": sample.writer_id,
                    "split": split,
                    "points_path": relative_points_path,
                    "num_points": int(metric_points.shape[0]),
                    "point_dim": 2,
                    "has_knot_labels": False,
                    "metadata": {
                        "source_version": UJI_DATASET_VERSION,
                        "source_url": UJI_DOWNLOAD_URL,
                        "source_sha256": source_sha256,
                        "preprocessing_version": UJI_PREPROCESSING_VERSION,
                        "group_strategy": "writer_disjoint_official_test",
                        "source_sample_index": sample.source_index,
                        "source_line": sample.source_line,
                        "source_sample_id": (
                            f"{sample.session_id}:U+"
                            + "-".join(f"{ord(char):04X}" for char in sample.character)
                        ),
                        "character": sample.character,
                        "session_id": sample.session_id,
                        "writer_id": sample.writer_id,
                        "short_writer_id": sample.short_writer_id,
                        "site": sample.site,
                        "official_partition": sample.official_partition,
                        "repetition": sample.repetition,
                        "stroke_index": stroke_index,
                        "stroke_count": len(sample.strokes),
                        "raw_num_points": int(raw_stroke.shape[0]),
                        "consecutive_duplicates_removed": int(
                            raw_stroke.shape[0] - deduplicated.shape[0]
                        ),
                        "coordinate_units": "millimetres",
                        "native_units_per_millimetre": scale,
                        "coordinate_orientation": "x_right_y_down",
                        "pen_up_joined": False,
                        "knot_label_policy": "absent_not_inferred_from_polyline_vertices",
                    },
                }
            )
            split_curve_counts[split] += 1

    write_curve_manifest(records, manifest, overwrite=overwrite)
    split_writers = {
        "train": sorted(train_writers),
        "val": sorted(validation_writers),
        "test": sorted(test_writers),
    }
    split_writer_counts = {
        name: len(writers) for name, writers in split_writers.items()
    }
    summary = {
        "dataset": UJI_DATASET_NAME,
        "source_version": UJI_DATASET_VERSION,
        "source_url": UJI_DOWNLOAD_URL,
        "source_file": source_path.name,
        "source_sha256": source_sha256,
        "preprocessing_version": UJI_PREPROCESSING_VERSION,
        "source_sample_count": len(samples),
        "curve_unit": "one continuous pen-down stroke",
        "curve_count": len(records),
        "skipped_stroke_count": len(skipped),
        "minimum_unique_points": minimum_unique_points,
        "split_seed": split_seed,
        "validation_writer_count": validation_writer_count,
        "split_writer_counts": split_writer_counts,
        "split_curve_counts": split_curve_counts,
        "writers": split_writers,
        "writer_disjoint": not bool(
            (train_writers & validation_writers)
            | (train_writers & test_writers)
            | (validation_writers & test_writers)
        ),
        "official_test_partition_preserved": True,
        "manifest": os.path.relpath(manifest.resolve(), split_summary.resolve().parent),
        "manifest_sha256": sha256_file(manifest),
        "has_knot_labels": False,
        "skipped_strokes": skipped,
    }
    _write_json(split_summary, summary, overwrite=overwrite)
    return UJIPreparationResult(
        manifest_path=manifest,
        split_summary_path=split_summary,
        source_sample_count=len(samples),
        curve_count=len(records),
        skipped_stroke_count=len(skipped),
        split_curve_counts=split_curve_counts,
        split_writer_counts=split_writer_counts,
    )
