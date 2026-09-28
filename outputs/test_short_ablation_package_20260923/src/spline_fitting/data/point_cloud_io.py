from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_ordered_point_cloud(
    path: str | Path,
    *,
    point_dim: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load one ordered 2D/3D point sequence from CSV, TXT, JSON, NPY or PT."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"point-cloud file does not exist: {source}")
    suffix = source.suffix.lower()
    raw: Any
    if suffix == ".npy":
        raw = np.load(source, allow_pickle=False)
    elif suffix in {".csv", ".txt", ".xyz"}:
        raw = np.loadtxt(source, delimiter="," if suffix == ".csv" else None)
    elif suffix == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("points", raw.get("point_cloud"))
            if raw is None:
                raise ValueError("JSON object must contain 'points' or 'point_cloud'")
    elif suffix in {".pt", ".pth"}:
        raw = torch.load(source, map_location="cpu", weights_only=True)
        if isinstance(raw, dict):
            raw = raw.get("points", raw.get("point_cloud"))
            if raw is None:
                raise ValueError("PT dictionary must contain 'points' or 'point_cloud'")
    else:
        raise ValueError(
            "unsupported point-cloud format; use .csv, .txt, .xyz, .json, .npy, .pt or .pth"
        )

    points = torch.as_tensor(raw, dtype=dtype)
    if points.ndim != 2:
        raise ValueError("point cloud must have shape [M, D]")
    if points.shape[0] < 4:
        raise ValueError("at least four ordered points are required for cubic fitting")
    if point_dim is not None and points.shape[1] != point_dim:
        raise ValueError(
            f"checkpoint expects point dimension {point_dim}, got {points.shape[1]}"
        )
    if points.shape[1] not in {2, 3}:
        raise ValueError("point coordinates must be 2D or 3D")
    if not torch.isfinite(points).all():
        raise ValueError("point cloud contains NaN or infinite coordinates")
    segment_lengths = (points[1:] - points[:-1]).norm(dim=-1)
    if float(segment_lengths.sum()) <= 1e-8:
        raise ValueError("point cloud has zero total chord length")
    return points.contiguous()


def normalize_ordered_point_cloud(points: torch.Tensor) -> dict[str, torch.Tensor]:
    """Apply the same centering, scaling and chord parameterization as training."""
    if points.ndim != 2:
        raise ValueError("points must have shape [M, D]")
    center = points.mean(dim=0)
    centered = points - center
    scale = centered.norm(dim=-1).amax().clamp_min(1e-8)
    normalized = centered / scale
    lengths = (normalized[1:] - normalized[:-1]).norm(dim=-1)
    chord_params = torch.cat(
        [
            normalized.new_zeros(1),
            torch.cumsum(lengths / lengths.sum().clamp_min(1e-8), dim=0),
        ]
    )
    return {
        "points": normalized,
        "chord_params": chord_params,
        "center": center,
        "scale": scale,
    }


def resample_ordered_point_cloud(
    points: torch.Tensor,
    num_points: int,
) -> torch.Tensor:
    """Linearly resample an ordered point sequence at uniform chord positions."""
    if points.ndim != 2:
        raise ValueError("points must have shape [M, D]")
    if num_points < 4:
        raise ValueError("num_points must be at least four")
    segment_lengths = (points[1:] - points[:-1]).norm(dim=-1)
    keep = torch.cat(
        [torch.ones(1, dtype=torch.bool, device=points.device), segment_lengths > 1e-8]
    )
    unique_points = points[keep]
    if unique_points.shape[0] < 2:
        raise ValueError("point cloud has zero total chord length")
    unique_lengths = (unique_points[1:] - unique_points[:-1]).norm(dim=-1)
    source = torch.cat(
        [
            unique_lengths.new_zeros(1),
            torch.cumsum(unique_lengths, dim=0),
        ]
    )
    source = source / source[-1].clamp_min(1e-8)
    target = torch.linspace(
        0.0,
        1.0,
        num_points,
        device=points.device,
        dtype=points.dtype,
    )
    right = torch.searchsorted(source, target, right=True).clamp(
        1, unique_points.shape[0] - 1
    )
    left = right - 1
    weight = (target - source[left]) / (source[right] - source[left]).clamp_min(1e-8)
    return unique_points[left] + weight.unsqueeze(-1) * (
        unique_points[right] - unique_points[left]
    )


def interpolate_parameters_by_chord(
    source_chord: torch.Tensor,
    source_parameters: torch.Tensor,
    target_chord: torch.Tensor,
) -> torch.Tensor:
    """Interpolate predicted parameters onto another ordered chord grid."""
    if source_chord.ndim != 1 or source_parameters.ndim != 1:
        raise ValueError("source_chord and source_parameters must be one-dimensional")
    if target_chord.ndim != 1:
        raise ValueError("target_chord must be one-dimensional")
    if source_chord.shape != source_parameters.shape:
        raise ValueError("source chord positions and parameters must share shape")
    if source_chord.numel() < 2:
        raise ValueError("at least two source positions are required")
    if target_chord.numel() < 2:
        raise ValueError("at least two target positions are required")
    if not source_chord.is_floating_point():
        raise ValueError("interpolation tensors must be floating point")
    if not (
        source_chord.device == source_parameters.device == target_chord.device
        and source_chord.dtype == source_parameters.dtype == target_chord.dtype
    ):
        raise ValueError("all interpolation tensors must share device and dtype")
    if not (
        torch.isfinite(source_chord).all()
        and torch.isfinite(source_parameters).all()
        and torch.isfinite(target_chord).all()
    ):
        raise ValueError("interpolation tensors must be finite")
    if torch.any(source_chord[1:] <= source_chord[:-1]):
        raise ValueError("source_chord must be strictly increasing")
    if torch.any(source_parameters[1:] < source_parameters[:-1]):
        raise ValueError("source_parameters must be non-decreasing")
    if torch.any(target_chord[1:] < target_chord[:-1]):
        raise ValueError("target_chord must be non-decreasing")
    tolerance = 8.0 * torch.finfo(source_chord.dtype).eps
    if (
        target_chord[0] < source_chord[0] - tolerance
        or target_chord[-1] > source_chord[-1] + tolerance
    ):
        raise ValueError("target_chord must lie inside the source chord domain")

    target = target_chord.clamp(source_chord[0], source_chord[-1])
    right = torch.searchsorted(source_chord, target, right=True).clamp(
        1, source_chord.numel() - 1
    )
    left = right - 1
    weight = (target - source_chord[left]) / (
        source_chord[right] - source_chord[left]
    )
    result = source_parameters[left] + weight * (
        source_parameters[right] - source_parameters[left]
    )
    result[0] = source_parameters[0]
    result[-1] = source_parameters[-1]
    return result
