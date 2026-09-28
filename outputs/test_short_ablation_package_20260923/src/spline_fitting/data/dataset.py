from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class CurvePointDataset(Dataset):
    """Dataset for ordered curve samples from files or in-memory arrays.

    Supported file formats:

    - ``.npy``: an ``[M, D]`` point array;
    - ``.txt`` / ``.csv``: ordered point rows;
    - ``.npz``: must contain ``points`` and may additionally contain
      ``parameters``, ``control_points``, ``knot_vector`` and ``degree``.

    Optional ``.npz`` spline metadata is returned when available. In
    particular, ``parameters`` may supervise ParameterHead during training.
    """

    def __init__(
        self,
        curves: Sequence[np.ndarray | torch.Tensor | Mapping[str, Any]] | None = None,
        files: Iterable[str | Path] | None = None,
        normalize: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if curves is None and files is None:
            raise ValueError("Either curves or files must be provided.")

        self.curves = list(curves) if curves is not None else None
        self.files = [Path(p) for p in files] if files is not None else None
        self.normalize = normalize
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.curves) if self.curves is not None else len(self.files)

    def _load_record(self, index: int) -> dict[str, Any]:
        if self.curves is not None:
            curve = self.curves[index]
            if isinstance(curve, Mapping):
                if "points" not in curve:
                    raise ValueError("In-memory curve mappings must contain 'points'.")
                return dict(curve)
            return {"points": curve}

        path = self.files[index]
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return {"points": np.load(path)}
        if suffix in {".txt", ".csv"}:
            return {
                "points": np.loadtxt(
                    path,
                    delimiter="," if suffix == ".csv" else None,
                )
            }
        if suffix == ".npz":
            with np.load(path) as archive:
                if "points" not in archive:
                    raise ValueError(f"NPZ file {path} does not contain 'points'.")
                return {name: archive[name] for name in archive.files}
        raise ValueError(f"Unsupported curve file: {path}")

    @staticmethod
    def _normalize(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center = points.mean(dim=0)
        centered = points - center
        scale = centered.norm(dim=-1).amax().clamp_min(1e-8)
        return centered / scale, center, scale

    @staticmethod
    def chord_length_parameters(points: torch.Tensor) -> torch.Tensor:
        segment_lengths = (points[1:] - points[:-1]).norm(dim=-1)
        total = segment_lengths.sum().clamp_min(1e-8)
        return torch.cat(
            [torch.zeros(1, dtype=points.dtype), torch.cumsum(segment_lengths / total, dim=0)]
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        record = self._load_record(index)
        points = torch.as_tensor(record["points"], dtype=self.dtype)
        if points.ndim != 2:
            raise ValueError(f"Curve must have shape [M, D], got {tuple(points.shape)}")
        if points.shape[0] < 4:
            raise ValueError("At least four ordered sample points are required.")

        if self.normalize:
            normalized, center, scale = self._normalize(points)
        else:
            normalized = points
            center = torch.zeros(points.shape[-1], dtype=points.dtype)
            scale = torch.ones((), dtype=points.dtype)

        result: dict[str, torch.Tensor | int] = {
            "points": normalized,
            "chord_params": self.chord_length_parameters(normalized),
            "center": center,
            "scale": scale,
            "sample_id": index,
        }

        # Preserve optional ground-truth spline metadata when available.
        if "parameters" in record:
            result["true_params"] = torch.as_tensor(record["parameters"], dtype=self.dtype)
        if "control_points" in record:
            control = torch.as_tensor(record["control_points"], dtype=self.dtype)
            if self.normalize:
                control = (control - center) / scale
            result["true_control_points"] = control
        if "knot_vector" in record:
            result["true_knot_vector"] = torch.as_tensor(record["knot_vector"], dtype=self.dtype)
        if "degree" in record:
            degree = np.asarray(record["degree"]).reshape(-1)[0]
            result["curve_degree"] = int(degree)
        return result
