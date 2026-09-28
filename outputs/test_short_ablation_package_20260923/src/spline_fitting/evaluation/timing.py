from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch


T = TypeVar("T")


def synchronize_device(device: torch.device | str) -> None:
    """Wait for queued CUDA work; CPU execution is already synchronous."""

    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""

    if not values:
        raise ValueError("values cannot be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("values must be finite")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class LatencyStatistics:
    """Synchronized wall-time observations for one explicitly scoped callable."""

    repeat_ms: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.repeat_ms:
            raise ValueError("repeat_ms cannot be empty")
        if any(not math.isfinite(value) or value < 0.0 for value in self.repeat_ms):
            raise ValueError("repeat_ms must contain finite non-negative values")

    @property
    def p50_ms(self) -> float:
        return float(statistics.median(self.repeat_ms))

    @property
    def median_ms(self) -> float:
        return self.p50_ms

    @property
    def mean_ms(self) -> float:
        return float(statistics.fmean(self.repeat_ms))

    @property
    def p95_ms(self) -> float:
        return percentile(self.repeat_ms, 0.95)

    @property
    def minimum_ms(self) -> float:
        return min(self.repeat_ms)

    @property
    def maximum_ms(self) -> float:
        return max(self.repeat_ms)

    def as_dict(self) -> dict[str, float | list[float]]:
        return {
            "p50_ms": self.p50_ms,
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
            "p95_ms": self.p95_ms,
            "minimum_ms": self.minimum_ms,
            "maximum_ms": self.maximum_ms,
            "repeat_ms": list(self.repeat_ms),
        }


@dataclass(frozen=True)
class TimedCall(Generic[T]):
    """Final callable result paired with synchronized latency statistics."""

    result: T
    latency: LatencyStatistics


def measure_synchronized_wall_time(
    function: Callable[[], T],
    *,
    synchronization_device: torch.device | str,
    warmup_repeats: int,
    timing_repeats: int,
) -> TimedCall[T]:
    """Measure an already-materialized callable with correct CUDA boundaries.

    The caller defines the semantic scope.  For pure network latency, inputs
    must already reside on ``synchronization_device`` and ``function`` must do
    nothing beyond the network forward.  Dataset work, host/device transfer,
    refitting, verification and search are therefore excluded by construction.
    """

    if warmup_repeats < 0:
        raise ValueError("warmup_repeats must be non-negative")
    if timing_repeats <= 0:
        raise ValueError("timing_repeats must be positive")
    device = torch.device(synchronization_device)
    result: T | None = None
    for _ in range(warmup_repeats):
        result = function()
    synchronize_device(device)

    durations: list[float] = []
    for _ in range(timing_repeats):
        synchronize_device(device)
        started_at = time.perf_counter_ns()
        result = function()
        synchronize_device(device)
        durations.append((time.perf_counter_ns() - started_at) / 1_000_000.0)
    assert result is not None
    return TimedCall(result=result, latency=LatencyStatistics(tuple(durations)))


__all__ = [
    "LatencyStatistics",
    "TimedCall",
    "measure_synchronized_wall_time",
    "percentile",
    "synchronize_device",
]
