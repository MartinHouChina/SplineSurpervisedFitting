from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spline_fitting.evaluation.timing import (  # noqa: E402
    LatencyStatistics,
    measure_synchronized_wall_time,
    percentile,
)


def test_percentile_interpolates_and_validates_inputs() -> None:
    assert percentile([0.0, 10.0], 0.95) == pytest.approx(9.5)
    with pytest.raises(ValueError, match="cannot be empty"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        percentile([1.0], 1.1)


def test_latency_statistics_exposes_serializable_summary() -> None:
    statistics = LatencyStatistics((1.0, 3.0, 2.0))
    assert statistics.p50_ms == 2.0
    assert statistics.p95_ms == pytest.approx(2.9)
    assert statistics.as_dict()["repeat_ms"] == [1.0, 3.0, 2.0]


def test_measure_cpu_excludes_warmups_from_repeat_sample() -> None:
    calls = 0

    def operation() -> int:
        nonlocal calls
        calls += 1
        return calls

    measured = measure_synchronized_wall_time(
        operation,
        synchronization_device="cpu",
        warmup_repeats=2,
        timing_repeats=3,
    )
    assert calls == 5
    assert measured.result == 5
    assert len(measured.latency.repeat_ms) == 3


def test_measure_cuda_synchronizes_before_and_after_each_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synchronized: list[torch.device] = []
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: synchronized.append(torch.device(device)),
    )
    measured = measure_synchronized_wall_time(
        lambda: "done",
        synchronization_device="cuda:0",
        warmup_repeats=2,
        timing_repeats=3,
    )
    assert measured.result == "done"
    # Once after all warmups, then immediately before and after every repeat.
    assert synchronized == [torch.device("cuda:0")] * 7


@pytest.mark.parametrize(
    ("warmup_repeats", "timing_repeats", "message"),
    [(-1, 1, "warmup_repeats"), (0, 0, "timing_repeats")],
)
def test_measure_validates_repeat_counts(
    warmup_repeats: int,
    timing_repeats: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        measure_synchronized_wall_time(
            lambda: None,
            synchronization_device="cpu",
            warmup_repeats=warmup_repeats,
            timing_repeats=timing_repeats,
        )
