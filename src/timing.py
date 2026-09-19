"""Reusable benchmark timer with correct accelerator synchronisation.

Each iteration is timed as   sync -> t0 -> fn() -> sync -> t1
so queued asynchronous GPU work is fully included and never leaks into the next
iteration. Timing uses time.perf_counter on every backend so numbers are
comparable; the (small, constant) synchronisation cost is part of the measurement.

Terminology used across the suite:
  latency      - wall time of ONE call, in milliseconds
  throughput   - work per second (GFLOPS, samples/s, images/s, tokens/s, env-steps/s)
  training time- wall time of a longer phase such as an epoch, in seconds
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .devices import synchronize
from .metrics import summarize


@dataclass
class TimingResult:
    warmup_ms: list[float] = field(default_factory=list)
    samples_ms: list[float] = field(default_factory=list)

    def stats(self) -> dict[str, float | None]:
        return summarize(self.samples_ms)


class DeviceTimer:
    """`with DeviceTimer(backend) as t: ...`  ->  t.elapsed_ms (synchronised on entry and exit)."""

    def __init__(self, backend: str):
        self.backend = backend
        self.elapsed_ms = float("nan")

    def __enter__(self) -> "DeviceTimer":
        synchronize(self.backend)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        synchronize(self.backend)
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0


def time_fn(
    fn: Callable[[], object],
    backend: str,
    warmup: int,
    iterations: int,
    on_sample: Callable[[str, int, float], None] | None = None,
) -> TimingResult:
    """Run `fn` warmup + iterations times. `on_sample(phase, index, latency_ms)` streams every
    individual measurement (phase is 'warmup' or 'measure') so callers can persist raw data."""
    result = TimingResult()
    for phase, count, sink in (("warmup", warmup, result.warmup_ms), ("measure", iterations, result.samples_ms)):
        for i in range(count):
            with DeviceTimer(backend) as t:
                fn()
            sink.append(t.elapsed_ms)
            if on_sample is not None:
                on_sample(phase, i, t.elapsed_ms)
    return result
