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
    reduced_for_budget: bool = False  # True if warmup/iterations were cut because the workload is very slow

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
    budget_s: float | None = None,
    min_iterations: int = 3,
) -> TimingResult:
    """Run `fn` warmup + iterations times. `on_sample(phase, index, latency_ms)` streams every
    individual measurement (phase is 'warmup' or 'measure') so callers can persist raw data.

    `budget_s` (optional) protects against pathologically slow configurations: if the FIRST call
    shows that warmup + iterations would exceed the budget, warmup is cut to 1 and the number of
    measured iterations to max(min_iterations, what fits), never more than requested. The result
    is flagged with `reduced_for_budget` so the reduction is never hidden."""
    result = TimingResult()
    planned_warmup, planned_iters = warmup, iterations
    i = 0
    while i < planned_warmup:
        with DeviceTimer(backend) as t:
            fn()
        result.warmup_ms.append(t.elapsed_ms)
        if on_sample is not None:
            on_sample("warmup", i, t.elapsed_ms)
        if i == 0 and budget_s and t.elapsed_ms / 1000 * (planned_warmup + planned_iters) > budget_s:
            fits = int(budget_s / (t.elapsed_ms / 1000)) - 1
            planned_warmup = 1
            planned_iters = min(iterations, max(min_iterations, fits))
            result.reduced_for_budget = True
        i += 1
    for i in range(planned_iters):
        with DeviceTimer(backend) as t:
            fn()
        result.samples_ms.append(t.elapsed_ms)
        if on_sample is not None:
            on_sample("measure", i, t.elapsed_ms)
    return result
