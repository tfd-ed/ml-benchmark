"""Statistics and platform-aware memory measurement.

Memory numbers from different backends come from DIFFERENT APIs and are NOT
equivalent. Every memory value is stored together with a `memory_kind` label.
"""
from __future__ import annotations

import math
import resource
import statistics
import sys
import threading
import time
from typing import Any, Sequence

import psutil
import torch

MB = 1024**2

# --------------------------------------------------------------------------- statistics


def summarize(values: Sequence[float]) -> dict[str, float | None]:
    """Median / mean / sample std / CV / min / max. std and cv are None for n < 2."""
    vals = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return {"n": 0, "median": None, "mean": None, "std": None, "cv": None, "min": None, "max": None}
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else None
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "mean": mean,
        "std": std,
        "cv": (std / mean) if (std is not None and mean != 0) else None,
        "min": min(vals),
        "max": max(vals),
    }


def speedup(baseline_time: float | None, backend_time: float | None) -> float | None:
    """speedup = baseline_time / backend_time (>1 means faster than the baseline)."""
    if not baseline_time or not backend_time:
        return None
    return baseline_time / backend_time


def throughput_degradation(values: Sequence[float], fraction: float = 0.1) -> dict[str, float | None]:
    """Compare median throughput of the first vs. last `fraction` of samples.
    `degradation_pct` > 0 means throughput dropped over time."""
    vals = [v for v in values if v is not None and math.isfinite(v)]
    k = max(1, int(len(vals) * fraction))
    if len(vals) < 2 * k:
        return {"first_median": None, "last_median": None, "degradation_pct": None, "window_samples": k}
    first, last = statistics.median(vals[:k]), statistics.median(vals[-k:])
    return {
        "first_median": first,
        "last_median": last,
        "degradation_pct": 100.0 * (first - last) / first if first else None,
        "window_samples": k,
    }


# --------------------------------------------------------------------------- memory


def process_rss_mb() -> float:
    return psutil.Process().memory_info().rss / MB


def peak_rss_mb() -> float:
    """Process-lifetime peak RSS (ru_maxrss: bytes on macOS, KiB on Linux). Cannot be reset."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (peak if sys.platform == "darwin" else peak * 1024) / MB


def memory_snapshot(backend: str) -> dict[str, Any]:
    """Raw counters for a backend. Keys differ per backend on purpose."""
    snap: dict[str, Any] = {"process_rss_mb": process_rss_mb()}
    if backend == "cuda":
        snap.update(
            cuda_allocated_mb=torch.cuda.memory_allocated() / MB,
            cuda_reserved_mb=torch.cuda.memory_reserved() / MB,
            cuda_peak_allocated_mb=torch.cuda.max_memory_allocated() / MB,
            cuda_peak_reserved_mb=torch.cuda.max_memory_reserved() / MB,
        )
    elif backend == "mps":
        snap.update(
            mps_current_allocated_mb=torch.mps.current_allocated_memory() / MB,
            mps_driver_allocated_mb=torch.mps.driver_allocated_memory() / MB,
        )
    return snap


def current_memory_mb(backend: str) -> tuple[float, str]:
    """Instantaneous memory reading with no side effects (does not touch peak counters)."""
    if backend == "cuda":
        return torch.cuda.memory_allocated() / MB, "cuda_memory_allocated (current, not peak)"
    if backend == "mps":
        return torch.mps.driver_allocated_memory() / MB, "mps_driver_allocated_memory (current snapshot; includes allocator cache)"
    return process_rss_mb(), "process_rss (current snapshot; includes Python/torch baseline)"


class MemoryTracker:
    """Context manager measuring memory over a block of work.

    CUDA : exact peak allocator statistics (reset on entry).
    MPS  : PyTorch exposes NO peak counter. We report driver_allocated_memory (bytes the
           Metal driver holds for this process, including allocator cache). With
           sample=True a background thread polls it and the maximum observed is reported;
           this is a sampled lower bound of the true peak, not an exact peak.
    CPU  : process RSS. With sample=True the maximum polled RSS is reported.
           `use_lifetime_peak=True` (only valid in a fresh process) reports ru_maxrss.
    """

    def __init__(self, backend: str, sample: bool = False, interval_s: float = 0.005, use_lifetime_peak: bool = False):
        self.backend = backend
        self.sample = sample and backend in ("mps", "cpu")
        self.interval_s = interval_s
        self.use_lifetime_peak = use_lifetime_peak and backend == "cpu"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_sampled = 0.0
        self._peak_current = 0.0  # MPS only: sampled max of tensor bytes (current_allocated_memory)
        self.result: dict[str, Any] = {}

    def _poll(self) -> float:
        if self.backend == "mps":
            return torch.mps.driver_allocated_memory() / MB
        return process_rss_mb()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._peak_sampled = max(self._peak_sampled, self._poll())
            if self.backend == "mps":
                self._peak_current = max(self._peak_current, torch.mps.current_allocated_memory() / MB)
            time.sleep(self.interval_s)

    def __enter__(self) -> "MemoryTracker":
        if self.backend == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.before = memory_snapshot(self.backend)
        if self.sample:
            self._peak_sampled = self._poll()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
        after = memory_snapshot(self.backend)
        if self.sample:
            self._peak_sampled = max(self._peak_sampled, self._poll())
            if self.backend == "mps":
                self._peak_current = max(self._peak_current, torch.mps.current_allocated_memory() / MB)
        self.result = self._finalise(after)

    def _finalise(self, after: dict[str, Any]) -> dict[str, Any]:
        out = dict(after)
        if self.backend == "cuda":
            out["memory_mb"] = after["cuda_peak_allocated_mb"]
            out["memory_kind"] = "cuda_max_memory_allocated (exact peak of tensor allocations)"
        elif self.backend == "mps":
            if self.sample:
                out["mps_current_allocated_sampled_peak_mb"] = self._peak_current  # tensor bytes only; excludes driver/cache overhead
                out["memory_mb"] = self._peak_sampled
                out["memory_kind"] = "mps_driver_allocated_memory (sampled max, lower bound of peak; includes allocator cache)"
            else:
                out["memory_mb"] = after["mps_driver_allocated_mb"]
                out["memory_kind"] = "mps_driver_allocated_memory (snapshot after work, not a peak; includes allocator cache)"
        else:
            if self.use_lifetime_peak:
                out["memory_mb"] = peak_rss_mb()
                out["memory_kind"] = "process_peak_rss (ru_maxrss, whole process lifetime incl. Python/torch baseline)"
            elif self.sample:
                out["memory_mb"] = self._peak_sampled
                out["memory_kind"] = "process_rss (sampled max; includes Python/torch baseline)"
            else:
                out["memory_mb"] = after["process_rss_mb"]
                out["memory_kind"] = "process_rss (snapshot after work; includes Python/torch baseline)"
        return out


def device_memory_capacity_mb(backend: str) -> tuple[float | None, str]:
    """Nominal memory the backend can address, with a label describing what it is."""
    if backend == "cuda":
        return torch.cuda.get_device_properties(0).total_memory / MB, "dedicated VRAM"
    if backend == "mps":
        return torch.mps.recommended_max_memory() / MB, "Metal recommended max working set (unified memory)"
    return psutil.virtual_memory().total / MB, "system RAM"
