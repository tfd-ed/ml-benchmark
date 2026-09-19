"""Explicit device selection. Nothing here ever falls back from one backend to another."""
from __future__ import annotations

import gc
import random
from dataclasses import dataclass

import numpy as np
import torch

from . import environment

BACKENDS = ("cpu", "mps", "cuda")


@dataclass(frozen=True)
class BackendStatus:
    backend: str
    available: bool
    reason: str | None  # why it is unavailable (None when available)
    device: str  # torch device string
    device_name: str


def _cpu_status() -> BackendStatus:
    return BackendStatus("cpu", True, None, "cpu", environment.cpu_name())


def _mps_status() -> BackendStatus:
    if not torch.backends.mps.is_built():
        return BackendStatus("mps", False, "this PyTorch build has no MPS support", "mps", "n/a")
    if not torch.backends.mps.is_available():
        return BackendStatus("mps", False, "MPS not available (needs Apple Silicon and macOS >= 12.3)", "mps", "n/a")
    return BackendStatus("mps", True, None, "mps", environment.apple_gpu_name())


def _cuda_status() -> BackendStatus:
    if torch.version.cuda is None:
        return BackendStatus("cuda", False, "this PyTorch build has no CUDA support (torch.version.cuda is None)", "cuda:0", "n/a")
    if not torch.cuda.is_available():
        return BackendStatus("cuda", False, "torch.cuda.is_available() is False (no NVIDIA GPU or driver)", "cuda:0", "n/a")
    return BackendStatus("cuda", True, None, "cuda:0", torch.cuda.get_device_name(0))


def detect_backends() -> dict[str, BackendStatus]:
    return {"cpu": _cpu_status(), "mps": _mps_status(), "cuda": _cuda_status()}


def resolve_backends(spec: str) -> list[BackendStatus]:
    """'auto' -> all three backends (unavailable ones are reported, not hidden).
    'cpu' / 'mps' / 'cuda' / comma-separated list -> exactly those, unavailable ones included."""
    detected = detect_backends()
    spec = spec.lower().strip()
    names = list(BACKENDS) if spec == "auto" else [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in detected]
    if unknown:
        raise ValueError(f"unknown backend(s) {unknown}; choose from {list(BACKENDS)} or 'auto'")
    return [detected[n] for n in dict.fromkeys(names)]


def get_device(status: BackendStatus) -> torch.device:
    if not status.available:
        raise RuntimeError(f"backend '{status.backend}' is unavailable: {status.reason}")
    return torch.device(status.device)


def synchronize(backend: str) -> None:
    """Block until all queued work on the backend has finished. No-op on CPU (already synchronous)."""
    if backend == "cuda":
        torch.cuda.synchronize()
    elif backend == "mps":
        torch.mps.synchronize()


def empty_cache(backend: str) -> None:
    if backend == "cuda":
        torch.cuda.empty_cache()
    elif backend == "mps":
        torch.mps.empty_cache()


def cleanup(backend: str) -> None:
    """Release cached allocations between configurations (call outside `except` blocks)."""
    gc.collect()
    empty_cache(backend)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU, and CUDA/MPS generators when present
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def configure_numerics(allow_tf32: bool) -> None:
    """Make FP32 semantics explicit and identical to what the config says."""
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def assert_on_device(device: torch.device, *objs) -> None:
    """Guard against silent cross-device placement: raise if any tensor/module is elsewhere."""
    for obj in objs:
        tensors = list(obj.parameters()) + list(obj.buffers()) if isinstance(obj, torch.nn.Module) else [obj]
        for t in tensors:
            if t.device.type != device.type:
                raise RuntimeError(f"expected tensor on {device}, found it on {t.device}")
