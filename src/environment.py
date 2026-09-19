"""Collect reproducibility metadata: host, software versions, hardware, power state."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import socket
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@lru_cache(maxsize=1)
def cpu_name() -> str:
    if sys.platform == "darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown"


@lru_cache(maxsize=1)
def apple_gpu_info() -> dict[str, Any]:
    """GPU name / core count from system_profiler (macOS only)."""
    if sys.platform != "darwin":
        return {}
    raw = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=30)
    if not raw:
        return {}
    try:
        gpus = json.loads(raw).get("SPDisplaysDataType", [])
    except json.JSONDecodeError:
        return {}
    if not gpus:
        return {}
    g = gpus[0]
    return {"name": g.get("sppci_model"), "cores": g.get("sppci_cores"), "metal": g.get("spdisplays_mtlgpufamilysupport")}


def apple_gpu_name() -> str:
    info = apple_gpu_info()
    if not info.get("name"):
        return "Apple GPU (unknown)"
    cores = info.get("cores")
    return f"{info['name']} ({cores}-core GPU)" if cores else str(info["name"])


def power_state() -> dict[str, Any]:
    """Power source and low-power mode. Plugged-in vs. battery changes laptop performance."""
    state: dict[str, Any] = {"power_source": None, "battery_percent": None, "low_power_mode": None}
    if sys.platform == "darwin":
        batt = _run(["pmset", "-g", "batt"])
        if batt:
            first = batt.splitlines()[0]
            state["power_source"] = first.replace("Now drawing from", "").strip(" '")
            for tok in batt.replace(";", " ").split():
                if tok.endswith("%"):
                    state["battery_percent"] = tok
                    break
        pm = _run(["pmset", "-g"])
        if pm:
            for line in pm.splitlines():
                if "lowpowermode" in line.lower():
                    state["low_power_mode"] = line.split()[-1] == "1"
    else:
        batt = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
        if batt is not None:
            state["power_source"] = "AC" if batt.power_plugged else "Battery"
            state["battery_percent"] = f"{batt.percent:.0f}%"
    return state


def git_info() -> dict[str, Any]:
    commit = _run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"])
    if commit is None:
        return {"git_commit": None, "git_dirty": None, "git_note": "not a git repository (or git unavailable)"}
    status = _run(["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"])
    return {"git_commit": commit, "git_dirty": bool(status), "git_note": None}


def code_hash() -> str:
    """SHA-256 over the benchmark sources, so runs are traceable even without git."""
    h = hashlib.sha256()
    for folder in ("src", "experiments", "scripts", "configs"):
        for path in sorted((PROJECT_ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml"} and "__pycache__" not in path.parts:
                h.update(str(path.relative_to(PROJECT_ROOT)).encode())
                h.update(path.read_bytes())
    return h.hexdigest()[:16]


def collect_metadata(status, cfg: dict[str, Any], seed: int, run_id: str, experiment: str) -> dict[str, Any]:
    """Full run metadata for one (experiment, backend). `status` is a devices.BackendStatus."""
    import numpy as np
    import torch

    from . import MPS_FALLBACK_ENV_ORIGINAL

    try:
        import torchvision

        tv = torchvision.__version__
    except Exception:  # torchvision is optional for most experiments
        tv = None

    vm = psutil.virtual_memory()
    meta: dict[str, Any] = {
        "run_id": run_id,
        "experiment": experiment,
        "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": tv,
        "numpy": np.__version__,
        "backend": status.backend,
        "device": status.device,
        "device_name": status.device_name,
        "backend_available": status.available,
        "backend_unavailable_reason": status.reason,
        "cpu": cpu_name(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "torch_num_threads": torch.get_num_threads(),
        "ram_gb": round(vm.total / 1024**3, 2),
        "gpu": None,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_cpu_fallback": "disabled (PYTORCH_ENABLE_MPS_FALLBACK removed by this suite)",
        "mps_fallback_env_original": MPS_FALLBACK_ENV_ORIGINAL,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "seed": seed,
        "code_sha256": code_hash(),
        **git_info(),
        **power_state(),
        "config": cfg,
    }
    if status.backend == "cuda" and status.available:
        props = torch.cuda.get_device_properties(0)
        meta["gpu"] = props.name
        meta["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)
        meta["cuda_compute_capability"] = f"{props.major}.{props.minor}"
    elif status.backend == "mps" and status.available:
        meta["gpu"] = apple_gpu_name()
        meta["mps_recommended_max_memory_gb"] = round(torch.mps.recommended_max_memory() / 1024**3, 2)
        meta["mps_memory_model"] = "unified memory shared with the CPU"
    return meta
