"""Shared harness so every experiment has the same CLI, failure handling and result recording."""
from __future__ import annotations

import argparse
import datetime as dt
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from . import devices
from .config import load_config, resolve_path
from .environment import collect_metadata
from .reporting import ResultWriter

OOM_MARKERS = ("out of memory", "invalid buffer size", "insufficient memory", "failed to allocate", "cannot allocate memory")
UNSUPPORTED_MARKERS = (
    "not implemented", "not supported", "unsupported", "does not support", "does not have support",
    "not currently supported", "only supported on", "not available for", "no kernel", "could not run",
)


def classify_exception(exc: BaseException) -> tuple[str, str]:
    """Map an exception to (status, error_type). Statuses: 'oom' | 'unsupported' | 'error'."""
    msg = str(exc).lower()
    name = type(exc).__name__
    if isinstance(exc, MemoryError) or name == "OutOfMemoryError" or any(m in msg for m in OOM_MARKERS):
        return "oom", name
    if isinstance(exc, NotImplementedError) or any(m in msg for m in UNSUPPORTED_MARKERS):
        return "unsupported", name
    return "error", name


def describe_exception(exc: BaseException, limit: int = 600) -> str:
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


class NonFiniteLoss(RuntimeError):
    pass


def check_finite(loss: float, where: str = "") -> None:
    import math

    if not math.isfinite(loss):
        raise NonFiniteLoss(f"loss became {loss} {where}".strip())


@dataclass
class RunContext:
    experiment: str
    cfg: dict[str, Any]
    exp_cfg: dict[str, Any]
    status: Any  # devices.BackendStatus
    writer: ResultWriter
    seed: int
    results_dir: Any = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def backend(self) -> str:
        return self.status.backend

    @property
    def device(self) -> torch.device:
        return devices.get_device(self.status)

    def sync(self) -> None:
        devices.synchronize(self.backend)

    def cleanup(self) -> None:
        devices.cleanup(self.backend)

    def record(self, **fields: Any) -> dict[str, Any]:
        return self.writer.write(**fields)

    def record_failure(self, exc: BaseException, **fields: Any) -> str:
        """Record a failed/unsupported/OOM configuration row and return its status."""
        status, err_type = classify_exception(exc)
        self.record(status=status, error_type=err_type, error=describe_exception(exc), phase="config", **fields)
        print(f"    [{self.backend}] {status.upper()}: {describe_exception(exc, 200)}", flush=True)
        return status


def build_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--backend", default="auto", help="auto | cpu | mps | cuda | comma list (default: auto)")
    p.add_argument("--config", default=None, help="YAML config merged over configs/default.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="override a config value, e.g. matmul.iterations=5")
    p.add_argument("--results-dir", default=None, help="output root (default: config results_dir)")
    p.add_argument("--run-id", default=None, help="identifier shared by all files of one suite run")
    return p


def experiment_main(experiment: str, run_fn: Callable[[RunContext], None], section: str | None = None,
                    description: str = "", argv: list[str] | None = None,
                    extra_args: Callable[[argparse.ArgumentParser], None] | None = None) -> int:
    parser = build_parser(description or f"{experiment} benchmark")
    if extra_args:
        extra_args(parser)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    results_dir = resolve_path(args.results_dir or cfg.get("results_dir", "results"))
    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    seed = int(cfg["seed"])
    exp_cfg = cfg.get(section or experiment, {})
    fatal = 0

    for status in devices.resolve_backends(args.backend):
        meta = collect_metadata(status, cfg, seed, run_id, experiment)
        writer = ResultWriter(results_dir / "raw", experiment, status, meta)
        print(f"== {experiment} on {status.backend} ({status.device_name}) ==", flush=True)
        try:
            if not status.available:
                writer.write(status="unavailable", phase="config", error=status.reason, error_type="BackendUnavailable")
                print(f"  unavailable: {status.reason}", flush=True)
                continue
            devices.seed_everything(seed)
            devices.configure_numerics(bool(cfg.get("numerics", {}).get("allow_tf32", False)))
            ctx = RunContext(experiment, cfg, exp_cfg, status, writer, seed, results_dir)
            run_fn(ctx)
        except BaseException as exc:  # a fatal harness error is recorded, never hidden, and never stops other backends
            if isinstance(exc, KeyboardInterrupt):
                writer.write(status="error", phase="config", error_type="KeyboardInterrupt", error="interrupted by user")
                writer.close()
                raise
            fatal += 1
            writer.write(status=classify_exception(exc)[0], phase="config", error_type=type(exc).__name__,
                         error=describe_exception(exc), traceback=traceback.format_exc(limit=8))
            print(f"  FATAL in {experiment}/{status.backend}: {describe_exception(exc, 300)}", flush=True)
        finally:
            writer.close()
            if status.available:
                devices.cleanup(status.backend)
        print(f"  rows by status: {writer.counts}  ->  {writer.csv_path}", flush=True)
    return 1 if fatal else 0
