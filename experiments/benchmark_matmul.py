"""Experiment 1: dense matrix multiplication  C = A @ B  across matrix sizes.

Every individual measurement (including warmup) is stored. Inputs are generated on the
CPU from a seeded generator and copied to the device ONCE before timing, so every backend
multiplies bit-identical matrices. Only `A @ B` is inside the timed region.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.devices import assert_on_device  # noqa: E402
from src.metrics import MemoryTracker  # noqa: E402
from src.runner import RunContext, experiment_main  # noqa: E402
from src.timing import time_fn  # noqa: E402

EXPERIMENT = "matmul"


def make_inputs(n: int, dtype: torch.dtype, seed: int):
    g = torch.Generator().manual_seed(seed + n)
    a = torch.randn(n, n, generator=g)
    b = torch.randn(n, n, generator=g)
    return a, b  # FP32 masters on CPU; cast to `dtype` per experiment


def run_size(ctx: RunContext, n: int, dtype: torch.dtype, dtype_name: str, warmup: int, iterations: int,
             verify: bool, sample_memory: bool = False, extra: dict | None = None, budget_s: float | None = None) -> None:
    flops = 2.0 * n**3
    a32, b32 = make_inputs(n, dtype, ctx.seed)
    a = a32.to(dtype).to(ctx.device)
    b = b32.to(dtype).to(ctx.device)
    assert_on_device(ctx.device, a, b)
    ctx.sync()

    rel_err = None
    if verify and ctx.backend != "cpu":
        ref = (a32.to(dtype).float() @ b32.to(dtype).float())
        got = (a @ b).float().cpu()
        rel_err = float((got - ref).abs().max() / ref.abs().max())
        del ref, got

    rows: list[tuple[str, int, float]] = []
    with MemoryTracker(ctx.backend, sample=sample_memory) as mem:
        res = time_fn(lambda: a @ b, ctx.backend, warmup, iterations, on_sample=lambda p, i, ms: rows.append((p, i, ms)), budget_s=budget_s)
    assert_on_device(ctx.device, a, b)

    for phase, i, ms in rows:
        ctx.record(
            model="matmul", batch_size=None, sequence_length=None, precision=dtype_name,
            phase=phase, iteration=i, latency_ms=ms,
            throughput=flops / (ms / 1000.0) / 1e9, throughput_unit="GFLOPS",
            memory_mb=mem.result["memory_mb"], memory_kind=mem.result["memory_kind"],
            matrix_size=n, matmuls_per_sec=1000.0 / ms, flops_per_matmul=flops,
            rel_err_vs_cpu_fp32=rel_err, reduced_for_time_budget=res.reduced_for_budget, **(extra or {}),
        )
    s = res.stats()
    print(f"    n={n:5d} {dtype_name}: median {s['median']:.3f} ms  ({flops / (s['median'] / 1000) / 1e9:,.0f} GFLOPS)  "
          f"cv={s['cv'] if s['cv'] is not None else float('nan'):.3f}", flush=True)


def run(ctx: RunContext) -> None:
    cfg = ctx.exp_cfg
    dtype_name = cfg.get("dtype", "float32")
    dtype = getattr(torch, dtype_name)
    for n in cfg["sizes"]:
        ov = cfg.get("overrides_by_size", {}).get(n, {})
        warmup, iters = ov.get("warmup", cfg["warmup"]), ov.get("iterations", cfg["iterations"])
        try:
            run_size(ctx, n, dtype, dtype_name, warmup, iters, verify=n <= cfg.get("verify_max_size", 0))
        except Exception as exc:  # OOM / unsupported / anything else is recorded, then we continue
            ctx.record_failure(exc, model="matmul", precision=dtype_name, matrix_size=n)
        ctx.cleanup()


def main(argv=None) -> int:
    return experiment_main(EXPERIMENT, run, description="Matrix multiplication benchmark", argv=argv)


if __name__ == "__main__":
    sys.exit(main())
