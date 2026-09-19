"""Experiment 5: FP32 vs FP16 vs BF16.

Two workloads per backend and precision:
  * matmul   - pure-dtype `A @ B` (inputs are cast to the dtype; no autocast)
  * training - GPT training step under `torch.autocast` (FP32 master weights, FP16 uses a GradScaler)

A precision is only recorded as supported if the workload actually ran. Anything that raises is stored
as `unsupported` (or `oom`/`error`); nothing is forced or emulated by this suite. Note that a mode that
*runs* on a backend can still be very slow (e.g. software-emulated half precision on a CPU).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from experiments.benchmark_matmul import run_size  # noqa: E402
from src.devices import assert_on_device  # noqa: E402
from src.metrics import MemoryTracker  # noqa: E402
from src.runner import RunContext, check_finite, experiment_main  # noqa: E402
from src.timing import time_fn  # noqa: E402
from src.workloads import build_gpt_workload  # noqa: E402

EXPERIMENT = "precision"
TORCH_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def run_training(ctx: RunContext, precision: str) -> None:
    tcfg, mcfg = ctx.exp_cfg["transformer"], ctx.cfg["transformer"]
    bs, seq = tcfg["batch_size"], tcfg["sequence_length"]
    w = build_gpt_workload(mcfg["model"], bs, seq, ctx.device, ctx.seed, mcfg["lr"], precision,
                           mcfg["data"]["num_sequences"], mcfg["data"]["branching"])
    assert_on_device(ctx.device, w.model)
    tokens = bs * seq
    common = dict(model="gpt", dataset="synthetic_markov", batch_size=bs, sequence_length=seq, precision=precision,
                  workload="training_autocast", n_params=w.n_params, tokens_per_step=tokens,
                  loss_scaling=w.scaler is not None)
    counter, last = {"i": 0}, []

    def one_step():
        last.append(w.step(counter["i"]))
        counter["i"] += 1
        check_finite(last[-1], f"(step {counter['i']})")

    with MemoryTracker(ctx.backend) as mem:
        res = time_fn(one_step, ctx.backend, tcfg["warmup_steps"], tcfg["steps"],
                      on_sample=lambda ph, i, ms: ctx.record(**common, phase=ph, iteration=i, latency_ms=ms, throughput=tokens / (ms / 1000.0),
                                                             throughput_unit="tokens/s", loss=last[-1], step_ms=ms,
                                                             memory_mb=None))
    ctx.record(**common, phase="config", memory_mb=mem.result["memory_mb"], memory_kind=mem.result["memory_kind"],
               **{k: v for k, v in mem.result.items() if k not in ("memory_mb", "memory_kind")})
    s = res.stats()
    print(f"    training {precision}: median step {s['median']:.1f} ms  {tokens / (s['median'] / 1000):,.0f} tokens/s  loss {last[0]:.4f}->{last[-1]:.4f}", flush=True)


def run(ctx: RunContext) -> None:
    cfg = ctx.exp_cfg
    m = cfg["matmul"]
    for precision in cfg["precisions"]:
        try:
            run_size(ctx, m["size"], TORCH_DTYPES[precision], precision, m["warmup"], m["iterations"], verify=False,
                     extra={"workload": "matmul_pure_dtype"})
        except Exception as exc:
            ctx.record_failure(exc, model="matmul", precision=precision, matrix_size=m["size"], workload="matmul_pure_dtype")
        ctx.cleanup()
    for precision in cfg["precisions"]:
        try:
            run_training(ctx, precision)
        except Exception as exc:
            ctx.record_failure(exc, model="gpt", dataset="synthetic_markov", precision=precision, workload="training_autocast",
                               batch_size=cfg["transformer"]["batch_size"], sequence_length=cfg["transformer"]["sequence_length"])
        ctx.cleanup()


def main(argv=None) -> int:
    return experiment_main(EXPERIMENT, run, description="FP32/FP16/BF16 precision benchmark", argv=argv)


if __name__ == "__main__":
    sys.exit(main())
