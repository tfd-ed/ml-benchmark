"""Experiment 3: small GPT-style transformer training over a grid of batch sizes x sequence lengths.

Each step = sync -> [host->device copy of tokens, forward, cross-entropy, backward, AdamW step,
loss.item()] -> sync. Data is a seeded synthetic Markov token stream (learnable: the loss can fall
towards ln(branching)), identical on every backend. Precision is FP32 here (see the precision experiment).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math  # noqa: E402

from src.devices import assert_on_device  # noqa: E402
from src.metrics import MemoryTracker  # noqa: E402
from src.runner import RunContext, check_finite, experiment_main  # noqa: E402
from src.timing import time_fn  # noqa: E402
from src.workloads import build_gpt_workload  # noqa: E402

EXPERIMENT = "transformer"


def run_config(ctx: RunContext, bs: int, seq: int) -> None:
    cfg = ctx.exp_cfg
    w = build_gpt_workload(cfg["model"], bs, seq, ctx.device, ctx.seed, cfg["lr"], "fp32",
                           cfg["data"]["num_sequences"], cfg["data"]["branching"])
    assert_on_device(ctx.device, w.model)
    tokens = bs * seq
    common = dict(model="gpt", dataset="synthetic_markov", batch_size=bs, sequence_length=seq, precision="fp32",
                  n_params=w.n_params, d_model=cfg["model"]["d_model"], n_layers=cfg["model"]["n_layers"],
                  n_heads=cfg["model"]["n_heads"], tokens_per_step=tokens, entropy_floor_loss=math.log(cfg["data"]["branching"]))
    counter = {"step": 0}
    rows = []

    def one_step():
        loss = w.step(counter["step"])
        counter["step"] += 1
        check_finite(loss, f"(step {counter['step']})")
        rows.append(loss)

    losses_by_call: list[float] = []
    with MemoryTracker(ctx.backend) as mem:
        res = time_fn(one_step, ctx.backend, cfg["warmup_steps"], cfg["steps"],
                      on_sample=lambda phase, i, ms: (losses_by_call.append(rows[-1]),
                                                      ctx.record(**common, phase=phase, iteration=i, latency_ms=ms,
                                                                 throughput=tokens / (ms / 1000.0), throughput_unit="tokens/s",
                                                                 loss=rows[-1], step_ms=ms)))
    assert_on_device(ctx.device, w.model)
    ctx.record(**common, phase="config", memory_mb=mem.result["memory_mb"], memory_kind=mem.result["memory_kind"],
               **{k: v for k, v in mem.result.items() if k not in ("memory_mb", "memory_kind")})
    s = res.stats()
    print(f"    bs={bs:3d} seq={seq:4d}: median step {s['median']:.1f} ms  {tokens / (s['median'] / 1000):,.0f} tokens/s  "
          f"loss {losses_by_call[0]:.3f}->{losses_by_call[-1]:.3f}  mem {mem.result['memory_mb']:.0f} MB", flush=True)


def run(ctx: RunContext) -> None:
    cfg = ctx.exp_cfg
    for bs in cfg["batch_sizes"]:
        for seq in cfg["sequence_lengths"]:
            try:
                run_config(ctx, bs, seq)
            except Exception as exc:
                ctx.record_failure(exc, model="gpt", dataset="synthetic_markov", batch_size=bs, sequence_length=seq, precision="fp32")
            ctx.cleanup()


def main(argv=None) -> int:
    return experiment_main(EXPERIMENT, run, description="Transformer training benchmark", argv=argv)


if __name__ == "__main__":
    sys.exit(main())
