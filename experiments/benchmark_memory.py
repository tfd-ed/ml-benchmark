"""Experiment 4: memory scaling.

Three ladders, each varying ONE dimension while the others stay at the base configuration:
    batch_size, sequence_length, model size.
Every configuration is a training step (forward+backward+AdamW) executed in a FRESH subprocess so that
(a) peak-memory numbers are not polluted by earlier configurations, and (b) an OOM, a hang or a hard
kill cannot take the suite down. A ladder stops at its first non-ok configuration; the last ok one is
the "largest successful configuration" for that dimension.

Safety limits (recorded, never hidden). Unified-memory machines can start swapping instead of failing:
  * MPS : torch.mps.set_per_process_memory_fraction(memory.mps_memory_fraction)
  * CPU : worker killed if its RSS exceeds memory.cpu_rss_cap_fraction of system RAM
  * all : worker killed if system available memory drops below MIN_AVAILABLE_GB
  * all : worker killed after memory.timeout_s seconds
A kill by these limits is stored as status 'oom'/'timeout' with error_type 'SafetyCap*' - it means
"stopped by this suite's safety limit", not "the hardware ran out of memory".

Memory capacity is NOT performance: this experiment only shows how large a workload fits, and how the
different memory architectures (dedicated VRAM / unified memory / system RAM) behave.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil  # noqa: E402

EXPERIMENT = "memory"
MIN_AVAILABLE_GB = 2.0
RESULT_PREFIX = "RESULT_JSON:"


# --------------------------------------------------------------------------- worker (child process)


def worker(spec: dict) -> None:
    import torch

    from src import devices
    from src.metrics import MemoryTracker, device_memory_capacity_mb, process_rss_mb, summarize
    from src.runner import check_finite, classify_exception, describe_exception
    from src.timing import DeviceTimer
    from src.workloads import build_gpt_workload

    out = {"status": "error", "error": None, "error_type": None}
    try:
        backend = spec["backend"]
        status = devices.detect_backends()[backend]
        if not status.available:
            out.update(status="unavailable", error=status.reason, error_type="BackendUnavailable")
        else:
            devices.seed_everything(spec["seed"])
            devices.configure_numerics(False)
            device = devices.get_device(status)
            if backend == "mps":
                torch.mps.set_per_process_memory_fraction(spec["mps_fraction"])
            cap_mb, cap_kind = device_memory_capacity_mb(backend)
            out.update(baseline_rss_mb=process_rss_mb(), capacity_mb=cap_mb, capacity_kind=cap_kind)
            step_ms, losses = [], []
            with MemoryTracker(backend, sample=True, use_lifetime_peak=True) as mem:
                w = build_gpt_workload(spec["model_cfg"], spec["batch_size"], spec["sequence_length"], device, spec["seed"],
                                       spec["lr"], "fp32", num_sequences=spec["batch_size"], max_len=spec["sequence_length"])
                out["n_params"] = w.n_params
                for i in range(spec["warmup_steps"] + spec["steps"]):
                    with DeviceTimer(backend) as t:
                        loss = w.step(i)
                    check_finite(loss)
                    if i >= spec["warmup_steps"]:
                        step_ms.append(t.elapsed_ms)
                        losses.append(loss)
            out.update(status="ok", step_ms=step_ms, losses=losses, memory=mem.result, step_stats=summarize(step_ms))
    except Exception as exc:
        status_name, err_type = classify_exception(exc)
        out.update(status=status_name, error_type=err_type, error=describe_exception(exc))
    print(RESULT_PREFIX + json.dumps(out, default=str), flush=True)


# --------------------------------------------------------------------------- parent


def run_worker(spec: dict, timeout_s: float, rss_cap_mb: float | None) -> dict:
    """Run one configuration in a subprocess; enforce timeout / memory watchdog; parse its result."""
    import subprocess

    with tempfile.TemporaryFile("w+") as fout, tempfile.TemporaryFile("w+") as ferr:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(spec)], stdout=fout, stderr=ferr, text=True)
        ps = psutil.Process(proc.pid)
        start, killed = time.time(), None
        while proc.poll() is None:
            time.sleep(0.2)
            try:
                rss_mb = ps.memory_info().rss / 1024**2
            except psutil.Error:
                break
            if time.time() - start > timeout_s:
                killed = ("timeout", "SafetyCapTimeout", f"killed after {timeout_s:.0f}s (memory.timeout_s); configuration did not finish in time")
            elif rss_cap_mb and rss_mb > rss_cap_mb:
                killed = ("oom", "SafetyCapRSS", f"killed by memory watchdog at RSS {rss_mb:,.0f} MB (cap {rss_cap_mb:,.0f} MB); a suite safety limit, not a native OOM")
            elif psutil.virtual_memory().available < MIN_AVAILABLE_GB * 1024**3:
                killed = ("oom", "SafetyCapSystemMemory", f"killed: system available memory fell below {MIN_AVAILABLE_GB} GB; a suite safety limit, not a native OOM")
            if killed:
                proc.kill()
                proc.wait()
                break
        wall = time.time() - start
        fout.seek(0)
        ferr.seek(0)
        stdout, stderr = fout.read(), ferr.read()
    if killed:
        return {"status": killed[0], "error_type": killed[1], "error": killed[2], "wall_s": wall}
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            res = json.loads(line[len(RESULT_PREFIX):])
            res["wall_s"] = wall
            return res
    tail = " | ".join(stderr.strip().splitlines()[-3:])[:400]
    return {"status": "error", "error_type": f"WorkerExit{proc.returncode}", "error": f"worker produced no result (exit code {proc.returncode}): {tail}", "wall_s": wall}


def ladders(cfg: dict):
    """Yield (ladder_name, [(model_name, batch, seq), ...]) for each ladder."""
    bm, bb, bs = cfg["base_model"], cfg["base_batch_size"], cfg["base_sequence_length"]
    lad = cfg["ladders"]
    yield "batch_size", [(bm, b, bs) for b in lad["batch_size"]]
    yield "sequence_length", [(bm, bb, s) for s in lad["sequence_length"]]
    yield "model", [(m, bb, bs) for m in lad["model"]]


def run(ctx) -> None:
    cfg, tcfg = ctx.exp_cfg, ctx.cfg["transformer"]
    ram_mb = psutil.virtual_memory().total / 1024**2
    rss_cap = cfg["cpu_rss_cap_fraction"] * ram_mb if ctx.backend == "cpu" else None
    for ladder_name, configs in ladders(cfg):
        print(f"  ladder: {ladder_name}", flush=True)
        for model_name, bs, seq in configs:
            model_cfg = {**cfg["models"][model_name], "vocab_size": cfg["vocab_size"], "mlp_ratio": 4}
            spec = dict(backend=ctx.backend, seed=ctx.seed, model_cfg=model_cfg, batch_size=bs, sequence_length=seq, lr=tcfg["lr"],
                        warmup_steps=cfg["warmup_steps"], steps=cfg["steps"], mps_fraction=cfg["mps_memory_fraction"])
            res = run_worker(spec, cfg["timeout_s"], rss_cap)
            common = dict(model=model_name, dataset="synthetic_markov", batch_size=bs, sequence_length=seq, precision="fp32", phase="config",
                          ladder=ladder_name, d_model=model_cfg["d_model"], n_layers=model_cfg["n_layers"], n_heads=model_cfg["n_heads"],
                          tokens_per_step=bs * seq, wall_s=res.get("wall_s"), n_params=res.get("n_params"),
                          capacity_mb=res.get("capacity_mb"), capacity_kind=res.get("capacity_kind"), baseline_rss_mb=res.get("baseline_rss_mb"))
            if res["status"] == "ok":
                st, mem = res["step_stats"], res["memory"]
                ctx.record(**common, status="ok", latency_ms=st["median"], throughput=bs * seq / (st["median"] / 1000.0), throughput_unit="tokens/s",
                           memory_mb=mem["memory_mb"], memory_kind=mem["memory_kind"], loss=res["losses"][-1],
                           step_ms_all=res["step_ms"], **{k: v for k, v in mem.items() if k not in ("memory_mb", "memory_kind")})
                print(f"    {model_name:6s} bs={bs:5d} seq={seq:5d}: OK  step {st['median']:8.1f} ms  mem {mem['memory_mb']:9.0f} MB", flush=True)
            else:
                ctx.record(**common, status=res["status"], error_type=res.get("error_type"), error=res.get("error"))
                print(f"    {model_name:6s} bs={bs:5d} seq={seq:5d}: {res['status'].upper()}  ({res.get('error_type')})  -> ladder stops", flush=True)
                break  # the first failure ends this ladder


def main(argv=None) -> int:
    from src.runner import experiment_main

    return experiment_main(EXPERIMENT, run, description="Memory scaling benchmark", argv=argv)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker(json.loads(sys.argv[sys.argv.index("--worker") + 1]))
        sys.exit(0)
    sys.exit(main())
