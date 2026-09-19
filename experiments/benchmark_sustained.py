"""Experiment 7: sustained transformer training (default 30 minutes, configurable).

The same training step runs back-to-back for `sustained.duration_minutes`. Every `log_interval_s`
seconds one row is written with the throughput / step time / loss measured over that window plus
whatever telemetry is available. Nothing is invented: a metric that cannot be collected is stored as
null and the reason is written to `telemetry_unavailable` (JSON in `extra`).

Telemetry sources (all optional, none require sudo):
  NVIDIA : pynvml (nvidia-ml-py) if installed, else `nvidia-smi` -> GPU utilisation, temperature, power
  Apple  : `pmset -g therm` (CPU_Speed_Limit when the OS is throttling); GPU utilisation / power /
           temperature need `sudo powermetrics` and are reported as unavailable
  CPU    : psutil process CPU utilisation; temperature only where psutil exposes sensors (not macOS)
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil  # noqa: E402

from src.devices import assert_on_device  # noqa: E402
from src.metrics import MemoryTracker, current_memory_mb, summarize, throughput_degradation  # noqa: E402
from src.runner import RunContext, check_finite, experiment_main  # noqa: E402
from src.timing import DeviceTimer  # noqa: E402
from src.workloads import build_gpt_workload  # noqa: E402

EXPERIMENT = "sustained"


class Telemetry:
    """Best-effort hardware telemetry. `read()` returns (values, reasons_for_missing)."""

    def __init__(self, backend: str):
        self.backend = backend
        self.proc = psutil.Process()
        self.proc.cpu_percent(None)
        self.nvml = None
        if backend == "cuda":
            try:
                import pynvml

                pynvml.nvmlInit()
                self.nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
            except Exception:
                self.nvml = None

    def _nvidia_smi(self):
        exe = shutil.which("nvidia-smi")
        if not exe:
            return None
        out = subprocess.run([exe, "--query-gpu=utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        u, t, p = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
        return float(u), float(t), float(p)

    def read(self) -> tuple[dict, dict]:
        vals: dict = {"process_cpu_percent": self.proc.cpu_percent(None), "gpu_utilization_percent": None,
                      "temperature_c": None, "power_w": None, "cpu_speed_limit_percent": None}
        why: dict = {}
        if self.backend == "cuda":
            try:
                if self.nvml:
                    nv, h = self.nvml
                    vals["gpu_utilization_percent"] = float(nv.nvmlDeviceGetUtilizationRates(h).gpu)
                    vals["temperature_c"] = float(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))
                    vals["power_w"] = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
                else:
                    smi = self._nvidia_smi()
                    if smi:
                        vals["gpu_utilization_percent"], vals["temperature_c"], vals["power_w"] = smi
                    else:
                        why.update({k: "neither pynvml nor nvidia-smi is available" for k in ("gpu_utilization_percent", "temperature_c", "power_w")})
            except Exception as exc:
                why.update({k: f"NVML/nvidia-smi query failed: {type(exc).__name__}" for k in ("gpu_utilization_percent", "temperature_c", "power_w")})
        elif self.backend == "mps":
            for k in ("gpu_utilization_percent", "temperature_c", "power_w"):
                why[k] = "Apple Silicon GPU sensors need `sudo powermetrics`; not collected (no root required by this suite)"
            if sys.platform == "darwin":
                try:
                    out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
                    for line in out.splitlines():
                        if "CPU_Speed_Limit" in line:
                            vals["cpu_speed_limit_percent"] = float(line.split("=")[1].strip())
                    if vals["cpu_speed_limit_percent"] is None:
                        vals["cpu_speed_limit_percent"] = 100.0  # pmset reports no limit when the OS is not throttling
                        why["cpu_speed_limit_percent"] = "pmset reported no thermal limit; 100 means 'no throttling reported by the OS', not a measured clock"
                except Exception as exc:
                    why["cpu_speed_limit_percent"] = f"pmset failed: {type(exc).__name__}"
        else:
            why["gpu_utilization_percent"] = "not applicable to the CPU backend"
            why["power_w"] = "CPU package power is not exposed without platform-specific privileged tools"
            try:
                temps = psutil.sensors_temperatures() if hasattr(psutil, "sensors_temperatures") else {}
                flat = [t.current for group in temps.values() for t in group]
                if flat:
                    vals["temperature_c"] = max(flat)
                else:
                    why["temperature_c"] = "no temperature sensors exposed by psutil on this OS"
            except Exception as exc:
                why["temperature_c"] = f"sensor query failed: {type(exc).__name__}"
        for k in ("gpu_utilization_percent", "temperature_c", "power_w"):
            if vals[k] is None and k not in why:
                why[k] = "not collected"
        return vals, why


def run(ctx: RunContext) -> None:
    cfg, mcfg = ctx.exp_cfg, ctx.exp_cfg["model"]
    bs, seq, tokens = cfg["batch_size"], cfg["sequence_length"], cfg["batch_size"] * cfg["sequence_length"]
    duration_s, interval_s = cfg["duration_minutes"] * 60.0, cfg["log_interval_s"]
    w = build_gpt_workload(mcfg, bs, seq, ctx.device, ctx.seed, cfg["lr"], "fp32", 2048, 4)
    assert_on_device(ctx.device, w.model)
    common = dict(model="gpt", dataset="synthetic_markov", batch_size=bs, sequence_length=seq, precision="fp32", n_params=w.n_params,
                  tokens_per_step=tokens, duration_s_target=duration_s)
    tel = Telemetry(ctx.backend)
    for i in range(5):  # warmup, not part of the measured duration
        w.step(i)
    ctx.sync()

    step, total_tokens, window_steps, window_ms, window_losses = 0, 0, 0, [], []
    windows: list[dict] = []
    with MemoryTracker(ctx.backend) as mem:
        t_start = win_start = time.perf_counter()
        while True:
            with DeviceTimer(ctx.backend) as t:
                loss = w.step(step + 5)
            check_finite(loss, f"(step {step})")
            step += 1
            total_tokens += tokens
            window_ms.append(t.elapsed_ms)
            window_losses.append(loss)
            now = time.perf_counter()
            done = now - t_start >= duration_s
            if now - win_start >= interval_s or done:
                span = now - win_start
                vals, why = tel.read()
                mem_mb, mem_kind = current_memory_mb(ctx.backend)
                thr = len(window_ms) * tokens / span
                row = dict(**common, phase="measure", iteration=len(windows), latency_ms=summarize(window_ms)["median"], throughput=thr,
                           throughput_unit="tokens/s (window)", loss=sum(window_losses) / len(window_losses), memory_mb=mem_mb,
                           memory_kind=mem_kind, elapsed_s=now - t_start, window_s=span, steps_in_window=len(window_ms),
                           step_ms_mean=summarize(window_ms)["mean"], step_ms_max=max(window_ms), samples_per_sec=thr / seq,
                           telemetry_unavailable=why or None, **vals)
                ctx.record(**row)
                windows.append(row)
                print(f"    t={now - t_start:7.0f}s  {thr:9,.0f} tokens/s  step {row['latency_ms']:.1f} ms  loss {row['loss']:.3f}", flush=True)
                window_ms, window_losses, win_start = [], [], now
            if done:
                break
    elapsed = time.perf_counter() - t_start
    thr = [r["throughput"] for r in windows]
    ctx.record(**common, phase="config", memory_mb=mem.result["memory_mb"], memory_kind=mem.result["memory_kind"], elapsed_s=elapsed,
               total_steps=step, total_tokens=total_tokens, total_samples=step * bs, average_throughput_tokens_per_s=total_tokens / elapsed,
               median_window_throughput=summarize(thr)["median"], **{f"degradation_{k}": v for k, v in throughput_degradation(thr).items()},
               summary="run summary; degradation compares median window throughput of first vs last 10% of windows")


def main(argv=None) -> int:
    return experiment_main(EXPERIMENT, run, description="Sustained training benchmark", argv=argv)


if __name__ == "__main__":
    sys.exit(main())
