# ml-hardware-benchmark

A reproducible PyTorch benchmark suite that compares **CPU**, **Apple Silicon (MPS)** and **NVIDIA (CUDA)** on
identical workloads. It was built for the video *"Can You Use a MacBook for Machine Learning Training in 2026?"*.

The suite makes **no assumption about which backend is faster**. It records raw measurements, states them
factually and leaves interpretation to you. It never computes an overall "winner".

* Backends are detected automatically. An unavailable backend is recorded as `unavailable`, never skipped silently.
* Nothing ever falls back from one device to another. `PYTORCH_ENABLE_MPS_FALLBACK` is removed at import time, so an
  operator that MPS does not support raises and is recorded as `unsupported` instead of quietly running on the CPU.
* OOM, unsupported dtypes/operators and other errors are stored as data (`status`, `error_type`, `error`); the run continues.
* Every individual measurement, warmup included, is saved. Statistics are computed afterwards from the raw rows.

> **Status of this checkout.** Everything here was developed and run on one machine: a MacBook Pro with an
> **Apple M1 Pro, 32 GB unified memory** (CPU + MPS). The **CUDA code path has never been executed on real
> NVIDIA hardware** (it is written against the documented `torch.cuda` API, and the unit tests cover the
> "unavailable" path). Run the suite on a CUDA machine before relying on CUDA numbers, and check
> `results/report.md` for what was actually measured.

## Setup

```bash
python3.12 -m venv .venv          # 3.10+ works; 3.12 has the widest wheel support
source .venv/bin/activate
pip install -r requirements.txt   # optional for CUDA telemetry: pip install nvidia-ml-py
```

The CNN experiment downloads CIFAR-10 (~170 MB) into `data/` on first use.

## Run everything

```bash
python scripts/run_all.py --backend auto --report      # every experiment on every available backend, then the report
```

`--backend auto` means "all three backends": unavailable ones get an `unavailable` row. Other forms:

```bash
python scripts/run_all.py --backend cpu
python scripts/run_all.py --backend mps
python scripts/run_all.py --backend cuda
python scripts/run_all.py --backend cpu,mps --config configs/default.yaml
```

Each (experiment, backend) pair runs in its own subprocess, so a crash in one cannot end the suite.
The default `sustained` experiment is **30 minutes per backend**; the full suite therefore takes on the order of hours on a laptop.

## Run individual experiments

```bash
python scripts/run_all.py --backend auto --experiment matmul
python scripts/run_all.py --backend auto --experiment cnn
python scripts/run_all.py --backend auto --experiment transformer
python scripts/run_all.py --backend auto --experiment memory
python scripts/run_all.py --backend auto --experiment precision
python scripts/run_all.py --backend auto --experiment rl
python scripts/run_all.py --backend auto --experiment sustained
# or call an experiment directly (same flags):
python experiments/benchmark_matmul.py --backend mps
```

Change any configuration value from the command line without editing YAML:

```bash
python scripts/run_all.py --backend mps --experiment sustained --set sustained.duration_minutes=5
python scripts/run_all.py --backend auto --experiment cnn --set cnn.train_samples=null --set cnn.epochs=5   # full CIFAR-10
```

## Generate the report

```bash
python scripts/generate_report.py            # uses the latest run of every (experiment, backend)
python scripts/generate_report.py --all-runs
```

It loads `results/raw/`, validates the data, writes aggregate tables, draws all plots and writes `results/report.md`.

## Where things are stored

| What | Where |
|---|---|
| Raw measurements (one row per measurement, CSV) | `results/raw/<experiment>__<backend>__<run_id>.csv` |
| Run metadata + full config (JSON) | `results/raw/<experiment>__<backend>__<run_id>.meta.json` |
| Suite manifest (return codes, durations) | `results/raw/run_all__<run_id>.json` |
| Aggregate tables (median/mean/std/CV/min/max, speedups) | `results/processed/*.csv` |
| Plots (PNG **and** SVG) | `results/plots/` |
| Markdown report | `results/report.md` |
| Configuration | `configs/default.yaml` |

## Project layout

```
configs/default.yaml     all experiment settings
src/devices.py           backend detection, explicit device selection, sync, seeding, placement guard
src/environment.py       metadata: host, OS, versions, CPU/GPU/RAM, power source, git commit + source hash
src/timing.py            synchronised timer (warmup, per-iteration samples)
src/metrics.py           statistics (median/mean/std/CV/min/max), speedup, degradation, memory tracking
src/reporting.py         result schema, streaming CSV writer, loader, validator, aggregation
src/runner.py            shared CLI + failure classification (oom / unsupported / error) for experiments
src/models.py            ResNet-18 (CIFAR), GPT-style transformer
src/workloads.py         seeded data pipelines, GPT training step
src/plotting.py          all plots (matplotlib, colours from the default cycle, one colour per backend)
experiments/             benchmark_matmul / resnet / transformer / memory / precision / rl / sustained
scripts/run_all.py       orchestrates experiments × backends
scripts/generate_report.py
tests/                   unit tests (python -m pytest)
```

## Result schema

Every raw CSV shares these columns; experiment-specific values go into the JSON column `extra`
(`generate_report.py` expands them to `x_<name>`).

`run_id, timestamp, experiment, backend, device, model, dataset, batch_size, sequence_length, precision, phase,
iteration, latency_ms, throughput, throughput_unit, memory_mb, memory_kind, loss, accuracy, status, error_type,
error, extra, metadata`

* `phase`: `warmup` (recorded, excluded from statistics), `measure`, or `config` (one row describing a whole
  configuration, used for memory summaries, failures and run summaries).
* `status`: `ok | oom | unsupported | unavailable | error | timeout | skipped`.
* `metadata` holds a compact fingerprint; the full metadata (timestamp, hostname, OS, Python/PyTorch/torchvision
  versions, device and GPU names, CPU, RAM, CUDA version, MPS availability, git commit, source hash, power source,
  seed, full configuration) is in the `.meta.json` file next to the CSV.

## The experiments

| # | Experiment | What is measured |
|---|---|---|
| 1 | `matmul` | `A @ B`, FP32, sizes 1024–8192: latency per call, GFLOPS, matmuls/s, memory; correctness vs. CPU FP32 |
| 2 | `cnn` | ResNet-18 (CIFAR variant) on a seeded CIFAR-10 subset, batch sizes 16–128: epoch time, images/s, train loss, val accuracy, memory |
| 3 | `transformer` | 4.2M-parameter GPT-style model, batch × sequence-length grid: step time, tokens/s, loss, memory |
| 4 | `memory` | Batch / sequence-length / model-size ladders until failure; each config in a fresh subprocess |
| 5 | `precision` | FP32 / FP16 / BF16: pure-dtype matmul and autocast training: throughput, step time, loss, memory |
| 6 | `rl` | PPO on CartPole, two policy widths: env steps/s, gradient steps/s, total runtime, losses, time split env / inference / update |
| 7 | `sustained` | Repeated training for `duration_minutes` (default 30): throughput over time, degradation, telemetry where accessible |

## Methodology

* **Same workload.** Model weights, data subsets, augmentation and token streams are created on the CPU from a fixed seed
  (`seed: 1234`) and copied to the device once; every backend gets identical inputs. GPU kernels are not bit-deterministic, so
  training trajectories can drift slightly between backends (visible in the loss columns).
* **Synchronised timing.** Each timed call is `sync → t0 → work → sync → t1` with `torch.cuda.synchronize()`,
  `torch.mps.synchronize()`, and nothing on CPU. The sync cost is part of every measurement on every backend.
* **Warm-up.** Warm-up iterations run first (GPU kernel compilation, clock ramp-up). Small workloads remain noisy;
  the coefficient of variation is reported so you can see it.
* **Baseline for speedups.** Always `speedup = cpu_time / backend_time` for the identical configuration on the same machine.
  Values below 1 mean *slower than the CPU baseline*. There is no aggregate score.
* **Eager mode, defaults.** No `torch.compile`, no fused optimizers, strict FP32 (TF32 disabled; `numerics.allow_tf32` in the config).
  Software-stack differences (Accelerate/AMX on Apple CPUs, MPS kernels, cuDNN) are part of what is compared, so results describe
  *hardware + PyTorch backend together*, not silicon alone.
* **Training-step scope.** Steps include the host→device copy of the batch and `loss.item()`; the CNN data augmentation is applied
  on the CPU per epoch *before* timing (identical, seeded) and is reported separately.

## Metric definitions and how to interpret them

| Metric | Meaning | Reading it |
|---|---|---|
| **latency** (`latency_ms`) | Wall time of one call / step | Lower is better. Compare medians, look at CV. |
| **throughput** (`throughput`) | Work per second; unit in `throughput_unit` | Higher is better. Only compare within the same unit and configuration. |
| **GFLOPS** | `2·n³ / latency` for `n×n` matmul | Achieved rate, not the hardware's theoretical peak. |
| **samples/s, images/s** | Training examples processed per second | Depends on batch size. |
| **tokens/s** | `batch × sequence / step time` | Transformer analogue of images/s. |
| **training time / epoch time** | Sum of step times of an epoch | Wall time of the training phase only. |
| **median / mean / std / CV** | Distribution of repeated measurements | Median is the headline; CV = std/mean flags noisy results. |
| **speedup vs CPU** | `cpu_time / backend_time` | Same configuration only; <1 = slower than the baseline. |
| **degradation %** | `(first-10% median − last-10% median) / first` for sustained throughput | Positive = slower at the end (e.g. thermal throttling). |
| **env steps/s (env only)** | Simulator speed, always on the CPU | Independent of the accelerator. |
| **end-to-end env steps/s** | Env steps / whole PPO iteration | Includes policy round trips and the update. |
| **gradient steps/s** | Minibatch updates / update time | Neural-network training speed only. |
| **largest successful configuration** | Last ok value on a ladder | Says how *large* a workload fits, not how *fast* it runs. |

### Memory metrics are NOT equivalent

`memory_mb` always comes with a `memory_kind` label. Do not compare absolute values across backends.

| Backend | What `memory_mb` is | Caveats |
|---|---|---|
| CUDA | `torch.cuda.max_memory_allocated` (exact peak of tensor allocations); reserved memory is stored in `extra` | Dedicated VRAM. Excludes CUDA context and allocator cache. |
| MPS | `torch.mps.driver_allocated_memory` (what the Metal driver holds for the process, including allocator cache). Where sampled, the maximum polled every 5 ms; `current_allocated_memory` (tensor bytes only) is stored in `extra` | **PyTorch exposes no peak counter for MPS**, so this is a snapshot or a sampled lower bound of the true peak. Includes a fixed floor of roughly 1 GiB, so small workloads look alike. Unified memory is shared with the CPU and the OS. |
| CPU | Process RSS: snapshot, sampled maximum, or lifetime peak (`ru_maxrss`), as labelled | Includes the Python/torch baseline; OS-managed, can include shared pages. |

In the memory-scaling experiment, unified-memory machines can start swapping instead of failing, so the suite enforces safety limits
(MPS memory fraction, CPU RSS cap, minimum system memory, per-configuration timeout). A ladder ended by one of these limits is
reported as `SafetyCap*` — *stopped by this suite*, not a native OOM.

## Limitations discovered while building this (Apple M1 Pro, 32 GB, PyTorch 2.14)

* **No CUDA hardware was available**; all CUDA results are absent, and CUDA telemetry code (NVML / `nvidia-smi`) is untested.
* **CPU matmul runs on Apple's AMX matrix units via Accelerate**, giving ~1.6 TFLOPS that is *independent of the thread count*
  (1 thread ≈ 10 threads). A pure-NEON CPU would be far slower, so "CPU" on Apple Silicon is unusually strong for large FP32 matmuls,
  and this is a property of the software stack too.
* **MPS has no peak-memory API** and no way to detect silent CPU fallback beyond disabling it (done here).
* **GPU utilisation / power / temperature on Apple Silicon require `sudo powermetrics`**, which the suite does not require; those
  telemetry fields are stored as `null` with the reason. Only the OS thermal speed limit (`pmset -g therm`) is collected.
* **Half precision does not imply speed-up.** On this machine FP16/BF16 behave very differently per backend (see the precision section of the report; CPU half-precision is software-emulated and far slower).
* **Laptop conditions matter.** The power source (AC/battery) and low-power mode are recorded per run; keep the machine plugged in and idle for the video runs.
* **The checkout is not a git repository**, so `git_commit` is `null`; a SHA-256 of all sources (`code_sha256`) is recorded instead.
* **CNN default uses a 4096-image CIFAR-10 subset** because the CPU trains at tens of images/s; set `cnn.train_samples=null` for the full dataset.
* **Sustained default is 30 minutes.** Shorter runs are possible (`--set sustained.duration_minutes=N`) and are labelled by their real duration in the report.

## Tests

```bash
python -m pytest -q
```
