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
> **Apple M1 Pro, 32 GB unified memory** (CPU + MPS). The CUDA path has since been run on an **NVIDIA GeForce
> RTX 3060** (Linux, PyTorch 2.14+cu130); see [Example results](#example-results). Every machine's numbers come from its own
> run, so check the report in `results/latest/report.md` for what was actually measured, and use
> `scripts/compare_runs.py` to put several machines side by side.

## Setup

```bash
python3.12 -m venv .venv          # 3.10+ works; 3.12 has the widest wheel support
source .venv/bin/activate
pip install -r requirements.txt   # optional for CUDA telemetry: pip install nvidia-ml-py
```

The CNN experiment downloads Fashion-MNIST (~30 MB) into `data/` on first use.

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
python scripts/run_all.py --backend auto --experiment cnn --set cnn.train_samples=null --set cnn.epochs=5   # full Fashion-MNIST
```

## Generate the report

```bash
python scripts/generate_report.py                          # the run results/latest points at
python scripts/generate_report.py --run-id 20260919T160927   # a specific run
python scripts/generate_report.py --all-runs               # keep every run inside the folder, not just the newest per (experiment, backend)
```

It loads `results/<run_id>/raw/`, validates the data, writes aggregate tables, draws all plots and writes `results/<run_id>/report.md`.

## Compare runs from different machines

Each machine runs the suite for the backend(s) it has and gives the run a meaningful ID:

```bash
python scripts/run_all.py --backend cuda --run-id rtx3060-20260919      # on the NVIDIA machine
python scripts/run_all.py --backend mps,cpu --run-id m1pro-20260920     # on the Mac
```

Copy the run folders (`results/<run_id>/`, `raw/` is enough) into one `results/` directory, then:

```bash
python scripts/compare_runs.py rtx3060-20260919 m1pro-20260920                       # baseline = first run's first backend
python scripts/compare_runs.py rtx3060-20260919 m1pro-20260920 --baseline "cpu · Apple M1 Pro"
```

* Every available (machine, backend) becomes one **series**, labelled `<backend> · <device>`: `cuda · NVIDIA GeForce RTX 3060`,
  `mps · Apple M1 Pro (16-core GPU)`, `cpu · Intel Core i7-8700K @ 3.70GHz`. (`cuda` = NVIDIA GPU through CUDA, `mps` = Apple GPU through Metal,
  `cpu` = the host CPU.) A host name is appended only when two machines would get the same label.
* Output: `results/comparison/comparison.md`, `processed/*.csv` (values and speedups) and `plots/`, using the same plots as the per-run report.
* **Speedup** = median throughput of a series / median throughput of the baseline series, for the identical configuration. Pick a CPU series as
  `--baseline` to get the usual "×faster than CPU" numbers.
* **Comparability check.** The script warns when the compared runs differ in `code_sha256`, seed or an experiment's configuration
  (subset size, epochs, batch sizes, durations …); those numbers are not directly comparable, so run every machine with the same code and flags.
  Differences in torch / Python / OS versions are listed as context only.

## Where things are stored

| What | Where |
|---|---|
| Raw measurements (one row per measurement, CSV) | `results/<run_id>/raw/<experiment>__<backend>__<run_id>.csv` |
| Run metadata + full config (JSON) | `results/<run_id>/raw/<experiment>__<backend>__<run_id>.meta.json` |
| Suite manifest (return codes, durations) | `results/<run_id>/raw/run_all.json` |
| Aggregate tables (median/mean/std/CV/min/max, speedups) | `results/<run_id>/processed/*.csv` |
| Plots (PNG **and** SVG) | `results/<run_id>/plots/` |
| Markdown report | `results/<run_id>/report.md` |
| Newest run | `results/latest` (symlink to its folder) |
| Cross-machine comparison | `results/comparison/` (`comparison.md`, `processed/`, `plots/`) |

Each suite run gets its own folder named after its run ID (a timestamp, or whatever you pass with `--run-id`), so raw data, tables, plots and the report of one run always belong together.
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
scripts/generate_report.py   report for one run folder
scripts/compare_runs.py      side-by-side comparison of several run folders (machines / backends)
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
| 2 | `cnn` | ResNet-18 (CIFAR variant) on a seeded Fashion-MNIST subset (28x28 padded to 32x32, 3 channels), batch sizes 16–128: epoch time, images/s, train loss, val accuracy, memory |
| 3 | `transformer` | 4.2M-parameter GPT-style model, batch × sequence-length grid: step time, tokens/s, loss, memory |
| 4 | `memory` | Batch / sequence-length / model-size ladders until failure; each config in a fresh subprocess |
| 5 | `precision` | FP32 / FP16 / BF16: pure-dtype matmul and autocast training: throughput, step time, loss, memory |
| 6 | `rl` | PPO on CartPole, two policy widths: env steps/s, gradient steps/s, total runtime, losses, time split env / inference / update |
| 7 | `sustained` | Repeated training for `duration_minutes` (default 30): throughput over time, degradation, telemetry where accessible |

## Example results

Backends are always labelled by what they are: **`cuda`** = NVIDIA GPU (CUDA), **`mps`** = Apple GPU (Metal), **`cpu`** = host CPU, followed by the device
name. The numbers below are the single series **`cuda · NVIDIA GeForce RTX 3060`** (Intel Core i7-8700K host, 62.7 GB RAM, Linux, PyTorch 2.14.0+cu130,
FP32 without TF32, seed 1234, run `20260919T163805`). No `cpu` or `mps` series was measured in this run, so there are no speedup columns; add
those runs with `compare_runs.py`. Full tables, CV/std and all plots are in that run's `report.md`.

**Matrix multiplication** (FP32; latency per `A @ B`, GFLOPS = 2n³ / latency)

| series | n | latency (ms, median) | GFLOPS |
|---|---|---|---|
| cuda · NVIDIA GeForce RTX 3060 | 1024 | 0.315 | 6,812 |
| cuda · NVIDIA GeForce RTX 3060 | 2048 | 2.286 | 7,516 |
| cuda · NVIDIA GeForce RTX 3060 | 4096 | 18.074 | 7,604 |
| cuda · NVIDIA GeForce RTX 3060 | 8192 | 116.543 | 9,435 |

![Matmul throughput](docs/example-results/01_matmul_throughput_vs_size.png)

**CNN training** (ResNet-18, Fashion-MNIST subset of 4,096 images, 3 epochs; epoch time = summed step time)

| series | batch size | epoch time (s) | images/s | final val. accuracy |
|---|---|---|---|---|
| cuda · NVIDIA GeForce RTX 3060 | 16 | 4.270 | 959 | 0.725 |
| cuda · NVIDIA GeForce RTX 3060 | 32 | 3.327 | 1,231 | 0.610 |
| cuda · NVIDIA GeForce RTX 3060 | 64 | 2.790 | 1,468 | 0.740 |
| cuda · NVIDIA GeForce RTX 3060 | 128 | 2.584 | 1,585 | 0.720 |

Validation accuracy uses a 1,000-image subset after 3 epochs, so it is a sanity check that training works, not a quality result.

![CNN images per second](docs/example-results/02_cnn_images_per_sec.png)

**Transformer training** (4.2M-parameter GPT-style model; tokens/s, median)

| series | batch \ sequence length | 128 | 256 | 512 |
|---|---|---|---|---|
| cuda · NVIDIA GeForce RTX 3060 | 8 | 121,354 | 134,980 | 141,402 |
| cuda · NVIDIA GeForce RTX 3060 | 16 | 143,655 | 149,717 | 138,309 |
| cuda · NVIDIA GeForce RTX 3060 | 32 | 164,162 | 154,633 | 143,958 |

![Transformer tokens per second](docs/example-results/04_transformer_tokens_per_sec.png)

**Precision** (median throughput)

| series | matmul 4096×4096 (GFLOPS) | GPT training step, autocast (tokens/s) |
|---|---|---|
| cuda · NVIDIA GeForce RTX 3060, FP32 | 7,599 | 158,684 |
| cuda · NVIDIA GeForce RTX 3060, FP16 | 23,998 | 283,008 |
| cuda · NVIDIA GeForce RTX 3060, BF16 | 23,941 | 291,882 |

![Precision](docs/example-results/08_precision_fp32_fp16_bf16.png)

**Memory scaling** (peak allocated CUDA memory, dedicated VRAM 11.6 GiB): batch size up to 256 fits (8,394 MB), 512 runs out of memory;
the sequence-length ladder reaches its ceiling of 4,096 tokens and the model ladder its ceiling of 206M parameters without failing.

**PPO / CartPole** (8 envs × 128 steps per iteration): end-to-end 9,448 env-steps/s (hidden width 64) and 9,107 (width 512);
gradient-step throughput 341 and 332 steps/s.

![PPO throughput](docs/example-results/09_ppo_throughput.png)

**Sustained training** (10 min): average 139,755 tokens/s; first→last 10% of windows changed by 4.10% (positive = slower at the end);
median GPU utilisation 98%, temperature 87 °C, power 121 W.

![Sustained throughput](docs/example-results/10_sustained_throughput_over_time.png)

### How to read these numbers

* **Median, CV.** Every number is the median of repeated synchronised measurements (warm-up excluded). CV = std / mean; values around 0.01 mean the
  repeats agree within about 1%, large values (e.g. 0.13 for the transformer at batch 32 × 128) flag a noisy configuration.
* **GFLOPS** is the *achieved* matmul rate for the given size and dtype, not the hardware's theoretical peak. Small matrices underuse a large GPU,
  which is why throughput changes with `n`.
* **Batch size** changes throughput (images/s, tokens/s) because larger batches keep the device busier; epoch time is what the same amount of work costs in seconds.
* **Precision.** Half precision is only faster if the hardware and kernels support it; FP32 here is strict FP32 (TF32 off), so FP16/BF16 gains are measured against that.
* **Memory** is the backend's own metric (`cuda` = peak allocated tensor memory) and must not be compared against `mps` or `cpu` memory numbers.
* **Speedup** (in `compare_runs.py` output and the per-run report) is throughput relative to a baseline series; it is meaningful only when both series ran the
  identical configuration.

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

* **One machine per run.** The numbers on this page come from a single RTX 3060 machine; `mps` results need an Apple-Silicon Mac and `cpu`
  results depend on the host CPU, so combine machines with `scripts/compare_runs.py`. CUDA telemetry (GPU utilisation, temperature, power) is read
  through `nvidia-smi` / NVML; the OS CPU speed limit is not collected on Linux.
* **CPU matmul runs on Apple's AMX matrix units via Accelerate**, giving ~1.6 TFLOPS that is *independent of the thread count*
  (1 thread ≈ 10 threads). A pure-NEON CPU would be far slower, so "CPU" on Apple Silicon is unusually strong for large FP32 matmuls,
  and this is a property of the software stack too.
* **MPS has no peak-memory API** and no way to detect silent CPU fallback beyond disabling it (done here).
* **GPU utilisation / power / temperature on Apple Silicon require `sudo powermetrics`**, which the suite does not require; those
  telemetry fields are stored as `null` with the reason. Only the OS thermal speed limit (`pmset -g therm`) is collected.
* **Half precision does not imply speed-up.** On this machine FP16/BF16 behave very differently per backend (see the precision section of the report; CPU half-precision is software-emulated and far slower).
* **Laptop conditions matter.** The power source (AC/battery) and low-power mode are recorded per run; keep the machine plugged in and idle for the video runs.
* **The checkout is not a git repository**, so `git_commit` is `null`; a SHA-256 of all sources (`code_sha256`) is recorded instead.
* **CNN default uses a 4096-image Fashion-MNIST subset** because the CPU trains at tens of images/s; set `cnn.train_samples=null` for the full dataset.
* **Sustained default is 30 minutes.** Shorter runs are possible (`--set sustained.duration_minutes=N`) and are labelled by their real duration in the report.

## Tests

```bash
python -m pytest -q
```
