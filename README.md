# ml-hardware-benchmark

A reproducible PyTorch benchmark suite that measures **Apple Silicon (`mps`)** and **NVIDIA (`cuda`)** on identical machine-learning workloads
(CPU is supported too). It was built for the video *"Can You Use a MacBook for Machine Learning Training in 2026?"*.

This README compares one MacBook Pro (**Apple M1 Pro, 16-core GPU**) with one desktop (**NVIDIA GeForce RTX 3060**). The suite records raw
measurements and states them factually. It never computes an overall "winner".

**Jump to:** [Results at a glance](#results-at-a-glance) · [Detailed results](#detailed-results) · [Quick start](#quick-start) ·
[What is measured](#what-is-measured-and-how) · [Metric definitions](#metric-definitions-what-how-it-is-calculated-why-it-matters) ·
[Limitations](#limitations-and-caveats)

---

## Results at a glance

Speedup = MPS throughput ÷ CUDA throughput for the same configuration. **Below 1 means the M1 Pro measured lower throughput than the RTX 3060.**

| Workload | Configuration shown | CUDA · RTX 3060 | MPS · M1 Pro | MPS ÷ CUDA | Range over all configurations |
|---|---|---|---|---|---|
| Matrix multiply, FP32 | 4096×4096 (GFLOPS) | 7,604 | 4,055 | **0.53** | 0.41 – 0.54 |
| CNN training (ResNet-18) | batch 64 (images/s) | 1,468 | 732 | **0.50** | 0.47 – 0.58 |
| Transformer training, FP32 | batch 16, seq 256 (tokens/s) | 149,717 | 60,055 | **0.40** | 0.38 – 0.41 |
| Reinforcement learning (PPO) | hidden 64 (env-steps/s) | 9,448 | 4,997 | **0.53** | 0.49 – 0.53 |
| Sustained training, 10 min | average (tokens/s) | 139,755 | 59,359 | **0.42** | – |
| Half precision, matmul | FP16 (GFLOPS) | 23,998 | 4,653 | **0.19** | BF16: 0.09 |
| Half precision, training | FP16 autocast (tokens/s) | 283,008 | 47,914 | **0.17** | BF16: 0.15 |

### What the numbers show

* **FP32:** on every FP32 workload measured, the M1 Pro reached between **0.38× and 0.58×** the RTX 3060's throughput. The ratio changes little with batch or matrix size within each workload (see the range column).
* **Half precision:** the gap is much wider. On the RTX 3060, FP16/BF16 is faster than FP32 (about 3× for matmul, 1.8× for training). On the M1 Pro through MPS,
  FP16/BF16 training is *slower* than FP32 (47.9k / 43.1k vs 60.1k tokens/s), and BF16 matmul is slower than FP32.
* **Sustained load:** over 10 minutes the M1 Pro's throughput stayed flat (−0.18 % change, first vs last 10 % of windows). The RTX 3060 slowed by 4.10 % (GPU at 87 °C, 121 W).
* **Memory capacity:** both series reached the same ceilings in the sequence-length and model-size ladders (4,096 tokens, 206M parameters). At batch 512 the RTX 3060 ran out of
  its 11.6 GiB of VRAM, while the M1 Pro run was stopped by the suite's own system-memory safety limit, so that is not a native out-of-memory error.
  Memory values are measured differently on each backend and are not compared as ratios.

> **Comparability caveat.** `compare_runs.py` reports that the two runs are **not strictly comparable**: `code_sha256` differs (the benchmark sources were not identical on
> the two machines; for `sustained` the Mac run also differs from the Mac's other experiments). Seeds and experiment settings matched. Treat the numbers as indicative and
> re-run both machines from one checkout for a like-for-like comparison. There is no `cpu` series in these runs, so all speedups are relative to the RTX 3060, not to a CPU.

### Test machines

| | `cuda · NVIDIA GeForce RTX 3060` | `mps · Apple M1 Pro (16-core GPU)` |
|---|---|---|
| Run ID | `20260919T163805` | `20260919T172639` |
| Host | Intel Core i7-8700K, Linux 6.8 | MacBook Pro, macOS (Darwin 25.6.0) |
| Memory | 62.7 GB system RAM, 11.6 GiB dedicated VRAM | 32 GB unified memory |
| Software | PyTorch 2.14.0+cu130, Python 3.12.5 | PyTorch 2.14.0, Python 3.12.5 |
| Numerics | FP32, TF32 off, seed 1234 | FP32, seed 1234 |

`cuda` = NVIDIA GPU through CUDA, `mps` = Apple GPU through Metal, `cpu` = the host CPU. Backends are detected automatically; an unavailable one is recorded as
`unavailable`. Nothing falls back from one device to another (`PYTORCH_ENABLE_MPS_FALLBACK` is removed), and OOM or unsupported operators are stored as data while the run continues.

---

## Detailed results

Every table shows both series; the speedup column is MPS ÷ CUDA. Full tables, CV/std and all plots are in each run's `report.md` and in `results/comparison/comparison.md`.
Metric definitions are [further down](#metric-definitions-what-how-it-is-calculated-why-it-matters).

### Matrix multiplication

*FP32; median GFLOPS = 2n³ / latency*

| n | cuda · RTX 3060 (GFLOPS) | mps · M1 Pro (GFLOPS) | speedup (mps vs cuda) |
|---|---|---|---|
| 1024 | 6,812 | 2,823 | 0.41 |
| 2048 | 7,516 | 4,040 | 0.54 |
| 4096 | 7,604 | 4,055 | 0.53 |
| 8192 | 9,435 | 4,059 | 0.43 |

![Matmul throughput](docs/example-results/01_matmul_throughput_vs_size.png)

### CNN training

*ResNet-18, Fashion-MNIST subset of 4,096 images, 3 epochs; median images/s per epoch*

| batch size | cuda · RTX 3060 (images/s) | mps · M1 Pro (images/s) | speedup (mps vs cuda) | mps epoch time (s) | mps final val. acc. |
|---|---|---|---|---|---|
| 16 | 959 | 553 | 0.58 | 7.41 | 0.751 |
| 32 | 1,231 | 644 | 0.52 | 6.36 | 0.743 |
| 64 | 1,468 | 732 | 0.50 | 5.60 | 0.776 |
| 128 | 1,585 | 746 | 0.47 | 5.49 | 0.711 |

The CUDA epoch times were 4.27 / 3.33 / 2.79 / 2.58 s. Validation accuracy uses a 1,000-image subset after 3 epochs, so it is a sanity check that training works,
not a quality result.

![CNN images per second](docs/example-results/02_cnn_images_per_sec.png)

### Transformer training

*4.2M-parameter GPT-style model; median tokens/s*

| batch \ sequence length | cuda 128 | cuda 256 | cuda 512 | mps 128 | mps 256 | mps 512 |
|---|---|---|---|---|---|---|
| 8 | 121,354 | 134,980 | 141,402 | 49,906 | 55,274 | 53,104 |
| 16 | 143,655 | 149,717 | 138,309 | 58,766 | 60,055 | 54,816 |
| 32 | 164,162 | 154,633 | 143,958 | 64,495 | 62,536 | 54,504 |

Across the nine configurations the M1 Pro reaches 0.38–0.41× the RTX 3060's tokens/s.

![Transformer tokens per second](docs/example-results/04_transformer_tokens_per_sec.png)

### Precision

*median throughput; speedup = mps vs cuda*

| workload | precision | cuda · RTX 3060 | mps · M1 Pro | speedup |
|---|---|---|---|---|
| matmul 4096×4096 (GFLOPS) | FP32 | 7,599 | 4,012 | 0.53 |
| matmul 4096×4096 (GFLOPS) | FP16 | 23,998 | 4,653 | 0.19 |
| matmul 4096×4096 (GFLOPS) | BF16 | 23,941 | 2,229 | 0.09 |
| GPT training step, autocast (tokens/s) | FP32 | 158,684 | 60,143 | 0.38 |
| GPT training step, autocast (tokens/s) | FP16 | 283,008 | 47,914 | 0.17 |
| GPT training step, autocast (tokens/s) | BF16 | 291,882 | 43,091 | 0.15 |

On the RTX 3060 half precision is faster than strict FP32 (about 3× for matmul, 1.8× for training). On the M1 Pro through MPS FP16/BF16 training is *slower* than FP32,
and BF16 matmul is slower than FP16 and FP32.

![Precision](docs/example-results/08_precision_fp32_fp16_bf16.png)

### Memory scaling

*largest configuration that ran; the two `memory_mb` values are different metrics and not comparable*

| ladder | cuda · RTX 3060 | mps · M1 Pro |
|---|---|---|
| batch size | up to 256 (8,394 MB); 512 out of memory (`OutOfMemoryError`) | up to 256 (11,561 MB); 512 stopped by the suite's system-memory safety cap (`SafetyCapSystemMemory`), not a native OOM |
| model size | ceiling reached: 206M parameters (4,568 MB) | ceiling reached: 206M parameters (6,267 MB) |
| sequence length | ceiling reached: 4,096 tokens (4,241 MB) | ceiling reached: 4,096 tokens (17,539 MB) |

CUDA memory is peak allocated tensor memory in dedicated VRAM (11.6 GiB); MPS memory is what the Metal driver holds in unified memory shared with the CPU and OS
(a sampled lower bound of the true peak).

### PPO / CartPole

*8 envs × 128 steps per iteration; median end-to-end env-steps/s*

| hidden width | cuda · RTX 3060 | mps · M1 Pro | speedup |
|---|---|---|---|
| 64 | 9,448 | 4,997 | 0.53 |
| 512 | 9,107 | 4,478 | 0.49 |

Gradient-step throughput: cuda 341 / 332 steps/s, mps 207 / 174 steps/s (width 64 / 512).

![PPO throughput](docs/example-results/09_ppo_throughput.png)

### Sustained training

*10 min*

| | cuda · RTX 3060 | mps · M1 Pro |
|---|---|---|
| average tokens/s | 139,755 | 59,359 |
| first→last 10% of windows (positive = slower at the end) | +4.10% | −0.18% |
| GPU telemetry | median utilisation 98%, 87 °C, 121 W | not collected (needs `sudo powermetrics`) |

![Sustained throughput](docs/example-results/10_sustained_throughput_over_time.png)

---

## Quick start

### Setup

```bash
python3.12 -m venv .venv          # 3.10+ works; 3.12 has the widest wheel support
source .venv/bin/activate
pip install -r requirements.txt   # optional for CUDA telemetry: pip install nvidia-ml-py
```

The CNN experiment downloads Fashion-MNIST (~30 MB) into `data/` on first use.

### Reproduce the MPS vs CUDA comparison

Run the suite on each machine for the backend it has, giving each run a meaningful ID:

```bash
python scripts/run_all.py --backend cuda --run-id rtx3060-20260919      # on the NVIDIA machine
python scripts/run_all.py --backend mps  --run-id m1pro-20260920        # on the Mac
```

Copy the run folders (`results/<run_id>/`, `raw/` is enough) into one `results/` directory, then compare:

```bash
python scripts/compare_runs.py rtx3060-20260919 m1pro-20260920                       # baseline = first run's first backend
python scripts/compare_runs.py rtx3060-20260919 m1pro-20260920 --baseline "cuda"     # or pick the baseline series explicitly
```

* Each available (machine, backend) becomes one **series**, labelled `<backend> · <device>` (a host name is appended only when two machines would get the same label).
* Output: `results/comparison/comparison.md`, `processed/*.csv` (values and speedups) and `plots/`.
* **Speedup** = median throughput of a series ÷ median throughput of the baseline series for the identical configuration. Use a CPU series as `--baseline` for "×faster than CPU".
* **Comparability check.** The script warns when the runs differ in `code_sha256`, seed or an experiment's configuration (subset size, epochs, batch sizes, durations …).
  Those numbers are not directly comparable, so run every machine with the same code and flags. Differences in torch / Python / OS versions are listed as context only.

### Run the suite

```bash
python scripts/run_all.py --backend auto --report      # every experiment on every available backend, then the report
python scripts/run_all.py --backend cpu                # or: mps, cuda, cpu,mps
python scripts/run_all.py --backend auto --experiment matmul   # one experiment: matmul, cnn, transformer, memory, precision, rl, sustained
python experiments/benchmark_matmul.py --backend mps           # or call an experiment directly (same flags)
```

`--backend auto` means "all three backends": unavailable ones get an `unavailable` row. Each (experiment, backend) pair runs in its own subprocess, so a crash in
one cannot end the suite. The default `sustained` experiment is **30 minutes per backend**, so the full suite takes on the order of hours on a laptop.

Change any configuration value from the command line without editing YAML:

```bash
python scripts/run_all.py --backend mps --experiment sustained --set sustained.duration_minutes=5
python scripts/run_all.py --backend auto --experiment cnn --set cnn.train_samples=null --set cnn.epochs=5   # full Fashion-MNIST
```

### Generate a per-run report

```bash
python scripts/generate_report.py                            # the run results/latest points at
python scripts/generate_report.py --run-id 20260919T172639   # a specific run
python scripts/generate_report.py --all-runs                 # keep every run in the folder, not just the newest per (experiment, backend)
```

It loads `results/<run_id>/raw/`, validates the data, writes aggregate tables, draws all plots and writes `results/<run_id>/report.md`.

---

## What is measured and how

### The seven experiments

| # | Experiment | What is measured |
|---|---|---|
| 1 | `matmul` | `A @ B`, FP32, sizes 1024–8192: latency per call, GFLOPS, matmuls/s, memory; correctness vs. CPU FP32 |
| 2 | `cnn` | ResNet-18 (CIFAR variant) on a seeded Fashion-MNIST subset (28x28 padded to 32x32, 3 channels), batch sizes 16–128: epoch time, images/s, train loss, val accuracy, memory |
| 3 | `transformer` | 4.2M-parameter GPT-style model, batch × sequence-length grid: step time, tokens/s, loss, memory |
| 4 | `memory` | Batch / sequence-length / model-size ladders until failure; each config in a fresh subprocess |
| 5 | `precision` | FP32 / FP16 / BF16: pure-dtype matmul and autocast training: throughput, step time, loss, memory |
| 6 | `rl` | PPO on CartPole, two policy widths: env steps/s, gradient steps/s, total runtime, losses, time split env / inference / update |
| 7 | `sustained` | Repeated training for `duration_minutes` (default 30): throughput over time, degradation, telemetry where accessible |

### Methodology

* **Same workload.** Model weights, data subsets, augmentation and token streams are created on the CPU from a fixed seed (`seed: 1234`) and copied to the device once;
  every backend gets identical inputs. GPU kernels are not bit-deterministic, so training trajectories can drift slightly between backends (visible in the loss columns).
* **Synchronised timing.** Each timed call is `sync → t0 → work → sync → t1` with `torch.cuda.synchronize()` / `torch.mps.synchronize()`. The sync cost is part of every measurement on every backend.
* **Warm-up.** Warm-up iterations run first (kernel compilation, clock ramp-up) and are recorded but excluded from statistics. Small workloads remain noisy; the coefficient of variation is reported.
* **Every measurement is saved.** Statistics are computed afterwards from the raw rows, never from a running total.
* **Speedups.** `series throughput ÷ baseline throughput` for the identical configuration. Values below 1 mean lower throughput than the baseline. There is no aggregate score.
* **Eager mode, defaults.** No `torch.compile`, no fused optimizers, strict FP32 (TF32 disabled; `numerics.allow_tf32` in the config). Software-stack differences (MPS kernels, cuDNN, …)
  are part of what is compared, so results describe *hardware + PyTorch backend together*, not silicon alone.
* **Training-step scope.** Steps include the host→device copy of the batch and `loss.item()`; the CNN data augmentation runs on the CPU per epoch *before* timing (identical, seeded) and is reported separately.

---

## Metric definitions: what, how it is calculated, why it matters

Formulas are rendered as LaTeX (GitHub renders `$…$` inline and `$$…$$` as display math). Notation used below: $t$ is the wall time of one timed call in
**seconds** (the raw data stores milliseconds, $t_\text{ms} = 1000 \cdot t$), $B$ the batch size, $L$ the sequence length and $n$ the matrix size.
Every number in the reports is computed from the per-measurement rows in `results/<run_id>/raw/`, never from a running total kept by the experiment.

Click a metric to expand it.

<details>
<summary><b>Timing and latency</b></summary>

**What.** *Latency* (`latency_ms`) is the wall time of one call or training step.

**How.** Every measurement is synchronised on both sides, so queued asynchronous GPU work is fully counted and cannot leak into the next iteration:

$$t = t_1 - t_0, \qquad \text{sync} \rightarrow t_0 \rightarrow \text{work} \rightarrow \text{sync} \rightarrow t_1$$

using `torch.cuda.synchronize()`, `torch.mps.synchronize()` (nothing on CPU) and `time.perf_counter()` on every backend. Warm-up iterations are recorded
(`phase = warmup`) but excluded from all statistics.

**Why it matters.** A GPU returns control to Python before the kernel has finished. Without the synchronisation the clock would measure only how fast work is
*queued*, which would flatter every GPU by a large and arbitrary factor. Excluding warm-up removes one-time costs (kernel compilation, clock ramp-up) that would
otherwise distort short benchmarks. Lower latency is better.

</details>

<details>
<summary><b>Summary statistics: median, mean, standard deviation, CV, min, max</b></summary>

**What.** Each configuration is measured $n$ times (matmul: 10–100 calls, CNN: one value per epoch, PPO: one per iteration). The distribution of those
values is summarised, and the **median** is the headline number.

**How.** With measurements $x_1,\dots,x_n$:

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i, \qquad s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}\left(x_i-\bar{x}\right)^2}, \qquad \mathrm{CV} = \frac{s}{\bar{x}}$$

$\tilde{x}$ (the median) is the middle value of the sorted measurements; $s$ is the *sample* standard deviation and is undefined (`null`) for $n < 2$.

**Why it matters.** The median is robust to the occasional slow outlier (an OS interrupt, a thermal dip, a garbage-collection pause), which would drag the mean
upward. The CV is a unit-free noise indicator: $\mathrm{CV}\approx 0.01$ means repeats agree within about 1%, while a large CV (e.g. 0.13 for the transformer at
batch 32 × sequence 128) flags a configuration whose number should not be over-interpreted. Report and read the CV alongside every median.

</details>

<details>
<summary><b>Throughput</b></summary>

**What.** Work done per second, in a unit that depends on the experiment (`throughput_unit`). Higher is better.

**How.** In general

$$\theta = \frac{W}{t}$$

where $W$ is the amount of work in one measured call. The concrete $W$ and unit for each experiment are given below. Because $\theta = W/t$ is a decreasing
function of $t$, the median throughput equals the work divided by the median latency.

**Why it matters.** Throughput lets you compare configurations that do different amounts of work per call (different batch sizes, matrix sizes) on one axis, and
it is the quantity that the speedup below is built from. Only compare throughputs that share the same unit *and* configuration.

</details>

<details>
<summary><b>Matrix multiplication: FLOPs and GFLOPS</b></summary>

**What.** The *achieved* floating-point rate for $C = AB$ with $A, B \in \mathbb{R}^{n\times n}$ (`matmul`, `matmuls_per_sec`).

**How.** Each of the $n^2$ output entries is a dot product of length $n$, costing $n$ multiplications and $n-1$ additions, so

$$\text{FLOPs}(n) \approx 2n^3, \qquad \text{GFLOPS} = \frac{2n^3}{t \cdot 10^{9}}, \qquad \text{matmuls/s} = \frac{1}{t}$$

*Example:* $n = 4096$ in $t = 18.074\ \text{ms}$ gives $\dfrac{2\cdot 4096^3}{0.018074\cdot 10^{9}} \approx 7{,}604$ GFLOPS (the `cuda · NVIDIA GeForce RTX 3060` row).

**Why it matters.** Matmul is the core operation of neural-network training, and for large $n$ it is compute-bound, so it isolates raw arithmetic throughput
from data-pipeline and framework overhead. It is the *achieved* rate for that size and dtype, **not** the hardware's theoretical peak: small matrices
cannot fill a large GPU, which is why GFLOPS grows with $n$. The pure-dtype matmul in the `precision` experiment uses the same formula for FP32, FP16 and BF16.

</details>

<details>
<summary><b>CNN training: epoch time, images/s, loss and accuracy</b></summary>

**What.** ResNet-18 trained on a seeded Fashion-MNIST subset (default 4,096 images, 3 epochs).

**How.** With $S$ steps per epoch and per-step synchronised times $t_1,\dots,t_S$:

$$T_\text{epoch} = \sum_{s=1}^{S} t_s, \qquad \theta_\text{img} = \frac{S \cdot B}{T_\text{epoch}}\ \ [\text{images/s}]$$

*Example:* batch 128 on the M1 Pro has $S \cdot B = 4096$ images and $T_\text{epoch} = 5.488$ s, so $\theta_\text{img} = 4096/5.488 \approx 746$ images/s.
Each step includes the host→device copy of the batch, forward, backward, optimiser update and `loss.item()`. Data augmentation and validation are timed
separately and are *not* part of $T_\text{epoch}$. The reported **training loss** is the mean cross-entropy over the epoch's steps, and **validation accuracy** is

$$\text{acc} = \frac{N_\text{correct}}{N_\text{val}}, \qquad N_\text{val} = 1000$$

**Why it matters.** Images/s (and its inverse, epoch time) is what determines how long a real training run takes. Batch size changes it because larger batches keep
the device busier. Loss and accuracy are only a sanity check that the run actually trained and the numerics are sane, not a model-quality result (3 epochs on a
1,000-image validation subset).

</details>

<details>
<summary><b>Transformer training: tokens/s</b></summary>

**What.** A 4.2M-parameter GPT-style model, one full training step per measurement (copy of tokens, forward, cross-entropy, backward, AdamW).

**How.**

$$\theta_\text{tok} = \frac{B \cdot L}{t_\text{step}}\ \ [\text{tokens/s}]$$

The `precision` experiment's `training_autocast` workload and the `sustained` experiment use the same definition. Loss is next-token cross-entropy; the synthetic data
has a known entropy floor $\ln(\text{branching})$ that the loss cannot go below, which makes divergence or a broken run visible.

**Why it matters.** Tokens/s is the transformer analogue of images/s and is the figure that translates directly into "how long does this training run take".
It depends on both $B$ and $L$ (longer sequences cost more per token because attention scales with $L^2$), so it is reported over a batch × sequence grid.

</details>

<details>
<summary><b>Precision: FP32, FP16, BF16</b></summary>

**What.** The same two throughput metrics (GFLOPS for a pure-dtype matmul, tokens/s for autocast training) measured per numeric format.

**How.** Formulas as above; FP32 is *strict* FP32 (TF32 disabled), so the half-precision gain is measured against a true FP32 baseline:

$$\text{gain}_{\text{FP16}} = \frac{\theta_{\text{FP16}}}{\theta_{\text{FP32}}}$$

(the same expression with BF16). A gain above 1 means the format is faster on that backend, below 1 that it is slower.

**Why it matters.** Half precision is the standard way to speed up and shrink training, but it is only faster where the hardware and kernels support it. The result
differs sharply per backend (see the [detailed results](#detailed-results)), so it should be measured, not assumed.

</details>

<details>
<summary><b>Reinforcement learning (PPO / CartPole): three rates</b></summary>

**What.** One PPO iteration collects $N = E \cdot T_r$ environment steps ($E$ parallel environments $\times$ $T_r$ steps each; 8 × 128 = 1024 by default) and then updates the policy.

**How.** The iteration time is split into four synchronised parts, and three rates are derived from them:

$$T_\text{iter} = T_\text{env} + T_\text{infer} + T_\text{gae} + T_\text{update}$$

$$\theta_\text{end-to-end} = \frac{N}{T_\text{iter}}, \qquad \theta_\text{env-only} = \frac{N}{T_\text{env}}, \qquad \theta_\text{grad} = \frac{G}{T_\text{update}}$$

where $G$ is the number of minibatch gradient steps in the update. $T_\text{env}$ is the simulator (always on the CPU), $T_\text{infer}$ the policy forward pass
including the host↔device round trip, $T_\text{gae}$ advantage estimation, and $T_\text{update}$ the gradient updates on the device.

*Example:* the M1 Pro at hidden width 64 has a median iteration of 204.9 ms, so $\theta_\text{end-to-end} = 1024/0.2049 \approx 4{,}997$ env-steps/s.

**Why it matters.** **End-to-end** steps/s is the rate that determines wall-clock time to train an RL agent. **Env-only** is independent of the accelerator and shows the
ceiling that the simulator imposes; **gradient steps/s** isolates the neural-network training speed. Together they explain *why* a GPU does or does not help:
with a tiny policy the per-step CPU↔device round trip and the CPU-bound simulator dominate, so the accelerator barely matters.

</details>

<details>
<summary><b>Sustained training: window throughput and degradation</b></summary>

**What.** The transformer step repeated for a fixed duration (default 30 min), summarised in fixed time windows.

**How.** For a window $w$ of length $\Delta t_w$ containing $k_w$ steps, and a run of total length $T$:

$$\theta_w = \frac{k_w \cdot B \cdot L}{\Delta t_w}, \qquad \bar{\theta} = \frac{\sum_w k_w \cdot B \cdot L}{T} = \frac{\text{total tokens}}{T}$$

Degradation compares the median window throughput of the first and last 10% of windows (at least one window each):

$$D = 100\cdot\frac{\tilde{\theta}_\text{first} - \tilde{\theta}_\text{last}}{\tilde{\theta}_\text{first}}\ \ [\text{percent}]$$

Positive $D$ means the run was slower at the end; negative means it sped up slightly. *Example:* $D = +4.10$ % for the RTX 3060 and $D = -0.18$ % for the M1 Pro
in the [detailed results](#detailed-results). GPU utilisation, temperature and power are read where accessible (NVML / `nvidia-smi` on CUDA; not collected on Apple Silicon
without `sudo powermetrics`).

**Why it matters.** Short benchmarks run on a cool, boosted device. Real training runs for hours, and sustained clocks are lower once the chip heats up or a laptop
throttles. $D$ quantifies that gap, which matters most for thin, fanless or laptop hardware.

</details>

<details>
<summary><b>Speedup</b></summary>

**What.** Throughput of one series relative to a chosen *baseline* series for the identical configuration (`compare_runs.py`, per-run report).

**How.**

$$S = \frac{\tilde{\theta}_\text{series}}{\tilde{\theta}_\text{baseline}} = \frac{\tilde{t}_\text{baseline}}{\tilde{t}_\text{series}}$$

the ratio of the two medians, computed per row of the table (per matrix size, batch size, …). $S > 1$ means higher throughput than the baseline, $S < 1$ lower,
and $S = 1$ equal. The baseline is the first series by default; pass `--baseline "cpu · Apple M1 Pro"` for the usual "×faster than CPU" numbers.
*Example:* matmul $n=4096$, M1 Pro vs RTX 3060: $S = 4054.8 / 7604.2 \approx 0.53$.

**Why it matters.** A ratio makes results across very different absolute scales (GFLOPS vs tokens/s vs env-steps/s) directly readable. It is meaningful only when both
series ran the *same* code, seed and settings, which is why `compare_runs.py` prints a comparability warning when `code_sha256`, the seed or an experiment's configuration
differ. There is deliberately no aggregate score: an average of speedups across unrelated workloads would hide exactly the per-workload differences this suite exists to show.

</details>

<details>
<summary><b>Memory and the largest successful configuration</b></summary>

**What.** `memory_mb` is the memory a backend reports for a workload. The *largest successful configuration* is the last `ok` rung of a memory ladder
(batch size, sequence length or model size) before the ladder fails or reaches its ceiling.

**How.** There is no single formula: each backend exposes a different counter (CUDA exact peak of tensor allocations; MPS driver-held memory, sampled; CPU process RSS),
and each value is stored with a `memory_kind` label. A ladder ends in one of three ways: a native out-of-memory error, a suite `SafetyCap*` limit, or reaching the ladder
ceiling with no failure.

**Why it matters.** Memory decides *how large* a model or batch you can train at all, independent of how fast it runs. Because the counters measure different things,
**absolute memory values must not be compared across backends**; see [Memory metrics are NOT equivalent](#memory-metrics-are-not-equivalent) below.

</details>

## Memory metrics are NOT equivalent

`memory_mb` always comes with a `memory_kind` label. Do not compare absolute values across backends.

| Backend | What `memory_mb` is | Caveats |
|---|---|---|
| CUDA | `torch.cuda.max_memory_allocated` (exact peak of tensor allocations); reserved memory is stored in `extra` | Dedicated VRAM. Excludes CUDA context and allocator cache. |
| MPS | `torch.mps.driver_allocated_memory` (what the Metal driver holds for the process, including allocator cache). Where sampled, the maximum polled every 5 ms; `current_allocated_memory` (tensor bytes only) is stored in `extra` | **PyTorch exposes no peak counter for MPS**, so this is a snapshot or a sampled lower bound of the true peak. Includes a fixed floor of roughly 1 GiB, so small workloads look alike. Unified memory is shared with the CPU and the OS. |
| CPU | Process RSS: snapshot, sampled maximum, or lifetime peak (`ru_maxrss`), as labelled | Includes the Python/torch baseline; OS-managed, can include shared pages. |

In the memory-scaling experiment, unified-memory machines can start swapping instead of failing, so the suite enforces safety limits
(MPS memory fraction, CPU RSS cap, minimum system memory, per-configuration timeout). A ladder ended by one of these limits is
reported as `SafetyCap*` — *stopped by this suite*, not a native OOM.

---

## Reference

### Where things are stored

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
| Configuration | `configs/default.yaml` |

Each suite run gets its own folder named after its run ID (a timestamp, or whatever you pass with `--run-id`), so raw data, tables, plots and the report of one run always belong together.

### Project layout

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

### Result schema

Every raw CSV shares these columns; experiment-specific values go into the JSON column `extra` (`generate_report.py` expands them to `x_<name>`).

`run_id, timestamp, experiment, backend, device, model, dataset, batch_size, sequence_length, precision, phase, iteration, latency_ms, throughput, throughput_unit, memory_mb, memory_kind, loss, accuracy, status, error_type, error, extra, metadata`

* `phase`: `warmup` (recorded, excluded from statistics), `measure`, or `config` (one row describing a whole configuration, used for memory summaries, failures and run summaries).
* `status`: `ok | oom | unsupported | unavailable | error | timeout | skipped`.
* `metadata` holds a compact fingerprint; the full metadata (timestamp, hostname, OS, Python/PyTorch/torchvision versions, device and GPU names, CPU, RAM, CUDA version, MPS availability,
  git commit, source hash, power source, seed, full configuration) is in the `.meta.json` file next to the CSV.

### Tests

```bash
python -m pytest -q
```

---

## Limitations and caveats

**About the MPS vs CUDA comparison**

* **Two machines, two builds.** The RTX 3060 and M1 Pro runs used different source hashes (`code_sha256`), so the comparison is indicative. Run both from one checkout for a like-for-like result.
* **One device of each kind, one run each.** These are single runs on one RTX 3060 desktop and one M1 Pro laptop; they say nothing about other GPUs or other Apple chips.
* **Hardware + software together.** Results include the PyTorch backend (MPS kernels vs CUDA/cuDNN), not silicon alone. FP32 is strict FP32 (TF32 off) on both.
* **Half precision does not imply speed-up.** FP16/BF16 behave very differently per backend (see the precision section of the detailed results).
* **Laptop conditions matter.** The power source (AC/battery) and low-power mode are recorded per run; keep the machine plugged in and idle.

**MPS-specific**

* **No peak-memory API.** PyTorch exposes no peak counter for MPS, so memory is a snapshot or sampled lower bound. There is also no way to detect silent CPU fallback beyond disabling it (done here).
* **No GPU telemetry without `sudo`.** GPU utilisation, power and temperature on Apple Silicon require `sudo powermetrics`, which the suite does not require; those fields are stored as `null` with the reason.
  Only the OS thermal speed limit (`pmset -g therm`) is collected.
* **Unified memory.** The memory ladder can start swapping instead of failing, so the suite stops it with safety limits (reported as `SafetyCap*`, not a native OOM).

**CUDA-specific**

* CUDA telemetry (GPU utilisation, temperature, power) is read through `nvidia-smi` / NVML; the OS CPU speed limit is not collected on Linux.

**General**

* **CPU note.** On Apple Silicon the CPU matmul runs on the AMX matrix units via Accelerate (~1.6 TFLOPS, independent of thread count), so "CPU" there is unusually strong for large FP32 matmuls. No `cpu` series is part of the comparison above.
* **CNN default uses a 4096-image Fashion-MNIST subset** because the CPU trains at tens of images/s; set `cnn.train_samples=null` for the full dataset.
* **Sustained default is 30 minutes** (the example above used 10). Shorter runs are labelled by their real duration in the report.
* **The checkout is not a git repository**, so `git_commit` is `null`; a SHA-256 of all sources (`code_sha256`) is recorded instead.
