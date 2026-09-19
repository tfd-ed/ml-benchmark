"""Load raw results -> validate -> aggregate -> plots -> Markdown report.

    python scripts/generate_report.py [--results-dir results] [--all-runs]

By default only the most recent run of every (experiment, backend) is reported.
The report states measurements only; it never ranks backends or declares a winner.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import plotting  # noqa: E402
from src.config import resolve_path  # noqa: E402
from src.reporting import add_speedup, aggregate_stats, latest_runs, load_results, md_table, validate  # noqa: E402

FAIL_STATUSES = ("oom", "unsupported", "unavailable", "error", "timeout", "skipped")


def num(v, digits=1):
    return "n/a" if v is None or pd.isna(v) else f"{v:,.{digits}f}"


class Report:
    def __init__(self, results_dir: Path):
        self.dir = results_dir
        self.plots_dir = results_dir / "plots"
        self.processed = results_dir / "processed"
        self.processed.mkdir(parents=True, exist_ok=True)
        self.parts: list[str] = []
        self.observations: list[str] = []

    def add(self, text: str) -> None:
        self.parts.append(text.rstrip() + "\n")

    def add_plots(self, paths) -> None:
        for p in paths:
            if p.suffix == ".png":
                self.add(f"![{p.stem}](plots/{p.name})  \n_(also saved as `plots/{p.stem}.svg`)_")

    def save_table(self, df: pd.DataFrame, name: str) -> None:
        df.to_csv(self.processed / f"{name}.csv", index=False)


# --------------------------------------------------------------------------- experiment sections


def section_matmul(df: pd.DataFrame, rep: Report) -> None:
    d = df[df["experiment"] == "matmul"]
    ok = d[(d["status"] == "ok") & (d["phase"] == "measure")].copy()
    rep.add("## Experiment 1: Matrix multiplication (A @ B)\n")
    if ok.empty:
        rep.add("No successful matmul measurements.\n")
        return
    ok["matrix_size"] = ok["x_matrix_size"].astype(int)
    keys = ["backend", "precision", "matrix_size"]
    lat = aggregate_stats(ok, keys, "latency_ms")
    gf = aggregate_stats(ok, keys, "throughput").rename(columns={c: f"gflops_{c}" for c in ("median", "mean", "std", "cv", "min", "max")})
    tbl = lat.rename(columns={c: f"lat_ms_{c}" for c in ("median", "mean", "std", "cv", "min", "max")})
    tbl = tbl.merge(gf[keys + ["gflops_median", "gflops_mean"]], on=keys)
    tbl["matmuls_per_sec_median"] = 1000.0 / tbl["lat_ms_median"]
    mem = ok.groupby(keys)["memory_mb"].first().reset_index()
    tbl = tbl.merge(mem, on=keys)
    tbl = add_speedup(tbl.rename(columns={"lat_ms_median": "median"}), ["precision", "matrix_size"]).rename(columns={"median": "lat_ms_median"})
    rep.save_table(tbl, "matmul_summary")
    show = tbl[["backend", "precision", "matrix_size", "n", "lat_ms_median", "lat_ms_mean", "lat_ms_std", "lat_ms_cv", "gflops_median", "matmuls_per_sec_median", "memory_mb", "speedup_vs_cpu"]]
    rep.add("Latency per `A @ B` (ms), throughput in GFLOPS (2n³ / latency). `speedup_vs_cpu` = cpu median latency / backend median latency "
            "(CPU is the baseline; <1 means slower than the CPU baseline). `memory_mb` is backend-specific, see the metric notes.\n")
    rep.add(md_table(show, 3))
    rep.add_plots(plotting.plot_matmul_throughput(df, rep.plots_dir))
    for n, g in tbl.groupby("matrix_size"):
        parts = [f"{r.backend} {num(r.gflops_median, 0)} GFLOPS ({num(r.lat_ms_median, 2)} ms)" for r in g.itertuples()]
        rep.observations.append(f"Matmul n={n} ({g['precision'].iloc[0]}): median throughput – " + "; ".join(parts) + ".")
    err = d[d["x_rel_err_vs_cpu_fp32"].notna()] if "x_rel_err_vs_cpu_fp32" in d else d.iloc[0:0]
    if not err.empty:
        e = err.groupby(["backend", "x_matrix_size"])["x_rel_err_vs_cpu_fp32"].first().reset_index()
        rep.observations.append("Max relative difference vs. CPU FP32 result: " + "; ".join(f"{r.backend} n={int(r.x_matrix_size)}: {r.x_rel_err_vs_cpu_fp32:.2e}" for r in e.itertuples()) + ".")


def stat_table(ok: pd.DataFrame, keys: list[str], cols: dict[str, str]) -> pd.DataFrame:
    """One row per group with n and median/mean/std/cv for every (label -> column) in `cols`."""
    out = None
    for i, (label, col) in enumerate(cols.items()):
        agg = aggregate_stats(ok, keys, col)
        keep = ["median", "mean", "std", "cv"] + (["n"] if i == 0 else [])
        agg = agg[keys + keep].rename(columns={c: (c if c == "n" else f"{label}_{c}") for c in keep})
        out = agg if out is None else out.merge(agg, on=keys, how="outer")
    return out


def observe(rep: Report, prefix: str, tbl: pd.DataFrame, cfg_cols: list[str], value_col: str, unit: str, digits: int = 0) -> None:
    for cfg, g in tbl.groupby(cfg_cols):
        cfg = cfg if isinstance(cfg, tuple) else (cfg,)
        label = ", ".join(f"{c}={v}" for c, v in zip(cfg_cols, cfg))
        parts = "; ".join(f"{r.backend} {num(getattr(r, value_col), digits)} {unit}" for r in g.itertuples())
        rep.observations.append(f"{prefix} [{label}]: median – {parts}.")


def cfg_str(*vals) -> str:
    return " ".join(str(v) for v in vals)


def section_cnn(df: pd.DataFrame, rep: Report) -> None:
    rep.add("## Experiment 2: CNN training (ResNet-18 / CIFAR-10)\n")
    d = df[df["experiment"] == "cnn"]
    ok = d[(d["status"] == "ok") & (d["phase"] == "measure") & (d["x_granularity"] == "epoch")].copy() if "x_granularity" in d else d.iloc[0:0]
    if ok.empty:
        rep.add("No successful CNN epochs.\n")
    else:
        ok["epoch_time_s"] = ok["x_epoch_time_s"]
        ok["batch_size"] = ok["batch_size"].astype(int)
        tbl = stat_table(ok, ["backend", "batch_size"], {"epoch_s": "epoch_time_s", "images_per_s": "throughput"})
        last = ok.sort_values("iteration").groupby(["backend", "batch_size"]).tail(1)[["backend", "batch_size", "loss", "accuracy"]].rename(columns={"loss": "final_train_loss", "accuracy": "final_val_acc"})
        mem = d[d["x_granularity"] == "memory_summary"][["backend", "batch_size", "memory_mb"]] if "x_granularity" in d else pd.DataFrame(columns=["backend", "batch_size", "memory_mb"])
        mem = mem.assign(batch_size=mem["batch_size"].astype(int))
        tbl = tbl.merge(last, on=["backend", "batch_size"]).merge(mem, on=["backend", "batch_size"], how="left")
        tbl = add_speedup(tbl.rename(columns={"epoch_s_median": "median"}), ["batch_size"]).rename(columns={"median": "epoch_s_median"})
        rep.save_table(tbl, "cnn_summary")
        rep.add("`n` = number of measured epochs. Epoch time = summed step time (host→device copy, forward, backward, optimizer step; synchronised). "
                "`speedup_vs_cpu` = cpu epoch time / backend epoch time (baseline = CPU). `memory_mb` is backend-specific (see notes). The first epoch includes warm-up effects not covered by the warm-up steps.\n")
        rep.add(md_table(tbl[["backend", "batch_size", "n", "epoch_s_median", "epoch_s_mean", "epoch_s_std", "epoch_s_cv", "images_per_s_median", "final_train_loss", "final_val_acc", "memory_mb", "speedup_vs_cpu"]], 3))
        rep.add_plots(plotting.plot_cnn_images_per_sec(df, rep.plots_dir) + plotting.plot_cnn_epoch_time(df, rep.plots_dir))
        observe(rep, "CNN training throughput", tbl, ["batch_size"], "images_per_s_median", "images/s")
        observe(rep, "CNN final validation accuracy", tbl, ["batch_size"], "final_val_acc", "(top-1)", 3)
    rep.add("")


def section_transformer(df: pd.DataFrame, rep: Report) -> None:
    rep.add("## Experiment 3: Transformer training\n")
    d = df[df["experiment"] == "transformer"]
    ok = d[(d["status"] == "ok") & (d["phase"] == "measure")].copy()
    if ok.empty:
        rep.add("No successful transformer steps.\n")
        return
    ok["batch_size"], ok["sequence_length"] = ok["batch_size"].astype(int), ok["sequence_length"].astype(int)
    keys = ["backend", "batch_size", "sequence_length"]
    tbl = stat_table(ok, keys, {"step_ms": "latency_ms", "tokens_per_s": "throughput"})
    first_last = ok.sort_values("iteration").groupby(keys).agg(loss_first=("loss", "first"), loss_last=("loss", "last")).reset_index()
    mem = d[d["phase"] == "config"][keys + ["memory_mb"]].dropna(subset=["batch_size"])
    mem = mem.assign(batch_size=mem["batch_size"].astype(int), sequence_length=mem["sequence_length"].astype(int))
    tbl = tbl.merge(first_last, on=keys).merge(mem, on=keys, how="left")
    tbl = add_speedup(tbl.rename(columns={"step_ms_median": "median"}), ["batch_size", "sequence_length"]).rename(columns={"median": "step_ms_median"})
    rep.save_table(tbl, "transformer_summary")
    rep.add("Step time = one synchronised training step (ms); `tokens_per_s` = batch × sequence / step time. `speedup_vs_cpu` = cpu step time / backend step time (baseline = CPU). "
            "`memory_mb` is backend-specific (see notes). Loss should fall towards ln(4) ≈ 1.39 on the synthetic Markov data.\n")
    rep.add(md_table(tbl[keys + ["n", "step_ms_median", "step_ms_mean", "step_ms_std", "step_ms_cv", "tokens_per_s_median", "loss_first", "loss_last", "memory_mb", "speedup_vs_cpu"]], 3))
    rep.add_plots(plotting.plot_transformer_tokens(df, rep.plots_dir) + plotting.plot_transformer_vs_batch(df, rep.plots_dir))
    observe(rep, "Transformer throughput", tbl, ["batch_size", "sequence_length"], "tokens_per_s_median", "tokens/s")
    rep.add("")


def section_memory(df: pd.DataFrame, rep: Report) -> None:
    rep.add("## Experiment 4: Memory scaling\n")
    d = df[df["experiment"] == "memory"]
    ladder_rows = d[d["status"] != "unavailable"].copy()
    if ladder_rows.empty:
        rep.add("No memory-scaling data.\n")
        return
    rep.add("Every configuration is one training step run in a fresh subprocess. A ladder stops at its first failure. Memory capacity says how *large* a workload fits, not how *fast* it runs. "
            "`ended_by_safety_cap` = the ladder was stopped by this suite's memory/time limit (needed because unified-memory machines swap instead of failing), not by a native out-of-memory error.\n")
    mx = plotting.max_successful(df)
    rep.save_table(mx, "memory_max_successful")
    rep.add("### Largest successful configuration per dimension\n")
    rep.add(md_table(mx[["backend", "ladder", "largest_ok_model", "largest_ok_batch", "largest_ok_seq", "largest_ok_params_M", "memory_mb_at_largest_ok", "ladder_ended_by", "ended_by_safety_cap"]], 1))
    cols = ["backend", "ladder", "model", "batch_size", "sequence_length", "x_n_params", "status", "error_type", "memory_mb", "x_mps_current_allocated_sampled_peak_mb", "latency_ms", "throughput"]
    full = ladder_rows[[c for c in cols if c in ladder_rows.columns]].rename(columns={"x_n_params": "params", "latency_ms": "step_ms", "throughput": "tokens_per_s", "x_mps_current_allocated_sampled_peak_mb": "mps_tensor_bytes_mb"})
    rep.save_table(full, "memory_all_configurations")
    rep.add("### All configurations tried\n")
    rep.add(md_table(full, 1))
    kinds = ladder_rows.dropna(subset=["memory_kind"]).drop_duplicates("backend")[["backend", "memory_kind"]]
    rep.add("Memory metric per backend: " + "; ".join(f"**{r.backend}** = {r.memory_kind}" for r in kinds.itertuples()) + ".\n")
    cap = ladder_rows.dropna(subset=["x_capacity_mb"]).drop_duplicates("backend") if "x_capacity_mb" in ladder_rows else ladder_rows.iloc[0:0]
    if len(cap):
        rep.add("Nominal memory capacity: " + "; ".join(f"**{r.backend}** {r.x_capacity_mb / 1024:.1f} GiB ({r.x_capacity_kind})" for r in cap.itertuples()) + ".\n")
    rep.add_plots(plotting.plot_memory_vs_batch(df, rep.plots_dir) + plotting.plot_max_config(df, rep.plots_dir))
    for r in mx.itertuples():
        if pd.notna(r.largest_ok_value):
            rep.observations.append(f"Memory ladder '{r.ladder}' on {r.backend}: largest successful value {r.largest_ok_value:,.4g}"
                                    + (f" ({r.largest_ok_model}, {r.largest_ok_params_M:.0f}M params)" if r.ladder == "model" and pd.notna(r.largest_ok_params_M) else "")
                                    + f"; ladder ended by: {r.ladder_ended_by}.")
    rep.add("")


def section_precision(df: pd.DataFrame, rep: Report) -> None:
    rep.add("## Experiment 5: Precision (FP32 / FP16 / BF16)\n")
    d = df[df["experiment"] == "precision"]
    ok = d[(d["status"] == "ok") & (d["phase"] == "measure")].copy()
    if ok.empty:
        rep.add("No successful precision measurements.\n")
        return
    ok["workload"] = ok["x_workload"]
    keys = ["backend", "workload", "precision"]
    tbl = stat_table(ok, keys, {"latency_ms": "latency_ms", "throughput": "throughput"})
    last_loss = ok[ok["workload"] == "training_autocast"].sort_values("iteration").groupby(keys).agg(loss_first=("loss", "first"), loss_last=("loss", "last")).reset_index()
    mem = d[d["phase"] == "config"][keys[:1] + ["precision", "x_workload", "memory_mb"]].rename(columns={"x_workload": "workload"}).dropna(subset=["workload"])
    tbl = tbl.merge(last_loss, on=keys, how="left").merge(mem, on=keys, how="left")
    mm = ok[ok["workload"] == "matmul_pure_dtype"].groupby(keys)["memory_mb"].first().reset_index().rename(columns={"memory_mb": "mm_mem"})
    tbl = tbl.merge(mm, on=keys, how="left")
    tbl["memory_mb"] = tbl["memory_mb"].fillna(tbl["mm_mem"])
    tbl = tbl.drop(columns="mm_mem")
    tbl = add_speedup(tbl.rename(columns={"latency_ms_median": "median"}), ["workload", "precision"]).rename(columns={"median": "latency_ms_median"})
    rep.save_table(tbl, "precision_summary")
    rep.add("`speedup_vs_cpu` compares like for like: same workload and precision against the CPU backend (baseline). GFLOPS for matmul, tokens/s for training. "
            "A precision appears here only if it actually ran; failures are listed in the failure section. Throughput in a lower precision is not automatically higher - see the numbers.\n")
    rep.add(md_table(tbl[["backend", "workload", "precision", "n", "latency_ms_median", "latency_ms_cv", "throughput_median", "loss_first", "loss_last", "memory_mb", "speedup_vs_cpu"]], 3))
    rep.add_plots(plotting.plot_precision(df, rep.plots_dir))
    for wl, unit in (("matmul_pure_dtype", "GFLOPS"), ("training_autocast", "tokens/s")):
        observe(rep, f"Precision ({wl})", tbl[tbl["workload"] == wl], ["precision"], "throughput_median", unit)
    rep.add("")


def section_rl(df: pd.DataFrame, rep: Report) -> None:
    rep.add("## Experiment 6: Reinforcement learning (PPO / CartPole)\n")
    d = df[df["experiment"] == "rl"]
    ok = d[(d["status"] == "ok") & (d["phase"] == "measure")].copy()
    if ok.empty:
        rep.add("No successful PPO iterations.\n")
        return
    ok["hidden"] = ok["x_hidden"].astype(int)
    keys = ["backend", "hidden"]
    tbl = stat_table(ok, keys, {"iter_ms": "latency_ms", "env_only_sps": "x_env_steps_per_sec_env_only", "end2end_sps": "x_env_steps_per_sec_end2end", "train_sps": "x_train_steps_per_sec"})
    parts = ok.groupby(keys).agg(total_runtime_s=("latency_ms", lambda s: s.sum() / 1000), env_s=("x_env_s", "mean"), infer_s=("x_infer_s", "mean"), gae_s=("x_gae_s", "mean"), update_s=("x_update_s", "mean"),
                                 policy_loss_last=("x_policy_loss", "last"), value_loss_last=("x_value_loss", "last"), return_last=("x_mean_episode_return", "last")).reset_index()
    tbl = tbl.merge(parts, on=keys)
    tbl = add_speedup(tbl.rename(columns={"iter_ms_median": "median"}), ["hidden"]).rename(columns={"median": "iter_ms_median"})
    rep.save_table(tbl, "rl_summary")
    rep.add("Times are means per PPO iteration (8 envs × 128 steps, then 4 epochs of minibatch updates); `total_runtime_s` sums all measured iterations. "
            "`speedup_vs_cpu` = cpu iteration time / backend iteration time (baseline = CPU). `return_last` = mean return of the last 20 finished episodes (learning sanity check).\n")
    rep.add(md_table(tbl[["backend", "hidden", "n", "total_runtime_s", "env_s", "infer_s", "gae_s", "update_s", "env_only_sps_median", "end2end_sps_median", "train_sps_median", "policy_loss_last", "value_loss_last", "return_last", "speedup_vs_cpu"]], 3))
    rep.add_plots(plotting.plot_ppo(df, rep.plots_dir) + plotting.plot_ppo_breakdown(df, rep.plots_dir))
    observe(rep, "PPO end-to-end throughput", tbl, ["hidden"], "end2end_sps_median", "env-steps/s")
    observe(rep, "PPO gradient-step throughput", tbl, ["hidden"], "train_sps_median", "grad-steps/s")
    rep.add("")


def section_sustained(df: pd.DataFrame, rep: Report) -> None:
    rep.add("## Experiment 7: Sustained training\n")
    d = df[df["experiment"] == "sustained"]
    ok = d[(d["status"] == "ok") & (d["phase"] == "measure")]
    summ = d[(d["status"] == "ok") & (d["phase"] == "config")]
    if ok.empty:
        rep.add("No sustained-run windows.\n")
        return
    rows = []
    for b, s in summ.groupby("backend"):
        s = s.iloc[0]
        w = ok[ok["backend"] == b]
        rows.append({"backend": b, "duration_min": s["x_elapsed_s"] / 60, "windows": len(w), "avg_tokens_per_s": s["x_average_throughput_tokens_per_s"], "median_window_tokens_per_s": s["x_median_window_throughput"],
                     "first10pct_median": s["x_degradation_first_median"], "last10pct_median": s["x_degradation_last_median"], "degradation_pct": s["x_degradation_degradation_pct"],
                     "total_tokens": s["x_total_tokens"], "total_samples": s["x_total_samples"], "final_loss": w.sort_values("iteration")["loss"].iloc[-1], "memory_mb": s["memory_mb"]})
    tbl = pd.DataFrame(rows)
    rep.save_table(tbl, "sustained_summary")
    rep.add("`degradation_pct` = 100 × (first-10%-windows median − last-10%-windows median) / first; positive means throughput fell during the run. "
            "Samples = sequences, tokens = samples × sequence length.\n")
    rep.add(md_table(tbl, 2))
    tel = []
    for b, w in ok.groupby("backend"):
        for m, label in (("gpu_utilization_percent", "GPU utilisation"), ("temperature_c", "temperature"), ("power_w", "power"), ("cpu_speed_limit_percent", "OS CPU speed limit")):
            col = f"x_{m}"
            have = col in w and w[col].notna().any()
            reasons = set()
            if "x_telemetry_unavailable" in w:
                for r in w["x_telemetry_unavailable"].dropna():
                    if isinstance(r, dict) and m in r:
                        reasons.add(r[m])
            tel.append({"backend": b, "metric": label, "collected": "yes" if have else "no",
                        "median_value": float(w[col].median()) if have else None, "reason / note": "; ".join(sorted(reasons))[:220] if reasons else ("" if have else "no reason recorded")})
    rep.add("### Telemetry availability\n")
    rep.add(md_table(pd.DataFrame(tel), 1))
    rep.add_plots(plotting.plot_sustained(df, rep.plots_dir))
    for r in tbl.itertuples():
        rep.observations.append(f"Sustained run on {r.backend}: {r.duration_min:.1f} min, average {num(r.avg_tokens_per_s, 0)} tokens/s, median window {num(r.median_window_tokens_per_s, 0)} tokens/s, "
                                f"first→last 10% windows {num(r.degradation_pct, 2)}% change (positive = slower); {r.total_tokens:,.0f} tokens ({r.total_samples:,.0f} samples) processed.")
    rep.add("")


SECTIONS = [("matmul", section_matmul), ("cnn", section_cnn), ("transformer", section_transformer), ("memory", section_memory),
            ("precision", section_precision), ("rl", section_rl), ("sustained", section_sustained)]

# --------------------------------------------------------------------------- generic sections


def md_hardware(metas: dict, df: pd.DataFrame) -> str:
    seen = {}
    for m in metas.values():
        seen[m["backend"]] = m  # latest wins (dict order = sorted filenames)
    rows = []
    for b, m in seen.items():
        rows.append({"backend": b, "available": m["backend_available"], "device": m["device_name"] if m["backend_available"] else f"– ({m['backend_unavailable_reason']})",
                     "CPU": m["cpu"], "GPU": m.get("gpu"), "RAM (GB)": m["ram_gb"], "power": m.get("power_source")})
    return md_table(pd.DataFrame(rows))


def md_environment(metas: dict) -> str:
    rows, seen = [], set()
    for m in metas.values():
        key = (m["experiment"], m["backend"], m["run_id"])
        if key in seen:
            continue
        seen.add(key)
        rows.append({"experiment": m["experiment"], "backend": m["backend"], "run_id": m["run_id"], "timestamp": m["timestamp"], "host": m["hostname"],
                     "OS": m["os"], "python": m["python"], "torch": m["torch"], "torchvision": m["torchvision"], "seed": m["seed"],
                     "torch_threads": m["torch_num_threads"], "git": m.get("git_commit") or "none", "code_sha256": m["code_sha256"]})
    return md_table(pd.DataFrame(rows))


METHODOLOGY = """\
* **Same workload, same seed.** Inputs and initial weights are generated on the CPU from a fixed seed and copied to the device once, so every backend starts from identical values. GPU kernels are not bit-deterministic, so training trajectories can drift slightly between backends.
* **Synchronised timing.** Each timed iteration is `sync → start → work → sync → stop` (`torch.cuda.synchronize()`, `torch.mps.synchronize()`, nothing on CPU). Warmup iterations are recorded but excluded from statistics. Every individual measurement is stored in `results/raw/`.
* **Statistics.** Median is the headline number; mean, sample standard deviation, coefficient of variation (std/mean), min and max are reported alongside it.
* **No silent fallback.** Each backend is selected explicitly. `PYTORCH_ENABLE_MPS_FALLBACK` is removed so operators unsupported on MPS raise instead of quietly running on the CPU; those are recorded as `unsupported`. Unavailable backends are recorded as `unavailable`.
* **Failures are data.** OOM, unsupported dtypes/operators and other errors are stored with `status`, `error_type` and `error` and the run continues.
* **Eager mode, default kernels.** No `torch.compile`, no fused optimizers, strict FP32 (TF32 disabled) unless the config says otherwise.
* **Baseline.** Speedups are always `cpu_time / backend_time` for the identical configuration on the same machine. There is no aggregate score.
"""

METRIC_NOTES = """\
* `latency_ms` – wall time of one call. `throughput` – work per second; its unit is in `throughput_unit` (GFLOPS, images/s, tokens/s, env-steps/s).
* **CUDA memory** – `torch.cuda.max_memory_allocated` (exact peak of tensor allocations; reserved memory is stored in `extra`). Dedicated VRAM.
* **MPS memory** – PyTorch has no peak counter. `driver_allocated_memory` is what the Metal driver holds for the process (including cache), taken as a snapshot or, where marked, a sampled maximum (lower bound of the true peak). Unified memory is shared with the CPU.
* **CPU memory** – process RSS (snapshot, sampled maximum, or lifetime peak as labelled in `memory_kind`); includes the Python/torch baseline.
* **These memory numbers are NOT equivalent across backends.** Compare trends within a backend, not absolute values across backends.
"""


def md_failures(df: pd.DataFrame) -> str:
    f = df[df["status"].isin(FAIL_STATUSES)]
    if f.empty:
        return "No failed, unsupported or unavailable configurations were recorded.\n"
    cfg_cols = [c for c in ("model", "batch_size", "sequence_length", "precision", "x_matrix_size", "x_workload", "x_hidden") if c in f.columns]
    g = f.groupby(["experiment", "backend", "status"] + cfg_cols + ["error_type", "error"], dropna=False).size().reset_index(name="rows")
    g["error"] = g["error"].astype(str).str.slice(0, 200)
    return md_table(g.dropna(axis=1, how="all"))


def md_mps_limitations(df: pd.DataFrame) -> str:
    lines = [
        "* PyTorch exposes no peak-memory counter for MPS; memory values are driver allocation snapshots/sampled maxima (see metric notes).",
        "* CPU fallback for unsupported MPS operators is disabled on purpose, so an unsupported operator shows up as a recorded failure instead of a hidden slowdown.",
        "* GPU utilisation, temperature and power on Apple Silicon need `sudo powermetrics`, which this suite does not require; those fields are stored as null.",
    ]
    m = df[(df["backend"] == "mps") & df["status"].isin(["unsupported", "oom", "error", "timeout"])]
    if m.empty:
        lines.append("* No MPS-specific failures were recorded in this data set.")
    else:
        lines.append("\nRecorded MPS failures:\n")
        lines.append(md_table(m.groupby(["experiment", "status", "error_type", "error"], dropna=False).size().reset_index(name="rows").assign(error=lambda x: x["error"].astype(str).str.slice(0, 200))))
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--all-runs", action="store_true", help="use every run instead of only the latest per (experiment, backend)")
    args = ap.parse_args()
    results_dir = resolve_path(args.results_dir)

    df, metas = load_results(results_dir / "raw")
    if df.empty:
        print(f"No raw results in {results_dir / 'raw'}")
        return 1
    issues = validate(df)
    print("validation:", "clean" if not issues else issues)
    if not args.all_runs:
        df = latest_runs(df)
        metas = {k: v for k, v in metas.items() if k.split("|")[2] in set(df["run_id"])}

    rep = Report(results_dir)
    rep.add("# ML hardware benchmark report\n")
    rep.add("_All numbers below come from the raw measurements in `results/raw/`. This report lists observations only; interpretation is left to the reader._\n")
    rep.add("## Hardware\n")
    rep.add(md_hardware(metas, df))
    rep.add("## Environment\n")
    rep.add(md_environment(metas))
    rep.add("**Data validation:** " + ("no problems found." if not issues else "\n" + "\n".join(f"- {i}" for i in issues)) + "\n")
    rep.add("## Methodology\n")
    rep.add(METHODOLOGY)
    rep.add("### Metric definitions and memory-metric differences\n")
    rep.add(METRIC_NOTES)
    rep.add("# Experiment results\n")
    for name, fn in SECTIONS:
        if (df["experiment"] == name).any():
            fn(df, rep)
    rep.add("## Failed, unsupported and unavailable configurations\n")
    rep.add(md_failures(df))
    rep.add("## MPS limitations encountered\n")
    rep.add(md_mps_limitations(df))
    rep.add("## Observations\n")
    rep.add("Factual statements generated from the data above; no overall ranking is implied.\n")
    rep.add("\n".join(f"* {o}" for o in rep.observations) if rep.observations else "_none_")
    out = results_dir / "report.md"
    out.write_text("\n".join(rep.parts))
    print(f"report -> {out}\nplots  -> {rep.plots_dir}\ntables -> {rep.processed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
