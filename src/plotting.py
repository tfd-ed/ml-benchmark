"""Matplotlib plots. Colours come from matplotlib's own property cycle (nothing hard-coded);
each backend always gets the same cycle slot so colour = backend in every figure.

Rules followed: zero-based axes for magnitudes (no truncated axes), one y-axis per chart
(no dual axes), individual measurements shown where they exist, every figure carries a
caption stating what is plotted and what the baseline is, PNG + SVG output.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BACKEND_ORDER = ("cpu", "mps", "cuda")
BASELINE_NOTE = "Baseline for any speedup is the CPU backend on the same machine (speedup = cpu_time / backend_time)."

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.labelsize": 10, "legend.frameon": False, "lines.linewidth": 1.8,
    "lines.markersize": 6,
})


def backend_color(backend: str) -> str:
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    idx = BACKEND_ORDER.index(backend) if backend in BACKEND_ORDER else len(BACKEND_ORDER)
    return cycle[idx % len(cycle)]


def ordered_backends(df: pd.DataFrame) -> list[str]:
    present = set(df["backend"].unique())
    return [b for b in BACKEND_ORDER if b in present] + sorted(present - set(BACKEND_ORDER))


def backend_label(df: pd.DataFrame, backend: str) -> str:
    names = df.loc[df["backend"] == backend, "device"].dropna().unique()
    return f"{backend} ({names[0]})" if len(names) else backend


def finish(fig, name: str, title: str, caption: str, out_dir: Path, has_legend: bool = True) -> list[Path]:
    """Add title/caption and save PNG + SVG. Returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    wrapped = "\n".join(textwrap.fill(p, 120) for p in caption.split("\n"))
    fig.text(0.01, 0.005, wrapped, ha="left", va="bottom", fontsize=8, color=plt.rcParams["text.color"], alpha=0.75)
    n_lines = wrapped.count("\n") + 1
    fig.tight_layout(rect=(0, 0.035 * n_lines + 0.02, 1, 0.94))
    paths = []
    for ext in ("png", "svg"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def empty_note(name: str, title: str, message: str, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    return finish(fig, name, title, "No successful measurements were available for this plot.", out_dir)


# --------------------------------------------------------------------------- 1. matmul


def plot_matmul_throughput(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    name, title = "01_matmul_throughput_vs_size", "Matrix multiplication throughput vs. matrix size"
    d = df[(df["experiment"] == "matmul") & (df["status"] == "ok") & (df["phase"] == "measure")].copy()
    if d.empty:
        return empty_note(name, title, "No matmul measurements", out_dir)
    d["n"] = d["x_matrix_size"].astype(int)
    backends = ordered_backends(d)
    sizes = sorted(d["n"].unique())
    dtype = ", ".join(sorted(d["precision"].dropna().unique()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, log in zip(axes, (False, True)):
        for k, b in enumerate(backends):
            g = d[d["backend"] == b]
            color = backend_color(b)
            offset = 1 + 0.02 * (k - (len(backends) - 1) / 2)  # tiny horizontal offset so dots don't overlap
            ax.scatter(g["n"] * offset, g["throughput"], s=10, alpha=0.35, color=color, linewidths=0)
            med = g.groupby("n")["throughput"].median()
            ax.plot(med.index, med.values, marker="o", color=color, label=backend_label(d, b))
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes, [str(s) for s in sizes])
        ax.minorticks_off()
        ax.set_xlabel("Matrix size n  (n × n × n matmul)")
        ax.set_ylabel("Throughput (GFLOPS)" + (", log scale" if log else ""))
        if log:
            ax.set_yscale("log")
            plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
            ax.yaxis.set_major_formatter(plain)
            ax.yaxis.set_minor_formatter(plain)
            ax.set_title("Log y-axis (to see all backends)", fontsize=10)
        else:
            ax.set_ylim(bottom=0)
            ax.set_title("Linear y-axis, starts at zero", fontsize=10)
    axes[0].legend(loc="best")
    caption = (f"Line = median of the measured iterations; faint dots = every individual measured iteration (warmup excluded). "
               f"Precision: {dtype}. GFLOPS = 2·n³ / latency, synchronised timing. Both panels show the same data. {BASELINE_NOTE}{avail_note(df)}")
    return finish(fig, name, title, caption, out_dir)


# --------------------------------------------------------------------------- shared helpers


def xcol(df: pd.DataFrame, name: str) -> pd.Series:
    """`x_<name>` column or an all-NaN series when the experiment never wrote it."""
    return df[f"x_{name}"] if f"x_{name}" in df.columns else pd.Series(np.nan, index=df.index)


def avail_note(df: pd.DataFrame) -> str:
    un = sorted(df.loc[df["status"] == "unavailable", "backend"].unique())
    return f" Backends unavailable on this machine (not plotted): {', '.join(un)}." if un else ""


def _measured(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    return df[(df["experiment"] == experiment) & (df["status"] == "ok") & (df["phase"] == "measure")].copy()


def _failures(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    return df[(df["experiment"] == experiment) & df["status"].isin(["oom", "unsupported", "error", "timeout"])].copy()


def grouped_bars(ax, groups: list, series: list[str], samples: dict, label_of, missing: dict | None = None, show_dots: bool = True, legend: bool = True) -> None:
    """Bars = median of `samples[(series, group)]`; dots = every individual sample.
    `missing[(series, group)]` holds a status string drawn as a marker where no bar exists."""
    width = 0.8 / max(len(series), 1)
    missing = missing or {}
    for k, s in enumerate(series):
        color = backend_color(s)
        for gi, g in enumerate(groups):
            x = gi + (k - (len(series) - 1) / 2) * width
            vals = samples.get((s, g))
            if vals:
                ax.bar(x, float(np.median(vals)), width * 0.9, color=color, alpha=0.85, label=label_of(s) if gi == 0 or (s, groups[0]) not in samples else None)
                if show_dots and len(vals) > 1:
                    ax.scatter(np.full(len(vals), x) + np.linspace(-width * 0.25, width * 0.25, len(vals)), vals, s=8, color=plt.rcParams["text.color"], alpha=0.45, linewidths=0, zorder=3)
            elif (s, g) in missing:
                ax.annotate(f"✕\n{missing[(s, g)]}", (x, 0), ha="center", va="bottom", fontsize=7, color=color, xytext=(0, 3), textcoords="offset points")
    ax.set_xticks(range(len(groups)), [str(g) for g in groups])
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.22)  # headroom so the legend never covers a bar
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    if uniq and legend:
        ax.legend(uniq.values(), uniq.keys(), loc="upper left", ncol=min(len(uniq), 2), fontsize=8)


def _samples(d: pd.DataFrame, series_col: str, group_col, value_col) -> tuple[dict, list]:
    out: dict = {}
    for (s, g), grp in d.groupby([series_col, group_col]):
        out[(s, g)] = [float(v) for v in grp[value_col].dropna()]
    return out, sorted({g for _, g in out})


def _missing(fail: pd.DataFrame, group_col) -> dict:
    return {(r["backend"], r[group_col]): r["status"] for _, r in fail.drop_duplicates(["backend", group_col]).iterrows()}


# --------------------------------------------------------------------------- 2/3. CNN


def _cnn_bars(df: pd.DataFrame, out_dir: Path, name: str, title: str, value: str, ylabel: str, note: str) -> list[Path]:
    d = _measured(df, "cnn")
    d = d[xcol(d, "granularity") == "epoch"]
    if d.empty:
        return empty_note(name, title, "No successful CNN epochs", out_dir)
    d["v"] = d["throughput"] if value == "throughput" else xcol(d, "epoch_time_s")
    backends = ordered_backends(d)
    samples, groups = _samples(d, "backend", "batch_size", "v")
    fail = _failures(df, "cnn")
    groups = sorted(set(groups) | set(fail["batch_size"].dropna().astype(int)))
    groups = [int(g) for g in groups]
    samples = {(b, int(g)): v for (b, g), v in samples.items()}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    grouped_bars(ax, groups, backends, samples, lambda b: backend_label(d, b), _missing(fail.assign(batch_size=fail["batch_size"].astype("Int64")), "batch_size"))
    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)
    n_tr = int(xcol(d, "train_samples").dropna().iloc[0]) if xcol(d, "train_samples").notna().any() else "?"
    cap = (f"ResNet-18 (CIFAR variant), CIFAR-10 subset of {n_tr} training images, identical seed/initial weights/augmentation/optimizer on every backend. "
           f"Bar = median over epochs, dots = individual epochs. {note}{avail_note(df)}")
    return finish(fig, name, title, cap, out_dir)


def plot_cnn_images_per_sec(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    return _cnn_bars(df, out_dir, "02_cnn_images_per_sec", "CNN training throughput (images/sec, higher = more work per second)", "throughput",
                     "Training throughput (images / second)", "Throughput = images / summed step time (host→device copy + forward/backward + optimizer, synchronised).")


def plot_cnn_epoch_time(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    return _cnn_bars(df, out_dir, "03_cnn_epoch_time", "CNN training time per epoch (seconds, lower = less time)", "epoch",
                     "Epoch training time (s)", "Epoch time = sum of training-step times; augmentation and validation are excluded (stored separately in the raw data).")


# --------------------------------------------------------------------------- 4/5. transformer


def plot_transformer_tokens(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    name, title = "04_transformer_tokens_per_sec", "Transformer training throughput by configuration (tokens/sec)"
    d = _measured(df, "transformer")
    if d.empty:
        return empty_note(name, title, "No transformer measurements", out_dir)
    d["cfg"] = list(zip(d["batch_size"].astype(int), d["sequence_length"].astype(int)))
    samples, groups = _samples(d, "backend", "cfg", "throughput")
    groups = sorted(groups, key=lambda c: (c[0] * c[1], c[1]))
    backends = ordered_backends(d)
    fail = _failures(df, "transformer")
    fail = fail.assign(cfg=list(zip(fail["batch_size"].astype(int), fail["sequence_length"].astype(int)))) if len(fail) else fail.assign(cfg=[])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    grouped_bars(ax, groups, backends, samples, lambda b: backend_label(d, b), _missing(fail, "cfg") if len(fail) else None)
    ax.set_xticks(range(len(groups)), [f"B={b}\nT={t}" for b, t in groups])
    ax.set_xlabel("Configuration: batch size B, sequence length T (sorted by tokens per step)")
    ax.set_ylabel("Training throughput (tokens / second)")
    n = int(xcol(d, "n_params").dropna().iloc[0])
    cap = (f"GPT-style decoder, {n / 1e6:.1f}M parameters, FP32, AdamW, synthetic Markov token data; bar = median over measured steps, dots = individual steps "
           f"(warmup excluded). Each step is synchronised and includes the host→device token copy.{avail_note(df)}")
    return finish(fig, name, title, cap, out_dir)


def plot_transformer_vs_batch(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    name, title = "05_transformer_throughput_vs_batch_size", "Transformer throughput vs. batch size, one panel per sequence length"
    d = _measured(df, "transformer")
    if d.empty:
        return empty_note(name, title, "No transformer measurements", out_dir)
    seqs = sorted(d["sequence_length"].dropna().astype(int).unique())
    bss = sorted(d["batch_size"].dropna().astype(int).unique())
    backends = ordered_backends(d)
    fig, axes = plt.subplots(1, len(seqs), figsize=(4.2 * len(seqs), 4.6), sharey=True, squeeze=False)
    for ax, seq in zip(axes[0], seqs):
        for b in backends:
            g = d[(d["backend"] == b) & (d["sequence_length"] == seq)]
            ax.scatter(g["batch_size"], g["throughput"], s=8, alpha=0.3, color=backend_color(b), linewidths=0)
            med = g.groupby("batch_size")["throughput"].median()
            ax.plot(med.index, med.values, marker="o", color=backend_color(b), label=backend_label(d, b))
        ax.set_xscale("log", base=2)
        ax.set_xticks(bss, [str(x) for x in bss])
        ax.minorticks_off()
        ax.set_title(f"sequence length {seq}", fontsize=10)
        ax.set_xlabel("Batch size")
        ax.set_ylim(bottom=0)
    axes[0][0].set_ylabel("Training throughput (tokens / second)")
    axes[0][0].legend(loc="best")
    cap = f"Line = median over measured steps, faint dots = individual steps. All panels share one zero-based y-axis.{avail_note(df)}"
    return finish(fig, name, title, cap, out_dir)


# --------------------------------------------------------------------------- 6/7. memory


def _mem_ok(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["experiment"] == "memory") & (df["status"] == "ok")].copy()
    d["ladder"] = xcol(d, "ladder")
    return d


def plot_memory_vs_batch(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    name, title = "06_memory_vs_batch_size", "Training memory vs. batch size (backend-specific memory metrics)"
    d = _mem_ok(df)
    d = d[d["ladder"] == "batch_size"]
    if d.empty:
        return empty_note(name, title, "No memory-scaling measurements", out_dir)
    fail = df[(df["experiment"] == "memory") & df["status"].isin(["oom", "timeout", "error", "unsupported"]) & (xcol(df, "ladder") == "batch_size")]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for b in ordered_backends(d):
        g = d[d["backend"] == b].sort_values("batch_size")
        kind = str(g["memory_kind"].iloc[0]).split(" (")[0]
        ax.plot(g["batch_size"], g["memory_mb"] / 1024, marker="o", color=backend_color(b), label=f"{b}: {kind}")
        if b == "mps" and xcol(g, "mps_current_allocated_sampled_peak_mb").notna().any():
            ax.plot(g["batch_size"], xcol(g, "mps_current_allocated_sampled_peak_mb") / 1024, marker="s", ms=4, ls="--", color=backend_color(b),
                    label="mps: current_allocated_memory (sampled max, tensor bytes only)")
        f = fail[fail["backend"] == b]
        for _, r in f.iterrows():
            ax.annotate(f"✕ {r['status']}\nbs={int(r['batch_size'])}", (r["batch_size"], 0), ha="center", va="bottom", fontsize=7, color=backend_color(b), xytext=(0, 4), textcoords="offset points")
        cap_mb = g["x_capacity_mb"].dropna()
        if len(cap_mb):
            ax.axhline(cap_mb.iloc[0] / 1024, color=backend_color(b), ls=":", lw=1, alpha=0.6)
            ax.annotate(f"{b} nominal capacity: {g['x_capacity_kind'].iloc[0]}", (g["batch_size"].max(), cap_mb.iloc[0] / 1024), fontsize=7, va="bottom", ha="right", color=backend_color(b))
    bss = sorted(d["batch_size"].astype(int).unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(bss, [str(x) for x in bss])
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Batch size (sequence length and model fixed at the base configuration)")
    ax.set_ylabel("Memory (GiB)")
    ax.legend(loc="center left", fontsize=8)
    cap = ("WARNING: the three memory metrics are NOT equivalent - CUDA: exact peak tensor allocations in dedicated VRAM; MPS: driver-allocated memory in unified memory "
           "(sampled maximum, includes allocator cache and a fixed ~1 GiB floor); CPU: process RSS lifetime peak in system RAM (includes the Python/torch baseline). "
           "Compare shapes within one backend, not absolute levels across backends. ✕ = configuration failed or was stopped by a suite safety limit." + avail_note(df))
    return finish(fig, name, title, cap, out_dir)


def max_successful(df: pd.DataFrame) -> pd.DataFrame:
    """Largest ok configuration per (backend, ladder) and how the ladder ended."""
    m = df[df["experiment"] == "memory"].copy()
    m["ladder"] = xcol(m, "ladder")
    rows = []
    for (b, lad), g in m[m["status"] != "unavailable"].groupby(["backend", "ladder"]):
        ok = g[g["status"] == "ok"]
        bad = g[g["status"].isin(["oom", "timeout", "error", "unsupported"])]
        if ok.empty and bad.empty:
            continue
        if lad == "batch_size":
            key = "batch_size"
        elif lad == "sequence_length":
            key = "sequence_length"
        else:
            key = "x_n_params"
        best = ok.sort_values(key).iloc[-1] if len(ok) else None
        stop = bad.iloc[0] if len(bad) else None
        rows.append({"backend": b, "ladder": lad, "largest_ok_value": None if best is None else best[key], "largest_ok_model": None if best is None else best["model"],
                     "largest_ok_batch": None if best is None else best["batch_size"], "largest_ok_seq": None if best is None else best["sequence_length"],
                     "largest_ok_params_M": None if best is None else (best["x_n_params"] / 1e6 if "x_n_params" in best and pd.notna(best.get("x_n_params")) else None),
                     "memory_mb_at_largest_ok": None if best is None else best["memory_mb"], "memory_kind": None if best is None else best["memory_kind"],
                     "ladder_ended_by": "ladder ceiling reached (no failure)" if stop is None else f"{stop['status']} ({stop['error_type']}) at {stop['model']} bs={int(stop['batch_size'])} seq={int(stop['sequence_length'])}",
                     "ended_by_safety_cap": bool(stop is not None and str(stop["error_type"]).startswith("SafetyCap"))})
    return pd.DataFrame(rows)


def plot_max_config(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    name, title = "07_max_successful_configuration", "Largest configuration that ran successfully (one dimension varied at a time)"
    t = max_successful(df)
    if t.empty:
        return empty_note(name, title, "No memory-scaling data", out_dir)
    panels = [("batch_size", "largest_ok_value", "Batch size"), ("sequence_length", "largest_ok_value", "Sequence length (tokens)"), ("model", "largest_ok_params_M", "Model size (M parameters)")]
    backends = ordered_backends(t)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (lad, col, label) in zip(axes, panels):
        g = t[t["ladder"] == lad]
        for k, b in enumerate(backends):
            r = g[g["backend"] == b]
            if r.empty or pd.isna(r.iloc[0][col]):
                continue
            r = r.iloc[0]
            ax.barh(k, r[col], height=0.5, color=backend_color(b), alpha=0.85)
            note = ("ceiling reached" if r["ladder_ended_by"].startswith("ladder ceiling") else ("stopped by safety cap" if r["ended_by_safety_cap"] else r["ladder_ended_by"].split(" (")[0]))
            model_txt = f" [{r['largest_ok_model']}]" if lad == "model" else ""
            ax.annotate(f" {r[col]:,.4g}{model_txt}\n {note}", (r[col], k), va="center", fontsize=7)
        ax.set_yticks(range(len(backends)), [backend_label(df, b) for b in backends], fontsize=8)
        ax.set_xlabel(label)
        ax.set_xlim(left=0, right=ax.get_xlim()[1] * 1.35)
        ax.invert_yaxis()
    cap = ("Each ladder increases one dimension (others at the base configuration) until the first failure. 'ceiling reached' = the ladder's largest tested value "
           "succeeded, so the true maximum is at least this large. 'stopped by safety cap' = the suite's memory/time limit ended the ladder, not a native OOM. "
           "A larger fitting configuration says nothing about speed." + avail_note(df))
    return finish(fig, name, title, cap, out_dir)


# --------------------------------------------------------------------------- 8. precision


def plot_precision(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    name, title = "08_precision_fp32_fp16_bf16", "Throughput by numeric precision"
    d = _measured(df, "precision")
    if d.empty:
        return empty_note(name, title, "No precision measurements", out_dir)
    d["workload"] = xcol(d, "workload")
    fail = _failures(df, "precision")
    fail = fail.assign(workload=xcol(fail, "workload"))
    order = ["fp32", "fp16", "bf16"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (wl, label) in zip(axes, [("matmul_pure_dtype", "Matmul 2048×2048, pure dtype (GFLOPS)"), ("training_autocast", "GPT training step, autocast (tokens / second)")]):
        g = d[d["workload"] == wl]
        samples, _ = _samples(g, "backend", "precision", "throughput")
        f = fail[fail["workload"] == wl]
        backends = ordered_backends(pd.concat([g, f])) if len(g) or len(f) else []
        grouped_bars(ax, order, backends, samples, lambda b: backend_label(d, b), _missing(f, "precision") if len(f) else None, legend=ax is axes[0])
        ax.set_ylabel(label)
        ax.set_xlabel("Precision")
    cap = ("Bar = median over measured iterations, dots = individual iterations. Matmul casts the inputs to the dtype; training uses torch.autocast with FP32 master weights "
           "(FP16 also uses a GradScaler). ✕ = precision not supported / failed on that backend. A precision that runs is not necessarily fast (e.g. software-emulated half precision)." + avail_note(df))
    return finish(fig, name, title, cap, out_dir)


# --------------------------------------------------------------------------- 9. PPO


def plot_ppo(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    name, title = "09_ppo_throughput", "PPO (CartPole): simulator, end-to-end and training throughput"
    d = _measured(df, "rl")
    if d.empty:
        return empty_note(name, title, "No PPO measurements", out_dir)
    panels = [("env_steps_per_sec_env_only", "Environment steps / s\n(simulator only, always CPU)"),
              ("env_steps_per_sec_end2end", "Environment steps / s\n(end-to-end incl. inference + update)"),
              ("train_steps_per_sec", "Gradient (minibatch) steps / s\n(update phase only)")]
    backends = ordered_backends(d)
    groups = sorted(d["x_hidden"].dropna().astype(int).unique())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, (col, label) in zip(axes, panels):
        d["v"] = d[f"x_{col}"]
        samples = {(b, int(h)): list(g["v"].dropna()) for (b, h), g in d.groupby(["backend", "x_hidden"])}
        grouped_bars(ax, [int(g) for g in groups], backends, samples, lambda b: backend_label(d, b), legend=ax is axes[0])
        ax.set_xlabel("Policy/value network hidden width")
        ax.set_ylabel(label)
    cap = ("Bar = median over PPO iterations, dots = individual iterations. Environment simulation always runs on the CPU, so it is independent of the accelerator; "
           "'end-to-end' adds the policy inference round trip (host→device→host every environment step) and the PPO update." + avail_note(df))
    return finish(fig, name, title, cap, out_dir)


def plot_ppo_breakdown(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    name, title = "09b_ppo_time_breakdown", "PPO: where each iteration's time goes"
    d = _measured(df, "rl")
    if d.empty:
        return empty_note(name, title, "No PPO measurements", out_dir)
    comps = [("env_s", "environment step (CPU)"), ("infer_s", "policy inference"), ("gae_s", "advantage estimation (CPU)"), ("update_s", "PPO update")]
    shades = plt.cm.Greys(np.linspace(0.25, 0.85, len(comps)))
    hatches = ["", "///", "..", "xx"]
    rows = sorted({(b, int(h)) for b, h in zip(d["backend"], d["x_hidden"])}, key=lambda r: (r[1], BACKEND_ORDER.index(r[0]) if r[0] in BACKEND_ORDER else 9))
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(rows) + 2.8))
    for i, (b, h) in enumerate(rows):
        g = d[(d["backend"] == b) & (d["x_hidden"] == h)]
        left = 0.0
        for (col, lab), shade, hatch in zip(comps, shades, hatches):
            v = float(g[f"x_{col}"].mean())
            ax.barh(i, v, left=left, color=shade, hatch=hatch, edgecolor=backend_color(b), linewidth=0.8, label=lab if i == 0 else None)
            left += v
        ax.annotate(f" {left:.3f} s", (left, i), va="center", fontsize=8)
    ax.set_yticks(range(len(rows)), [f"{b}, hidden {h}" for b, h in rows])
    for tick, (b, _) in zip(ax.get_yticklabels(), rows):
        tick.set_color(backend_color(b))
    ax.invert_yaxis()
    ax.set_xlim(left=0, right=ax.get_xlim()[1] * 1.12)
    ax.set_xlabel("Mean wall time per PPO iteration (s); iteration = 8 envs × 128 steps + 16 gradient steps")
    ax.legend(loc="lower right", fontsize=8)
    cap = "Mean over measured iterations; segment style = phase, bar-edge colour = backend. Every phase is synchronised." + avail_note(df)
    return finish(fig, name, title, cap, out_dir)


# --------------------------------------------------------------------------- 10. sustained


def plot_sustained(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    name, title = "10_sustained_throughput_over_time", "Sustained training throughput over time"
    d = _measured(df, "sustained")
    if d.empty:
        return empty_note(name, title, "No sustained-run measurements", out_dir)
    summ = df[(df["experiment"] == "sustained") & (df["phase"] == "config") & (df["status"] == "ok")]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for b in ordered_backends(d):
        g = d[d["backend"] == b].sort_values("x_elapsed_s")
        lab = backend_label(d, b)
        s = summ[summ["backend"] == b]
        if len(s) and pd.notna(xcol(s, "degradation_degradation_pct").iloc[0]):
            lab += f" – first→last 10% windows: {float(xcol(s, 'degradation_degradation_pct').iloc[0]):+.1f}% change (positive = slower)"
        ax.plot(g["x_elapsed_s"] / 60.0, g["throughput"], marker="o", ms=3, color=backend_color(b), label=lab)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_xlabel("Elapsed time (minutes)")
    ax.set_ylabel("Throughput (tokens / s)")
    ax.legend(loc="lower right", fontsize=8)
    minutes = float(d["x_elapsed_s"].max()) / 60
    cap = (f"One point per logging window; run length {minutes:.1f} min. FP32 GPT training step repeated back-to-back. "
           f"Degradation compares the median window throughput of the first vs. last 10% of windows.{avail_note(df)}")
    return finish(fig, name, title, cap, out_dir)
