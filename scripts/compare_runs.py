"""Compare runs made on different machines / backends.

    python scripts/compare_runs.py rtx3060-20260919 m1pro-20260920
    python scripts/compare_runs.py rtx3060-20260919 m1pro-20260920 --baseline "cpu · Apple M1 Pro"
    python scripts/compare_runs.py results/a results/b --out results/comparison

Every argument is a run folder (results/<run_id>/, or just <run_id> below --results-dir). Each available
(machine, backend) pair in those folders becomes one *series*, labelled "<backend> · <device>" (e.g.
"cuda · NVIDIA GeForce RTX 3060"); a host name is appended only when two machines share a label.
Writes <out>/comparison.md, <out>/processed/*.csv and <out>/plots/ using the same plots as the
per-run report. Speedup = median throughput / median throughput of the baseline series (default: the
first run given). The script warns when runs differ in code hash, seed or experiment configuration,
because such numbers are not directly comparable. It reports measurements only; it ranks nothing.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import plotting  # noqa: E402
from src.config import resolve_path  # noqa: E402
from src.reporting import latest_runs, load_results, md_table  # noqa: E402

MUST_MATCH = ("seed", "code_sha256")
INFO_ONLY = ("torch", "python", "os")


def short_device(name: str) -> str:
    """'Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz' -> 'Intel Core i7-8700K @ 3.70GHz' (keeps series labels readable)."""
    return re.sub(r"\s+", " ", re.sub(r"\((R|TM)\)|\bCPU\b", "", name)).strip()


def resolve_run(arg: str, root: Path) -> Path:
    p = Path(arg)
    for cand in (p, root / arg):
        if (cand / "raw").is_dir():
            return cand.resolve()
    raise SystemExit(f"no run folder '{arg}' (looked for {p}/raw and {root / arg}/raw)")


def load_series(run_dirs: list[Path]) -> tuple[pd.DataFrame, list[dict], list[str]]:
    """One series per available (run folder, backend). Returns (rows with a `series` column, series info, notes)."""
    found, notes = [], []
    for rd in run_dirs:
        df, metas = load_results(rd / "raw")
        df = latest_runs(df)
        for backend in plotting.ordered_backends(df):
            rows = df[df["backend"] == backend]
            if not (rows["status"] == "ok").any():
                notes.append(f"`{rd.name}`: backend **{backend}** has no successful measurements (not compared).")
                continue
            device = rows["device"].dropna().mode()
            m = {k.split("|")[0]: v for k, v in metas.items() if k.split("|")[1] == backend and k.split("|")[2] in set(rows["run_id"])}
            host = next((v.get("hostname") for v in m.values()), None)
            found.append({"run": rd.name, "backend": backend, "device": device.iloc[0] if len(device) else "?", "host": host, "metas": m, "rows": rows})
    labels = [f"{s['backend']} · {short_device(s['device'])}" for s in found]
    for i, s in enumerate(found):  # disambiguate identical labels: host first, then run folder
        if labels.count(labels[i]) > 1:
            s["label"] = f"{labels[i]} @ {s['host'] or s['run']}"
        else:
            s["label"] = labels[i]
    taken = [s["label"] for s in found]
    for s in found:
        if taken.count(s["label"]) > 1:
            s["label"] = f"{s['label']} [{s['run']}]"
    if not found:
        raise SystemExit("no successful measurements in the given run folders")
    frames = [s["rows"].assign(series=s["label"]) for s in found]
    return pd.concat(frames, ignore_index=True), found, notes


def md_series(series: list[dict]) -> str:
    rows = []
    for s in series:
        m = next(iter(s["metas"].values()), {})
        rows.append({"series": s["label"], "run folder": s["run"], "host": s["host"], "CPU": m.get("cpu"), "GPU": m.get("gpu"),
                     "RAM (GB)": m.get("ram_gb"), "OS": m.get("os"), "torch": m.get("torch"), "python": m.get("python")})
    return md_table(pd.DataFrame(rows))


def comparability(series: list[dict]) -> tuple[list[str], list[str]]:
    """(problems, info): problems make numbers non-comparable; info is stack/version context."""
    problems, info = [], []
    experiments = sorted({e for s in series for e in s["metas"]})
    for exp in experiments:
        have = [(s["label"], s["metas"][exp]) for s in series if exp in s["metas"]]
        if len(have) < 2:
            continue
        ref_label, ref = have[0]
        for label, m in have[1:]:
            diffs = []
            a, b = ref.get("config", {}).get(exp, {}), m.get("config", {}).get(exp, {})
            keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            if keys:
                diffs.append("configuration differs in " + ", ".join(f"`{k}` ({a.get(k)} vs {b.get(k)})" for k in keys))
            for f in MUST_MATCH:
                if ref.get(f) != m.get(f):
                    diffs.append(f"`{f}` differs ({ref.get(f)} vs {m.get(f)})")
            if diffs:
                problems.append(f"**{exp}**: {label} vs {ref_label}: " + "; ".join(diffs))
            same_stack = [f for f in INFO_ONLY if ref.get(f) != m.get(f)]
            if same_stack:
                info.append(f"{exp}: {label} vs {ref_label}: " + ", ".join(f"{f} {m.get(f)} vs {ref.get(f)}" for f in same_stack))
    return problems, sorted(set(info))


# --------------------------------------------------------------------------- tables


def wide(d: pd.DataFrame, keys: list[str], value: str, labels: list[str]) -> pd.DataFrame:
    """Median of `value` per config (rows) and series (columns), columns in the order the runs were given."""
    w = d.groupby([*keys, "series"])[value].median().unstack("series")
    return w.reindex(columns=[c for c in labels if c in w.columns]).reset_index()


def speedup_table(w: pd.DataFrame, keys: list[str], baseline: str) -> pd.DataFrame:
    vals = [c for c in w.columns if c not in keys]
    out = w[keys].copy()
    for c in vals:
        if c != baseline:  # the baseline's own column would be all 1.00
            out[c] = w[c] / w[baseline] if baseline in w.columns else None
    return out


def build_tables(df: pd.DataFrame, labels: list[str]) -> dict[str, tuple[str, list[str], pd.DataFrame, str]]:
    """experiment -> (heading, config keys, wide value table, unit/description)."""
    ok = df[(df["status"] == "ok") & (df["phase"] == "measure")]
    out = {}
    d = ok[ok["experiment"] == "matmul"].assign(matrix_size=lambda x: x["x_matrix_size"].astype("Int64"))
    if len(d):
        out["matmul"] = ("Matrix multiplication", ["precision", "matrix_size"], wide(d, ["precision", "matrix_size"], "throughput", labels), "median GFLOPS (2n³ / latency)")
    d = ok[(ok["experiment"] == "cnn") & (ok.get("x_granularity") == "epoch")].assign(batch_size=lambda x: x["batch_size"].astype("Int64"))
    if len(d):
        out["cnn"] = ("CNN training (ResNet-18 / Fashion-MNIST)", ["batch_size"], wide(d, ["batch_size"], "throughput", labels), "median images/s per epoch")
    d = ok[ok["experiment"] == "transformer"].assign(batch_size=lambda x: x["batch_size"].astype("Int64"), sequence_length=lambda x: x["sequence_length"].astype("Int64"))
    if len(d):
        out["transformer"] = ("Transformer training", ["batch_size", "sequence_length"], wide(d, ["batch_size", "sequence_length"], "throughput", labels), "median tokens/s")
    d = ok[ok["experiment"] == "precision"].assign(workload=lambda x: x["x_workload"])
    if len(d):
        out["precision"] = ("Precision", ["workload", "precision"], wide(d, ["workload", "precision"], "throughput", labels), "median throughput (GFLOPS for matmul_pure_dtype, tokens/s for training_autocast)")
    d = ok[ok["experiment"] == "rl"].assign(hidden=lambda x: x["x_hidden"].astype("Int64"))
    if len(d):
        d = d.assign(sps=d["x_env_steps_per_sec_end2end"])
        out["rl"] = ("Reinforcement learning (PPO / CartPole)", ["hidden"], wide(d, ["hidden"], "sps", labels), "median end-to-end environment steps/s (simulator + inference + update)")
    summ = df[(df["experiment"] == "sustained") & (df["status"] == "ok") & (df["phase"] == "config")]
    if len(summ):
        s = summ.groupby("series").first()
        rows = pd.DataFrame({"metric": ["avg_tokens_per_s", "degradation_pct (positive = slower at the end)", "duration_min"],
                             **{lab: [s.loc[lab, "x_average_throughput_tokens_per_s"], s.loc[lab, "x_degradation_degradation_pct"], s.loc[lab, "x_elapsed_s"] / 60]
                                for lab in labels if lab in s.index}})
        out["sustained"] = ("Sustained training", ["metric"], rows, "average tokens/s over the run; degradation compares first vs. last 10% of windows")
    return out


def render_plots(df: pd.DataFrame, out_dir: Path) -> dict[str, list[Path]]:
    have = set(df["experiment"])
    calls = {
        "matmul": [plotting.plot_matmul_throughput],
        "cnn": [plotting.plot_cnn_images_per_sec, plotting.plot_cnn_epoch_time],
        "transformer": [plotting.plot_transformer_tokens, plotting.plot_transformer_vs_batch],
        "memory": [plotting.plot_memory_vs_batch, plotting.plot_max_config],
        "precision": [plotting.plot_precision],
        "rl": [plotting.plot_ppo, plotting.plot_ppo_breakdown],
        "sustained": [plotting.plot_sustained],
    }
    return {exp: [p for fn in fns for p in fn(df, out_dir) if p.suffix == ".png"] for exp, fns in calls.items() if exp in have}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders (or run ids below --results-dir); order = column order and default baseline")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=None, help="output folder (default: <results-dir>/comparison)")
    ap.add_argument("--baseline", default=None, help="series label (or a unique substring of it) that speedups are relative to (default: first series)")
    args = ap.parse_args()
    root = resolve_path(args.results_dir)
    out = resolve_path(args.out) if args.out else root / "comparison"

    run_dirs = [resolve_run(r, root) for r in args.runs]
    df, series, notes = load_series(run_dirs)
    labels = [s["label"] for s in series]
    hits = [lab for lab in labels if args.baseline and args.baseline in lab] if args.baseline else labels[:1]
    if len(hits) != 1:
        raise SystemExit(f"--baseline '{args.baseline}' must match exactly one series; matches: {hits or 'none'}; available: {labels}")
    baseline = hits[0]

    # one colour slot and one label per series in every reused plot
    df = df.assign(backend=df["series"], device=None)
    plotting.BACKEND_ORDER = tuple(labels)
    plotting.BASELINE_NOTE = f"Speedups in the tables are relative to the baseline series '{baseline}'."
    df = df[df["status"] != "unavailable"]

    (out / "processed").mkdir(parents=True, exist_ok=True)
    plots = render_plots(df, out / "plots")
    tables = build_tables(df, labels)
    problems, info = comparability(series)

    md = ["# Cross-machine comparison\n",
          f"_{len(series)} series from {len(run_dirs)} run folder(s). Speedup = median throughput / median throughput of the baseline series **{baseline}**. "
          "Measurements only; nothing is ranked._\n",
          "## Series\n", md_series(series)]
    if notes:
        md += ["\n".join(f"* {n}" for n in notes) + "\n"]
    md += ["## Comparability\n"]
    if problems:
        md += ["**These runs used different code, seeds or experiment settings, so the affected numbers are NOT directly comparable:**\n",
               "\n".join(f"* {p}" for p in problems) + "\n"]
    else:
        md += ["No differences in code hash, seed or experiment configuration were found between the compared runs.\n"]
    if info:
        md += ["Software stack differences (expected across machines; part of what is being compared):\n", "\n".join(f"* {i}" for i in info) + "\n"]
    md += ["## Results\n",
           "_Memory numbers are backend-specific and not equivalent across backends (see the per-run reports). Throughput for very small workloads is noisy; check `lat_ms_cv` in the per-run reports._\n"]
    for exp, (heading, keys, tbl, desc) in tables.items():
        md += [f"### {heading}\n", f"Values: {desc}.\n", md_table(tbl, 1)]
        tbl.to_csv(out / "processed" / f"{exp}_comparison.csv", index=False)
        if exp != "sustained":
            sp = speedup_table(tbl, keys, baseline)
            sp.to_csv(out / "processed" / f"{exp}_speedup.csv", index=False)
            md += [f"Speedup vs **{baseline}** (>1 = higher throughput than the baseline):\n", md_table(sp, 2)]
        md += [f"![{p.stem}](plots/{p.name})" for p in plots.get(exp, [])] + [""]
    if "memory" in plots:
        mx = plotting.max_successful(df).rename(columns={"backend": "series"})
        mx.to_csv(out / "processed" / "memory_max_successful.csv", index=False)
        cols = [c for c in ["series", "ladder", "largest_ok_model", "largest_ok_batch", "largest_ok_seq", "largest_ok_params_M", "memory_mb_at_largest_ok", "ladder_ended_by"] if c in mx.columns]
        md += ["### Memory scaling: largest successful configuration\n",
               "How *large* a workload fits, not how fast it runs; depends on the device's memory capacity.\n", md_table(mx[cols], 1)]
        md += [f"![{p.stem}](plots/{p.name})" for p in plots["memory"]] + [""]
    (out / "comparison.md").write_text("\n".join(md))
    print(f"series: {labels}\nbaseline: {baseline}")
    for p in problems:
        print("WARNING not comparable:", p)
    print(f"comparison -> {out / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
