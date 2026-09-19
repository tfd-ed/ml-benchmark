"""Run the benchmark suite: every (experiment, backend) pair in its own subprocess.

    python scripts/run_all.py --backend auto
    python scripts/run_all.py --backend mps --experiment matmul --experiment cnn
    python scripts/run_all.py --backend cpu,mps --experiment sustained --set sustained.duration_minutes=5

A subprocess per pair keeps allocator state clean between experiments and means a hard crash
(segfault, kill) is recorded as an 'error' row instead of ending the suite.
"""
import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import devices  # noqa: E402
from src.config import load_config, mark_latest, resolve_path, run_dir  # noqa: E402
from src.environment import collect_metadata  # noqa: E402
from src.reporting import ResultWriter  # noqa: E402

EXPERIMENTS = {
    "matmul": "benchmark_matmul.py",
    "cnn": "benchmark_resnet.py",
    "transformer": "benchmark_transformer.py",
    "memory": "benchmark_memory.py",
    "precision": "benchmark_precision.py",
    "rl": "benchmark_rl.py",
    "sustained": "benchmark_sustained.py",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="auto", help="auto | cpu | mps | cuda | comma list")
    ap.add_argument("--experiment", action="append", choices=[*EXPERIMENTS, "all"], help="repeatable; default: all")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--report", action="store_true", help="run scripts/generate_report.py afterwards")
    args = ap.parse_args()

    names = list(EXPERIMENTS) if not args.experiment or "all" in args.experiment else list(dict.fromkeys(args.experiment))
    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    cfg = load_config(args.config, args.overrides)
    results_dir = resolve_path(args.results_dir or cfg.get("results_dir", "results"))
    out_dir = run_dir(results_dir, run_id)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    mark_latest(results_dir, run_id)
    backends = devices.resolve_backends(args.backend)
    print(f"run_id {run_id}: experiments {names} on backends {[b.backend for b in backends]}", flush=True)

    manifest, worst = [], 0
    for name in names:
        for st in backends:
            cmd = [sys.executable, str(ROOT / "experiments" / EXPERIMENTS[name]), "--backend", st.backend, "--run-id", run_id]
            if args.config:
                cmd += ["--config", args.config]
            if args.results_dir:
                cmd += ["--results-dir", args.results_dir]
            for o in args.overrides:
                cmd += ["--set", o]
            t0 = time.time()
            rc = subprocess.run(cmd).returncode
            manifest.append({"experiment": name, "backend": st.backend, "returncode": rc, "seconds": round(time.time() - t0, 1)})
            if rc < 0 or rc > 1:  # killed by a signal / interpreter crash: the experiment could not record it itself
                meta = collect_metadata(st, cfg, int(cfg["seed"]), run_id, name)
                with ResultWriter(out_dir / "raw", name, st, {**meta, "run_id": run_id + "_crash"}) as w:
                    w.write(status="error", phase="config", error_type="SubprocessCrash", error=f"experiment process exited with code {rc}")
                worst = 1
    out = out_dir / "raw" / "run_all.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run_id": run_id, "backend_arg": args.backend, "overrides": args.overrides, "runs": manifest}, indent=2))
    print(f"manifest -> {out}")
    for m in manifest:
        print(f"  {m['experiment']:12s} {m['backend']:5s} rc={m['returncode']} {m['seconds']}s")
    if args.report:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_report.py"), "--run-id", run_dir(results_dir, run_id).name] + (["--results-dir", args.results_dir] if args.results_dir else []))
    return worst


if __name__ == "__main__":
    sys.exit(main())
