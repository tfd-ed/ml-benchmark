"""Normalised result schema, incremental writer, loader and validator.

Layout on disk (results/raw/):
    <experiment>__<backend>__<run_id>.csv        one row per measurement (streamed, flushed per row)
    <experiment>__<backend>__<run_id>.meta.json  full run metadata + configuration

Common columns are shared by every experiment; experiment-specific values go in the
JSON-encoded `extra` column so the common schema is never broken.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

COLUMNS = [
    "run_id", "timestamp", "experiment", "backend", "device", "model", "dataset",
    "batch_size", "sequence_length", "precision", "phase", "iteration",
    "latency_ms", "throughput", "throughput_unit", "memory_mb", "memory_kind",
    "loss", "accuracy", "status", "error_type", "error", "extra", "metadata",
]
STATUSES = ("ok", "oom", "unsupported", "unavailable", "error", "timeout", "skipped")
PHASES = ("warmup", "measure", "config")  # 'config' = one row describing a whole configuration
REQUIRED_NON_NULL = ["run_id", "timestamp", "experiment", "backend", "status"]


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class ResultWriter:
    """Streams rows to CSV (so a crash keeps everything measured so far)."""

    def __init__(self, raw_dir: Path, experiment: str, status, metadata: dict[str, Any]):
        import datetime as dt

        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.experiment = experiment
        self.status = status
        self.metadata = metadata
        stem = f"{experiment}__{status.backend}__{metadata['run_id']}"
        self.csv_path = self.raw_dir / f"{stem}.csv"
        self.meta_path = self.raw_dir / f"{stem}.meta.json"
        self.meta_path.write_text(json.dumps(metadata, indent=2, default=str))
        self._fh = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=COLUMNS)
        self._writer.writeheader()
        self._fh.flush()
        self._now = lambda: dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
        self._meta_compact = json.dumps(
            {k: metadata.get(k) for k in ("torch", "device_name", "seed", "git_commit", "code_sha256", "hostname")},
            separators=(",", ":"),
        )
        self.counts: dict[str, int] = {}

    def write(self, **fields: Any) -> dict[str, Any]:
        """Write one row. Unknown keyword arguments are folded into `extra`."""
        row = {c: None for c in COLUMNS}
        extra = dict(fields.pop("extra", None) or {})
        for k, v in fields.items():
            if k in row:
                row[k] = _clean(v)
            else:
                extra[k] = v
        row.update(
            run_id=self.metadata["run_id"], timestamp=row["timestamp"] or self._now(),
            experiment=self.experiment, backend=self.status.backend, device=self.status.device_name,
            status=row["status"] or "ok", metadata=self._meta_compact,
        )
        row["extra"] = json.dumps({k: _clean(v) for k, v in extra.items()}, default=str) if extra else None
        self._writer.writerow(row)
        self._fh.flush()
        self.counts[row["status"]] = self.counts.get(row["status"], 0) + 1
        return row

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- loading / validation


def load_results(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Load every raw CSV. Returns (dataframe with extra_* columns, {run_id|backend|experiment: metadata})."""
    frames, metas = [], {}
    for csv_path in sorted(Path(raw_dir).glob("*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["source_file"] = csv_path.name
        frames.append(df)
        meta_path = csv_path.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            metas[f"{meta['experiment']}|{meta['backend']}|{meta['run_id']}"] = meta
    if not frames:
        return pd.DataFrame(columns=COLUMNS), metas
    df = pd.concat(frames, ignore_index=True)
    return explode_extra(df), metas


def explode_extra(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the JSON `extra` column into `x_<key>` columns."""
    if "extra" not in df.columns:
        return df
    parsed = df["extra"].map(lambda s: json.loads(s) if isinstance(s, str) and s else {})
    extra_df = pd.DataFrame(list(parsed), index=df.index).add_prefix("x_")
    return pd.concat([df.drop(columns=["extra"]), extra_df], axis=1)


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems (empty list = clean)."""
    issues: list[str] = []
    missing = [c for c in COLUMNS if c not in df.columns and c != "extra"]
    if missing:
        return [f"missing columns: {missing}"]
    for col in REQUIRED_NON_NULL:
        n = int(df[col].isna().sum())
        if n:
            issues.append(f"{n} rows have null '{col}'")
    bad = sorted(set(df["status"].dropna()) - set(STATUSES))
    if bad:
        issues.append(f"unknown status values: {bad}")
    bad_phase = sorted(set(df["phase"].dropna()) - set(PHASES))
    if bad_phase:
        issues.append(f"unknown phase values: {bad_phase}")
    ok = df[df["status"] == "ok"]
    for col in ("latency_ms", "throughput"):
        vals = pd.to_numeric(ok[col], errors="coerce").dropna()
        if (vals < 0).any():
            issues.append(f"{int((vals < 0).sum())} ok rows have negative {col}")
    measured = ok[ok["phase"] == "measure"]
    n_nan = int((measured["latency_ms"].isna() & measured["throughput"].isna()).sum())
    if n_nan:
        issues.append(f"{n_nan} 'ok' measured rows have neither latency nor throughput")
    failed = df[~df["status"].isin(["ok"])]
    no_msg = failed[failed["error"].isna() & (failed["status"] != "skipped")]
    if len(no_msg):
        issues.append(f"{len(no_msg)} non-ok rows have no error message")
    return issues


# --------------------------------------------------------------------------- markdown helpers


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, float):
        if v != 0 and abs(v) >= 1e5:
            return f"{v:,.0f}"
        return f"{v:,.{digits}f}" if abs(v) >= 0.01 or v == 0 else f"{v:.3g}"
    return str(v)


def md_table(df: pd.DataFrame, digits: int = 2) -> str:
    if df.empty:
        return "_no data_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c], digits).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- aggregation


def aggregate_stats(df: pd.DataFrame, keys: list[str], value_col: str) -> pd.DataFrame:
    """Per-group n / median / mean / std / cv / min / max of `value_col`."""
    from .metrics import summarize

    rows = []
    for key, g in df.groupby(keys, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(keys, key)), **summarize(pd.to_numeric(g[value_col], errors="coerce").dropna().tolist())})
    return pd.DataFrame(rows)


def add_speedup(summary: pd.DataFrame, config_keys: list[str], time_col: str = "median", baseline: str = "cpu") -> pd.DataFrame:
    """Add `speedup_vs_<baseline>` = baseline_time / backend_time for identical configurations.
    `summary[time_col]` must be a TIME-like quantity (lower is better), e.g. latency or epoch time."""
    from .metrics import speedup

    out = summary.copy()
    base = out[out["backend"] == baseline].set_index(config_keys)[time_col] if (out["backend"] == baseline).any() else None
    col = f"speedup_vs_{baseline}"
    if base is None:
        out[col] = None
        return out
    out[col] = [
        speedup(base.get(tuple(r[k] for k in config_keys) if len(config_keys) > 1 else r[config_keys[0]]), r[time_col])
        for _, r in out.iterrows()
    ]
    return out


def latest_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most recent run for every (experiment, backend). A '<run_id>_crash' marker
    written by run_all belongs to the same run as its partial data."""
    if df.empty:
        return df
    base = df["run_id"].astype(str).str.replace("_crash$", "", regex=True)
    latest = base.groupby([df["experiment"], df["backend"]]).transform("max")
    return df[base == latest].copy()
