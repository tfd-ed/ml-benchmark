"""YAML configuration loading with `--set a.b.c=value` command-line overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply 'section.key=value' strings; values are parsed as YAML (so 5, 1.5, [1,2], true work)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like section.key=value")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load the default config, merge an optional user config over it, then apply overrides."""
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if path and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        cfg = deep_merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    return apply_overrides(cfg, overrides)


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


# --------------------------------------------------------------------------- per-run result folders

LATEST = "latest"


def base_run_id(run_id: str) -> str:
    """The '<run_id>_crash' marker written by run_all belongs to the same run as its partial data."""
    return run_id[: -len("_crash")] if run_id.endswith("_crash") else run_id


def run_dir(results_dir: str | Path, run_id: str) -> Path:
    """results/<run_id>/ holds raw/, processed/, plots/ and report.md of one suite run."""
    return Path(results_dir) / base_run_id(run_id)


def mark_latest(results_dir: str | Path, run_id: str) -> None:
    """Point the relative symlink results/latest at this run (atomically replaced; best effort)."""
    root = Path(results_dir)
    tmp = root / f".{LATEST}.tmp"
    try:
        root.mkdir(parents=True, exist_ok=True)
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(base_run_id(run_id))
        tmp.replace(root / LATEST)
    except OSError:
        pass  # e.g. a filesystem without symlinks; --run-id still works


def resolve_run_dir(results_dir: str | Path, run_id: str | None = None) -> Path:
    """The folder of `run_id`, or of the run `latest` points at (falling back to the newest run folder)."""
    root = Path(results_dir)
    if run_id:
        return root / run_id
    link = root / LATEST
    if link.exists():
        return link.resolve()
    runs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "raw").is_dir()) if root.is_dir() else []
    if not runs:
        raise FileNotFoundError(f"no run folders under {root}")
    return runs[-1]
