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
