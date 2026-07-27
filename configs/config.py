"""Tiny config loader: YAML -> attribute-accessible nested namespace.

Deliberately avoids pulling in a heavyweight config framework (Hydra/OmegaConf)
for a project this size -- plain YAML + dotted overrides is enough.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _to_dict(obj: Any) -> Any:
    if isinstance(obj, SimpleNamespace):
        return {k: _to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    return obj


def _set_dotted(d: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    # best-effort type coercion so CLI overrides don't turn ints into strings
    last = keys[-1]
    if isinstance(cur.get(last), bool):
        value = str(value).lower() in ("1", "true", "yes")
    elif isinstance(cur.get(last), int) and not isinstance(cur.get(last), bool):
        try:
            value = int(value)
        except ValueError:
            pass
    elif isinstance(cur.get(last), float):
        try:
            value = float(value)
        except ValueError:
            pass
    cur[last] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None):
    """Load YAML config and apply `key.subkey=value` overrides.

    Returns a SimpleNamespace tree so callers write `cfg.train.lr` instead of
    `cfg["train"]["lr"]`.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = copy.deepcopy(raw)

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Bad override '{ov}', expected key.subkey=value")
        key, value = ov.split("=", 1)
        _set_dotted(raw, key.strip(), value.strip())

    return _to_namespace(raw)


def config_to_dict(cfg) -> dict:
    return _to_dict(cfg)
