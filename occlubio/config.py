"""Tiny YAML config loader -> attribute-accessible namespace.

Zero extra deps beyond PyYAML. Access nested keys with dots: ``cfg.gallery.top_k``.
"""
from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return types.SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def _to_dict(obj: Any) -> Any:
    if isinstance(obj, types.SimpleNamespace):
        return {k: _to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    return obj


def load_config(path: Optional[str | Path] = None) -> types.SimpleNamespace:
    """Load a config file (defaults to configs/default.yaml)."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _to_ns(raw)


def config_to_dict(cfg: types.SimpleNamespace) -> dict:
    return _to_dict(cfg)
