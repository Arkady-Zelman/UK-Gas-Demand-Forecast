"""Configuration loader — reads settings.yaml into a plain dict."""

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"

_cache: dict[str, Any] | None = None


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load and cache the YAML configuration."""
    global _cache
    if _cache is not None and path is None:
        return _cache

    path = Path(path) if path else _DEFAULT_PATH
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if path == _DEFAULT_PATH:
        _cache = cfg
    return cfg


def get(key: str, default: Any = None) -> Any:
    """Dot-separated key lookup, e.g. ``get("project.base_unit")``."""
    cfg = load_config()
    parts = key.split(".")
    node: Any = cfg
    for part in parts:
        if isinstance(node, dict):
            node = node.get(part)
        else:
            return default
        if node is None:
            return default
    return node
