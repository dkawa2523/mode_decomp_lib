"""Shared config helpers."""
from __future__ import annotations

from typing import Any, Mapping


def cfg_get(cfg: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


__all__ = ["cfg_get"]
