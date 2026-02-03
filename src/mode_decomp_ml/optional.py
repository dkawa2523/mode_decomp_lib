"""Helpers for optional dependencies with consistent error messages."""
from __future__ import annotations

from typing import Any


def require_dependency(
    module: Any,
    *,
    name: str,
    pip_name: str,
    extra_hint: str | None = None,
) -> None:
    if module is not None:
        return
    msg = f"{name} requires {pip_name} (pip install {pip_name})"
    if extra_hint is None:
        extra_hint = "If you do not need this component, choose another option or remove it from config."
    if extra_hint:
        msg = f"{msg}. {extra_hint}"
    raise ImportError(msg)


__all__ = ["require_dependency"]
