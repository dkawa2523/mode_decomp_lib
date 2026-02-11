"""Metrics helpers for the decomposition process (split from processes.decomposition)."""
from __future__ import annotations

def _grid_spacing_from_domain(domain_spec) -> tuple[float, float]:
    x = domain_spec.coords.get("x")
    y = domain_spec.coords.get("y")
    if x is None or y is None:
        raise ValueError("domain_spec must include x/y for div/curl metrics")
    dx = float(abs(x[0, 1] - x[0, 0])) if x.shape[1] > 1 else 1.0
    dy = float(abs(y[1, 0] - y[0, 0])) if y.shape[0] > 1 else 1.0
    if dx <= 0 or dy <= 0:
        raise ValueError("domain_spec spacing must be positive for div/curl metrics")
    return dx, dy


