"""Immutable coefficient structure, separate from per-operation diagnostics."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CoefficientLayout:
    method: str
    raw_shape: tuple[int, ...]
    raw_layout: str
    raw_format: str | None
    component_offsets: tuple[int, ...]
    real_dimension: int
    synthesis_kind: str

    @classmethod
    def from_meta(cls, meta: Mapping[str, Any], dimension: int, synthesis_kind: str):
        return cls(
            str(meta["method"]),
            tuple(int(x) for x in meta.get("coeff_shape", ())),
            str(meta.get("coeff_layout", "")),
            meta.get("coeff_format"),
            tuple(int(x) for x in meta.get("component_offsets", ())),
            int(dimension),
            synthesis_kind,
        )
