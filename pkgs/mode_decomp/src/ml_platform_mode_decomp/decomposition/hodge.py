"""Weighted Hodge decomposition of edge 1-cochains on a fixed masked Cartesian grid.

The native orthogonal decomposition lives on edges, not collocated node vectors.
The node adapter preserves its sampling nullspace as a separate residual.
"""

from __future__ import annotations

from ml_platform_mode_decomp.config import constructor_config

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsmr

from ..geometry import DomainSpec, validate_decomposer_compatibility
from ..geometry.identity import geometry_fingerprint
from .base import BaseDecomposer, _mask_values


def validate_hodge_geometry(geometry: DomainSpec):
    """Validate Cartesian physical inputs without constructing the cochain operators."""
    validate_decomposer_compatibility(geometry, {"name": "hodge"})
    if geometry.meta.get("periodic_axes"):
        raise ValueError("hodge requires nonperiodic geometry; use helmholtz for periodic vector projection")
    shape = geometry.grid_shape
    x, y = np.asarray(geometry.coords["x"]), np.asarray(geometry.coords["y"])
    if (
        x.shape != shape
        or y.shape != shape
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.array_equal(x, np.broadcast_to(x[0], x.shape))
        or not np.array_equal(y, np.broadcast_to(y[:, 0][:, None], y.shape))
        or np.any(np.diff(x[0]) <= 0)
        or np.any(np.diff(y[:, 0]) <= 0)
    ):
        raise ValueError("hodge requires separable, strictly increasing Cartesian x/y coordinates")
    units = geometry.meta.get("coordinate_units", geometry.meta.get("units"))
    if isinstance(units, Mapping) and units.get("x") != units.get("y"):
        raise ValueError("hodge x/y must use the same coordinate unit; convert axes before decomposition")
    mask = np.ones(shape, dtype=bool) if geometry.mask is None else np.array(geometry.mask, dtype=bool)
    area = geometry.integration_weights()
    if area is None:
        raise ValueError("hodge requires physical node integration weights")
    area = np.asarray(area, dtype=float)
    if area.shape != shape or not np.isfinite(area[mask]).all() or np.any(area[mask] <= 0):
        raise ValueError("hodge needs positive finite integration weights at every domain node")
    if not np.any(mask):
        raise ValueError("hodge requires a nonempty fixed domain mask")
    return x, y, mask, area


class HodgeDecomposition:
    """A reusable finite cochain complex and positive edge metric.

    Natural boundary means all vertex potentials and all retained face cochains
    participate. No boundary node value, pointwise velocity, or external flux is
    prescribed. d0 maps vertex potentials to edge integrals, d1 maps edge
    integrals to oriented face circulation, and d1 @ d0 == 0.
    """

    parts = ("exact", "coexact", "harmonic")

    def __init__(self, geometry: DomainSpec, *, tolerance: float = 1e-11, maxiter: int | None = None):
        if not np.isfinite(tolerance) or not 0 < tolerance < 1:
            raise ValueError("hodge tolerance must lie between zero and one")
        if maxiter is not None and (isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 1):
            raise ValueError("hodge maxiter must be a positive integer or null")
        self.tolerance, self.maxiter = tolerance, maxiter
        self.grid_shape = geometry.grid_shape
        x, y, self.mask, area = validate_hodge_geometry(geometry)
        self.node_indices = np.flatnonzero(self.mask)
        self.node_weights = area.reshape(-1)[self.node_indices].copy()
        self.vertex_coordinates = np.column_stack((x.reshape(-1)[self.node_indices], y.reshape(-1)[self.node_indices]))
        vertices = np.full(self.grid_shape, -1, dtype=int)
        vertices[self.mask] = np.arange(len(self.node_indices))
        horizontal = self.mask[:, :-1] & self.mask[:, 1:]
        vertical = self.mask[:-1] & self.mask[1:]
        tail = np.concatenate((vertices[:, :-1][horizontal], vertices[:-1][vertical]))
        head = np.concatenate((vertices[:, 1:][horizontal], vertices[1:][vertical]))
        self.edge_vertices = np.column_stack((tail, head))
        self.edge_axes = np.repeat([0, 1], [horizontal.sum(), vertical.sum()])
        self.edge_lengths = np.concatenate((np.diff(x, axis=1)[horizontal], np.diff(y, axis=0)[vertical]))
        vertex_count, edge_count = len(self.node_indices), len(tail)
        self.d0 = csr_matrix(
            (np.tile([-1.0, 1.0], edge_count), (np.repeat(np.arange(edge_count), 2), self.edge_vertices.reshape(-1))),
            shape=(edge_count, vertex_count),
        )
        horizontal_ids = np.full(horizontal.shape, -1, dtype=int)
        vertical_ids = np.full(vertical.shape, -1, dtype=int)
        horizontal_ids[horizontal] = np.arange(horizontal.sum())
        vertical_ids[vertical] = np.arange(vertical.sum()) + horizontal.sum()
        faces = self.mask[:-1, :-1] & self.mask[1:, :-1] & self.mask[:-1, 1:] & self.mask[1:, 1:]
        self.face_origins = np.column_stack(np.nonzero(faces))
        boundary_edges = np.column_stack(
            (
                horizontal_ids[:-1][faces],
                vertical_ids[:, 1:][faces],
                horizontal_ids[1:][faces],
                vertical_ids[:, :-1][faces],
            )
        )
        face_count = len(boundary_edges)
        self.d1 = csr_matrix(
            (
                np.tile([1.0, 1.0, -1.0, -1.0], face_count),
                (np.repeat(np.arange(face_count), 4), boundary_edges.reshape(-1)),
            ),
            shape=(face_count, edge_count),
        )
        adjacency = csr_matrix(
            (np.ones(2 * edge_count), (np.concatenate((tail, head)), np.concatenate((head, tail)))),
            shape=(vertex_count, vertex_count),
        )
        self.connected_components = int(connected_components(adjacency, directed=False, return_labels=False))
        # A finite planar grid subcomplex has no nonzero 2-cycles: rank(d1)=F.
        self.harmonic_dimension = edge_count - vertex_count + self.connected_components - face_count
        incident_length = np.zeros((2, vertex_count))
        np.add.at(incident_length, (self.edge_axes, tail), self.edge_lengths)
        np.add.at(incident_length, (self.edge_axes, head), self.edge_lengths)
        allocated_area = self.edge_lengths * (
            self.node_weights[tail] / incident_length[self.edge_axes, tail]
            + self.node_weights[head] / incident_length[self.edge_axes, head]
        )
        self.edge_weights = allocated_area / self.edge_lengths**2
        if not np.isfinite(self.edge_weights).all() or np.any(self.edge_weights <= 0):
            raise ValueError("hodge edge metric must be finite and strictly positive")
        self._sqrt_weight = np.sqrt(self.edge_weights)
        self._exact_operator = diags(self._sqrt_weight) @ self.d0
        self._coexact_operator = diags(1 / self._sqrt_weight) @ self.d1.T
        self._chains: list[tuple[np.ndarray, np.ndarray, int]] = []
        for row in range(self.grid_shape[0]):
            self._add_chains(vertices[row], horizontal_ids[row], 0)
        for column in range(self.grid_shape[1]):
            self._add_chains(vertices[:, column], vertical_ids[:, column], 1)

    def _add_chains(self, vertices: np.ndarray, edges: np.ndarray, axis: int):
        positions = np.flatnonzero(vertices >= 0)
        for run in np.split(positions, np.flatnonzero(np.diff(positions) != 1) + 1):
            if len(run) > 1:
                self._chains.append((vertices[run], edges[run[:-1]], axis))

    def _edges(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value)
        if value.dtype.kind not in "fiu" or value.shape != self.edge_weights.shape or not np.isfinite(value).all():
            raise ValueError(f"Expected finite edge line integrals with shape {self.edge_weights.shape}")
        return value.astype(float, copy=False)

    def _least_squares(self, matrix: csr_matrix, rhs: np.ndarray) -> np.ndarray:
        scale = float(np.max(np.abs(rhs), initial=0))
        if matrix.shape[1] == 0 or scale == 0:
            return np.zeros(matrix.shape[1])
        result = lsmr(
            matrix,
            rhs / scale,
            atol=self.tolerance,
            btol=self.tolerance,
            conlim=0,
            maxiter=self.maxiter or 4 * max(matrix.shape),
        )
        if result[1] not in {0, 1, 2, 4, 5}:
            raise RuntimeError(f"hodge weighted least-squares did not converge (LSMR status {result[1]})")
        return result[0] * scale

    def decompose_edges(self, edge_flow: np.ndarray) -> dict[str, np.ndarray]:
        """Return the W-orthogonal exact, coexact and closed/coclosed harmonic parts."""
        flow = self._edges(edge_flow)
        potential = self._least_squares(self._exact_operator, self._sqrt_weight * flow)
        exact = np.asarray(self.d0 @ potential)
        stream = self._least_squares(self._coexact_operator, self._sqrt_weight * (flow - exact))
        coexact = np.asarray(self._coexact_operator @ stream) / self._sqrt_weight
        return {"exact": exact, "coexact": coexact, "harmonic": flow - exact - coexact}

    def reconstruct_edges(self, parts: Mapping[str, np.ndarray]) -> np.ndarray:
        """Sum native edge components; no node interpolation is involved."""
        return sum((self._edges(parts[name]) for name in self.parts), start=np.zeros_like(self.edge_weights))

    def energy(self, edge_flow: np.ndarray) -> float:
        """Squared norm in the declared lumped edge metric, not a collocated-node energy."""
        flow = self._edges(edge_flow)
        return float(np.dot(self.edge_weights * flow, flow))

    def sample_nodes(self, field: np.ndarray) -> np.ndarray:
        """Endpoint-average tangential values times oriented edge lengths."""
        field = np.asarray(field)
        if field.dtype.kind not in "fiu" or field.shape != (*self.grid_shape, 2):
            raise ValueError("hodge node fields must have shape (H,W,2), ordered as x/y components")
        field = field.astype(float, copy=False)
        values = field.reshape(-1, 2)[self.node_indices]
        if not np.isfinite(values).all():
            raise ValueError("hodge requires observations at all fixed domain nodes")
        tail, head = self.edge_vertices.T
        return self.edge_lengths * (values[tail, self.edge_axes] + values[head, self.edge_axes]) / 2

    def lift_edges(self, edge_flow: np.ndarray) -> np.ndarray:
        """Minimum node-area-norm right inverse of endpoint-average sampling.

        Each x row / y column is a path, so its one checkerboard gauge is fixed
        analytically. Isolated unsampled components have minimum-norm value zero.
        """
        flow = self._edges(edge_flow)
        values = np.zeros((len(self.node_indices), 2))
        for nodes, edges, axis in self._chains:
            sign = 1.0 - 2.0 * (np.arange(len(nodes)) % 2)
            particular = sign * np.r_[0.0, np.cumsum(sign[1:] * 2 * flow[edges] / self.edge_lengths[edges])]
            weight = self.node_weights[nodes]
            gauge = -np.dot(weight * sign, particular) / weight.sum()
            values[nodes, axis] = particular + sign * gauge
        field = np.zeros((*self.grid_shape, 2))
        field.reshape(-1, 2)[self.node_indices] = values
        return field


class HodgeDecomposer(BaseDecomposer[np.ndarray]):
    """Node-vector adapter: three edge component lifts plus a sampling residual."""

    synthesis_kind = "component"

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any] | None = None,
        maxiter: Any = None,
        boundary: Any = "natural",
        tolerance: Any = 1e-11,
    ) -> None:
        cfg = constructor_config(cfg, locals(), name="hodge")
        super().__init__(cfg=cfg)
        self.name = "hodge"
        if cfg["boundary"] != "natural":
            raise ValueError("hodge supports boundary='natural' (full finite cochains) only")
        self._tolerance = float(cfg["tolerance"])
        self._maxiter = cfg["maxiter"]
        self.hodge: HodgeDecomposition | None = None
        self._geometry_sha256: str | None = None

    def fit(self, dataset=None, domain_spec: DomainSpec | None = None):
        if domain_spec is None:
            raise ValueError("hodge.fit requires its fixed geometry")
        self.hodge = HodgeDecomposition(domain_spec, tolerance=self._tolerance, maxiter=self._maxiter)
        self._geometry_sha256 = geometry_fingerprint(domain_spec)
        return self

    def validate_field_spec(self, spec, *, preprocess_name="none") -> None:
        spec.require_vector(self.name)
        if not np.isfinite(self._tolerance) or not 0 < self._tolerance < 1:
            raise ValueError("hodge tolerance must lie between zero and one")
        if self._maxiter is not None and (
            isinstance(self._maxiter, bool) or not isinstance(self._maxiter, int) or self._maxiter < 1
        ):
            raise ValueError("hodge maxiter must be a positive integer or null")
        if preprocess_name != "none":
            raise ValueError("Physical hodge decomposition requires field_preprocess.name: none")

    def _complex(self, domain_spec: DomainSpec) -> HodgeDecomposition:
        if self.hodge is None:
            self.fit(domain_spec=domain_spec)
        if geometry_fingerprint(domain_spec) != self._geometry_sha256:
            raise ValueError("hodge cannot change the fitted coordinates, mask or integration weights")
        assert self.hodge is not None
        return self.hodge

    def transform(self, field: np.ndarray, mask: np.ndarray | None, domain_spec: DomainSpec) -> np.ndarray:
        hodge = self._complex(domain_spec)
        if mask is not None:
            mask = _mask_values(mask)
            if mask.shape != hodge.grid_shape or not mask[hodge.mask].all():
                raise ValueError("hodge requires observations at every fixed domain node; it is not an imputer")
        values = np.asarray(field)
        if values.dtype.kind not in "fiu":
            raise ValueError("hodge requires real node vectors")
        values = values.astype(float, copy=False)
        flow = hodge.sample_nodes(values)
        parts = hodge.decompose_edges(flow)
        lifted = [hodge.lift_edges(parts[name]) for name in hodge.parts]
        residual = np.zeros_like(values)
        residual[hodge.mask] = values[hodge.mask] - np.sum(lifted, axis=0)[hodge.mask]
        coefficients = np.stack([*lifted, residual])
        self._coeff_meta = {
            "method": self.name,
            "field_shape": [*hodge.grid_shape, 2],
            "field_ndim": 3,
            "field_layout": "HWC",
            "channels": 2,
            "coeff_shape": list(coefficients.shape),
            "coeff_layout": "PHWC",
            "flatten_order": "C",
            "complex_format": "real",
            "parts": ["exact_edge_lift", "coexact_edge_lift", "harmonic_edge_lift", "sampling_residual"],
            "part_axis": 0,
            "boundary": "natural",
            "projection": "finite_cochain_hodge",
            "projection_inner_product": "edge_lumped_domain_area",
            "orthogonality_scope": "native_edge_cochains_only",
            "node_bridge": "endpoint_average_line_integrals_and_node_area_minimum_norm_lift",
            "harmonic_dimension": hodge.harmonic_dimension,
            "complex_size": {
                "vertices": len(hodge.node_indices),
                "edges": len(hodge.edge_weights),
                "faces": hodge.d1.shape[0],
            },
        }
        return coefficients

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        hodge = self._complex(domain_spec)
        coeff = np.asarray(coeff)
        shape = (4, *hodge.grid_shape, 2)
        if coeff.dtype.kind not in "fiu" or coeff.shape != shape or not np.isfinite(coeff).all():
            raise ValueError(f"hodge expects finite component arrays with shape {shape}")
        return coeff.sum(axis=0)
