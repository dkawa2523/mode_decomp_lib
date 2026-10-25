"""Linear maps between explicit 2-D observations and a fixed Cartesian grid."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.spatial import Delaunay, KDTree, QhullError


def _real(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{label} must contain real numeric values; complex values are not converted.")
    return array.astype(np.float64, copy=False)


def _points(value: Any, label: str) -> np.ndarray:
    points = _real(value, label)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError(f"{label} must contain finite Cartesian coordinates with shape (M,2).")
    return points


def _mask(value: Any, count: int, label: str) -> np.ndarray:
    if value is None:
        return np.ones(count, dtype=bool)
    mask = np.asarray(value)
    if mask.shape != (count,) or mask.dtype.kind not in "bui" or not np.isin(mask, (0, 1)).all():
        raise ValueError(f"{label} must contain one boolean or 0/1 value per position.")
    return mask.astype(bool, copy=False)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _triangle_overlap(query: np.ndarray, triangle: np.ndarray) -> bool:
    """Positive-area intersection; shared edges/vertices alone are permitted."""
    polygon = list(query)
    for start, stop in zip(triangle, np.roll(triangle, -1, axis=0), strict=True):
        if not polygon:
            return False
        clipped = []
        previous = polygon[-1]
        previous_side = float(_cross(stop - start, previous - start))
        for current in polygon:
            current_side = float(_cross(stop - start, current - start))
            if (current_side >= 0) != (previous_side >= 0):
                clipped.append(previous + previous_side / (previous_side - current_side) * (current - previous))
            if current_side >= 0:
                clipped.append(current)
            previous, previous_side = current, current_side
        polygon = clipped
    if len(polygon) < 3:
        return False
    vertices = np.asarray(polygon)
    # Center the coordinates before the area sum to avoid cancellation at a boundary.
    shifted = vertices - vertices[0]
    twice_area = abs(float(_cross(shifted, np.roll(shifted, -1, axis=0)).sum()))
    reference = max(
        abs(float(_cross(query[1] - query[0], query[2] - query[0]))),
        abs(float(_cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))),
    )
    return twice_area > 256 * np.finfo(float).eps * reference


def _segment_interval(query: np.ndarray, triangle: np.ndarray) -> tuple[float, float] | None:
    """Closed-triangle intersection as a segment-parameter interval."""
    start, stop = query
    low, high = 0.0, 1.0
    for a, b in zip(triangle, np.roll(triangle, -1, axis=0), strict=True):
        initial = float(_cross(b - a, start - a))
        delta = float(_cross(b - a, stop - start))
        if delta == 0:
            if initial < 0:
                return None
        elif delta > 0:
            low = max(low, -initial / delta)
        else:
            high = min(high, -initial / delta)
    return (low, high) if high >= low else None


class _TriangleSupport:
    """AABB-filtered source triangles that inverse stencils must not cross."""

    def __init__(self, triangles: np.ndarray):
        self.triangles = triangles.copy()
        clockwise = _cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) < 0
        self.triangles[clockwise] = self.triangles[clockwise][:, [0, 2, 1]]
        self.minimum, self.maximum = triangles.min(axis=1), triangles.max(axis=1)
        centers = (self.minimum + self.maximum) / 2
        self.radius = float(np.max((self.maximum - self.minimum) / 2)) if len(triangles) else 0.0
        self.tree = KDTree(centers)

    def _candidates(self, vertices: np.ndarray) -> np.ndarray:
        minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
        radius = float(np.max((maximum - minimum) / 2))
        candidates = np.asarray(
            self.tree.query_ball_point((minimum + maximum) / 2, radius + self.radius, p=np.inf), dtype=int
        )
        return candidates[((self.maximum[candidates] >= minimum) & (self.minimum[candidates] <= maximum)).all(axis=1)]

    def crosses_triangle(self, vertices: np.ndarray) -> bool:
        return any(_triangle_overlap(vertices, self.triangles[index]) for index in self._candidates(vertices))

    def covers_segment(self, vertices: np.ndarray) -> bool:
        # Allowed-boundary edges are valid, but an excluded wall along a target
        # grid edge must not become supported simply because its area is zero.
        intervals = [_segment_interval(vertices, self.triangles[index]) for index in self._candidates(vertices)]
        covered = 0.0
        for low, high in sorted(interval for interval in intervals if interval is not None):
            if low > covered + 256 * np.finfo(float).eps:
                return False
            covered = max(covered, high)
        return covered >= 1 - 256 * np.finfo(float).eps


@dataclass(frozen=True)
class BarycentricMap:
    """A precomputed linear map. Unsupported queries stay NaN, with support=False."""

    source_count: int
    vertices: np.ndarray
    weights: np.ndarray
    support: np.ndarray
    simplices: np.ndarray
    simplex_support: np.ndarray

    @classmethod
    def build(cls, source_xy: Any, query_xy: Any, *, simplex_support: Any = None) -> BarycentricMap:
        points, queries = _points(source_xy, "source_xy"), _points(query_xy, "query_xy")
        if len(points) < 3 or len(np.unique(points, axis=0)) != len(points):
            raise ValueError("Linear interpolation requires at least three unique non-collinear source points.")
        try:
            tri = Delaunay(points)
        except QhullError as exc:
            raise ValueError("Source points do not define a stable 2-D triangulation; no jitter is applied.") from exc
        if tri.coplanar.size:
            raise ValueError("Delaunay omitted source points; resolve near-degenerate coordinates explicitly.")
        allowed = np.ones(len(tri.simplices), dtype=bool)
        if simplex_support is not None:
            declared = np.asarray(simplex_support)
            if declared.ndim == 2 and declared.shape[1] == 3 and declared.dtype.kind in "iu":
                candidates = {tuple(row) for row in np.sort(tri.simplices, axis=1).tolist()}
                selected = {tuple(row) for row in np.sort(declared, axis=1).tolist()}
                if not selected <= candidates:
                    raise ValueError("Allowed simplex vertex triples must belong to the source Delaunay triangulation.")
                allowed = np.asarray([tuple(row) in selected for row in np.sort(tri.simplices, axis=1).tolist()])
            else:
                allowed = _mask(declared, len(tri.simplices), "simplex_support")
        found = tri.find_simplex(queries)
        support = found >= 0
        support[support] &= allowed[found[support]]
        vertices = np.zeros((len(queries), 3), dtype=np.int64)
        weights = np.zeros((len(queries), 3), dtype=np.float64)
        active = found[support]
        transform = tri.transform[active]
        first = np.einsum("qij,qj->qi", transform[:, :2], queries[support] - transform[:, 2])
        barycentric = np.column_stack([first, 1 - first.sum(axis=1)])
        # Qhull boundary tolerance can produce tiny negative coordinates.
        barycentric = np.maximum(barycentric, 0)
        barycentric /= barycentric.sum(axis=1, keepdims=True)
        vertices[support], weights[support] = tri.simplices[active], barycentric
        return cls(len(points), vertices, weights, support, tri.simplices.copy(), allowed)

    def forward(self, values: Any, observation_mask: Any = None) -> tuple[np.ndarray, np.ndarray]:
        array = _real(values, "values")
        if array.ndim != 2 or array.shape[0] != self.source_count or array.shape[1] < 1:
            raise ValueError("values must have shape (source_count,C), with C >= 1.")
        observed = _mask(observation_mask, self.source_count, "observation_mask")
        if not np.isfinite(array[observed]).all():
            raise ValueError("Observed source values must be finite; provide an explicit mask for missing values.")
        required = self.weights > 1e-13
        support = self.support & np.all(observed[self.vertices] | ~required, axis=1)
        clean = np.where(observed[:, None], array, 0.0)
        result = np.einsum("qv,qvc->qc", self.weights, clean[self.vertices])
        result[~support] = np.nan
        return result, support


@dataclass(frozen=True)
class ScatteredGridMap:
    """Forward Delaunay interpolation and local-cell evaluation at source points.

    x/y are physical Cartesian axes. No point-spacing area, nearest fill, or
    coordinate rescaling is inferred. Grid-to-point evaluation never spans a hole.
    """

    x: np.ndarray
    y: np.ndarray
    point_to_grid: BarycentricMap
    grid_to_point: BarycentricMap
    source_points: np.ndarray

    @classmethod
    def build(
        cls, points: Any, x: Any, y: Any, *, domain_mask: Any = None, simplex_support: Any = None
    ) -> ScatteredGridMap:
        points = _points(points, "points")
        axes = [_real(axis, label) for axis, label in ((x, "x"), (y, "y"))]
        if any(
            axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all() or not (np.diff(axis) > 0).all()
            for axis in axes
        ):
            raise ValueError("Target x/y need finite, strictly increasing axes with at least two coordinates.")
        x, y = axes
        xx, yy = np.meshgrid(x, y, indexing="xy")
        if domain_mask is not None and np.asarray(domain_mask).shape != xx.shape:
            raise ValueError("domain_mask must match the target (len(y),len(x)) shape.")
        domain = _mask(None if domain_mask is None else np.asarray(domain_mask).reshape(-1), xx.size, "domain_mask")
        forward = BarycentricMap.build(
            points, np.column_stack([xx.ravel(), yy.ravel()]), simplex_support=simplex_support
        )
        forward = replace(forward, support=forward.support & domain)
        xi = np.clip(np.searchsorted(x, points[:, 0], side="right") - 1, 0, len(x) - 2)
        yi = np.clip(np.searchsorted(y, points[:, 1], side="right") - 1, 0, len(y) - 2)
        fx = (points[:, 0] - x[xi]) / (x[xi + 1] - x[xi])
        fy = (points[:, 1] - y[yi]) / (y[yi + 1] - y[yi])
        lower = fx + fy <= 1
        base = yi * len(x) + xi
        vertices = np.column_stack([base, base + 1, base + len(x)])
        vertices[~lower] = np.column_stack([base + len(x) + 1, base + len(x), base + 1])[~lower]
        weights = np.column_stack([1 - fx - fy, fx, fy])
        weights[~lower] = np.column_stack([fx + fy - 1, 1 - fx, 1 - fy])[~lower]
        cells = np.column_stack([base, base + 1, base + len(x), base + len(x) + 1])
        inside = (fx >= 0) & (fx <= 1) & (fy >= 0) & (fy <= 1) & domain[cells].all(axis=1)
        reverse = BarycentricMap(
            xx.size, vertices, weights, inside, np.empty((0, 3), dtype=int), np.empty(0, dtype=bool)
        )
        return cls(x, y, forward, reverse, points.copy())

    def forward(self, values: Any, observation_mask: Any = None) -> tuple[np.ndarray, np.ndarray]:
        values, support = self.point_to_grid.forward(values, observation_mask)
        return values.reshape(len(self.y), len(self.x), -1), support.reshape(len(self.y), len(self.x))

    def to_points(
        self, grid_values: Any, grid_mask: Any = None, *, observation_mask: Any = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate without crossing declared source gaps or missing observations.

        Pass the original observation_mask when some source values were missing;
        a nodal grid_mask cannot describe an unsupported sub-cell source region.
        """
        array = np.asarray(grid_values)
        if array.ndim != 3 or array.shape[:2] != (len(self.y), len(self.x)):
            raise ValueError("grid_values must have shape (len(y),len(x),C).")
        support = self.point_to_grid.support.copy()
        if grid_mask is not None:
            if np.asarray(grid_mask).shape != array.shape[:2]:
                raise ValueError("grid_mask must match the target spatial grid shape.")
            support &= _mask(np.asarray(grid_mask).reshape(-1), len(support), "grid_mask")
        result, valid = self.grid_to_point.forward(array.reshape(-1, array.shape[-1]), support)
        observed = _mask(observation_mask, len(self.source_points), "observation_mask")
        triangles = self.point_to_grid.simplices
        forbidden = ~self.point_to_grid.simplex_support | ~observed[triangles].all(axis=1)
        if forbidden.any():
            source_support = _TriangleSupport(self.source_points[triangles[forbidden]])
            permitted_support = _TriangleSupport(self.source_points[triangles[~forbidden]])
            xx, yy = np.meshgrid(self.x, self.y, indexing="xy")
            grid_points = np.column_stack((xx.ravel(), yy.ravel()))
            checked: dict[tuple[int, ...], bool] = {}
            for index in np.flatnonzero(valid):
                required = self.grid_to_point.weights[index] > 1e-13
                stencil = tuple(sorted(self.grid_to_point.vertices[index][required].tolist()))
                if stencil not in checked:
                    vertices = grid_points[list(stencil)]
                    checked[stencil] = (
                        source_support.crosses_triangle(vertices)
                        if len(stencil) == 3
                        else not permitted_support.covers_segment(vertices)
                        if len(stencil) == 2
                        else False
                    )
                if checked[stencil]:
                    valid[index] = False
            result[~valid] = np.nan
        return result, valid


def point_scores(
    observed: Any,
    predicted: Any,
    support: Any,
    *,
    observation_mask: Any = None,
    point_weights: Any = None,
) -> dict[str, Any]:
    """Component-wise point errors and coverage. Weights are never inferred areas."""
    truth, values = _real(observed, "observed"), _real(predicted, "predicted")
    if truth.ndim != 2 or values.shape != truth.shape:
        raise ValueError("Observed and predicted point values must have matching (M,C) shapes.")
    known = _mask(observation_mask, len(truth), "observation_mask")
    valid = known & _mask(support, len(truth), "support")
    weights = np.ones(len(truth)) if point_weights is None else _real(point_weights, "point_weights")
    if weights.shape != (len(truth),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("point_weights must be finite, non-negative, and have shape (M,).")
    if not np.isfinite(truth[known]).all() or not np.isfinite(values[valid]).all():
        raise ValueError("Known observations and supported predictions must be finite.")
    known = known & (weights > 0)
    valid = valid & (weights > 0)
    result: dict[str, Any] = {
        "known_points": int(known.sum()),
        "scored_points": int(valid.sum()),
        "coverage": float(valid.sum() / known.sum()) if known.any() else 0.0,
        "weighting": "point_mean" if point_weights is None else "explicit_point_weights",
        "rmse": None,
        "mae": None,
    }
    if valid.any():
        error = values[valid] - truth[valid]
        selected_weights = weights[valid]
        selected_weights = selected_weights / selected_weights.max()
        result["rmse"] = np.sqrt(np.average(error**2, axis=0, weights=selected_weights)).tolist()
        result["mae"] = np.average(np.abs(error), axis=0, weights=selected_weights).tolist()
    return result
