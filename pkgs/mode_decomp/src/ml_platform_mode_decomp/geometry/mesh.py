"""Mesh domain helpers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ml_platform_mode_decomp.config import cfg_get


def compute_vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    verts = _normalize_vertices(vertices)
    face_arr = _normalize_faces(faces, verts.shape[0])
    v0 = verts[face_arr[:, 0]]
    v1 = verts[face_arr[:, 1]]
    v2 = verts[face_arr[:, 2]]
    if verts.shape[1] == 2:
        a = v1 - v0
        b = v2 - v0
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        area = 0.5 * np.abs(cross)
    else:
        cross = np.cross(v1 - v0, v2 - v0)
        area = 0.5 * np.linalg.norm(cross, axis=1)
    if not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("mesh faces must have finite positive area")
    weights = np.zeros(verts.shape[0], dtype=np.float64)
    share = area / 3.0
    for idx in range(3):
        np.add.at(weights, face_arr[:, idx], share)
    return weights


def mesh_measure(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Use complete active triangles to define topology and lumped physical mass.

    Vertex indices retain the original field layout. A requested vertex without
    an incident active triangle has zero measure and is outside the domain.
    Observation masks must not be passed here: they restrict projection data,
    rather than changing the fixed physical mesh.
    """
    vertices = _normalize_vertices(vertices)
    faces = _normalize_faces(faces, len(vertices))
    mask = (
        np.ones(len(vertices), dtype=bool)
        if vertex_mask is None
        else _normalize_vertex_mask(vertex_mask, len(vertices), "domain mask")
    )
    active_faces = faces[mask[faces].all(axis=1)]
    if not len(active_faces):
        raise ValueError("mesh domain mask removed all faces")
    weights = compute_vertex_areas(vertices, active_faces)
    active_vertices = mask & (weights > 0)
    return active_faces, None if active_vertices.all() else active_vertices, weights


def compact_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Return the measured active mesh with compact indices for linear algebra."""
    active_faces, mask, weights = mesh_measure(vertices, faces, vertex_mask)
    if mask is None:
        return np.asarray(vertices), active_faces, None, weights
    indices = np.flatnonzero(mask)
    mapping = np.full(len(vertices), -1, dtype=np.int64)
    mapping[indices] = np.arange(len(indices))
    return np.asarray(vertices)[indices], mapping[active_faces], indices, weights[indices]


def _normalize_vertices(vertices: Any) -> np.ndarray:
    verts = np.asarray(vertices)
    if verts.ndim != 2 or verts.shape[1] not in {2, 3}:
        raise ValueError(f"mesh vertices must be (N,2) or (N,3), got shape {verts.shape}")
    if verts.dtype.kind not in "fiu" or not np.isfinite(verts).all():
        raise ValueError("mesh vertices must contain finite real coordinates")
    return verts.astype(np.float64, copy=False)


def _normalize_faces(faces: Any, n_vertices: int) -> np.ndarray:
    face_arr = np.asarray(faces)
    if face_arr.ndim != 2 or face_arr.shape[1] != 3:
        raise ValueError(f"mesh faces must be (F, 3), got shape {face_arr.shape}")
    if face_arr.size == 0:
        raise ValueError("mesh faces must be non-empty")
    if (
        face_arr.dtype.kind not in "fiu"
        or not np.isfinite(face_arr).all()
        or not np.equal(face_arr, np.floor(face_arr)).all()
    ):
        raise ValueError("mesh faces must contain finite integer vertex indices")
    min_idx = int(face_arr.min())
    max_idx = int(face_arr.max())
    if min_idx < 0 or max_idx >= n_vertices:
        raise ValueError(f"mesh faces index out of range: [{min_idx}, {max_idx}] vs {n_vertices}")
    return face_arr.astype(np.int64, copy=False)


def _normalize_vertex_mask(mask: Any, n_vertices: int, label: str) -> np.ndarray:
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 2 and mask_arr.shape[1] == 1:
        mask_arr = mask_arr[:, 0]
    if mask_arr.ndim != 1:
        raise ValueError(f"{label} must be 1D (N,) or (N,1), got shape {mask_arr.shape}")
    if mask_arr.shape[0] != n_vertices:
        raise ValueError(f"{label} length {mask_arr.shape[0]} does not match vertices {n_vertices}")
    if mask_arr.dtype.kind not in "bui" or not np.isin(mask_arr, (0, 1)).all():
        raise ValueError(f"{label} must contain boolean or 0/1 values")
    return mask_arr.astype(bool)


def mesh_inputs(
    domain_cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Normalize in-memory mesh arrays; file resolution belongs to data loading."""
    if any(key in domain_cfg for key in ("mesh_path", "vertex_mask_path", "mask_path", "mesh")):
        raise ValueError("Numerical mesh geometry accepts arrays; resolve file references through data loading.")
    vertices = _normalize_vertices(cfg_get(domain_cfg, "vertices", None))
    faces = _normalize_faces(cfg_get(domain_cfg, "faces", None), len(vertices))
    mask_value = cfg_get(domain_cfg, "vertex_mask", cfg_get(domain_cfg, "mask", None))
    mask = None if mask_value is None else _normalize_vertex_mask(mask_value, len(vertices), "mask")
    return vertices, faces, mask, {}
