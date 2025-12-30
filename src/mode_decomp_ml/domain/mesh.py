"""Mesh domain helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _cfg_get(cfg: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


def _normalize_vertices(vertices: Any) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] not in {2, 3}:
        raise ValueError(f"mesh vertices must be (N,2) or (N,3), got shape {verts.shape}")
    return verts


def _normalize_faces(faces: Any, n_vertices: int) -> np.ndarray:
    face_arr = np.asarray(faces, dtype=np.int64)
    if face_arr.ndim != 2 or face_arr.shape[1] != 3:
        raise ValueError(f"mesh faces must be (F, 3), got shape {face_arr.shape}")
    if face_arr.size == 0:
        raise ValueError("mesh faces must be non-empty")
    min_idx = int(face_arr.min())
    max_idx = int(face_arr.max())
    if min_idx < 0 or max_idx >= n_vertices:
        raise ValueError(f"mesh faces index out of range: [{min_idx}, {max_idx}] vs {n_vertices}")
    return face_arr


def _normalize_vertex_mask(mask: Any, n_vertices: int, label: str) -> np.ndarray:
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 2 and mask_arr.shape[1] == 1:
        mask_arr = mask_arr[:, 0]
    if mask_arr.ndim != 1:
        raise ValueError(f"{label} must be 1D (N,) or (N,1), got shape {mask_arr.shape}")
    if mask_arr.shape[0] != n_vertices:
        raise ValueError(f"{label} length {mask_arr.shape[0]} does not match vertices {n_vertices}")
    return mask_arr.astype(bool)


def _load_mesh_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as data:
        if "vertices" not in data or "faces" not in data:
            raise ValueError("mesh_path .npz must contain vertices and faces arrays")
        vertices = np.asarray(data["vertices"])
        faces = np.asarray(data["faces"])
        mask = None
        if "vertex_mask" in data:
            mask = np.asarray(data["vertex_mask"])
        elif "mask" in data:
            mask = np.asarray(data["mask"])
    return vertices, faces, mask


def load_mesh_inputs(
    domain_cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    mesh_path = _cfg_get(domain_cfg, "mesh_path", None)
    vertices_inline = _cfg_get(domain_cfg, "vertices", None)
    faces_inline = _cfg_get(domain_cfg, "faces", None)
    if mesh_path is not None and (vertices_inline is not None or faces_inline is not None):
        raise ValueError("mesh_path cannot be combined with vertices/faces")
    if mesh_path is None and (vertices_inline is None or faces_inline is None):
        raise ValueError("mesh requires mesh_path or vertices/faces")

    if mesh_path is not None:
        mesh_path = Path(str(mesh_path))
        if not mesh_path.exists():
            raise FileNotFoundError(f"mesh_path not found: {mesh_path}")
        vertices, faces, mask_from_mesh = _load_mesh_npz(mesh_path)
        mesh_source = "file"
    else:
        vertices = vertices_inline
        faces = faces_inline
        mask_from_mesh = None
        mesh_source = "inline"

    vertices = _normalize_vertices(vertices)
    faces = _normalize_faces(faces, vertices.shape[0])

    mask_inline = _cfg_get(domain_cfg, "vertex_mask", None)
    if mask_inline is None:
        mask_inline = _cfg_get(domain_cfg, "mask", None)
    mask_path = _cfg_get(domain_cfg, "vertex_mask_path", None)
    if mask_path is None:
        mask_path = _cfg_get(domain_cfg, "mask_path", None)

    if mask_inline is not None and mask_path is not None:
        raise ValueError("Provide only one of vertex_mask/mask or vertex_mask_path/mask_path")

    mask = None
    mask_source = None
    if mask_path is not None:
        mask_path = Path(str(mask_path))
        if not mask_path.exists():
            raise FileNotFoundError(f"mask_path not found: {mask_path}")
        loaded = np.load(mask_path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            loaded.close()
            raise ValueError("mask_path must point to a .npy array, not a .npz archive")
        mask = _normalize_vertex_mask(loaded, vertices.shape[0], "mask")
        mask_source = "file"
    elif mask_inline is not None:
        mask = _normalize_vertex_mask(mask_inline, vertices.shape[0], "mask")
        mask_source = "inline"
    elif mask_from_mesh is not None:
        mask = _normalize_vertex_mask(mask_from_mesh, vertices.shape[0], "mesh mask")
        mask_source = "mesh_file"

    meta: dict[str, Any] = {"mesh_source": mesh_source}
    if mesh_path is not None:
        meta["mesh_path"] = str(mesh_path)
    if mask_source is not None:
        meta["mask_source"] = mask_source
    if mask_path is not None:
        meta["mask_path"] = str(mask_path)

    return vertices, faces, mask, meta
