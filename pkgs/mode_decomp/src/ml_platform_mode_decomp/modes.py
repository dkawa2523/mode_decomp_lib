"""Inspect synthesis coordinates and reconstruct selected ones on a fitted geometry.

The default space is packed A, before coefficient postprocessing. These functions
do not equate coefficient squares with physical energy, assume orthogonality or
linear independence, or order modes by importance. Synthesis responses are
generated only when requested.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from operator import index as integer_index
from typing import TYPE_CHECKING, Any

import numpy as np

from .coefficients.codecs.real_fft import RealFFTCodec

if TYPE_CHECKING:
    from .representation import FieldRepresentation

_LINEAR_CODECS = {
    "none",
    "real_fft",
    "wavelet_pack_v1",
}
_AFFINE_POSTS = {"none", "standardize", "pca", "dict_learning"}
_COMPOSITES = {"offset_residual"}
_COMPONENTS = {"helmholtz", "hodge"}


def _dimension(representation: FieldRepresentation, space: str) -> int:
    representation._require_fitted()
    if space not in {"a", "z"}:
        raise ValueError("space must be 'a' or 'z'")
    return representation.packed_dim if space == "a" else representation.target_dim


def _indices(indices: Sequence[int] | None, dimension: int) -> list[int]:
    if indices is None:
        return list(range(dimension))
    values = []
    for value in indices:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Mode indices must be integers")
        position = integer_index(value)
        if not 0 <= position < dimension:
            raise ValueError(f"Mode index {position} is outside [0,{dimension})")
        values.append(position)
    if len(set(values)) != len(values):
        raise ValueError("Mode indices must be unique")
    return values


def _codec_meta(representation: FieldRepresentation) -> dict[str, Any]:
    # A broadcast view supplies the vector size; listing metadata allocates no basis or dense vector.
    return dict(
        representation.codec.coeff_meta(representation.raw_meta, np.broadcast_to(0.0, (representation.packed_dim,)))
    )


def _semantics(representation: FieldRepresentation, space: str, codec: dict[str, Any]) -> tuple[str, bool]:
    method = representation.raw_meta["method"]
    codec_linear = codec.get("name") in _LINEAR_CODECS or (
        codec.get("name") == "fft_complex_codec_v1" and codec.get("mode") == "real_imag"
    )
    field_affine = representation.preprocess.name in {"none", "basic"}
    layout = getattr(representation, "layout", None)
    synthesis = (
        layout.synthesis_kind if layout is not None else getattr(representation.decomposer, "synthesis_kind", "unknown")
    )
    additive = (
        synthesis in {"linear", "affine", "component"}
        and codec_linear
        and field_affine
        and (space == "a" or representation.post.name in _AFFINE_POSTS)
    )
    if space == "z" or method in {"autoencoder", "quadratic_manifold"}:
        return "latent_coordinate", additive
    if method in _COMPOSITES:
        return "composite_coordinate", False
    if not additive:
        return "perturbation_coordinate", False
    if method in _COMPONENTS:
        return "component_coordinate", True
    if codec.get("name") == "real_fft":
        return "fourier_mode", True
    return "basis_coordinate", True


def list_modes(
    representation: FieldRepresentation, indices: Sequence[int] | None = None, *, space: str = "a"
) -> list[dict[str, Any]]:
    """List coordinates, their interpretation and available method-specific labels.

    additive describes affine *synthesis about inverse(0)*, not an orthogonal
    projection or energy decomposition. Z coordinates are always labelled latent,
    even if a linear PCA inverse makes their synthesis additive.
    """
    selected = _indices(indices, _dimension(representation, space))
    packed_meta = _codec_meta(representation)
    codec = packed_meta["codec"]
    interpretation, additive = _semantics(representation, space, codec)
    result = [
        {
            "index": i,
            "id": f"{space}:{i}",
            "space": space,
            "method": representation.raw_meta["method"],
            "interpretation": interpretation,
            "additive": additive,
            "reference_required": not additive,
            "field_units": list(representation.field_units),
            "response_definition": "inverse(reference + amplitude*e_i) - inverse(reference)",
        }
        for i in selected
    ]
    if space == "z":
        for item in result:
            item["coefficient_post"] = representation.post.name
        return result
    if codec["name"] == "real_fft":
        coordinates = RealFFTCodec.coordinate_metadata(representation.raw_meta, selected)
        for item, coordinate in zip(result, coordinates, strict=True):
            item.update(coordinate)
            _fourier_labels(representation, item)
    else:
        _raw_labels(representation, result, packed_meta)
    return result


def _raw_labels(representation: FieldRepresentation, result: list[dict[str, Any]], packed_meta: dict[str, Any]):
    raw = representation.raw_meta
    if raw.get("coeff_format") == "component_blocks_v1":
        offsets = np.asarray(raw["component_offsets"])
        for item in result:
            channel = int(np.searchsorted(offsets[1:], item["index"], side="right"))
            _channel_label(representation, item, channel)
            item["mode_index"] = int(item["index"] - offsets[channel])
            item["raw_index"] = [item["index"]]
            item["mode_order"] = "descending_training_energy_within_component"
        return
    shape = tuple(packed_meta.get("coeff_shape", ()))
    layout = packed_meta.get("coeff_layout", "")
    order = raw.get("flatten_order", "C")
    # Special structured formats may have no one-to-one array index. Do not invent one.
    direct = packed_meta["codec"]["name"] in _LINEAR_CODECS - {"real_fft", "wavelet_pack_v1"}
    if not direct or not shape or int(np.prod(shape)) != representation.packed_dim:
        return
    for item in result:
        position = [int(x) for x in np.unravel_index(item["index"], shape, order="F" if order == "F" else "C")]
        item["raw_index"] = position
        if raw["method"] == "tucker":
            item["core_index"] = dict(zip(("y", "x", "component"), position, strict=True))
            item["component_interpretation"] = "learned_component_factor; may mix physical channels"
        if raw.get("parts") is not None and raw.get("part_axis") is not None:
            part = position[raw["part_axis"]]
            item["part"] = raw["parts"][part]
        joint = raw["method"] in {"pod_joint", "pod_joint_em"}
        if layout.startswith("C") and not joint:
            _channel_label(representation, item, position[0])
        if layout.endswith("K"):
            item["mode_index"] = position[-1]
            for key in ("nm_list", "nm_kind_list", "lm_kind_list"):
                if raw.get(key) is not None:
                    item[key.removesuffix("_list")] = raw[key][position[-1]]
        if raw.get("mode_order") is not None:
            item["mode_order"] = raw["mode_order"]


def _channel_label(representation: FieldRepresentation, item: dict[str, Any], channel: int):
    item["channel"] = channel
    if representation.components:
        item["component_name"] = representation.components[channel]


def _coordinate_unit(representation: FieldRepresentation, axis: str) -> str | None:
    units = representation.geometry.meta.get("coordinate_units", representation.geometry.meta.get("units"))
    value = units.get(axis) if isinstance(units, dict) else units
    return value if isinstance(value, str) else None


def _fourier_labels(representation: FieldRepresentation, item: dict[str, Any]):
    raw = representation.raw_meta
    original_shape = raw.get("orig_hw")
    if original_shape is None:
        original_shape = raw["coeff_shape"][-2:]
    height, width = (int(x) for x in original_shape)
    retained_h, retained_w = raw["coeff_shape"][-2:]
    stored_y, stored_x = item["raw_index"][-2:]
    bins = []
    for original, retained, stored in ((height, retained_h, stored_y), (width, retained_w, stored_x)):
        frequencies = np.rint(np.fft.fftfreq(original) * original).astype(int)
        if raw.get("fft_shift", False):
            frequencies = np.fft.fftshift(frequencies)
            stored += original // 2 - retained // 2
        bins.append(int(frequencies[stored]))
    ky, kx = bins
    coords = representation.geometry.coords
    dx = float(coords["x"][0, 1] - coords["x"][0, 0]) if width > 1 else None
    dy = float(coords["y"][1, 0] - coords["y"][0, 0]) if height > 1 else None
    fx = kx / (width * dx) if dx is not None else 0.0
    fy = ky / (height * dy) if dy is not None else 0.0
    self_conjugate = item["self_conjugate"]
    kind = (
        "dc"
        if kx == ky == 0
        else "nyquist"
        if self_conjugate
        else ("cosine" if item["component"] == "real" else "sine")
    )
    item.update(
        {
            "kx": kx,
            "ky": ky,
            "kind": kind,
            "nyquist_axes": [
                axis for axis, k, n in (("x", kx, width), ("y", ky, height)) if n % 2 == 0 and k == -n // 2
            ],
            "frequency_x": fx,
            "frequency_y": fy,
            "angular_wavenumber_x": 2 * np.pi * fx,
            "angular_wavenumber_y": 2 * np.pi * fy,
            "period_x": width * dx if dx is not None else None,
            "period_y": height * dy if dy is not None else None,
            "phase_origin": [float(coords["x"][0, 0]), float(coords["y"][0, 0])],
            "phase_sign": -1 if item["component"] == "imag" else 1,
            "basis_amplitude_before_field_inverse": (1.0 if self_conjugate else np.sqrt(2)) / np.sqrt(height * width),
            "basis_units_before_field_inverse": "1",
            "fourier_basis_domain": "ambient_periodic_grid_before_domain_mask",
            "fft_periodic_extension": True,
            "declared_periodic_axes": representation.geometry.meta.get("periodic_axes", []),
        }
    )
    for axis in ("x", "y"):
        unit = _coordinate_unit(representation, axis)
        item[f"coordinate_unit_{axis}"] = unit
        item[f"frequency_unit_{axis}"] = f"cycles/{unit}" if unit is not None else "cycles/coordinate_unit"
    _channel_label(representation, item, item["raw_index"][0] if len(item["raw_index"]) == 3 else 0)
    channel = item["channel"]
    field_unit = representation.field_units[channel] if representation.field_units else None
    preprocess_ops = representation.preprocess.state().get("ops", [])
    item["coefficient_unit"] = "1" if preprocess_ops else field_unit
    item["field_response_unit_per_coefficient_unit"] = field_unit if preprocess_ops else "1"


def _batch(values: np.ndarray, dimension: int) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "fiu" or array.ndim != 2 or array.shape[1] != dimension or not np.isfinite(array).all():
        raise ValueError(f"Expected finite coefficient batch with shape (N,{dimension})")
    return array


def _reference(reference: np.ndarray | None, count: int, dimension: int) -> np.ndarray:
    if reference is None:
        return np.zeros((count, dimension))
    reference = np.asarray(reference)
    if reference.ndim == 1:
        reference = reference[None, :]
    reference = _batch(reference, dimension).astype(np.float64, copy=False)
    if len(reference) not in {1, count}:
        raise ValueError("Reference must be one coordinate vector or one vector per sample")
    return np.broadcast_to(reference, (count, dimension)).copy()


def mode_fields(
    representation: FieldRepresentation,
    indices: Sequence[int],
    *,
    space: str = "a",
    reference: np.ndarray | None = None,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Return requested field responses (M,*field_shape), with the affine baseline removed.

    For fixed affine synthesis these are amplitude times its basis directions.
    Otherwise an explicit reference is required and responses are finite coordinate
    perturbations, not a global basis or a derivative. The result uses original
    field units. Existing means from preprocessing/POD are never repeated per mode.
    """
    selected = _indices(indices, _dimension(representation, space))
    fields = None
    for index, response in enumerate(
        _mode_responses(representation, selected, space=space, reference=reference, amplitude=amplitude)
    ):
        if fields is None:
            fields = np.empty((len(selected), *representation.field_shape), dtype=response.dtype)
        fields[index] = response
    return fields if fields is not None else np.empty((0, *representation.field_shape))


def _mode_responses(representation, selected, *, space="a", reference=None, amplitude=1.0):
    """Yield validated selected coordinates without retaining a field per mode."""
    dimension = _dimension(representation, space)
    additive = _semantics(representation, space, _codec_meta(representation)["codec"])[1]
    if not additive and reference is None:
        raise ValueError("Nonlinear or composite coordinate responses require an explicit reference")
    if not np.isfinite(amplitude) or amplitude == 0:
        raise ValueError("amplitude must be finite and non-zero")
    anchor = _reference(reference, 1, dimension)
    if not selected:
        return
    baseline = representation.inverse_transform(anchor, space=space)[0]
    for position in selected:
        values = anchor.copy()
        values[0, position] += amplitude
        yield representation.inverse_transform(values, space=space)[0] - baseline


def reconstruct_modes(
    representation: FieldRepresentation,
    coefficients: np.ndarray,
    indices: Sequence[int],
    *,
    space: str = "a",
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct with selected coordinate values; other values equal reference (default zero).

    This is coordinate selection, including for latent/composite coordinates. It is
    not a physical projection or an additive component claim for nonlinear inverses.
    """
    dimension = _dimension(representation, space)
    selected = _indices(indices, dimension)
    values = _batch(coefficients, dimension)
    retained = _reference(reference, len(values), dimension)
    retained[:, selected] = values[:, selected]
    return representation.inverse_transform(retained, space=space)


def cumulative_reconstruct(
    representation: FieldRepresentation,
    coefficients: np.ndarray,
    order: Sequence[int],
    *,
    space: str = "a",
    reference: np.ndarray | None = None,
) -> Iterator[np.ndarray]:
    """Yield reconstruction after each requested coordinate, in explicit caller order.

    No field stack for all prefixes is allocated. With nonorthogonal/nonlinear
    synthesis a prefix's physical error need not decrease monotonically.
    """
    dimension = _dimension(representation, space)
    selected = _indices(order, dimension)
    values = _batch(coefficients, dimension)
    retained = _reference(reference, len(values), dimension)
    for position in selected:
        retained[:, position] = values[:, position]
        yield representation.inverse_transform(retained, space=space)
