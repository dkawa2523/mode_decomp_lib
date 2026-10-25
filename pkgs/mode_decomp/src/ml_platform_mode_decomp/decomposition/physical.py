"""Physical component operations on supplied fields, independent of learned bases."""

import numpy as np

from ..geometry.identity import geometry_fingerprint
from ..geometry import validate_decomposer_compatibility
from .hodge import HodgeDecomposer, validate_hodge_geometry
from .helmholtz import HelmholtzDecomposer, _grid_spacing


def validate_components(spec, geometry, *, method="hodge", params=None):
    """Check a requested physical operation without building its cochain matrices."""
    if method == "hodge":
        operator = HodgeDecomposer(**(params or {}))
        operator.validate_field_spec(spec)
        validate_hodge_geometry(geometry)
    elif method == "helmholtz":
        operator = HelmholtzDecomposer(**(params or {}))
        operator.validate_field_spec(spec)
        validate_decomposer_compatibility(geometry, operator._cfg)
        _grid_spacing(geometry)
        if operator._mask_policy == "error" and geometry.mask is not None and not geometry.mask.all():
            raise ValueError(
                "helmholtz mask_policy=error requires the full periodic grid; zero_fill is an ambient extension"
            )
    else:
        raise ValueError("Physical components.method must be hodge or helmholtz")


def analyze_components(spec, geometry, fields, sample_ids, *, method="hodge", params=None):
    validate_components(spec, geometry, method=method, params=params)
    fields, sample_ids = np.asarray(fields), np.asarray(sample_ids, dtype=str)
    if sample_ids.ndim != 1 or not len(sample_ids) or fields.shape != (len(sample_ids), *spec.shape):
        raise ValueError("Component fields and sample IDs must form a nonempty matching batch")
    if method == "hodge":
        arrays, rows, summary = _hodge_components(geometry, fields, sample_ids, params or {})
    elif method == "helmholtz":
        arrays, rows, summary = _helmholtz_components(geometry, fields, sample_ids, params or {})
    else:
        raise ValueError("Physical components.method must be hodge or helmholtz")
    arrays.update(
        sample_ids=sample_ids,
        components=np.asarray(spec.components, dtype=str),
        field_units=np.asarray(spec.units, dtype=str),
    )
    summary.update(
        method=method,
        sample_count=len(sample_ids),
        components=list(spec.components),
        field_units=list(spec.units),
        geometry_sha256=geometry_fingerprint(geometry),
        coordinate_units=geometry.meta.get("coordinate_units", geometry.meta.get("units")),
        coordinate_source=geometry.meta.get("coordinate_source"),
        weight_measure=geometry.meta.get("weight_measure"),
    )
    return arrays, rows, summary


def _hodge_components(geometry, fields, sample_ids, params):
    operator = HodgeDecomposer(**params).fit(domain_spec=geometry)
    hodge = operator.hodge
    assert hodge is not None
    names = (
        "edge_sampled",
        "edge_exact",
        "edge_coexact",
        "edge_harmonic",
        "node_exact_edge_lift",
        "node_coexact_edge_lift",
        "node_harmonic_edge_lift",
        "node_sampling_residual",
    )
    arrays, rows = {name: [] for name in names}, []
    for sample_id, field in zip(sample_ids, fields, strict=True):
        flow = hodge.sample_nodes(field)
        parts = hodge.decompose_edges(flow)
        lifted = [hodge.lift_edges(parts[name]) for name in hodge.parts]
        residual = np.zeros((*hodge.grid_shape, 2))
        residual[hodge.mask] = field[hodge.mask] - np.sum(lifted, axis=0)[hodge.mask]
        arrays["edge_sampled"].append(flow)
        for name, node_values in zip(hodge.parts, lifted, strict=True):
            arrays[f"edge_{name}"].append(parts[name])
            arrays[f"node_{name}_edge_lift"].append(node_values)
        arrays["node_sampling_residual"].append(residual)
        energies = {f"{name}_edge_energy": hodge.energy(parts[name]) for name in hodge.parts}
        rows.append(
            {
                "sample_id": str(sample_id),
                "input_edge_energy": hodge.energy(flow),
                **energies,
                "sum_component_edge_energy": sum(energies.values()),
                "node_sampling_residual_norm": float(
                    np.sqrt(np.sum(hodge.node_weights[:, None] * residual.reshape(-1, 2)[hodge.node_indices] ** 2))
                ),
            }
        )
    arrays = {name: np.stack(values) for name, values in arrays.items()}
    arrays.update(
        domain_mask=hodge.mask,
        node_flat_indices=hodge.node_indices,
        node_coordinates=hodge.vertex_coordinates,
        node_weights=hodge.node_weights,
        edge_vertex_indices=hodge.edge_vertices,
        edge_axes=hodge.edge_axes,
        edge_lengths=hodge.edge_lengths,
        edge_weights=hodge.edge_weights,
        face_origins=hodge.face_origins,
    )
    summary = {
        "boundary": "natural",
        "native_parts": list(hodge.parts),
        "native_inner_product": "edge_lumped_domain_area",
        "physical_guarantee_scope": "native_edge_cochains_resampled_from_supplied_fields",
        "energy_definition": "sum(edge_weights * edge_integrals**2); no density or one-half factor",
        "node_output_interpretation": "three edge-component lifts plus sampling_residual; node orthogonality is not claimed",
        "node_sampling_residual_norm": "sqrt(sum(node_area * sum_cartesian_components(residual**2)))",
        "boundary_interpretation": "full finite cochains; no prescribed wall velocity, flux or Dirichlet/Neumann data",
        "harmonic_dimension": hodge.harmonic_dimension,
        "complex_size": {
            "vertices": len(hodge.node_indices),
            "edges": len(hodge.edge_weights),
            "faces": hodge.d1.shape[0],
        },
        "edge_vertex_indexing": "indices into node_coordinates; node_flat_indices map to flattened H/W storage",
        "edge_orientation": "positive x or positive y; edge_axes: x=0,y=1",
    }
    return arrays, rows, summary


def _helmholtz_components(geometry, fields, sample_ids, params):
    operator = HelmholtzDecomposer(**params)
    parts = np.stack([operator.transform(field, None, geometry) for field in fields])
    arrays = {
        "node_curl_free": parts[:, 0],
        "node_div_free": parts[:, 1],
        "x": geometry.coords["x"],
        "y": geometry.coords["y"],
    }
    if geometry.mask is not None:
        arrays["domain_mask"] = geometry.mask
    rows = []
    for index, sample_id in enumerate(sample_ids):
        energies = {
            f"{name}_grid_energy": float(np.sum(parts[index, axis] ** 2))
            for axis, name in enumerate(("curl_free", "div_free"))
        }
        rows.append(
            {
                "sample_id": str(sample_id),
                "input_grid_energy": float(np.sum(parts[index].sum(axis=0) ** 2)),
                **energies,
                "sum_component_grid_energy": sum(energies.values()),
            }
        )
    summary = {
        "boundary": "periodic",
        "native_parts": ["curl_free", "div_free"],
        "native_inner_product": "euclidean_grid",
        "physical_guarantee_scope": "periodic_uniform_grid_projection_of_supplied_fields",
        "energy_definition": "sum(Cartesian component values**2); no density, area or one-half factor",
        "boundary_interpretation": "periodic ambient grid; zero_fill is an extension, not a nonperiodic wall solution",
        "derivative_convention": operator.coeff_meta()["derivative_convention"],
        "derivative_nullspace_assignment": "div_free",
        "mask_policy": operator._mask_policy,
    }
    return arrays, rows, summary
