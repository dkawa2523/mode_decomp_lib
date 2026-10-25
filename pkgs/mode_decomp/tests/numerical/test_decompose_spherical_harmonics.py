from __future__ import annotations


import numpy as np
import pytest

from ml_platform_mode_decomp.geometry import build_domain_spec
from ml_platform_mode_decomp.coefficients.codecs import build_coeff_codec
from ml_platform_mode_decomp.decomposition import build_decomposer


def _domain_cfg() -> dict[str, object]:
    return {
        "name": "sphere_grid",
        "lat_range": [-90.0, 90.0],
        "lon_range": [0.0, 360.0],
        "angle_unit": "deg",
        "radius": 1.0,
    }


def test_spherical_harmonics_roundtrip_constant() -> None:
    field = np.ones((6, 12), dtype=np.float64)
    domain = build_domain_spec(_domain_cfg(), field.shape)

    decomposer = build_decomposer(
        {
            "name": "spherical_harmonics",
            "l_max": 0,
            "real_form": True,
            "norm": "ortho",
            "backend": "scipy",
        }
    )
    codec = build_coeff_codec({"name": "none", "dtype_policy": "float32"})

    raw_coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    raw_meta = decomposer.coeff_meta()
    vector = codec.encode(raw_coeff, raw_meta)
    raw_back = codec.decode(vector, raw_meta)
    field_hat = decomposer.inverse_transform(raw_back, domain_spec=domain)

    assert field_hat.shape == field.shape
    assert np.allclose(field_hat, field, atol=1e-5)


def test_spherical_harmonics_scipy_basis_matches_analytic_l1_modes() -> None:
    field = np.ones((6, 12), dtype=np.float64)
    domain = build_domain_spec(_domain_cfg(), field.shape)
    decomposer = build_decomposer(
        {
            "name": "spherical_harmonics",
            "l_max": 1,
            "real_form": True,
            "norm": "ortho",
            "backend": "scipy",
        }
    )

    # Private access is OK in tests: we explicitly want to validate the basis implementation.
    basis = decomposer._get_basis(domain)  # type: ignore[attr-defined]
    lm_kind_list = decomposer._lm_kind_list  # type: ignore[attr-defined]

    theta = np.asarray(domain.coords["theta"], dtype=np.float64)
    phi = np.asarray(domain.coords["phi"], dtype=np.float64)
    # The real l=1,m=1 modes independently fix azimuth/polar angle order.
    scale = np.sqrt(3.0 / (4.0 * np.pi))
    expected_cos = scale * np.sin(phi) * np.cos(theta)
    expected_sin = scale * np.sin(phi) * np.sin(theta)

    idx_cos = lm_kind_list.index([1, 1, "cos"])
    idx_sin = lm_kind_list.index([1, 1, "sin"])
    assert np.allclose(basis[idx_cos], expected_cos, atol=1e-12)
    assert np.allclose(basis[idx_sin], expected_sin, atol=1e-12)


def test_spherical_harmonics_rejects_unimplemented_pyshtools_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        build_decomposer(
            {
                "name": "spherical_harmonics",
                "l_max": 1,
                "backend": "pyshtools",
            }
        )
