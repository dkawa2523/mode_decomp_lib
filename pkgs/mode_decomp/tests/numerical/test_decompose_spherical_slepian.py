from __future__ import annotations

import pathlib

from ml_platform_core.io import dump_joblib, load_joblib

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


def _cap_cfg() -> dict[str, float | str]:
    return {"lat": 0.0, "lon": 0.0, "radius": 90.0, "angle_unit": "deg"}


def test_spherical_slepian_scipy_real_sh_basis_matches_analytic_l1_modes() -> None:
    field = np.ones((6, 12), dtype=np.float64)
    domain = build_domain_spec(_domain_cfg(), field.shape)

    import ml_platform_mode_decomp.decomposition.spherical_slepian as mod

    theta = np.asarray(domain.coords["theta"], dtype=np.float64)
    phi = np.asarray(domain.coords["phi"], dtype=np.float64)
    basis, lm_kind_list = mod._build_real_sh_basis(l_max=1, theta=theta, phi=phi)

    # The real l=1,m=1 modes independently fix azimuth/polar angle order.
    scale = np.sqrt(3.0 / (4.0 * np.pi))
    expected_cos = scale * np.sin(phi) * np.cos(theta)
    expected_sin = scale * np.sin(phi) * np.sin(theta)

    idx_cos = lm_kind_list.index([1, 1, "cos"])
    idx_sin = lm_kind_list.index([1, 1, "sin"])
    assert np.allclose(basis[idx_cos], expected_cos, atol=1e-12)
    assert np.allclose(basis[idx_sin], expected_sin, atol=1e-12)


def test_spherical_slepian_roundtrip_constant() -> None:
    field = np.ones((6, 12), dtype=np.float64)
    domain = build_domain_spec(_domain_cfg(), field.shape)

    decomposer = build_decomposer(
        {
            "name": "spherical_slepian",
            "l_max": 1,
            "k": 4,
            "backend": "scipy",
            "region_cap": _cap_cfg(),
        }
    )
    decomposer.fit(dataset=None, domain_spec=domain)
    codec = build_coeff_codec({"name": "none", "dtype_policy": "float32"})

    raw_coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    raw_meta = decomposer.coeff_meta()
    vector = codec.encode(raw_coeff, raw_meta)
    raw_back = codec.decode(vector, raw_meta)
    field_hat = decomposer.inverse_transform(raw_back, domain_spec=domain)

    assert field_hat.shape == field.shape
    assert np.allclose(field_hat, field, atol=1e-4)

    eigvals = np.asarray(raw_meta.get("eigenvalues"))
    assert eigvals.size == 4
    assert np.all(np.diff(eigvals) <= 1e-6)
    assert np.all(eigvals >= -1e-6)
    assert np.all(eigvals <= 1.0 + 1e-6)


def test_spherical_slepian_rejects_unimplemented_pyshtools_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        build_decomposer(
            {
                "name": "spherical_slepian",
                "l_max": 1,
                "backend": "pyshtools",
                "k": 2,
                "region_cap": _cap_cfg(),
            }
        )


def test_spherical_slepian_basis_survives_core_joblib(tmp_path: pathlib.Path) -> None:
    field = np.ones((6, 12), dtype=np.float64)
    domain = build_domain_spec(_domain_cfg(), field.shape)
    decomposer = build_decomposer(
        {
            "name": "spherical_slepian",
            "l_max": 1,
            "k": 4,
            "backend": "scipy",
            "region_cap": _cap_cfg(),
        }
    )

    decomposer.fit(dataset=None, domain_spec=domain)
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    path = dump_joblib(decomposer, tmp_path / "decomposer.joblib")
    restored = load_joblib(path)
    assert np.allclose(restored.coeff_meta()["eigenvalues"], decomposer.coeff_meta()["eigenvalues"])
    assert np.allclose(restored.transform(field, mask=None, domain_spec=domain), coeff)
    assert np.allclose(restored.inverse_transform(coeff, domain_spec=domain), field, atol=1e-4)
