from __future__ import annotations

import importlib.util
import pathlib
import warnings

import numpy as np
import pytest
from scipy import special

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.plugins.codecs import build_coeff_codec
from mode_decomp_ml.plugins.decomposers import build_decomposer


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


def test_spherical_slepian_scipy_real_sh_basis_matches_deprecated_sph_harm() -> None:
    # The SciPy backend uses `sph_harm_y` under the hood on SciPy 1.15+, which has different
    # theta/phi conventions compared to the deprecated `sph_harm`. Ensure our wrapper preserves
    # `sph_harm` semantics to avoid silent basis corruption.
    field = np.ones((6, 12), dtype=np.float64)
    domain = build_domain_spec(_domain_cfg(), field.shape)

    import mode_decomp_ml.plugins.decomposers.spherical_slepian as mod

    theta = np.asarray(domain.coords["theta"], dtype=np.float64)
    phi = np.asarray(domain.coords["phi"], dtype=np.float64)
    basis, lm_kind_list = mod._build_real_sh_basis(l_max=1, theta=theta, phi=phi)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        y11 = special.sph_harm(1, 1, theta, phi)
    factor = np.sqrt(2.0) * (-1.0)
    expected_cos = factor * y11.real
    expected_sin = factor * y11.imag

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
    codec = build_coeff_codec({"name": "slepian_pack_v1", "dtype_policy": "float32"})

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


def test_spherical_slepian_requires_pyshtools_backend() -> None:
    if importlib.util.find_spec("pyshtools") is not None:
        pytest.skip("pyshtools is installed")

    field = np.ones((4, 8), dtype=np.float64)
    domain = build_domain_spec(_domain_cfg(), field.shape)
    decomposer = build_decomposer(
        {
            "name": "spherical_slepian",
            "l_max": 1,
            "k": 2,
            "backend": "pyshtools",
            "region_cap": _cap_cfg(),
        }
    )

    with pytest.raises(ImportError):
        decomposer.fit(dataset=None, domain_spec=domain)


def test_spherical_slepian_state_saves_basis(tmp_path: pathlib.Path) -> None:
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
    decomposer.save_state(tmp_path)
    basis_path = tmp_path / "outputs" / "states" / "decomposer" / "slepian_basis.npz"
    assert basis_path.exists()
    data = np.load(basis_path)
    assert "eigenvalues" in data
    assert "eigenvectors" in data
