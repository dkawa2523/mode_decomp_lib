from __future__ import annotations

import importlib.util

import numpy as np
import pytest

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
    codec = build_coeff_codec({"name": "sh_pack_v1", "dtype_policy": "float32"})

    raw_coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    raw_meta = decomposer.coeff_meta()
    vector = codec.encode(raw_coeff, raw_meta)
    raw_back = codec.decode(vector, raw_meta)
    field_hat = decomposer.inverse_transform(raw_back, domain_spec=domain)

    assert field_hat.shape == field.shape
    assert np.allclose(field_hat, field, atol=1e-5)


def test_spherical_harmonics_requires_pyshtools_backend() -> None:
    if importlib.util.find_spec("pyshtools") is not None:
        pytest.skip("pyshtools is installed")

    field = np.ones((4, 8), dtype=np.float64)
    domain = build_domain_spec(_domain_cfg(), field.shape)
    decomposer = build_decomposer(
        {
            "name": "spherical_harmonics",
            "l_max": 1,
            "real_form": True,
            "norm": "ortho",
            "backend": "pyshtools",
        }
    )

    with pytest.raises(ImportError):
        decomposer.transform(field, mask=None, domain_spec=domain)
