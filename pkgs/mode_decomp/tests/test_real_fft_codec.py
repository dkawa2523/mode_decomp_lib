"""Independent real FFT coordinates preserve norm and any regressed vector is real."""

import numpy as np
import pytest

from ml_platform_mode_decomp.catalog import build_coeff_codec, build_decomposer
from ml_platform_mode_decomp.geometry import build_domain_spec
from ml_platform_mode_decomp.config import resolve_component


@pytest.mark.parametrize("shape", [(5, 7, 1), (6, 8, 2), (5, 8, 3), (6, 7, 1)])
@pytest.mark.parametrize(
    "method,params", [("fft2", {}), ("fft2_lowpass", {"block_size": 3}), ("fft2_lowpass", {"block_size": 2})]
)
def test_independent_fft_coordinates_are_lossless_isometric_and_conjugate_closed(shape, method, params):
    rng = np.random.default_rng(42)
    field = rng.normal(size=shape)
    h, w, _ = shape
    domain = build_domain_spec(
        {
            "kind": "rectangle",
            "axes": {
                "x": {"start": 0, "step": 1, "count": w, "periodic": True},
                "y": {"start": 0, "step": 1, "count": h, "periodic": True},
            },
        },
        shape,
    )
    decomposition = build_decomposer(resolve_component({"name": method, "params": params}, "decomposition", 42))
    raw = decomposition.transform(field, None, domain)
    metadata = dict(decomposition.coeff_meta())
    codec = build_coeff_codec({"name": "auto", "params": {"dtype_policy": "float64"}})
    packed = codec.encode(raw, metadata)
    assert len(packed) == raw.size
    assert np.sum(packed**2) == pytest.approx(float(np.sum(abs(raw) ** 2)), rel=1e-12)
    np.testing.assert_allclose(codec.decode(packed, metadata), raw, atol=1e-12)
    arbitrary = rng.normal(size=packed.shape)
    spectrum = codec.decode(arbitrary, metadata)
    unshifted = np.fft.ifftshift(spectrum, axes=(-2, -1)) if metadata["fft_shift"] else spectrum
    assert np.max(abs(np.fft.ifft2(unshifted, axes=(-2, -1)).imag)) < 1e-12
    np.testing.assert_allclose(codec.encode(spectrum, metadata), arbitrary, atol=1e-12)
    assert codec.coeff_meta(metadata, packed)["codec"]["independent_real_coordinates"]


def test_real_fft_does_not_silently_discard_a_nonconjugate_spectrum():
    codec = build_coeff_codec({"name": "real_fft"})
    raw = np.zeros((1, 4, 4), complex)
    raw[0, 0, 1] = 1j
    with pytest.raises(ValueError, match="conjugate"):
        codec.encode(raw, {"method": "fft2", "coeff_shape": list(raw.shape), "fft_shift": False})


def test_real_fft_batch_conjugacy_is_checked_at_each_samples_own_scale():
    codec = build_coeff_codec({"name": "real_fft"})
    raw = np.zeros((2, 1, 4, 4), complex)
    raw[0, 0, 0, 0] = 1e15
    raw[1, 0, 0, 1] = 1j
    meta = {"method": "fft2", "coeff_shape": [1, 4, 4], "fft_shift": False}
    with pytest.raises(ValueError, match="conjugate"):
        codec.encode_batch(raw, meta)


@pytest.mark.parametrize("mode", ["real_imag", "ri", "real-imag"])
def test_auto_real_fft_dispatch_uses_the_normalized_complex_mode(mode):
    raw = np.fft.fft2(np.arange(32).reshape(2, 4, 4), axes=(-2, -1))
    meta = {"method": "fft2", "coeff_shape": list(raw.shape), "complex_format": "complex", "fft_shift": False}
    codec = build_coeff_codec({"name": "auto", "params": {"complex_mode": mode, "dtype_policy": "float64"}})
    assert codec.encode(raw, meta).size == 32


def test_offset_wrapped_fft_also_preserves_real_residual_spectra():
    from ml_platform_mode_decomp.data.schema import FieldDataset
    from ml_platform_mode_decomp.representation import FieldRepresentation

    rng = np.random.default_rng(42)
    fields = rng.normal(size=(6, 8, 8, 1))
    domain = build_domain_spec({"kind": "rectangle", "x_range": [-1, 1], "y_range": [-1, 1]}, (8, 8, 1))
    dataset = FieldDataset(fields, domain, np.asarray([f"s{i}" for i in range(6)]))
    model = FieldRepresentation(
        {
            "decomposition": {
                "name": "offset_residual",
                "params": {"enabled": True, "inner": {"name": "fft2", "disk_policy": "error"}},
            }
        }
    ).fit(dataset)
    decoded = model.codec.decode(rng.normal(size=model.packed_dim), model.raw_meta)
    assert np.max(abs(np.fft.ifft2(decoded["residual"], axes=(-2, -1)).imag)) < 1e-12
    # The wrapper retains its own offset coordinates; this is a local residual guarantee.
    assert model.codec.coeff_meta(model.raw_meta, np.zeros(model.packed_dim))["residual_coeff_meta"]["codec"][
        "independent_real_coordinates"
    ]
