"""Small counterexamples for the numerical defects found during migration."""

from types import SimpleNamespace

import numpy as np
import pytest

from ml_platform_mode_decomp.decomposition.annular_zernike import AnnularZernikeDecomposer
from ml_platform_mode_decomp.decomposition.fft2_lowpass import FFT2LowpassDecomposer
from ml_platform_mode_decomp.decomposition.pod_em import PODEMDecomposer
from ml_platform_mode_decomp.decomposition.pod_joint import JointPODDecomposer
from ml_platform_mode_decomp.decomposition.pod_joint_em import JointPODEMDecomposer
from ml_platform_mode_decomp.decomposition.pod_utils import svd_snapshots_basis
from ml_platform_mode_decomp.decomposition.spherical_harmonics import SphericalHarmonicsDecomposer
from ml_platform_mode_decomp.decomposition.spherical_slepian import SphericalSlepianDecomposer
from ml_platform_mode_decomp.geometry import DomainSpec


def rectangle(height=16, width=16):
    x, y = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    return DomainSpec("rectangle", {"x": x, "y": y}, None, np.ones_like(x), {})


@pytest.mark.parametrize("shape,block", [((16, 16), 8), ((15, 17), 6), ((8, 8), 8), ((16, 16), 5)])
def test_lowpass_is_an_idempotent_real_projection(shape, block):
    domain = rectangle(*shape)
    field = np.random.default_rng(5).normal(size=shape)
    decomp = FFT2LowpassDecomposer(cfg={"name": "fft2_lowpass", "disk_policy": "error", "block_size": block})
    encoded = decomp.transform(field, None, domain)
    reconstructed = decomp.inverse_transform(encoded, domain)
    np.testing.assert_allclose(decomp.transform(reconstructed, None, domain), encoded, atol=1e-12)


def test_lowpass_retains_both_cutoff_frequencies():
    field = np.tile(np.cos(2 * np.pi * 4 * np.arange(16) / 16), (16, 1))
    domain = rectangle()
    decomp = FFT2LowpassDecomposer(cfg={"name": "fft2_lowpass", "disk_policy": "error", "block_size": 8})
    coeff = decomp.transform(field, None, domain)
    assert coeff.shape == (1, 9, 9)
    assert decomp.coeff_meta()["retained_hw"] == [9, 9]
    np.testing.assert_allclose(decomp.inverse_transform(coeff, domain), field, atol=1e-12)


def test_annular_degree_cap_keeps_the_radial_mode():
    domain = rectangle(21, 21)
    x = (domain.coords["x"] - 10) / 10
    y = (domain.coords["y"] - 10) / 10
    r = np.hypot(x, y)
    mask = (r >= 0.3) & (r <= 1)
    domain = DomainSpec(
        "annulus", {"r": r, "theta": np.arctan2(y, x)}, mask, np.where(mask, 1.0, 0.0), {"r_inner": 0.3, "r_outer": 1.0}
    )
    decomp = AnnularZernikeDecomposer(
        cfg={
            "name": "annular_zernike",
            "n_max": 2,
            "m_max": 1,
            "ordering": "n_then_m",
            "normalization": "orthonormal",
            "boundary_condition": "none",
        }
    )
    field = np.ones((21, 21))
    coeff = decomp.transform(field, None, domain)
    assert decomp._nm_list == [(0, 0), (1, -1), (1, 1), (2, 0)]
    np.testing.assert_allclose(decomp.inverse_transform(coeff, domain)[mask], 1.0, atol=1e-12)


@pytest.mark.parametrize("norm", ["schmidt", "4pi"])
def test_unimplemented_spherical_normalization_is_rejected(norm):
    with pytest.raises(ValueError, match="norm"):
        SphericalHarmonicsDecomposer(cfg={"l_max": 2, "norm": norm})


@pytest.mark.parametrize(
    "cls,params", [(SphericalHarmonicsDecomposer, {"l_max": 2}), (SphericalSlepianDecomposer, {"l_max": 2, "k": 2})]
)
def test_unimplemented_spherical_backend_is_rejected(cls, params):
    with pytest.raises(ValueError, match="backend"):
        cls(cfg={**params, "backend": "pyshtools"})


@pytest.mark.parametrize("scale", [1.0, 1e-8, 1e8])
def test_snapshot_rank_is_relative_and_never_creates_zero_modes(scale):
    matrix = np.outer([1.0, 2.0, -1.0, -2.0], [1.0, 0.0, 2.0, 3.0]) * scale
    with pytest.raises(ValueError, match="rank"):
        svd_snapshots_basis(matrix, n_modes=2, method="test")
    basis = svd_snapshots_basis(matrix, n_modes=None, method="test")
    assert basis.shape == (4, 1)
    np.testing.assert_allclose(basis.T @ basis, np.eye(1), atol=1e-14)
    np.testing.assert_allclose(matrix @ basis @ basis.T, matrix, atol=1e-6 * scale)


def masked_samples(fill):
    fields = [
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[2.0, 3.0], [4.0, 5.0]]),
        np.array([[3.0, 4.0], [5.0, 6.0]]),
    ]
    masks = [
        np.array([[True, False], [True, True]]),
        np.array([[True, True], [True, False]]),
        np.array([[False, True], [True, True]]),
    ]
    return [SimpleNamespace(field=np.where(m, f, fill)[..., None], mask=m) for f, m in zip(fields, masks, strict=True)]


@pytest.mark.parametrize("cls,name", [(PODEMDecomposer, "pod_em"), (JointPODEMDecomposer, "pod_joint_em")])
def test_em_fit_is_independent_of_masked_storage_values(cls, name):
    domain = rectangle(2, 2)
    config = {
        "name": name,
        "n_modes": 1,
        "n_iter": 3,
        "ridge_alpha": 1e-6,
        "inner_product": "euclidean",
        "mask_policy": "allow",
    }
    baseline = cls(cfg=config).fit(masked_samples(0), domain)
    missing = cls(cfg=config).fit(masked_samples(np.nan), domain)
    sample = masked_samples(0)[0]
    a = baseline.transform(sample.field, sample.mask, domain)
    b = missing.transform(sample.field, sample.mask, domain)
    np.testing.assert_allclose(baseline.inverse_transform(a, domain), missing.inverse_transform(b, domain))


@pytest.mark.parametrize("cls,name", [(JointPODDecomposer, "pod_joint"), (JointPODEMDecomposer, "pod_joint_em")])
@pytest.mark.parametrize("center", [True, False])
def test_joint_pod_center_setting_matches_the_computation(cls, name, center):
    domain = rectangle(2, 2)
    direction = np.arange(1.0, 9.0).reshape(2, 2, 2)
    offset = np.array([2.0, -1.0, 0.0, 3.0, 1.0, 1.0, -2.0, 4.0]).reshape(2, 2, 2)
    samples = [SimpleNamespace(field=offset + a * direction, mask=None) for a in [-2.0, -1.0, 1.0, 2.0]]
    config = {
        "name": name,
        "n_modes": 1 if center else 2,
        "inner_product": "euclidean",
        "mask_policy": "allow" if "em" in name else "error",
        "mean_centered": center,
    }
    if "em" in name:
        config.update(n_iter=2, ridge_alpha=0.0)
    decomp = cls(cfg=config).fit(samples, domain)
    if not center:
        np.testing.assert_array_equal(decomp._mu, np.zeros(8))
    for sample in samples:
        coeff = decomp.transform(sample.field, None, domain)
        assert decomp.coeff_meta()["mean_centered"] is center
        np.testing.assert_allclose(decomp.inverse_transform(coeff, domain), sample.field, atol=1e-12)


def test_offset_ignores_unobserved_nan_and_rejects_zero_epsilon():
    from ml_platform_mode_decomp.decomposition.offset_residual import _compute_offset_residual

    domain = rectangle(2, 2)
    mask = np.array([[True, False], [True, False]])
    field = np.array([[1.0, 0.0], [3.0, 0.0]])
    first = _compute_offset_residual(field, mask, domain_spec=domain, min_residual_rms=1e-8)
    field[~mask] = np.nan
    second = _compute_offset_residual(field, mask, domain_spec=domain, min_residual_rms=1e-8)
    np.testing.assert_allclose(first.offset, [2.0])
    np.testing.assert_allclose(first.offset, second.offset)
    np.testing.assert_allclose(first.residual, second.residual)
    with pytest.raises(ValueError, match="min_residual_rms"):
        _compute_offset_residual(np.ones((2, 2)), None, domain_spec=domain, min_residual_rms=0)


def test_modred_backend_reads_the_named_results_and_reconstructs():
    pytest.importorskip("modred")
    from ml_platform_mode_decomp.decomposition.pod import PODDecomposer

    domain = rectangle(3, 3)
    direction = np.arange(9.0).reshape(3, 3, 1)
    samples = [SimpleNamespace(field=3.0 + amplitude * direction, mask=None) for amplitude in [-2.0, -1.0, 1.0, 2.0]]
    decomp = PODDecomposer(
        cfg={
            "name": "pod",
            "backend": "modred",
            "solver": "snapshots",
            "n_modes": 1,
            "mask_policy": "error",
            "inner_product": "domain_weights",
        }
    )
    decomp.fit(samples, domain)
    for sample in samples:
        coeff = decomp.transform(sample.field, None, domain)
        np.testing.assert_allclose(decomp.inverse_transform(coeff, domain), sample.field, atol=1e-11)


def test_autoencoder_fit_preserves_rng_and_refit_starts_fresh():
    torch = pytest.importorskip("torch")
    from ml_platform_mode_decomp.decomposition.autoencoder import AutoencoderDecomposer

    domain = rectangle(4, 4)
    samples = [
        SimpleNamespace(field=np.arange(16, dtype=np.float32).reshape(4, 4, 1) * amplitude, mask=None)
        for amplitude in [0.5, 1.0, 1.5]
    ]
    decomp = AutoencoderDecomposer(
        cfg={
            "name": "autoencoder",
            "mask_policy": "error",
            "latent_dim": 2,
            "hidden_channels": [2],
            "epochs": 1,
            "batch_size": 2,
            "seed": 7,
        }
    )
    before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    decomp.fit(samples, domain)
    assert torch.equal(before, torch.random.get_rng_state())
    initial_state = {key: value.clone() for key, value in decomp._model.state_dict().items()}
    decomp.fit(samples, domain)
    assert torch.equal(before, torch.random.get_rng_state())
    if cuda_before:
        assert all(
            torch.equal(before_state, after_state)
            for before_state, after_state in zip(cuda_before, torch.cuda.get_rng_state_all(), strict=True)
        )
    for key, value in decomp._model.state_dict().items():
        assert torch.equal(initial_state[key], value)
