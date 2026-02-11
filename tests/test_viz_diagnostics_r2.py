import numpy as np

from mode_decomp_ml.viz import masked_weighted_r2, per_pixel_r2_map


def test_masked_weighted_r2_perfect_is_one() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=(100,))
    r2 = masked_weighted_r2(y, y)
    assert np.isfinite(r2)
    assert abs(r2 - 1.0) < 1e-12


def test_masked_weighted_r2_constant_true_is_nan() -> None:
    y_true = np.ones((10,), dtype=float)
    y_pred = np.zeros((10,), dtype=float)
    r2 = masked_weighted_r2(y_true, y_pred)
    assert not np.isfinite(r2)


def test_masked_weighted_r2_respects_mask() -> None:
    y_true = np.array([0.0, 1.0, 2.0], dtype=float)
    y_pred = np.array([0.0, 1.0, 0.0], dtype=float)
    mask = np.array([True, True, False])
    r2 = masked_weighted_r2(y_true, y_pred, mask=mask)
    assert np.isfinite(r2)
    assert abs(r2 - 1.0) < 1e-12


def test_per_pixel_r2_map_shape_and_values() -> None:
    # field_true varies across samples, so ss_tot > 0 for all pixels.
    n, h, w = 8, 3, 4
    base = np.arange(n, dtype=float)[:, None, None]
    field_true = base + np.zeros((n, h, w), dtype=float)
    field_pred = field_true.copy()
    r2_map = per_pixel_r2_map(field_true, field_pred)
    assert r2_map.shape == (h, w)
    assert np.allclose(r2_map, 1.0, atol=1e-12, equal_nan=False)


def test_per_pixel_r2_map_mask_can_make_nan() -> None:
    n, h, w = 5, 2, 2
    base = np.arange(n, dtype=float)[:, None, None]
    field_true = base + np.zeros((n, h, w), dtype=float)
    field_pred = field_true.copy()
    mask = np.ones((n, h, w), dtype=bool)
    # For pixel (0,0), leave only one valid sample -> undefined R^2 (NaN).
    mask[:, 0, 0] = False
    mask[0, 0, 0] = True
    r2_map = per_pixel_r2_map(field_true, field_pred, mask=mask)
    assert np.isfinite(r2_map[0, 1])
    assert not np.isfinite(r2_map[0, 0])

