import numpy as np

from ml_platform_mode_decomp.field_preprocessing import build_preprocess


def test_preprocess_none_roundtrip():
    rng = np.random.default_rng(13)
    fields = rng.normal(size=(3, 8, 8, 2)).astype(np.float32)
    masks = rng.random((3, 8, 8)) > 0.5
    pre = build_preprocess({"name": "none"})
    pre.fit(fields, masks)
    out_fields, out_masks = pre.transform(fields, masks)
    expected = np.where(masks[..., None], fields, 0)
    assert np.allclose(out_fields, expected)
    assert np.array_equal(out_masks, masks)
    inv_fields, inv_masks = pre.inverse_transform(out_fields, out_masks)
    assert np.allclose(inv_fields, expected)
    assert np.array_equal(inv_masks, masks)
