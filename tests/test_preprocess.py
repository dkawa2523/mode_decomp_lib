import numpy as np

from mode_decomp_ml.preprocess import build_preprocess


def test_preprocess_none_roundtrip():
    fields = np.random.randn(3, 8, 8, 2).astype(np.float32)
    masks = np.random.rand(3, 8, 8) > 0.5
    pre = build_preprocess({"name": "none"})
    pre.fit(fields, masks, split="train")
    out_fields, out_masks = pre.transform(fields, masks)
    assert np.allclose(out_fields, fields)
    assert np.array_equal(out_masks, masks)
    inv_fields, inv_masks = pre.inverse_transform(out_fields, out_masks)
    assert np.allclose(inv_fields, fields)
    assert np.array_equal(inv_masks, masks)


def test_preprocess_field_standardize_roundtrip():
    fields = np.random.randn(4, 6, 6, 1).astype(np.float32)
    pre = build_preprocess({"name": "basic", "ops": [{"name": "field_standardize"}]})
    pre.fit(fields, None, split="train")
    out_fields, _ = pre.transform(fields, None)
    mean = out_fields.mean()
    std = out_fields.std()
    assert abs(mean) < 1e-5
    assert abs(std - 1.0) < 1e-5
    inv_fields, _ = pre.inverse_transform(out_fields, None)
    assert np.allclose(inv_fields, fields)
