from __future__ import annotations


def test_preprocess_registry_integrated_into_plugins():
    # Importing preprocess should populate plugins registry.
    import mode_decomp_ml.preprocess as _preprocess  # noqa: F401
    from mode_decomp_ml.plugins import build_preprocess, list_preprocess

    names = list_preprocess()
    assert "none" in names
    assert "basic" in names

    obj = build_preprocess({"name": "none"})
    assert getattr(obj, "name", None) == "none"

