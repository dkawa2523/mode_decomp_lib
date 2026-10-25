# Standalone core dependency

The versioned core wheel in this directory is the dependency used by `uv.lock`.
Normal environment setup and mode wheel builds do not clone the platform.
`core.json` records the artifact hash, upstream revision and patch hash.

Maintainers rebuild it with `python integration/build_core.py`, then update the
lock and verify both the existing strict core API and the mode bundle readers.
Core 0.2.1 adds `validate_artifact_integrity` for hash and requested provenance;
`validate_model_integrity` retains its existing strict runtime behavior.

Do not replace a released artifact with different code under the same version.
The upstream changes are recorded in `integration/ml_platform_core.patch`.
