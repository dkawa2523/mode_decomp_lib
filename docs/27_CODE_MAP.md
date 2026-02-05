# Code map (structure & data flow)

This document is a quick map of "where to look" and "how data moves".

## Entry points
- CLI: `src/mode_decomp_ml/cli/run.py`
- Programmatic: `src/mode_decomp_ml/run.py`
- Process entrypoints: `src/processes/*.py`

## High-level flow (typical run)
1. CLI loads config (Hydra) → `cli/run.py`
2. Task routes to process → `src/processes/{decomposition,preprocessing,train,inference,pipeline,leaderboard}.py` (I/O + run_dir)
3. Process uses pipeline helpers → `src/mode_decomp_ml/pipeline/*`
4. Pipeline calls plugins → `src/mode_decomp_ml/plugins/*`
5. Artifacts are written → `runs/<name>/<process>/...`

## Core modules
- Pipeline utilities: `src/mode_decomp_ml/pipeline/`
  - `loaders.py`: dataset/materialization helpers
  - `steps.py`: step tracking
  - `artifacts.py`: read/write artifacts
  - `utils.py`: run dir, config helpers
- Domain & geometry: `src/mode_decomp_ml/domain/`
- Data: `src/mode_decomp_ml/data/`
- Preprocess: `src/mode_decomp_ml/preprocess/`
- Evaluate: `src/mode_decomp_ml/evaluate/`
- Visualization: `src/mode_decomp_ml/viz/` (library utilities called by process layer)
- Tracking (optional): `src/mode_decomp_ml/tracking/`
  - sphere_grid helpers: `src/mode_decomp_ml/domain/sphere_grid.py`

## Plugin system
- Registry: `src/mode_decomp_ml/plugins/registry.py`
- Decomposers: `src/mode_decomp_ml/plugins/decomposers/` (flat layout)
- Coeff post: `src/mode_decomp_ml/plugins/coeff_post/`
- Codecs: `src/mode_decomp_ml/plugins/codecs/`
- Models: `src/mode_decomp_ml/plugins/models/`


## Shim (deprecated paths)
- `src/mode_decomp_ml/models/` → `plugins/models`
- `src/mode_decomp_ml/coeff_post/` → `plugins/coeff_post`

## Artifacts (contracts)
See `docs/04_ARTIFACTS_AND_VERSIONING.md` for required outputs and run dir structure.

## Add-on notes
- Special-function suite docs: `docs/addons/32_SPECIAL_FUNCTION_SUITE.md`
