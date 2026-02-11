# Code Tour

This is a pragmatic map for third-party engineers to understand where things live and how data flows.

## Start Here (Suggested Reading Order)

1. `docs/USER_QUICKSTART.md` (how to run stages)
2. `docs/01_ARCHITECTURE.md` (high-level concepts)
3. `docs/10_PROCESS_CATALOG.md` + `docs/13_TASK_FLOW.md` (how stages connect)
4. `docs/11_PLUGIN_REGISTRY.md` + `docs/20_METHOD_CATALOG.md` (what methods exist)
5. `docs/28_COEFF_META_CONTRACT.md` (coeff/meta invariants)

## Key Directories

- `src/processes/`
  - CLI-style entrypoints for each stage (`decomposition.py`, `preprocessing.py`, `train.py`, `inference.py`, etc.)
  - These focus on orchestration and artifact writing, not math details.

- `src/mode_decomp_ml/pipeline/`
  - Shared run directory/artifact conventions and process helpers.
  - Important:
    - `artifacts.py`: `ArtifactWriter` + standardized run layout
    - `steps.py`: `StepRecorder` and step tracking contract
    - `utils.py`: path/config helpers, dataset manifest helpers, `build_meta()`
    - `process_base.py`: common init/finalize for processes (keeps entrypoints consistent)

- `src/mode_decomp_ml/plugins/`
  - All extensible “method” components:
    - `decomposers/`: basis builders and (inverse) transforms
    - `codecs/`: raw_coeff <-> vector packing
    - `models/`: regressors (cond -> coeff)
    - `coeff_post/`: coefficient post-processing (e.g., PCA)
  - Note: `preprocess` implementations live in `src/mode_decomp_ml/preprocess/` (pickle compatibility), but they are
    registered via `mode_decomp_ml.plugins.registry` like other plugin types.

- `src/mode_decomp_ml/domain/`
  - Domain specs, masks/weights, coordinate grids, and compatibility checks.

- `tools/bench/`
  - Benchmark runners and report generation.
  - `tools/bench/report/` holds plotting/report building blocks used by summary generation.

## Data Flow (Mental Model)

1. Dataset provides `(cond, field, mask)` per sample.
2. Decomposition:
   - `decomposer.fit(dataset, domain_spec)` builds a basis/state
   - `raw_coeff = decomposer.transform(field, mask, domain_spec)`
   - `coeff = codec.encode(raw_coeff, coeff_meta)`
3. Optional preprocessing:
   - `coeff_post.fit(coeff)` and `coeff_z = coeff_post.transform(coeff)`
4. Training:
   - model learns `cond -> coeff_{a or z}`
5. Inference:
   - predict coeff, then decode and inverse-transform back to field for diagnostics or downstream use.

## About `docs_manual/`

`docs/` is treated as the canonical documentation set. `docs_manual/` may contain legacy or exploratory notes and
should not be assumed up-to-date unless explicitly referenced.
