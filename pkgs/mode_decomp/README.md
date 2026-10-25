# Mode Decomposition 0.11

A standalone package for field decomposition, condition-to-field prediction and missing-field reconstruction. Requires Python 3.11+ and `ml-platform-core>=0.2.1,<0.3`; no execution SDK or ClearML is required.

The same `FieldSurrogate` also accepts paired fields with `fit(target, inputs=source)` and an `input_representation` alongside the target `representation`. The two fixed geometries may differ. `data.input` selects fields or checked saved Z coefficients for the existing pipeline, surrogate analysis and inference tasks. Source/target IDs must match exactly; field input representations are fitted inside each training partition. Saved coefficient inputs instead use their fixed external representation. The consumer ZIP includes `docs/FIELD_MAPPING.md` and a field mapping example.

```shell
python -m pip install ml_platform_core-0.2.1-py3-none-any.whl ml_platform_mode_decomp-0.11.1-py3-none-any.whl
python -m ml_platform_mode_decomp run.yaml
```

One run YAML can contain input fields, geometry, optional conditions and numerical methods. A profile and separate dataset manifest are optional. CSV/scattered formats are converted once into retained run-local canonical data before the normal workflows execute.

Steady-field representations also include `tucker` (shared spatial/component factors and one core per sample) and
`quadratic_manifold` (joint POD coordinates with an orthogonal quadratic decoder correction). Both use NumPy/SciPy,
require complete training observations on their fixed domain, and support partially observed queries. Tucker uses a
rectangular tensor-product grid; quadratic manifolds also support fixed masked grids and meshes. Their existing
coefficient codecs, regressors, model bundles and held-out imputation workflows remain shared.

## Array API

```python
from ml_platform_mode_decomp import FieldDataset, FieldRepresentation, FieldSurrogate

data = FieldDataset.from_arrays(fields, conditions)  # fields: NHWC, conditions: (N,P)
representation = FieldRepresentation({"decomposition": {"name": "dct2"}})
coefficients = representation.fit_transform(data)
restored = representation.inverse_transform(coefficients)
surrogate = FieldSurrogate({"representation": {"decomposition": {"name": "dct2"}}}).fit(data)
predictions = surrogate.predict(new_conditions)
```

`surrogate.iter_predict(new_conditions)` yields field batches in input order; `intervals=True` yields prediction/lower/upper batches for calibrated models. Normal array-returning prediction methods also bound query intermediates. `output.prediction_name: predictions.npy` selects stream-written, memory-mappable output with automatic ID/band/metadata sidecars. The default `predictions.npz` retains the existing NumPy archive format. NPZ writing temporarily needs both uncompressed staging files and the compressed archive on disk.

`layout="HWC"` identifies one grid field; mesh NVC/VC requires explicit geometry. Omitted sample IDs, component names and condition names receive positional names. Omitted grid coordinates use array indices without inferred physical units. Supply x/y or a DomainSpec for physical geometry. Physical vectors require `field_kind="vector"`; `components` and `field_units` specify their names and units. The default field kind is scalar for C=1 and multichannel otherwise. Constructor calls neither fit nor split data.

## Run purposes

- `mode_decomp_representation`: directly fit/decompose all supplied fields, including one field; save A/Z coefficients, reconstruction and the representation bundle. Metrics have scope `fitted_input_reconstruction`, not unseen-sample performance.
- `mode_decomp_analysis`: compare fixed candidates on shared holdouts. `analysis.kind` selects representation, surrogate or imputation; it does not choose a winner from test scores.
- `mode_decomp_pipeline`: fit a fixed candidate once, or select among candidates with inner holdout/CV and refit the winner. Evaluate independent final test samples in both cases.
- `mode_decomp_infer`: predict from a saved surrogate and new conditions only.
- `mode_decomp_import`: optional explicit conversion for reusable canonical datasets.

```python
from ml_platform_mode_decomp import load_config, run_task

result = run_task(load_config("run.yaml"))
```

Optional reports use `analysis.modes.enabled`, `analysis.components.enabled`, `analysis.scattered.enabled` and `report.enabled`. Modes and physical components are available in direct decomposition and held-out analysis. A/Z coordinates belong to each fitted representation; compare methods using the same original-field reference, support and weights.

Detailed `analyze_representation()` and `analyze_surrogate()` retain diagnostic fields by default; use `retain_fields=False` when only statistics and final predictions are needed. YAML analyses do this automatically. Mode reports consume responses one at a time; `mode_fields()` retains the final requested array without a second stack of all responses. Joint/EM fitting retains only selected basis vectors and reuses its work matrices; direct SVD still needs full factorization memory.

Split-conformal field bands use `uncertainty.enabled`. Calibration samples are reserved before all fitting, selection and CV. `FieldSurrogate.predict_interval` returns unchanged point predictions plus saved lower/upper bounds. Simultaneous coverage concerns one whole fixed supported field; pointwise coverage is coordinate-wise. Coverage requires exchangeable calibration/future samples and is not guaranteed by time/group splitting or for OOD conditions. Bands are not Gaussian standard deviations.

Surrogate calibration consumes prediction batches once. Simultaneous calibration retains one score per sample; exact pointwise calibration retains the smaller order-statistic tail per coordinate, so its memory still grows with calibration size and support. `FieldCalibrator.fit_batches(data, predictions, training=proper_train)` accepts a prediction iterator in sample order; `fit()` remains available for existing arrays. Coverage reports retain sample statistics; file output generates band endpoints by batch.

`HodgeDecomposition` exposes exact/coexact/harmonic edge cochains with a positive geometry-derived metric on a nonperiodic Cartesian mask. The node adapter retains a separate sampling residual. `natural` is the finite-cochain boundary convention; pointwise normal velocity or prescribed PDE boundary values are not imposed. The old `helmholtz_poisson` implementation was removed without an alias.

The bundle contains geometry, transforms, coefficient layout, components/units, condition schema, regressor and optional calibration. Normal inference needs no training data. Optional backends use wavelet/pod/torch/mtgp/gbdt extras.

New models use field schema 4. Older schemas 1/2/3 retain their original validation and coefficient ordering; actual 0.7.1 and 0.9.0 bundles cover representations, surrogates, calibrated bands and geometry-dependent methods. Independent POD/Gappy/pod_svd retain each component's mode count, including zero modes for constant centered profiles. Numerical backend versions remain required, including the same sklearn version for stored sklearn estimators. Package provenance does not require an exact mode/core patch version.

The source checkout includes the fixed core dependency. `uv sync --locked` needs no platform checkout; `scripts/build_wheels.py` stages and builds mode only. Core rebuilding is a separate maintainer operation in `integration/build_core.py`.

SDK integration remains deferred. Continuous/adaptive HPO, repeated outer nested CV, 3D scattered interpolation, changing geometry and arbitrary PDE boundary conditions remain outside this release.
