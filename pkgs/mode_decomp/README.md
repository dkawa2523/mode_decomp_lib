# Mode Decomposition

A standalone Python package for steady spatial fields: decomposition, condition-to-field or paired-field prediction, and missing-field reconstruction. Requires Python 3.11+ and `ml-platform-core>=0.2.1,<0.3`. No execution SDK or ClearML is required.

Install the core and mode wheels from the consumer ZIP, then run one YAML:

```shell
python -m pip install ml_platform_core-0.2.1-py3-none-any.whl ml_platform_mode_decomp-0.11.2-py3-none-any.whl
python -m ml_platform_mode_decomp run.yaml
```

The YAML defines fields, geometry, optional conditions, and methods. Profiles and separate dataset manifests are optional. CSV/scattered inputs are converted once into run-local canonical data.

```python
from ml_platform_mode_decomp import FieldDataset, FieldRepresentation, FieldSurrogate

data = FieldDataset.from_arrays(fields, conditions)  # fields: (N,H,W,C)
representation = FieldRepresentation({"decomposition": {"name": "dct2"}})
z = representation.fit_transform(data)
restored = representation.inverse_transform(z)
model = FieldSurrogate({"representation": {"decomposition": {"name": "dct2"}}}).fit(data)
predictions = model.predict(new_conditions)
```

`FieldSurrogate.fit(target, inputs=source)` maps paired fields with independent fixed geometries; add `input_representation` to the candidate. Identified saved source Z coefficients are also supported. Source and target case IDs must match exactly. Field-input representations are fitted within each training partition; saved coefficient inputs use their frozen external representation.

Tasks cover direct decomposition, held-out representation/surrogate/imputation comparison, final training, and saved-model inference. Direct decomposition metrics describe reconstruction of fitted inputs. Unknown-case performance uses held-out samples and original-field metrics. `predict_coefficients()` returns target Z; `predict()` returns the target field in its stored units. Optional `output.coefficients_name` exports identified target Z for reuse.

POD, Tucker and quadratic manifolds share the coefficient API and model bundles. Optional extras are wavelet/pod/torch/mtgp/gbdt. Selected methods determine geometry, observation and runtime requirements. Model saving records runtime pins in a companion requirements file; transferring a model requires its recorded Python minor and numerical backends.

The consumer ZIP includes runnable configurations, input examples and guides for data formats, field mapping, methods, evaluation, uncertainty and distribution. The repository README is the development entry point; those guides are the detailed specification.
