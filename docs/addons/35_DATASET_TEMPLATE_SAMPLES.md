# Dataset template samples

## NPY dataset layout (with manifest)

```
data/my_dataset/
  cond.npy
  field.npy
  mask.npy            # optional
  manifest.json
```

Example `manifest.json` (sphere_grid):

```json
{
  "field_kind": "scalar",
  "grid": {
    "H": 18,
    "W": 36,
    "x_range": [-1.0, 1.0],
    "y_range": [-1.0, 1.0]
  },
  "domain": {
    "type": "sphere_grid",
    "n_lat": 18,
    "n_lon": 36,
    "angle_unit": "deg",
    "radius": 1.0
  }
}
```

Notes:
- `n_lat/n_lon` を指定すれば `lat_range/lon_range` は自動補完されます。
- sphere_grid の domain 作成は `mode_decomp_ml.domain.sphere_grid` のユーティリティを使うこと。

## run.yaml example (npy_dir)

```yaml
dataset:
  name: npy_dir
  root: data/my_dataset
  mask_policy: allow_none
task: pipeline
params:
  domain:
    name: sphere_grid
    n_lat: 18
    n_lon: 36
    angle_unit: deg
    radius: 1.0
```

## CSV dataset layout (conditions + per-sample fields)

```
data/my_dataset/
  conditions.csv
  fields/
    <id>.csv
```

`conditions.csv` columns:
- `id` (required)
- feature columns (arbitrary names)

`fields/<id>.csv` columns:
- `x`, `y`, `f` (fixed)

Vector fields (C=2) use two files:
- `fields/<id>_fx.csv`
- `fields/<id>_fy.csv`
and set `field_components: [fx, fy]`.

## run.yaml example (csv_fields)

```yaml
dataset:
  conditions_csv: data/my_dataset/conditions.csv
  fields_dir: data/my_dataset/fields
  id_column: id
  mask_policy: allow_none
  field_components: [fx, fy]   # vector only
  grid:
    H: 64
    W: 64
task: decomposition
params:
  domain:
    name: rectangle
    x_range: [-1.0, 1.0]
    y_range: [-1.0, 1.0]
```
