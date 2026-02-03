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
task: train
params:
  domain:
    name: sphere_grid
    n_lat: 18
    n_lon: 36
    angle_unit: deg
    radius: 1.0
```
