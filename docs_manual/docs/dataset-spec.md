# データセット仕様

## 推奨フォーマット（npy_dir）

dataset root に以下がある想定です:

| ファイル | 役割 | 形状 |
|---|---|---|
| `cond.npy` | 条件 | `N×D` |
| `field.npy` | field | `N×H×W×C` |
| `mask.npy` | mask | `N×H×W` または `H×W` |
| `manifest.json` | domain/grid/field_kind | JSON |

### `manifest.json`（最低限の役割）

`manifest.json` は「domain を自動解決するためのメタ情報」です。

- domain type（例: rectangle/disk/annulus/arbitrary_mask/sphere_grid/mesh）
- grid 解像度と範囲（x/y_range や lat/lon_range など）
- field kind（scalar/vector）
- mask の source（domain 側で固定マスクを持つ場合は `mask_path` を持てる）

実装の source-of-truth:

- スキーマ: `src/mode_decomp_ml/data/manifest.py`
- 読み込み: `src/mode_decomp_ml/data/datasets.py`（`NpyDirDataset`）
- domain 自動解決: `src/mode_decomp_ml/pipeline/utils.py`（`resolve_domain_cfg`）

## legacy（csv_fields）

外部データ互換のため残していますが、巨大化しやすいので Git 管理は推奨しません。

## 取得/生成

- eval dataset: `data/mode_decomp_eval_dataset_v1/`
- benchmark datasets: `tools/bench/generate_benchmark_datasets_v1.py`

## shape の補足（domainにより異なる）

| domain | field の典型 shape（サンプル1件） | 備考 |
|---|---|---|
| rectangle/disk/annulus/arbitrary_mask | `H×W×C` | 2D格子 |
| sphere_grid | `H×W×C` | 緯度経度格子（経度は周期に注意） |
| mesh | `V×C`（実装によって `V×1×C` など） | 頂点上の値。faces/vertices は manifest に含める |

`mask.npy` は `N×H×W` を推奨します（可変 mask を扱えるため）。
