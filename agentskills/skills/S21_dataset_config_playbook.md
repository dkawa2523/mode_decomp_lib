# S21: データセット設定プレイブック（npy_dir / csv_fields / manifest）

## 目的
- 別環境の Copilot でも、`dataset` と `domain` をコード契約どおりに最短で設定する。
- `mask_policy` と `manifest` の扱いを固定し、手法比較で設定ブレを防ぐ。

## このスキルを使う場面
- 新しいデータセットを `run.yaml` または Hydra で実行したい。
- `mask_policy` / `domain` / `grid` の設定でエラーになっている。
- `npy_dir` と `csv_fields` のどちらで設定すべきか迷っている。

## 入力として先に確定する情報
- データ形式: `npy_dir` か `csv_fields`。
- field 種別: scalar (`C=1`) か vector (`C>=2`)。
- ドメイン: `rectangle` / `disk` / `annulus` / `arbitrary_mask` / `sphere_grid` / `mesh`。
- mask の運用方針: 必須か任意か禁止か。

## 手順
1. loader を決める。
- `cond.npy` と `field.npy` があるなら `dataset.name=npy_dir`。
- `conditions.csv` + `fields/*.csv` なら `dataset.name=csv_fields`。

2. `npy_dir` の最低契約を満たす。
- 必須: `cond.npy`, `field.npy`。
- 任意: `mask.npy`（`mask_policy` と整合必須）。
- 推奨: `manifest.json`（`resolve_domain_cfg` が domain を自動解決）。

3. `csv_fields` の最低契約を満たす。
- 必須: `conditions_csv`, `fields_dir`, `id_column`, `grid.H`, `grid.W`。
- `grid.x_range`, `grid.y_range`（sphere_grid は `lon_range`, `lat_range`）を埋める。
- vector は `field_components: [fx, fy]` 等を指定。
- 注意: CSV 補間に SciPy が必要。

4. `mask_policy` を明示する。
- `require`: mask が無いと失敗。
- `allow_none`: mask が無くても可。
- `forbid`: mask があると失敗。

5. domain 設定の優先順位を守る。
- `manifest.json` がある `npy_dir` は manifest を優先。
- manifest が無い場合は `configs/domain/*.yaml` または `params.domain.*` で明示。
- `sphere_grid` は `n_lat/n_lon` を入れると `lat/lon_range` を自動補完可能。

6. dry-run で設定を検証する。
- `PYTHONPATH=src python -m mode_decomp_ml.run --config <run.yaml> --dry-run`
- ここで `dataset` / `domain` / `run_dir` が想定どおりか確認。

## 最小テンプレ（npy_dir, run.yaml）
```yaml
dataset:
  name: npy_dir
  root: data/my_dataset
  mask_policy: allow_none

task: pipeline

pipeline:
  decomposer: dct2
  codec: auto
  coeff_post: none
  model: ridge

output:
  root: runs
  name: my_dataset_baseline

params:
  domain:
    name: rectangle
    x_range: [-1.0, 1.0]
    y_range: [-1.0, 1.0]
```

## 最小テンプレ（csv_fields, run.yaml）
```yaml
dataset:
  name: csv_fields
  conditions_csv: data/my_dataset/conditions.csv
  fields_dir: data/my_dataset/fields
  id_column: id
  mask_policy: allow_none
  field_components: [fx, fy]
  grid:
    H: 64
    W: 64
    x_range: [-1.0, 1.0]
    y_range: [-1.0, 1.0]

task: decomposition
```

## 事故りやすい点
- `mask_policy=require` なのに mask ファイルが存在しない。
- `csv_fields` で `grid.H/W` を未設定。
- `sphere_grid` なのに `lat/lon` と `x/y` の意味を混在。
- manifest があるのに手動 `domain` と矛盾する値を入れる。

## 実装の根拠（読む場所）
- `src/mode_decomp_ml/data/datasets.py`
- `src/mode_decomp_ml/pipeline/utils.py` (`resolve_domain_cfg`)
- `src/mode_decomp_ml/domain/__init__.py`
- `src/mode_decomp_ml/run.py` (`_normalize_dataset`)
