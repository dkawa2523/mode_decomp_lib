# Config Conventions（Hydra + run.yaml）

本プロジェクトは Hydra を内部実装として使いますが、**非DSユーザーは run.yaml を入口**にします。

## 1. configの階層（現行）
- `configs/config.yaml`: Hydra 入口（defaults + seed + run_dir）
- `configs/dataset/*.yaml`: dataset preset（`synthetic`, `npy_dir`, `csv_fields`）
- `configs/split/*.yaml`: split preset（`all`）
- `configs/domain/*.yaml`: domain preset（`rectangle`, `disk`, `annulus`, `sphere_grid`, `arbitrary_mask`）
- `configs/preprocess/*.yaml`: preprocess pipeline（`basic`）
- `configs/decompose/{analytic,data_driven}/*.yaml`: decomposer preset（`fft2`, `zernike`, `pod_svd`, `annular_zernike`, `wavelet2d`, `spherical_harmonics`）
- `configs/codec/*.yaml`: coeff codec preset（`none`, `fft_complex_codec_v1`, `zernike_pack_v1`, `wavelet_pack_v1`, `sh_pack_v1`）
- `configs/coeff_post/*.yaml`: coeff post preset（`none`, `pca`, `quantile`, `power_yeojohnson`）
- `configs/model/*.yaml`: regressor preset（`ridge`, `multitask_lasso`, `mtgp`, `xgb`, `lgbm`, `catboost`）
- `configs/train/*.yaml`: train eval/CV/tuning/viz preset（`basic`）
- `configs/eval/*.yaml`: metrics/report settings（`basic`）
- `configs/viz/*.yaml`: visualization settings（`basic`）
- `configs/uncertainty/*.yaml`: uncertainty settings（`gpr_mc`）
- `configs/clearml/*.yaml`: tracking settings（`basic`）
- `configs/task/*.yaml`: task routing（doctor/decomposition/preprocessing/train/inference/pipeline/leaderboard）

> NOTE: 旧プリセットは cleanup で削除済み。必要なら task/RFC で復活させる。

## 2. 命名規約
- configキーは snake_case
- method名は registry key と一致（例: `decompose.name: zernike`）

## 3. 追加ルール（yaml増殖停止）
- **新しい手法を追加しても YAML を増やさない。**
  - まずはコード側の safe default と `run.yaml` の `params:` で調整する
  - Hydra sweep が必要な場合のみ **最小の preset** を追加する
- 既存の非最小 preset は削除し、必要なら最小 preset を追加する

## 4. manifest / domain
- `manifest.json` がある場合は **manifest が domain の真実**（domain presetは参照されない）
- manifest が無い legacy dataset は `domain` が必須
  - run.yaml の場合は `params.domain` に明示する

### sphere_grid の設定例
range を手で持たず、`n_lat/n_lon` から自動補完するのが標準です。

```yaml
params:
  domain:
    name: sphere_grid
    n_lat: 18
    n_lon: 36
    angle_unit: deg
    radius: 1.0
```

詳細な domain 概念は `docs/02_DOMAIN_MODEL.md` を参照。
dataset テンプレは `docs/addons/35_DATASET_TEMPLATE_SAMPLES.md` を参照。

### csv_fields（vector）
- ベクトル場は `fields/<id>_fx.csv` / `<id>_fy.csv` の2ファイル
- `field_components: [fx, fy]` を指定する

補完される範囲（内部で利用）:
- `lat_range`: [-90.0, 90.0]
- `lon_range`: [-180.0, 180.0 - 360/n_lon]（周期重複を避ける）

dataset 生成スクリプトを追加する場合は、`mode_decomp_ml.domain.sphere_grid` の
`sphere_grid_domain_cfg` を使って domain を作ること（range の再実装禁止）。

## 5. seed
- `seed` は top-level に置く
- split / model init / torch seed を必ず統制

## 6. run dir
- 出力は `runs/<name>/<process>/` 固定（docs/04）
- run_id は廃止し、同一 `<name>/<process>` は常に上書き

## 7. Multi-run（比較）
- pipeline で method sweep を実行する（decompose_list / model_list）
- 例: `python -m mode_decomp_ml.cli.run -m task=pipeline task.decompose_list=fft2,zernike task.coeff_post_list=none,pca`

## 8. 追加の運用ルール
- preprocessing では decomposer によって `coeff_post` が自動で無効化される場合がある（例: POD系は PCA を無効化）。明示的に強制する場合は `coeff_post.force: true` を指定する。
- pipeline は `task.stages` と `task.energy_threshold`（energy_cumsumの閾値）をサポートする。
