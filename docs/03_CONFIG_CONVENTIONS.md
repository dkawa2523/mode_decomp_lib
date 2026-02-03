# Config Conventions（Hydra + run.yaml）

本プロジェクトは Hydra を内部実装として使いますが、**非DSユーザーは run.yaml を入口**にします。

## 1. configの階層（現行）
- `configs/config.yaml`: Hydra 入口（defaults + seed + run_dir）
- `configs/dataset/*.yaml`: dataset preset（`synthetic`, `npy_dir`）
- `configs/split/*.yaml`: split preset（`all`）
- `configs/domain/*.yaml`: domain preset（`rectangle`, `disk`, `annulus`, `sphere_grid`, `arbitrary_mask`）
- `configs/preprocess/*.yaml`: preprocess pipeline（`basic`）
- `configs/decompose/{analytic,data_driven}/*.yaml`: decomposer preset（`fft2`, `zernike`, `pod_svd`, `annular_zernike`, `wavelet2d`, `spherical_harmonics`）
- `configs/codec/*.yaml`: coeff codec preset（`none`, `fft_complex_codec_v1`, `zernike_pack_v1`, `wavelet_pack_v1`, `sh_pack_v1`）
- `configs/coeff_post/*.yaml`: coeff post preset（`none`, `pca`, `quantile`, `power_yeojohnson`）
- `configs/model/*.yaml`: regressor preset（`ridge`, `multitask_lasso`, `mtgp`, `xgb`, `lgbm`, `catboost`）
- `configs/eval/*.yaml`: metrics/report settings（`basic`）
- `configs/viz/*.yaml`: visualization settings（`basic`）
- `configs/uncertainty/*.yaml`: uncertainty settings（`gpr_mc`）
- `configs/clearml/*.yaml`: tracking settings（`basic`）
- `configs/task/*.yaml`: task routing（doctor/train/predict/reconstruct/eval/benchmark/leaderboard/viz）

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

補完される範囲（内部で利用）:
- `lat_range`: [-90.0, 90.0]
- `lon_range`: [-180.0, 180.0 - 360/n_lon]（周期重複を避ける）

dataset 生成スクリプトを追加する場合は、`mode_decomp_ml.domain.sphere_grid` の
`sphere_grid_domain_cfg` を使って domain を作ること（range の再実装禁止）。

## 5. seed
- `seed` は top-level に置く
- split / model init / torch seed を必ず統制

## 6. run dir
- 出力は `runs/<tag>/<run_id>/` 固定（docs/04）
- Hydra run dir でも `run_dir` を最終保存先として扱う

## 7. Multi-run（比較）
- Hydra multirun で method sweep を可能にする
- ただし比較可能性（docs/00）を壊す sweep（datasetを混ぜる等）は禁止
- 例: `python -m mode_decomp_ml.cli.run -m task=benchmark task.decompose_list=fft2,zernike task.coeff_post_list=none,pca`
