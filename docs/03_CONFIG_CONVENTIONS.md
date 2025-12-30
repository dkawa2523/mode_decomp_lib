# Config Conventions（Hydra前提）

本プロジェクトは Hydra を前提にします（導入タスクは work/tasks に含む）。

## 1. configの階層（推奨）
- `conf/config.yaml`: 入口（defaults + seed + run_dir）
- `conf/data/*.yaml`: dataset, split, paths
- `conf/domain/*.yaml`: rect/disk/mask/points/mesh
- `conf/preprocess/*.yaml`: preprocess pipeline（順序あり）
- `conf/vector/*.yaml`: vector transform（必要な場合のみ）
- `conf/decompose/*.yaml`: decomposer（method + params）
- `conf/coeff_post/*.yaml`: coeff_post（PCA等）
- `conf/model/*.yaml`: regressor
- `conf/train/*.yaml`: training params
- `conf/eval/*.yaml`: metrics/report settings
- `conf/viz/*.yaml`: visualization settings

## 2. 命名規約
- configキーは snake_case
- method名は registry key と一致させる（例: `decompose.method: zernike`）

## 3. seed
- `seed` は top-level に置く
- split / model init / torch seed を必ず統制

## 4. run dir
- Hydra run dir は `outputs/<process>/<YYYY-MM-DD>/<HH-MM-SS>_<tag>_<jobnum>/` を推奨
  - single run は jobnum=0、multirun では 0,1,2... が付与される
- すべての artifact は run dir 配下に保存する（docs/04）

## 5. Multi-run（比較）
- Hydra multirun で method sweep を可能にする
- ただし比較可能性（docs/00）を壊す sweep（datasetを混ぜる等）は禁止
- 例: `python -m mode_decomp_ml.cli.run -m task=benchmark task.decompose_list=fft2,zernike task.coeff_post_list=none,pca`
