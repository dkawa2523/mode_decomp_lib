# mode_decomp_eval_dataset_v1

2次元空間分布の **モード分解＋学習** を評価するための synthetic データセットです。

## サブセット一覧
- scalar_rect / scalar_disk / scalar_mask / scalar_annulus / scalar_sphere
- vector_rect / vector_disk / vector_mask

各サブセットは以下を含みます:
- cond.npy   : 条件ベクトル（legacy / npy）
- field.npy  : 2D場（スカラーC=1 / ベクトルC=2, legacy / npy）
- mask.npy   : 領域マスク（0/1, legacy / npy）
- conditions.csv : 条件テーブル（CSV, id列 + 任意列）
- fields/    : 各条件の `x,y,f` CSV（CSV, 1条件1ファイル）
  - vector の場合は `<id>_fx.csv` / `<id>_fy.csv` の2ファイル
- README.md  : サブセット説明
- example_config.yaml : Hydra上書き例（参考）

## 推奨配置
プロジェクトルートの `data/` 配下に展開してください。

例:
```
<project_root>/
  data/
    mode_decomp_eval_dataset_v1/
      scalar_rect/
      ...
```

## 最小の動作確認（例, CSV）
プロジェクト側のCLI入口が `python -m mode_decomp_ml.cli.run` の場合:

```
PYTHONPATH=src python -m mode_decomp_ml.cli.run task=doctor \
  dataset=csv_fields dataset.conditions_csv=data/mode_decomp_eval_dataset_v1/scalar_rect/conditions.csv \
  dataset.fields_dir=data/mode_decomp_eval_dataset_v1/scalar_rect/fields \
  dataset.grid.H=64 dataset.grid.W=64 dataset.grid.x_range=[0.0,1.0] dataset.grid.y_range=[0.0,1.0] \
  domain=rectangle
```

※ dataset/domain のキーは、プロジェクト側の configs/ 実装に合わせて必要に応じて調整してください。

## 典型の評価実行（例）
### 1) FFT/DCT/DST を比較（scalar_rect）
```
PYTHONPATH=src python -m mode_decomp_ml.cli.run -m task=pipeline \
  dataset=csv_fields dataset.conditions_csv=data/mode_decomp_eval_dataset_v1/scalar_rect/conditions.csv \
  dataset.fields_dir=data/mode_decomp_eval_dataset_v1/scalar_rect/fields \
  dataset.grid.H=64 dataset.grid.W=64 dataset.grid.x_range=[0.0,1.0] dataset.grid.y_range=[0.0,1.0] \
  domain=rectangle preprocess=basic coeff_post=none model=ridge \
  decompose=fft2,dct2,dst2
```

### 2) Zernike vs Fourier-Bessel vs POD（scalar_disk）
```
PYTHONPATH=src python -m mode_decomp_ml.cli.run -m task=pipeline \
  dataset=csv_fields dataset.conditions_csv=data/mode_decomp_eval_dataset_v1/scalar_disk/conditions.csv \
  dataset.fields_dir=data/mode_decomp_eval_dataset_v1/scalar_disk/fields \
  dataset.grid.H=64 dataset.grid.W=64 dataset.grid.x_range=[-1.0,1.0] dataset.grid.y_range=[-1.0,1.0] \
  domain=disk preprocess=basic coeff_post=none model=ridge \
  decompose=zernike,fourier_bessel,pod_svd
```

### 3) 任意形状mask：Graph Fourier / POD（scalar_mask）
```
PYTHONPATH=src python -m mode_decomp_ml.cli.run -m task=pipeline \
  dataset=csv_fields dataset.conditions_csv=data/mode_decomp_eval_dataset_v1/scalar_mask/conditions.csv \
  dataset.fields_dir=data/mode_decomp_eval_dataset_v1/scalar_mask/fields dataset.mask_file=mask.npy \
  dataset.grid.H=64 dataset.grid.W=64 dataset.grid.x_range=[-1.0,1.0] dataset.grid.y_range=[-1.0,1.0] \
  domain=arbitrary_mask preprocess=basic coeff_post=none model=ridge \
  decompose=graph_fourier,pod_svd
```

### 3b) Annulus（scalar_annulus）
```
PYTHONPATH=src python -m mode_decomp_ml.cli.run -m task=pipeline \
  dataset=csv_fields dataset.conditions_csv=data/mode_decomp_eval_dataset_v1/scalar_annulus/conditions.csv \
  dataset.fields_dir=data/mode_decomp_eval_dataset_v1/scalar_annulus/fields dataset.mask_file=mask.npy \
  dataset.grid.H=64 dataset.grid.W=64 dataset.grid.x_range=[-1.0,1.0] dataset.grid.y_range=[-1.0,1.0] \
  domain=annulus preprocess=basic coeff_post=none model=ridge \
  decompose=annular_zernike
```

### 3c) Sphere grid（scalar_sphere）
```
PYTHONPATH=src python -m mode_decomp_ml.cli.run -m task=pipeline \
  dataset=csv_fields dataset.conditions_csv=data/mode_decomp_eval_dataset_v1/scalar_sphere/conditions.csv \
  dataset.fields_dir=data/mode_decomp_eval_dataset_v1/scalar_sphere/fields \
  dataset.grid.H=18 dataset.grid.W=36 dataset.grid.lon_range=[-180.0,180.0] dataset.grid.lat_range=[-90.0,90.0] \
  domain=sphere_grid preprocess=basic coeff_post=none model=ridge \
  decompose=spherical_harmonics
```

### 4) 学習モデル比較（例：Ridge vs GPR）
```
PYTHONPATH=src python -m mode_decomp_ml.cli.run -m task=pipeline \
  dataset=csv_fields dataset.conditions_csv=data/mode_decomp_eval_dataset_v1/scalar_disk/conditions.csv \
  dataset.fields_dir=data/mode_decomp_eval_dataset_v1/scalar_disk/fields \
  dataset.grid.H=64 dataset.grid.W=64 dataset.grid.x_range=[-1.0,1.0] dataset.grid.y_range=[-1.0,1.0] \
  domain=disk preprocess=basic coeff_post=pca \
  decompose=zernike \
  model=ridge,gpr
```
