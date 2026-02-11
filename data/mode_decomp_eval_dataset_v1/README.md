# mode_decomp_eval_dataset_v1

2次元空間分布の **モード分解＋学習** を評価するための synthetic データセットです。

## サブセット一覧
- scalar_rect / scalar_disk / scalar_mask / scalar_annulus / scalar_sphere
- vector_rect / vector_disk / vector_mask

各サブセットは以下を含みます:
- cond.npy   : 条件ベクトル（float32）
- field.npy  : 2D場（float32, shape: N×H×W×C。scalarはC=1、vectorはC=2）
- mask.npy   : 領域マスク（0/1, uint8。shape: N×H×W または H×W）
- manifest.json : `npy_dir` 用のメタ情報（grid/domain/field_kind）
- README.md  : サブセット説明
- example_config.yaml : Hydra上書き例（参考）

補足（legacy / optional）:
- conditions.csv / fields/*.csv（`csv_fields` dataset用）は互換目的で残しますが、巨大化するため **Git管理しません**。

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

## 最小の動作確認（推奨: npy_dir）
プロジェクト側のCLI入口が `python3 -m mode_decomp_ml.cli.run` の場合:

```
PYTHONPATH=src python3 -m mode_decomp_ml.cli.run task=doctor \
  dataset=npy_dir dataset.root=data/mode_decomp_eval_dataset_v1/scalar_rect dataset.mask_policy=allow_none
```

※ manifest.json が無い場合は `domain=...` の指定が必要です。

## 典型の評価実行（例）
### 1) FFT/DCT/DST を比較（scalar_rect）
```
PYTHONPATH=src python3 -m mode_decomp_ml.cli.run -m task=pipeline \
  dataset=npy_dir dataset.root=data/mode_decomp_eval_dataset_v1/scalar_rect dataset.mask_policy=allow_none \
  preprocess=basic coeff_post=none model=ridge \
  decompose=fft2,dct2,dst2
```

### 2) Zernike vs Fourier-Bessel vs POD（scalar_disk）
```
PYTHONPATH=src python3 -m mode_decomp_ml.cli.run -m task=pipeline \
  dataset=npy_dir dataset.root=data/mode_decomp_eval_dataset_v1/scalar_disk dataset.mask_policy=allow_none \
  preprocess=basic coeff_post=none model=ridge \
  decompose=zernike,fourier_bessel,pod_svd
```

### 3) 任意形状mask：Graph Fourier / POD（scalar_mask）
```
PYTHONPATH=src python3 -m mode_decomp_ml.cli.run -m task=pipeline \
  dataset=npy_dir dataset.root=data/mode_decomp_eval_dataset_v1/scalar_mask dataset.mask_policy=require \
  preprocess=basic coeff_post=none model=ridge \
  decompose=graph_fourier,pod_svd
```

### 3b) Annulus（scalar_annulus）
```
PYTHONPATH=src python3 -m mode_decomp_ml.cli.run -m task=pipeline \
  dataset=npy_dir dataset.root=data/mode_decomp_eval_dataset_v1/scalar_annulus dataset.mask_policy=allow_none \
  preprocess=basic coeff_post=none model=ridge \
  decompose=annular_zernike
```

### 3c) Sphere grid（scalar_sphere）
```
PYTHONPATH=src python3 -m mode_decomp_ml.cli.run -m task=pipeline \
  dataset=npy_dir dataset.root=data/mode_decomp_eval_dataset_v1/scalar_sphere dataset.mask_policy=allow_none \
  preprocess=basic coeff_post=none model=ridge \
  decompose=spherical_harmonics
```

### 4) 学習モデル比較（例：Ridge vs GPR）
```
PYTHONPATH=src python3 -m mode_decomp_ml.cli.run -m task=pipeline \
  dataset=npy_dir dataset.root=data/mode_decomp_eval_dataset_v1/scalar_disk dataset.mask_policy=allow_none \
  preprocess=basic coeff_post=pca \
  decompose=zernike \
  model=ridge,gpr
```
