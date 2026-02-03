# mode_decomp_eval_dataset_v1

2次元空間分布の **モード分解＋学習** を評価するための synthetic データセットです。

## サブセット一覧
- scalar_rect / scalar_disk / scalar_mask
- vector_rect / vector_disk / vector_mask

各サブセットは以下を含みます:
- cond.npy   : 条件ベクトル
- field.npy  : 2D場（スカラーC=1 / ベクトルC=2）
- mask.npy   : 領域マスク（0/1）
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

## 最小の動作確認（例）
プロジェクト側のCLI入口が `python -m mode_decomp_ml.cli.run` の場合:

```
PYTHONPATH=src python -m mode_decomp_ml.cli.run task=doctor \
  dataset=npy_dir dataset.root=data/mode_decomp_eval_dataset_v1/scalar_rect \
  domain=rectangle
```

※ dataset/domain のキーは、プロジェクト側の configs/ 実装に合わせて必要に応じて調整してください。

## 典型の評価実行（例）
### 1) FFT/DCT/DST を比較（scalar_rect）
```
PYTHONPATH=src python -m mode_decomp_ml.cli.run -m task=benchmark \
  dataset=npy_dir dataset.root=data/mode_decomp_eval_dataset_v1/scalar_rect \
  domain=rectangle preprocess=basic coeff_post=none model=ridge \
  decompose=fft2,dct2,dst2
```

### 2) Zernike vs Fourier-Bessel vs POD（scalar_disk）
```
PYTHONPATH=src python -m mode_decomp_ml.cli.run -m task=benchmark \
  dataset=npy_dir dataset.root=data/mode_decomp_eval_dataset_v1/scalar_disk \
  domain=disk preprocess=basic coeff_post=none model=ridge \
  decompose=zernike,fourier_bessel,pod_svd
```

### 3) 任意形状mask：Graph Fourier / POD（scalar_mask）
```
PYTHONPATH=src python -m mode_decomp_ml.cli.run -m task=benchmark \
  dataset=npy_dir dataset.root=data/mode_decomp_eval_dataset_v1/scalar_mask \
  domain=arbitrary_mask preprocess=basic coeff_post=none model=ridge \
  decompose=graph_fourier,pod_svd
```

### 4) 学習モデル比較（例：Ridge vs GPR）
```
PYTHONPATH=src python -m mode_decomp_ml.cli.run -m task=benchmark \
  dataset=npy_dir dataset.root=data/mode_decomp_eval_dataset_v1/scalar_disk \
  domain=disk preprocess=basic coeff_post=pca \
  decompose=zernike \
  model=ridge,gpr
```

