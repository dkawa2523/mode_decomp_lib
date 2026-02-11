# 学習機能

## 概要

学習は基本的に `cond -> coeff(a|z)` を回帰します。評価は `coeff` 空間だけでなく、復元した `field_hat` に対する指標も重要です。

### PCA（係数の次元削減）について

本コードの標準ワークフローとして、モード分解（`field -> coeff(a)`）の **後段に PCA を挟む**選択肢があります。

- 位置づけ: `coeff_post`（例: `pca`）
- 目的: 係数次元の削減、学習の安定化、ノイズ低減
- 実装/設定:
  - 実装: `src/mode_decomp_ml/plugins/coeff_post/`
  - 設定: `configs/coeff_post/`

ベンチマークでも `coeff_post=pca` を有効にしたケースは、係数空間の学習指標だけでなく **field 空間（復元後）**の指標で評価するのが重要です（`val_field_r2` 等）。

## 前処理（テンプレ）

| 区分 | 名前 | 入力 | 出力 | 目的 | 依存 |
|---|---|---|---|---|---|
| field preprocess | basic | field | field | 欠損/型/正規化など | numpy |
| coeff_post | pca | coeff(a) | coeff(z) | 次元削減 | sklearn |
| coeff_post | none | coeff(a) | coeff(a) | 何もしない（baseline） | なし |

## 学習モデル（テンプレ）

| model | 入力 | 出力 | 概要 | 想定利用 | 依存 |
|---|---|---|---|---|---|
| ridge | cond | coeff | 線形回帰 | baseline | sklearn |
| gpr | cond | coeff + std | ガウス過程 | 小規模/不確かさ | sklearn |
| mtgp | cond | coeff + std | 多出力GP | 依存重い | torch/gpytorch |

## 評価 metrics（テンプレ）

| metric | 空間 | 概要 | Tips |
|---|---|---|---|
| rmse | coeff/field | 誤差の二乗平均平方根 | mask内評価 |
| r2 | coeff/field | 決定係数 | 定数系列は NaN になり得る |
| energy_cumsum | coeff | 係数エネルギー累積 | 並び順に注意（topkも併用） |

### 「coeff 指標」と「field 指標」を分けて見る

decomposer/codec/coeff_post の組合せで係数のスケールが変わるため、手法間比較は **field 指標**が基本になります。

- coeff 指標（例: `val_r2`）: モデルの学習が進んでいるかの内部診断
- field 指標（例: `val_field_r2`）: 最終目的（場の再現）の比較軸
