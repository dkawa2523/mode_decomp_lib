# 概要（背景/課題/機能一覧）

## 想定背景 / 課題

- 条件ベクトル `cond` から、空間場 `field` を予測したい（回帰・不確かさ推定・最適化など）。
- `field` は高次元（H×W×C）なので、そのまま学習するとコストが高い／汎化が難しい。
- ドメイン（rectangle/disk/annulus/arbitrary_mask/sphere_grid/mesh）やマスク（欠損/観測領域）により、妥当な基底・評価が変わる。

## 本コードの概要

本コードは、`field` をモード分解して係数 `coeff` に変換し、学習は `cond -> coeff` として行います。推論後は `coeff -> field` に復元し、評価・可視化・ベンチマークを一貫した artifact 契約で出力します。

## 機能一覧（高レベル）

| 機能 | 役割 | 主な実装場所 |
|---|---|---|
| データセット | `(cond, field, mask)` の供給 | `src/mode_decomp_ml/data/` |
| ドメイン | 座標/マスク/重み（積分近似） | `src/mode_decomp_ml/domain/` |
| モード分解 | `field -> raw_coeff` | `src/mode_decomp_ml/plugins/decomposers/` |
| codec | `raw_coeff <-> coeff(a)` | `src/mode_decomp_ml/plugins/codecs/` |
| coeff_post | `coeff(a) <-> coeff(z)` | `src/mode_decomp_ml/plugins/coeff_post/` |
| 学習 | `cond -> coeff(a|z)` | `src/mode_decomp_ml/models/` / `src/processes/train.py` |
| 推論 | 係数予測・復元・可視化 | `src/processes/inference.py` |
| 評価 | rmse/r2/energy 等 | `src/mode_decomp_ml/evaluate/` |
| 可視化 | plots 生成 | `src/mode_decomp_ml/viz/` |
| ベンチ/レポート | ケース×手法の実行/集計 | `tools/bench/` |

## この課題にどう効くか（要点）

- ドメインに合う基底（特殊関数・グラフ・FFT 等）を選べる。
- マスクを含む評価（domain mask + dataset mask）で、誤解を減らす。
- artifact が標準化され、方法間比較・失敗原因の特定が容易。

補足:

- offset（定数）が支配的なデータでは、`offset_residual` により “定数成分” と “不均一成分” を分けて扱えます。
- `tools/bench/` で domain×手法のベンチとサマリー（`summary_benchmark.md`）を自動生成できます。
