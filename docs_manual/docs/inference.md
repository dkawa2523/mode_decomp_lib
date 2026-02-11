# 推論機能

## 概要

推論は `cond -> coeff -> field_hat` を生成し、可視化・統計・（必要なら）最適化を行います。

推論は通常 “真値 field が無い” 前提でも動くため、出力は次の2系統になります。

- 生成結果の保存（`coeff_pred`, `field_hat`）
- 健全性チェック用の図（分布、外れ値、係数のスケールなど）

## 単一条件

| 入力 | 出力 | 想定ユースケース |
|---|---|---|
| `cond` | `coeff`, `field_hat`, plots | 1条件の予測/診断 |

## 範囲 Grid 実行

| 入力 | 出力 | 想定ユースケース |
|---|---|---|
| cond の grid | `field_hat` のバッチ | 条件スイープ、感度分析 |

## 最適化

| 入力 | 出力 | 想定ユースケース |
|---|---|---|
| cond 範囲 + objective | best cond + field | 設計最適化 |

sampler / objective / loss は `configs/` の該当 group に集約されています（最適化が必要な場合のみ参照）。
