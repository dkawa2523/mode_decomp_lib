# アーキテクチャ（Greenfield）

本プロジェクトは「2D場のモード分解 → 係数を条件から回帰 → 逆変換で再構成」の比較基盤です。
既存実装の改修ではなく、**0から**以下の “不変契約” を中心に設計します（docs/00）。

## 依存方向（原則）
- `configs/` が “真実”
- `src/processes/*` は **パイプラインの組み立てのみ**（アルゴリズムを持たない）
- アルゴリズムは `src/mode_decomp_ml/*` に閉じる
  - preprocess / domain / decompose / coeff_post / models / evaluate / viz / tracking

## パイプライン（抽象）
1. dataset: sample = {cond, field, mask, meta}
2. domain: 座標生成・mask検証・正規化
3. preprocess: 欠損/ノイズ処理（train/test同一）
4. decompose: field → a（係数） + coeff_meta
5. codec: a(raw) → a_vec（lossless/圧縮） + codec meta
6. coeff_post: a_vec → z（任意） + inverse_transform
7. model: cond → (a_vec or z)
8. reconstruct: 予測係数 → field_hat
9. evaluate: coeff誤差 + field誤差 + 解釈性補助
10. artifacts: config/meta/metrics/preds/model を統一保存

## P0で扱う領域と手法
- domain: rectangle / disk
- decompose: FFT2 / Zernike
- codec: none / fft_complex_codec_v1 / zernike_pack_v1
- coeff_post: none / standardize / PCA
- model: Ridge（多出力）

## 拡張ポイント（plugins）
- decomposer/codec/coeff_post/model は `src/mode_decomp_ml/plugins/` に集約し、registry で解決する。
- 拡張の起点は `configs/*/*.yaml` と `scripts/bench/matrix.yaml`。
- 詳細な互換表・推奨用途は docs/11_PLUGIN_REGISTRY.md を参照。

## P1/P2の接続点（実装済み + TODO）
- Helmholtz（vector field）は `plugins/decomposers/helmholtz.py` にあり、通常の decomposer と同じ流れで使える。
- Autoencoder は `plugins/decomposers/autoencoder.py` にあり、codec/coeff_post を介して回帰へ接続する。
- VAE は未実装（現状は Autoencoder のみ）。
- ClearML は `mode_decomp_ml/tracking/clearml.py` と `configs/clearml/` で接続。`maybe_log_run` 経由で opt-in。

## Bench（quick/full）
- `scripts/bench/matrix.yaml` が sweep の真実。quick/full は profile で切替える。
- `scripts/bench/run_p0p1_p2ready.sh`（quick）と `scripts/bench/run_full.sh`（full）を入口にする。

P1/P2は backlog（queue_p1/queue_p2）で管理し、P0の比較基盤を壊さずに拡張します。
