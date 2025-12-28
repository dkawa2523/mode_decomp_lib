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
5. coeff_post: a → z（任意） + inverse_transform
6. model: cond → (a or z)
7. reconstruct: 予測係数 → field_hat
8. evaluate: coeff誤差 + field誤差 + 解釈性補助
9. artifacts: config/meta/metrics/preds/model を統一保存

## P0で扱う領域と手法
- domain: rectangle / disk
- decompose: FFT2 / Zernike
- coeff_post: none / standardize / PCA
- model: Ridge（多出力）

P1/P2は backlog（queue_p1/queue_p2）で管理し、P0の比較基盤を壊さずに拡張します。
