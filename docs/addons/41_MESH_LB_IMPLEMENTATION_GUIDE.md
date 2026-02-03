# Mesh + Laplace–Beltrami 実装ガイド（Task 310 補助）

Task 310 は “曲面/メッシュ” を扱うための最初の大きな拡張です。
Codex が迷走しやすいので、ここで **最小実装の設計** を固定します。

## ゴール（最小）
- **単一の固定メッシュ**（同一トポロジ）上のスカラー/ベクトル値を展開できる
- Discrete Laplace–Beltrami の最小固有対を取得し、
  - transform: f → a
  - inverse: a → f_hat
  ができる

## データモデル（推奨）
- メッシュ: (V,F)
  - V: shape [nV, 3]（2D平面でも3Dで持つと将来が楽）
  - F: shape [nF, 3] int（三角形）
- 場: per-vertex 値を基本にする
  - field: shape [nV, C] （scalar: C=1, vector: C=2）
- これらを `.npz` にまとめるのが最小:
  - `V`, `F`, `field`, optional `maskV`（頂点mask）

## 離散化（最小の数学）
- cotan Laplacian（よく使われる離散LB）を用意
  - L: sparse [nV, nV]
- mass matrix（Voronoi/ barycentric でもよい）
  - M: diagonal sparse [nV, nV]

一般化固有値問題:
- L φ = λ M φ
- 小さい λ の固有ベクトルが “低周波モード”

## transform / inverse（最小）
- M直交規格化を前提（または計算で補正）
- transform:
  - a_i = φ_i^T M f
- inverse:
  - f_hat = Σ a_i φ_i

## 実装の落とし穴
- 最小固有値 λ0 ≈ 0 の定数モードは扱いを決める（保持/除外）
- 辺ケース（境界ありメッシュ）は境界条件の意味が入る
  - 最小実装では “そのまま” とし、docsに制約を書く（後で改善）
- 固有分解は重い:
  - まず toy mesh（~1k頂点）で成立させる
  - k=16〜64程度から始める

## 依存（最小）
- できるだけ SciPy（scipy.sparse, scipy.sparse.linalg.eigsh）で完結
- 追加ライブラリは P2 preflight の方針に従う（新規依存は必要最小限）

## 検証（最小）
- transform→inverse の再構成誤差が小さい
- k を増やすほど誤差が下がる
- 係数スペクトルが “低周波から高周波” の順になっている（固有値で確認）
