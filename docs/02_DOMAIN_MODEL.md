# Domain Model（概念）

本プロジェクトの中心概念を定義します。**コード設計の不変の語彙**です。

---

## Data
### FieldSample
- `sample_id`: str（主キー）
- `field`: ndarray shape (H, W, C)
  - スカラー場: C=1
  - ベクトル場: C=2（vx,vy）など
- `mask`: ndarray shape (H, W) bool / {0,1}
  - True=有効領域、False=欠損/領域外
- `domain`: DomainSpec（座標系・境界）
- `cond` (condition): dict / vector
  - 学習の入力（温度、圧力、設計パラメータなど）。**時間ではない**。
- `meta`: dict（任意）

### DomainSpec
- `type`: {rect, disk, mask, points, mesh, sphere_grid}
- `coords`: 座標情報
  - rect: x,y grid
  - disk: r,theta grid / または x,y + disk mask
  - sphere_grid: lat/lon grid（theta/phi も保持）
    - `n_lat/n_lon` 指定時は `lat_range/lon_range` を自動補完（range の手書きは不要）
    - 設定例は `docs/03_CONFIG_CONVENTIONS.md` の sphere_grid セクションを参照
  - points: (N,2)
  - mesh: vertices, faces
- `boundary_condition`: optional
  - Zernikeは通常disk上で定義（境界条件は暗黙）
  - Fourier–Bessel / Laplace固有などは Dirichlet/Neumann 等を明示

---

## Decomposition
### Decomposer（一次モード分解）
- `fit(dataset)`: 基底/辞書を学習（固定基底はno-opでも良い）
- `transform(sample) -> coeff a`
- `inverse_transform(a) -> field_hat`
- 係数 `a` は「モード係数ベクトル」（実数 or 複素）で、shapeは method ごとに定義
- 係数の metadata（(n,m)対応、周波数インデックス等）も保存する

例：
- Zernike: a[(n,m)]（mの符号含む）
- FFT: A[kx,ky]（複素）
- DCT: C[u,v]（実数）
- RBF: w[j]（中心ごと）

### CoeffPost（係数後処理 = 特徴量化）
- 入力: 係数 `a`
- 出力: 学習用特徴 `z`
- `fit(A_train)` / `transform(A)` / `inverse_transform(Z)`
  - PCAなど **fitが必要** な手法は必ず学習データでfitし、状態を保存する

---

## Model
### Regressor（学習モデル）
- 入力: condition（+必要ならdomain params）
- 出力: `z` もしくは `a`
- Multi-output regression を前提（係数次元が大きい）
- scikit-learn系とPyTorch系の両方をプラグイン化

---

## Process（単独実行単位）
- preprocess: raw -> cleaned（mask維持）
- decompose.fit: decomposerのfit（必要なら）
- decompose.transform: field -> a
- coeff_post.fit: coeff -> z（学習データで）
- train: condition -> (z or a)
- predict: condition -> (z_hat or a_hat)
- reconstruct: (z_hat -> a_hat -> field_hat)
- eval: field_hat vs field の評価
- viz: 可視化・レポート
- leaderboard: 結果集計
- doctor: 環境/データ/再現性チェック

---

## Artifacts（保存物）
- config snapshot（Hydra）
- dataset meta（hash/shape）
- decomposer state（basis/indices）
- coeff_post state（PCA等）
- model weights
- metrics / predictions / recon images
