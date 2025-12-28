# Task 053 (P1): 円領域：Fourier–Bessel（Bessel/Dini）分解の実装

## 目的
円領域でZernike以外の直交基底として有用な **Fourier–Bessel（Bessel/Dini）分解** を導入し、
円形境界条件の違い（Dirichlet/Neumann）を明示して扱えるようにする。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/050_decomposer_registry.md
- depends_on: work/tasks/040_preprocess_pipeline.md

## スコープ
### In
- `fourier_bessel` decomposer を追加
- boundary_condition を config で指定（dirichlet/neumann）
- Bessel零点の計算と基底生成をキャッシュ
- 入力がcartesian gridの場合、polarへ変換する方針を確定（preprocessで推奨）
- coeff_meta に (m,n) と零点、規約を保存

### Out
- 高速化（θ方向FFT分離などの最適化）はP2

## 実装方針（Codex向け）
### 1) 基底の定義（実装前に規約を固定）
- 例：Dirichlet 境界 `f(r=1)=0` のとき
  - α_{m,n} = J_m の n番目の零点
  - 基底: Φ_{m,n}(r,θ) = J_m(α_{m,n} r) * e^{i m θ}
- Neumann 境界 `∂f/∂r(r=1)=0` のとき
  - α_{m,n} = J_m' の零点（導関数の零点）

### 2) 実装ステップ（シンプル版）
- domainから `r_grid, theta_grid` を得る（または preprocess の `to_polar_grid` を前提）
- `scipy.special.jn_zeros(m, nmax)` または `jnp_zeros` で零点を取得
- (m,n) を列挙して basis を作る（複素）
- 係数 a_{m,n} を離散内積で計算
  - 重みは `r` を含む（面積要素）
  - mask がある場合は mask内で重み付き最小二乗に切替できるようにする（オプション）

### 3) inverse_transform
- `f_hat = Σ a_{m,n} Φ_{m,n}`
- 実数場なら最終的に real を取る（規約をcoeff_metaに残す）

### 4) config パラメータ例
- `m_max`, `n_max`
- `boundary_condition: dirichlet|neumann`
- `use_complex: true`
- `polar_grid: {nr, ntheta}`（固定）

### 5) 注意点
- 入力がcartesianのままだと、r,θのサンプリングが歪む → polar resample 推奨
- 離散直交性は格子/重みに依存 → `recon_error` テストで担保

## ライブラリ候補
- scipy.special（jn, jn_zeros, jnp_zeros）
- numpy（複素演算）

## Acceptance Criteria（完了条件）
- [ ] fourier_bessel decomposer が registry に登録される
- [ ] boundary_condition が config/coeff_meta に記録される
- [ ] 合成データで transform->inverse の再構成誤差が小さい

## Verification（検証手順）
- [ ] 既知の( m,n )係数で合成した場を再構成できる（unit test）
- [ ] disk domain 以外で実行すると明確にエラーになる
