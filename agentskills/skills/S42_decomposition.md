# S42: モード分解プラグイン実装（Decomposer）

## 目的
- Zernike / Fourier–Bessel / FFT / DCT / RBF / Wavelet / Graph Fourier / Laplace系など、
  **分解手法を追加しても仕様がぶれない**ように、同一I/O契約で実装する。

## やること手順
1. **domain制約を明記**
   - `disk` 限定か、`rect` か、`mask/points/mesh` か
   - boundary condition が必要なら config に必ず追加（Dirichlet/Neumann 等）

2. **係数表現（a）と meta を設計**
   - a の shape と dtype（実数/複素）
   - index の意味（(n,m)、(kx,ky)、center id…）を `coeff_meta.json` に保存

3. **transform の実装**
   - 入力：field(H,W,C), mask(H,W), domain_spec
   - 出力：a（必要ならチャネルごとに連結）
   - maskを勝手に変更しない（silent fill禁止）
   - 直交基底なら内積、欠損が多いなら最小二乗などを選べるようにする

4. **inverse_transform の実装**
   - a から `field_hat` を生成できること（少なくとも近似）
   - 返す shape を入力と一致させる

5. **state 保存**
   - 固定基底は state 無しでもよいが、meta は必須
   - 学習基底（POD/NMF/辞書学習等）は state を必ず保存

6. **テスト**
   - 合成データで round-trip（transform→inverse）誤差が小さい
   - domain不一致時は明確にエラー

## 事故りやすい点
- 係数の並び順を変えたのに meta を更新しない（比較不能）
- FFTの shift / norm の規約を統一しない
- disk系で r の重み（面積要素）を忘れる
- maskがあるのに暗黙で0埋めする

## DoD（Definition of Done）
- registry 登録（Hydraから選べる）
- coeff_meta.json が出る（意味が説明可能）
- round-trip テストがある
- docs/20_METHOD_CATALOG.md が更新される（必要なら）

## よくある差分
- 直交内積 vs 欠損対応（最小二乗）
- complex係数を返す vs 実数化（実数化は原則 coeff_post で）
- ベクトル場の扱い（componentwise / div-curl）
