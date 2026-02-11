# 実行フロー（ドメイン別）

このページは domain ごとの「違い（mask/weights/coords）」と、それがモード分解・評価にどう影響するかを図で示します。

## rectangle

<div class="mermaid">
flowchart TD
  A[field on grid] --> B[FFT/DCT/Wavelet/POD/GraphFourier]
  B --> C[coeff]
</div>

domain 固有の注意:

- 座標: 一般に `x_range/y_range` の等間隔格子（dx/dy は評価や補間に影響）。
- マスク: 典型的には全点有効（dataset mask があれば combine）。
- 境界/周期:
  - `fft2` は暗黙に周期境界（端で不連続だと高周波が増えやすい）。
  - `dct2` は偶対称延長（境界での勾配条件に相当）で、定数成分（DC）を 1成分で表しやすい。

## disk

<div class="mermaid">
flowchart TD
  A[field + domain mask] --> B[Zernike / Fourier-Bessel / PolarFFT / GraphFourier / POD]
  B --> C[coeff]
</div>

domain 固有の注意:

- 座標: `r/theta` が使える（解析系基底はこれに強く依存）。
- マスク: 円の外は無効。評価は **disk mask 内だけ**で行う。
- 近似手法:
  - `fft2` を使う場合は `disk_policy=mask_zero_fill`（円外を 0埋めする近似）になる。
  - `polar_fft` は補間誤差が出るので、ベンチでは許容誤差や `n_r/n_theta` が重要。

## annulus

<div class="mermaid">
flowchart TD
  A[field + annulus mask] --> B[Annular-Zernike / PolarFFT / GraphFourier]
  B --> C[coeff]
</div>

domain 固有の注意:

- マスク: 内側の穴（r<r_inner）が無効。評価は **annulus mask 内だけ**で行う。
- `polar_fft` は annulus に自然拡張できるが、やはり補間誤差が支配的になり得る。

## arbitrary_mask

<div class="mermaid">
flowchart TD
  A[field + mask] --> B[Gappy/Graph/RBF/POD系]
  B --> C[coeff]
</div>

domain 固有の注意:

- マスク: 不規則形状・欠損点を含む。手法によって “固定マスク前提” がある。
- 可変マスク（サンプルごとに mask が変わる）では:
  - fixed basis + ridge 推定（例: gappy 系）
  - RBF/局所基底
  - 欠損対応 POD（EM/ALS 系）
 などが有効になりやすい。

## sphere_grid

<div class="mermaid">
flowchart TD
  A[lon/lat grid] --> B[SphericalHarmonics / Slepian]
  B --> C[coeff]
</div>

domain 固有の注意:

- 座標: 緯度経度格子。経度方向は周期（seam の不連続に注意）。
- 重み: 面積要素（例: `cos(lat)`）を使った weighted 指標が重要になりやすい。
- `spherical_slepian` は ROI（cap 等）前提のことが多いので、テストケース側の “問題設定” と揃える必要がある。

## mesh

<div class="mermaid">
flowchart TD
  A[mesh vertices] --> B[Laplace-Beltrami / mesh basis]
  B --> C[coeff]
</div>

domain 固有の注意:

- データは格子ではなく頂点上（V点）。`field` の shape や可視化（triangulation）が rectangle とは異なる。
- `laplace_beltrami` は “メッシュ上の離散ラプラシアン固有基底” を使うため、メッシュ品質（非一様性、穴）で数値が変わる。
