# モード分解手法

## 全体像

モード分解は `field -> coeff` を行い、学習は基本的に `cond -> coeff` を回帰します。

分類:
- 特殊関数（解析）系
- データ駆動系（POD/DictLearning/AE 等）
- グラフ/離散固有系
- 近似変換（補間を含む）系

## 分解前の空間補完について

現状は「補完を必須にする」設計ではなく、mask/weights を用いた推定（weighted LS 等）を基本とします。
必要な補完（interpolation）がある手法は、その decomposer 実装側で完結させます。

## scalar / vector の扱い

| 種別 | 入力 | 出力 coeff の形 | 注意 |
|---|---|---|---|
| scalar | `H×W×1` | `CK` / `CHW` / `K` | 評価は mask 内（domain+dataset）のみ |
| vector | `H×W×2` | `CK` / `CHW` / `K` / parts | channel 合算の指標（energy/n_req）に注意 |

## domain × 手法（テンプレ表）

このマニュアルでは “第三者が最初に選べる” ことを優先し、代表的な組み合わせだけ先に固定します。
詳細・網羅表は canonical docs を参照してください。

参照（網羅）:

- 手法カタログ: `docs/20_METHOD_CATALOG.md`
- plugin 互換: `docs/11_PLUGIN_REGISTRY.md`

### 代表的な選択肢（まずここから）

| domain | 優先候補（解析/直交寄り） | 代替（汎用/データ駆動） | 注意 |
|---|---|---|---|
| rectangle | `dct2`, `fft2`, `pswf2d_tensor` | `pod_svd`, `dict_learning`, `autoencoder`, `graph_fourier` | `fft2` は周期境界。`autoencoder` は依存（torch）と学習設定に注意 |
| disk | `zernike`, `pseudo_zernike`, `fourier_bessel`, `fourier_jacobi` | `polar_fft`, `graph_fourier`, `pod_em` | `polar_fft` は補間誤差。`fft2` を使う場合は zero-fill 近似 |
| annulus | `annular_zernike` | `polar_fft`, `graph_fourier`, `pod_em` | annulus mask 外を評価に含めない（指標の水増し防止） |
| arbitrary_mask | （解析系は不向き） | `gappy_graph_fourier`, `rbf_expansion`, `pod_em` | 可変maskでは gappy/EM が有利になりやすい |
| sphere_grid | `spherical_harmonics` | `spherical_slepian` | `spherical_slepian` は ROI 前提のことが多い（問題設定を揃える） |
| mesh | `laplace_beltrami` | （必要なら）POD/学習系 | 可視化/評価が格子と異なる（triangulation 等） |

### “offset優勢” なデータでの推奨

ベンチのように「オフセット（定数）が支配的」な場合は、残差（不均一成分）と分けると比較が安定します。

- decomposer wrapper: `offset_residual`
  - offset（各サンプルの mean）を分離し、残差だけを inner decomposer で分解します
  - inner の選択は上表と同じ
