# Task 052 (P0): 矩形基底：FFT2 / DCT2 / DST2 をDecomposerとして実装

## 目的
矩形領域の代表的分解（2D FFT / DCT / DST）を `Decomposer` として実装し、
円領域以外のベースラインを作る。

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
- `fft2` decomposer（複素係数）
- `dct2` decomposer（実数係数）
- （P1扱いでも可）`dst2` decomposer
- maskがある場合のポリシーを明示（require_full をデフォルトに）
- coeff_meta に周波数/インデックス対応、規約（シフト、正規化）を保存

### Out
- 任意マスク上での厳密な周波数分解（NUFFT等、P2）

## 実装方針（Codex向け）
### 1) FFT2
- `A = fft2(field)`（必要なら `fftshift` する）
- `coeff_meta` に
  - `shifted: true/false`
  - `freq_kx`, `freq_ky`（サンプル間で固定なら省略可）
  - 正規化（numpyのfftは非正規化なので `norm` オプションを統一）
- `inverse_transform` は `ifft2`

### 2) DCT2 / DST2
- scipy.fft の `dctn` / `idctn` を使用（type=2/3等を明示）
- DCTは境界不連続を緩和できることがある（比較候補として重要）
- DSTはDirichlet境界相当の用途で候補

### 3) mask扱い（重要）
- FFT/DCTは全格子が有効な前提が強い
- `mask_handling` を config に置く：
  - `require_full`: maskが全Trueでないとエラー（デフォルト）
  - `zero_fill`: mask外を0埋め（比較不能になりやすいので明示した時のみ）
  - `inpaint`: preprocessでinpaint済みを前提（maskは保持）
- silentに0埋めは禁止（docs/00）

### 4) 係数の実数化（後段）
- FFTは複素なので、学習入力としては
  - `coeff_post` で `mag/phase` または `real/imag` に変換する
- ここでは decomposer は複素を返す（規約を固定）

## ライブラリ候補
- numpy.fft（FFT）
- scipy.fft（dctn/idctn/dstn/idstn）
- numpy（shift/freq等）

## Acceptance Criteria（完了条件）
- [ ] fft2/dct2 が registry に登録され、Hydraで選べる
- [ ] coeff_meta に周波数/正規化/shift規約が保存される
- [ ] inverse_transform が動き、合成データで再構成誤差が小さい

## Verification（検証手順）
- [ ] 矩形上の合成周波数パターンで transform->inverse の誤差が小さい
- [ ] maskが全Trueでない場合、require_fullでエラーになることを確認
