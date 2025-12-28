# Task 055 (P2): 多解像：Wavelet（DWT2/SWT2）分解のDecomposer

## 目的
局所構造や多スケール性が強いデータに対して、Wavelet分解を Decomposer として導入する。
（まずは矩形・mask全True前提の実装から）

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/050_decomposer_registry.md

## スコープ
### In
- `wavelet_dwt2` decomposer を実装（pywt.wavedec2 / waverec2）
- level, wavelet名（dbN等）, mode を config 化
- 係数ツリーを1次元ベクトルにflattenし、inverse用の shapes/meta を保存
- mask_handling を明示（require_fullをデフォルト）

### Out
- 任意マスク上のwavelet（補完込み）はP3
- wavelet packet はP3

## 実装方針（Codex向け）
### 1) DWT2
- `coeffs = pywt.wavedec2(field, wavelet, level=..., mode=...)`
- coeffs は (cA, (cH,cV,cD), ...) のツリー構造
- flatten して a にする：
  - `a = np.concatenate([cA.ravel(), cH.ravel(), cV.ravel(), cD.ravel(), ...])`
- `coeff_meta` に treeの shapes と順序を保存（逆変換に必須）

### 2) inverse_transform
- flatten a を meta に従って元のツリー構造に戻す
- `field_hat = pywt.waverec2(coeffs, wavelet, mode=...)`

### 3) 注意
- waveletは境界モード（symmetric等）で結果が変わる → 必ず meta に記録
- dtype/float精度で差が出るのでテストはtolを緩める

## ライブラリ候補
- PyWavelets（pywt）
- numpy

## Acceptance Criteria（完了条件）
- [ ] wavelet decomposer が registry に登録される
- [ ] coeff_meta に shapes/treeが保存され、inverseが可能
- [ ] 合成データで transform->inverse の誤差が小さい

## Verification（検証手順）
- [ ] 矩形のテスト画像で round-trip の誤差が小さいことを確認
- [ ] maskが全Trueでない場合 require_fullでエラー
