# Task 051 (P0): Zernike分解をDecomposerプラグイン化（既存実装の移植/整理）

## 目的
既存repoの Zernike 展開を `Decomposer` として整理し、
係数の並び・正規化・逆変換・maskの扱いを明文化して保存できるようにする。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/050_decomposer_registry.md

## スコープ
### In
- 既存Zernike分解コードを `src/decompose/zernike.py` に移植またはラップする
- `transform` で係数 a を返し、`coeff_meta` で (n,m)対応と正規化規約を保存
- `inverse_transform` で係数から場を再構成できるようにする
- disk domain（r<=1）への座標正規化を domain 側で提供し、decomposerはそれを使う

### Out
- Pseudo-Zernike / Annular Zernike（P1/P2）
- 回転不変化（coeff_post側で実装）

## 実装方針（Codex向け）
### 1) 数学/実装の規約を固定（最重要）
- Zernike基底の定義（符号、正規化）
- 係数の並び（(n,m)の列挙順、mの符号の扱い）
- 離散内積（重み r）をどう近似するか
- mask外（円外/欠損）の扱い

### 2) transform の実装方針
- (H,W) の格子上で r,theta を計算（domainから供給）
- 指定次数 `n_max` までの基底を生成（キャッシュ推奨）
- 内積で係数を計算
  - 直交性が成り立つ前提なら単純内積
  - mask欠損が多い場合は最小二乗（design matrix）も選べるようにする（P1でも可）

### 3) inverse_transform
- `field_hat = sum_k a[k] * basis_k`
- dtype/shape を transform と一致させる

### 4) 性能
- basis生成は高コスト：`BasisCache` を作り、(H,W,n_max)でキャッシュ
- 複数サンプルの一括処理では basis を共通利用

### 5) 保存するもの
- `coeff_meta.json`: nm_list, normalization, grid, ordering
- `state.pkl`: basis cacheは保存不要（再生成できる）だが、必要なら保存可

## ライブラリ候補
- numpy
- scipy.special（必要なら factorial 等）
- 既存実装（優先：既に正しく動くものを尊重）

## Acceptance Criteria（完了条件）
- [ ] Zernike decomposer が registry に登録され、Hydraで選べる
- [ ] coeff_meta.json に (n,m) 対応と規約が保存される
- [ ] inverse_transform により再構成ができ、再構成誤差が許容範囲

## Verification（検証手順）
- [ ] 合成Zernike係数から生成した場を transform->inverse して誤差が小さい
- [ ] tinyデータで zernike 分解が走り、coeff.npy が生成される
