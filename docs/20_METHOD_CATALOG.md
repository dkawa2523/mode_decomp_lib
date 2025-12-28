# Method Catalog（一次分解 + 係数後処理）

この文書は “候補一覧” です。実装は `work/tasks/` で段階的に行います。

## 一次分解（Decomposer）カテゴリ
### 円（disk）
- Zernike（P0）
- Fourier–Bessel（P1）

### 矩形（rect）
- FFT2（P0）
- DCT2（P0）
- DST2（P1）

### 任意マスク/不規則点
- RBF expansion（P0–P1）
- Graph Fourier（P1）

### 多解像
- Wavelet（P1）

### 曲面/メッシュ（将来）
- Laplace–Beltrami eigs（P2）

## 係数後処理（CoeffPost）
- normalize（P0）
- complex -> mag/phase（FFT採用時にP0）
- PCA / TruncatedSVD（P0–P1）
- ICA / NMF（P2）
- energy threshold / low-order selection（P1）

## 参考（このチャットで整理したパイプライン）
- 添付の `mode_decomposition.md`（会話要約）を同梱します。
