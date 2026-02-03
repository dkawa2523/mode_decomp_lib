# S43: 係数後処理（CoeffPost：PCA/ICA/normalize/複素→実数）

## 目的
- 本プロジェクトの“特徴量化”を **分解後係数に対する後処理** として固定し、
  PCA等の状態を保存して train/serve skew を防ぐ。

## やること手順
1. **入出力を固定**
   - 入力：A (N,K)
   - 出力：Z (N,K2)
   - inverse_transform で A_hat に戻せる（可逆または近似可逆）

2. **fit/transform を分離**
   - fit は train split のみ
   - predict は保存した state をロードして transform のみ

3. **state 保存**
   - `artifacts/coeff_post/state.pkl`
   - 次元（K2）やエネルギー閾値を meta に保存

4. **標準化**
   - per-mode 標準化はP0で有効（学習安定）

5. **複素係数の扱い**
   - FFTなどは複素 → 実数表現（mag/phase or real/imag）をここで行う

6. **テスト**
   - 合成Aで round-trip（fit->transform->inverse）誤差が小さい

## 事故りやすい点
- PCAを全データでfitしてしまう（データ漏洩）
- predict時に別stateでtransformしてしまう（skew）
- K2が自動で変わったのに保存しない（比較不能）
- complexをそのまま sklearn に渡す

## DoD
- registry 登録
- state 保存/ロードが動く
- round-trip テスト
- docs/09 と docs/04 の契約に沿っている

## よくある差分
- PCA vs TruncatedSVD
- whiten の有無
- ICA/NMF など初期値依存手法のseed固定
