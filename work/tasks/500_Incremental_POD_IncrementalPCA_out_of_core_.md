# TASK 500: Incremental POD（IncrementalPCA、out-of-core運用の足場）

## 目的
Out-of-core/継続運用を見越して Incremental POD を追加します。
PODDecomposerの “solver=incremental” として提供し、他部分に分岐を出さない。

## 作業内容
1. sklearn IncrementalPCA を用いて `solver=incremental` を実装
2. batch_size の扱いは options で最小露出（デフォルトは自動）
3. state保存（components/mean/var等）を統一
4. `fit` と `partial_fit` の差を docs に明記（同一結果にならない可能性）
5. 出力メタ（metrics）に batch_size / n_batches / fit_time を追加

## 受け入れ条件
- `solver=incremental` が動き、メモリ使用が抑えられる（少なくとも設計上）
- 既存の train/predict/reconstruct に統合され、追加分岐が増えていない

## 検証
- 同じdatasetで full vs incremental の再構成誤差を比較し、極端に悪化していない
