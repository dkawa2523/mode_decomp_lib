# TASK 502: Decomposer: GappyPODDecomposer（欠損/部分観測→係数推定、正則化options）

## 目的
Gappy POD を **別クラス**として追加し、欠損・部分観測から係数推定→再構成を実現します。

## 作業内容
1. `GappyPODDecomposer` を `data_driven/gappy_pod.py` 等に追加（PODDecomposerと分離）
2. 入力：
   - field（欠損はmaskで表現、またはNaNでも良いが内部で統一）
   - mask（観測領域）
3. 推定：
   - 係数 `a` を最小二乗で推定（必要ならTikhonov λ）
   - 係数推定は sample ごとに行う（バッチ最適化はv2）
4. inverseで full field を復元（POD基底は事前にfit済み想定）
5. optionsで以下をON/OFFできるようにする
   - `options.gappy.enable_reg`（正則化）
   - `options.gappy.lambda`（小さなデフォルト）
6. 可視化（docs/addons/32のGappy図）を追加

## 受け入れ条件
- `decomposer=gappy_pod` が registry から選べる
- 任意maskで、観測点のみから係数推定→復元が可能
- 正則化ON/OFFで挙動が変わる（不安定回避）

## 検証
- mask付きdatasetで gappy reconstruct を実行し、観測領域と欠損領域で誤差を分けて出力
