# TASK 499: Randomized POD（高速SVDの導入、seed再現性、sweep最小追加）

## 目的
Randomized SVD / Randomized POD を「設定が増えず」「比較が簡単」な形で提供します。

## 作業内容
1. sklearn backend: `svd_solver="randomized"` を正式サポートし、seedで再現可能にする
2. modred backend: v1では snapshots 法のみでも良いが、randomized相当が必要なら
   - correlation matrix の低ランク近似 or sklearn.randomized_svd 利用を検討
   - ただし実装が増えるなら v2 に回し、docsに TODO を残す
3. benchmark matrix に randomized を追加（過剰な組合せ爆発は避ける）
4. 速度測定（簡易）を出力（metrics.json に fit_time_sec を追加）

## 受け入れ条件
- `solver=randomized` がPODDecomposerで選べる
- 再現性（seed固定）と速度改善（目視でもOK）が確認できる
- 組合せ爆発を起こさず sweep 可能

## 検証
- 同一seedで 2回実行して modes/coeff が一致（許容誤差内）
- 既存POD（full）より fit が速い（目安でOK）

## Review Map
- 変更ファイル一覧: `src/processes/train.py`, `configs/decompose/data_driven/pod.yaml`, `configs/decompose/data_driven/pod_randomized.yaml`, `scripts/bench/matrix.yaml`, `tests/test_processes_e2e.py`
- 重要な入口/関数: `src/processes/train.py:main`（fit_decomposer の計時 + metrics.json 出力）
- 設計判断: `pod_randomized` を追加して既存 `pod` デフォルトは維持。benchmark は quick だけ追加して組合せ爆発を回避。POD の seed は ${seed} に追従させる。
- 検証結果: `pytest tests/test_processes_e2e.py -q`
- 削除一覧: なし
