# Process: train/predict/reconstruct/eval（artifact契約）

**ID:** 080  
**Priority:** P0  
**Status:** done  
**Depends on:** 070  
**Unblocks:** 090  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
プロジェクトのCLI入口（process）を揃え、artifact契約どおりに保存する。

必要process:
- train: model学習（必要ならcoeff_post/decomposerもfit）
- predict: cond→(a|z)予測
- reconstruct: 予測係数→field_hat
- eval: metrics出力（coeff/field）
- doctor: 環境/最小実行の健全性チェック

artifact（docs/04）:
- config（.hydra）
- meta（実行条件、git hash等）
- metrics（json/csv）
- preds（npy/csv）
- model（pickle/pt）

## Acceptance Criteria
- [x] `task=train` → `task=predict` → `task=reconstruct` → `task=eval` が一連で動く
- [x] outputs に artifact が契約どおり保存される
- [x] 再構成ができる（field_hatが生成される）

## Verification
- [x] synthetic dataset で一連を実行し、outputs配下に成果物が揃う

## Review Map
- 変更ファイル一覧: `src/mode_decomp_ml/pipeline/utils.py`, `src/mode_decomp_ml/pipeline/__init__.py`, `src/mode_decomp_ml/evaluate/__init__.py`, `src/mode_decomp_ml/decompose/__init__.py`, `src/processes/train.py`, `src/processes/predict.py`, `src/processes/reconstruct.py`, `src/processes/eval.py`, `configs/config.yaml`, `configs/split/all.yaml`, `configs/task/predict.yaml`, `configs/task/reconstruct.yaml`, `configs/task/eval.yaml`, `tests/test_processes_e2e.py`
- 重要な関数/クラス: `src/processes/train.py` の `main`, `src/processes/predict.py` の `main`, `src/processes/reconstruct.py` の `main`, `src/processes/eval.py` の `main`, `src/mode_decomp_ml/pipeline/utils.py` の `build_dataset_meta`/`build_meta`, `src/mode_decomp_ml/evaluate/__init__.py` の `compute_metrics`
- 設計判断: split は最小の `all` だけを実装し、run dir 受け渡しは task config に明示。decomposer/coeff_post/model は pickle state を artifact に保存し、reconstruct/eval で同一状態を再利用する。
- リスク/注意点: 予測/評価は dataset 全件をロードするため大規模データでメモリ負荷あり。`model.target_space=a` と `coeff_post!=none` は禁止。run dir を誤指定すると復元・評価に失敗する。
- 検証コマンドと結果: `pytest tests/test_processes_e2e.py`（PASS）
- 削除一覧: なし
