# Hydra最小導入（config真実・task入口統一）

**ID:** 010  
**Priority:** P0  
**Status:** todo  
**Depends on:** 000  
**Unblocks:** 020  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
Hydraを “設定が真実” になる形で導入し、process入口（train/eval/predict...）を config/task で切り替えられるようにする。

- `configs/config.yaml` を root とし、`task=train` 等で入口を切替
- run_dir, seed, output_dir 等は docs/03 に従う
- `python -m processes.doctor` ではなく、統一入口 `python -m mode_decomp_ml.cli.run task=doctor` を用意

## Acceptance Criteria
- [ ] `python -m mode_decomp_ml.cli.run task=doctor` が動き、configが保存される
- [ ] `configs/task/*.yaml` が最低限存在し、taskの切替ができる
- [ ] 重要なパラメータ（seed/run_dir/output_dir）が config に集約されている

## Verification
- [ ] `python -m mode_decomp_ml.cli.run task=doctor` を実行し、outputsに `.hydra/config.yaml` が残る
