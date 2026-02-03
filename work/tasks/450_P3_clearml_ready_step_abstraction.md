# Task: 450 Design+Prep: ClearMLタスク化を見越した step 抽象（依存なし）

- Priority: P2
- Status: done
- Depends on: 401, 400
- Unblocks: 490

## Intent
将来的な ClearML タスク化を見越して、process を “小さなステップ” に分解できる内部抽象を導入する。
ただし今は ClearML を導入しない（依存なし）。設計と接続点だけを作る。

## Context / Constraints
- 現在の実行入口（train/predict/eval 等）は維持
- 将来: dataset/task/model の管理に必要な最小情報（config, hashes, artifacts）を取り出せる形にする
- “ステップ分割” のためにパイプラインの責務が増えすぎないよう注意

## Plan
- [ ] 内部的に `Step` / `TaskNode`（仮）を定義し、run内で step を順に実行できるようにする
- [ ] 各 step は入力/出力 artifact を宣言できる（manifest_run.json に記録）
- [ ] docs/12 のチェックリストを更新し、必要情報がどこにあるかを明示

## Acceptance Criteria
- [ ] ClearMLなしで、step情報が artifact に記録される（どの処理をしたか追える）
- [ ] 既存入口が壊れない
- [ ] docs が更新され、ClearML導入時の変更点が最小になる設計になっている

## Verification
- 1 run を実行し、manifest_run.json に step 履歴が入っていることを確認
