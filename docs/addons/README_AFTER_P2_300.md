# Add-on: After P2 Task 300（残件整理＋追加タスク）

このzipは、P2の進捗が以下の状態になったプロジェクトに対して、
**残りタスク（310〜390）の指示書を強化**し、さらに仕様/計画に照らして
必要になりやすい “追加の仕上げタスク” を **有限個** 追加します。

## 想定進捗
- P2: 299 ✅ / 300 ✅ / 310 in_progress / 320〜390 todo

## 目的
- Mesh/Laplace–Beltrami（310）がブレずに完了するように “最小実装” を明確化
- AE/VAE（320）・DictLearning（330）・Helmholtz（340）・ClearML（350）を
  P0/P1の契約を壊さずに追加できるよう、実装範囲とI/Oを固定
- Cleanup（390）で “後からの大規模リファクタ” を回避する
- 追加タスク（395, 398）で **基盤としての品質** を一段上げる
  - artifact validator（契約逸脱検知）
  - release-ready packaging（依存/実行コマンド/再現性の固定）

## 適用方法（上書き展開）
プロジェクトルートで:
```bash
unzip -o mode_decomp_greenfield_addon_after_p2_300_v1_flat.zip -d .
```

## queue更新（推奨：自動パッチ）
現在の `work/queue.json` をバックアップしてから、
- 299/300をdoneに揃える
- 395/398を追加（未登録の場合）
- 310〜390の task md が存在することを前提に、queue参照を整える
を行います。

```bash
python tools/_legacy/apply_addon_after_p2_300.py
```

## Autopilot再開
```bash
LIVE_TEE=1 PYTHON_BIN=/usr/bin/python3 ./autopilot.sh 80
```

## このzipで追加される主なファイル
- docs/addons/40_PLAN_STATUS_MATRIX_AFTER_P2_300.md（計画→残件マトリクス）
- docs/addons/41_MESH_LB_IMPLEMENTATION_GUIDE.md（310の実装ガイド）
- docs/addons/42_ARTIFACTS_FOR_GEOMETRY_AND_DL.md（P2手法のartifact方針）
- work/tasks_p2_v2/*.md（310/320/330/340/350/390 を強化）
- work/tasks_p2_v2/395_artifact_validator.md（追加）
- work/tasks_p2_v2/398_release_ready_packaging.md（追加）
- tools/_legacy/apply_addon_after_p2_300.py（queueパッチ）
