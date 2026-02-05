# 処理フロー

## 操作フロー（mermaid）
```mermaid
flowchart TD
  U[ユーザー: run.yaml 作成] --> R[run.py 起動]
  R --> P[パイプライン選択]
  P --> D[モード分解]
  D --> PP[前処理]
  PP --> T[学習]
  T --> I[推論]
  D --> V[可視化/評価]
  PP --> V
  T --> V
  I --> V
```

## ClearML 連携
- `configs/clearml` に設定がある場合、プロジェクト/タスク階層で結果を保存
- 実行単位は process ごとに分割

## 操作ぶれへの工夫
- `run.yaml` のシンプルな指定で Hydra config を統合
- 出力ディレクトリは一貫構成（再実行時は上書き）
- YAML スナップショットを configuration に保存
