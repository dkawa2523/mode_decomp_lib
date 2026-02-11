# 処理フロー（操作、処理フロー）

## 全体フロー（Mermaid）

<div class="mermaid">
flowchart LR
  U[User config] --> R[RunDir resolve]
  R --> D[Decomposition]
  D --> P[Preprocessing]
  P --> T[Train]
  T --> I[Inference]
  D --> S[Summary/Report]
  T --> S
  I --> S
</div>

## ClearML（有効時）

現状は “将来統合できるように hook を置いてある” 状態で、デフォルトは無効です。

- 設定: `configs/clearml/basic.yaml`
  - `enabled: true` にすると有効化されます
- 設計方針: `docs/12_INTEGRATIONS_READY.md`
  - process entrypoint で Task を開始できる位置を固定
  - artifacts を run_dir に必ず保存（外部基盤に移しても再現できる）
  - `manifest_run.json` に step 履歴（inputs/outputs）を保存し、外部Taskへ写像可能にする

## ユーザー操作ぶれ対策（例）

- `run_dir` の解決（固定化）
- `manifest.json` による domain 自動解決
- `auto_codec_v1` による raw_coeff 多様性の吸収
- `mask` の合成（domain+dataset）
- `coeff_meta` 契約により、手法ごとの差分を meta で比較できる
