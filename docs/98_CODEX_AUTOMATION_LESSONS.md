# Codex Automation Lessons（運用メモ）

## 原則
- 1回の指示で全部やらせない。`work/tasks/*.md` を最小単位にして回す
- 毎回、以下をCodexに要求する：
  - 変更点のサマリ
  - 追加/変更ファイル一覧
  - 実行コマンド（smoke）
  - 影響するdocs（契約）があるか

## よく詰まる点
- 係数の並び順（coeff_meta）を保存しない → 比較不能になる
- train時のPCA fit と predict時 transform を混同 → skew
- maskが壊れる（補間がdomain外を埋める） → 不変条件違反

## 推奨のCodexプロンプト
- `codex/prompt_templates/01_execute_task.md` を使う
- タスクのAcceptance CriteriaとVerificationをコピペして、完了判定を明確にする
