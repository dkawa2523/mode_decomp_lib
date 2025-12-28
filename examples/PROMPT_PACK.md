# Prompt Pack（ChatGPTに “repo専用 devkit” を生成させる）

## 使い方
1) 対象repoで REPO_CONTEXT.md を作成して貼る
2) 以下のプロンプトを順に投げる（推奨：観測→docs→work→skills→tools）

---

## Prompt 0: 観測（Process/I/O/地雷の抽出）
あなたはリポジトリ解析担当です。添付の REPO_CONTEXT.md と、貼り付けたソースのみを根拠にしてください（捏造禁止）。
不足情報は TODO として列挙してください。

出力:
1) 現状処理を Process にマッピングした表（build_dataset/preprocess/featurize/train/evaluate/predict/visualize/leaderboard/audit）
2) 比較不能になる地雷（split漏洩、seed未固定、skew、artifact不足、混在スクリプト等）
3) P0で直すべきものを優先度順に 10個（理由付き）
4) docsに固定すべき“不変契約”候補

---

## Prompt 1: docs生成（仕様/ポリシー/契約）
あなたはアーキテクトです。Prompt0を根拠に、対象repo向けの docs/ を新規作成してください（捏造禁止、根拠が無い点は TODO）。

作成:
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/03_CONFIG_CONVENTIONS.md
- docs/04_ARTIFACTS_AND_VERSIONING.md
- docs/09_EVALUATION_PROTOCOL.md
- docs/10_PROCESS_CATALOG.md
- docs/11_PLUGIN_REGISTRY.md
- docs/05_SECURITY_SECRETS.md
- docs/12_INTEGRATIONS_READY.md
- docs/README.md

出力形式:
File: <path>
```md
...
```

---

## Prompt 2: work生成（tasks/queue起票）
あなたはPMです。docsに従い、P0中心に work/queue.json と work/tasks/*.md を生成してください。
blocked運用：順番待ちは depends_on、blockedは解除子タスク（unblocks付き）。

---

## Prompt 3: AgentSkills生成
タスク種別に応じた手順カードを agentskills/ に作成してください（ROUTER + skills）。

---

## Prompt 4: Codex運用と自動化
codex/ と tools/（codex_prompt.py, autopilot.sh）を生成してください。
“確認要求/未実装/差分なし/flag差/doctor失敗”で止まりにくい仕組みを入れてください。


## 実行エラー/自動化の落とし穴（プロンプトに入れるべき前提）

自動化（autopilot）を前提にする場合、以下をプロンプトに明記すると “確認要求で止まる” を防げます。

- 質問禁止（確認要求で止まらない）
- 状態の真実は `work/queue.json`
- “順番待ち”は blocked にせず `depends_on` を使う
- blocked にしたら解除子タスクを必ず作り、子に `unblocks:["親ID"]` を付ける
- 進捗が見えない場合は `work/.autopilot/<ts>/codex_err_*.txt` を見る

詳細は `docs/99_AUTOPILOT_TROUBLESHOOTING.md` を参照。
