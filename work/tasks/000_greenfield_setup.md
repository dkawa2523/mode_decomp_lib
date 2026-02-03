# Greenfield初期化（スコープ固定・最小骨格・doctor通過）

**ID:** 000  
**Priority:** P0  
**Status:** done  
**Depends on:** None  
**Unblocks:** 010  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
このプロジェクトを「0から作る」前提で、Autopilotが止まらず進むための最小準備を行う。

- docs を読む順にざっと確認し、P0スコープを固定（docs/18）
- `src/` / `configs/` / `work/` が存在することを確認し、最低限 import が通る状態にする
- `./doctor.sh` が **no issue** で完了すること（P0の入口）

## Implementation Notes
- 旧実装を参照したい場合は `legacy/` に配置する（任意）
- ここでは機能実装はしない。**開発が終わらない問題を先に潰す**

## Acceptance Criteria
- [x] `LIVE_TEE=1 PYTHON_BIN=/usr/bin/python3 ./doctor.sh` が no issue で完了する
- [x] `work/queue.json` の P0 scope が docs と一致している（P0で不要に広げない）
- [x] `README.md` の手順で `autopilot.sh` が起動できる（Codex実行直前まで）

## Verification
- [x] doctorログ（work/.autopilot/.../doctor_initial.txt）に no issue が記録される

## Review Map
- 変更ファイル一覧: `work/tasks/000_greenfield_setup.md`, `work/queue.json`, `work/.autopilot/20251228T132235Z/doctor_initial.txt`, `work/.autopilot/20251228T132401Z/doctor_initial.txt`, `work/.autopilot/20251228T132401Z/_meta.txt`, `work/.autopilot/20251228T132401Z/codex_help.txt`, `work/.autopilot/20251228T132401Z/codex_exec_help.txt`, `work/.autopilot/20251228T132401Z/codex_version.txt`
- 重要な関数/クラス: `tools/codex_prompt.py`（doctor実行で整合性確認）, `tools/autopilot.sh`（preflight/doctorログ生成）
- 設計判断: doctorログは作業証跡として残し、P0スコープはdocsに合わせて変更しない方針で固定
- リスク/注意点: `work/.autopilot/` は検証ログのため残置（機密がある場合は運用で除外）
- 検証コマンドと結果: `LIVE_TEE=1 PYTHON_BIN=/usr/bin/python3 ./doctor.sh` → `no issues found`; `MAX_ITERS=0 LIVE_TEE=1 PYTHON_BIN=/usr/bin/python3 ./autopilot.sh 0` → preflight/doctorログ生成
- 削除一覧: なし
