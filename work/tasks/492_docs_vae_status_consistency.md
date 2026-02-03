# Task: 492 Docs: VAE status consistency (Autoencoder only)

- Priority: P2
- Status: done
- Depends on: 491
- Unblocks: (none)

## Intent
VAE の実装有無について、主要ドキュメントの記述を現状（Autoencoderのみ）に揃える。

## Context / Constraints
- 既存の不変契約: docs/00_INVARIANTS.md, docs/04_ARTIFACTS_AND_VERSIONING.md, docs/09_EVALUATION_PROTOCOL.md
- スパゲッティ化禁止：プラグインI/Fと責務分離を守る
- 途中で止めない：Acceptance Criteria が満たされるまで次タスクへ進まない
- 不要ファイル/コードは削除（Deletion policy）

## Plan
- [x] 主要ドキュメント内の VAE 記述を確認する
- [x] Autoencoder のみ実装されている旨に統一する
- [x] 簡易テストを実行する

## Acceptance Criteria (必須)
- [x] docs/01_ARCHITECTURE.md で VAE が未実装である旨が明確になっている
- [x] docs/20_METHOD_CATALOG.md で VAE が未実装である旨が明確になっている
- [x] docs/addons/42_ARTIFACTS_FOR_GEOMETRY_AND_DL.md の見出しが現状に整合している

## Verification (必須)
- Command:
  - `python3 -m pytest tests/test_decompose_autoencoder.py -q`
- Expected:
  - テストが成功する

## Notes
- 追加の実装（VAE本体）は別タスクで扱う
