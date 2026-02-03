# Task: 520 directory_refactor_phase2_flatten_decomposers

- Priority: High
- Status: Todo
- Depends on: work/rfcs/RFC_directory_refactor_plan.md
- Unblocks: 490_cleanup_simplify_refactor, 491_docs_architecture_plugin_catalog_update

## Intent
ディレクトリ階層を浅くして decomposer の導線を単純化し、既存の import 互換を維持する。

## Context / Constraints
- 既存の不変契約: docs/00_INVARIANTS.md, docs/04_ARTIFACTS_AND_VERSIONING.md, docs/09_EVALUATION_PROTOCOL.md
- スパゲッティ化禁止：プラグインI/Fと責務分離を守る
- 途中で止めない：Acceptance Criteria が満たされるまで次タスクへ進まない
- 不要ファイル/コードは削除（Deletion policy）
- 既存の plugin 名 / config group 名は維持

## Plan
- [ ] RFCに沿って decomposer をフラット構成へ移動
- [ ] 旧パスに shim を追加して import 互換を維持
- [ ] docs/27_CODE_MAP.md と docs/11_PLUGIN_REGISTRY.md を更新
- [ ] import 互換のテストを追加/更新

## Acceptance Criteria (必須)
- [ ] `plugins/decomposers/` がフラット構成になり、旧パスは shim 経由で動作する
- [ ] 旧 import パス（例: `plugins.decomposers.grid.fft_dct`）が引き続き解決できる
- [ ] docs で「開発者が見るべき起点」が更新されている

## Verification (必須)
- Command:
  - `python3 -m pytest tests/test_plugin_shims.py -q`
  - `python3 -m pytest tests/test_decompose_fft_dct.py -q`
- Expected:
  - 全て pass

## Notes
- RFCのPhase 2「最終構成案」に従うこと
