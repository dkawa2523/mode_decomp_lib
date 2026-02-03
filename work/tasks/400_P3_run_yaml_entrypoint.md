# Task: 400 Add: run.yaml 単一設定入口（CLI adapter + dry-run）

- Priority: P0
- Status: todo
- Depends on: 398, 399
- Unblocks: 401, 402

## Intent
非データサイエンティスト向けに、**run.yaml 1枚**で train/predict/eval/viz/bench を実行できる入口を提供する。
内部がHydraでも、ユーザー体験は run.yaml に統一する。

## Context / Constraints
- 既存の CLI (`python -m mode_decomp_ml.cli.run task=...`) を壊さない
- run.yaml のキーは `docs/20` の確定仕様に従う
- sweep（-m）等の高度利用は従来のHydra経路を残してよい

## Plan
- [ ] 新しい入口 `python -m mode_decomp_ml.run --config run.yaml`（仮）を追加
- [ ] run.yaml を読み、内部のConfig（Hydra/OmegaConf）へ変換して既存processを呼ぶ
- [ ] `--dry-run` を追加し、解決された設定と出力dirを表示して終了できるようにする
- [ ] docs: `docs/USER_QUICKSTART.md`（最小でよい）を追加し run.yaml 運用を説明
- [ ] smoke test: run.yaml で doctor/train が回ること

## Acceptance Criteria
- [ ] run.yaml 1枚で doctor/train/eval/viz の少なくとも1本が実行できる
- [ ] 既存の Hydra CLI 入口はそのまま動く
- [ ] `--dry-run` で「どの decomposer/model を使うか」「出力先」が見える
- [ ] 最小ドキュメントが追加される（読む順が分かる）

## Verification
- Command:
  - `python -m mode_decomp_ml.run --config examples/run_scalar_disk_zernike.yaml --dry-run`
  - `python -m mode_decomp_ml.run --config examples/run_scalar_disk_zernike.yaml`
- Expected:
  - dry-run で resolved config と run_dir が表示される
  - 実行で `runs/<tag>/<run_id>/metrics.json` 等が生成される
