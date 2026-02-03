# Task: 402 Refactor: configs/整理（yaml増殖停止、プリセット最小化、deprecated隔離）

- Priority: P0
- Status: todo
- Depends on: 398, 400, 401
- Unblocks: 490

## Intent
Hydra/yamlの配置を整理し、ユーザーが迷わない最小プリセット構造に縮約する。
「手法追加のたびにyamlが増殖する」状態を止める。

## Context / Constraints
- run.yaml 経路を最優先（非DS向け）
- sweep用途に最低限のHydra groupは残してよい（ただし数を増やさない）
- 既存yamlは deprecate → 削除の手順を明確にする（cleanupに繋げる）

## Plan
- [ ] 現状の configs/ を棚卸し（重複/未使用/似たもの）
- [ ] “残すべき最小プリセット” を決める（例: domain=rectangle/disk/mask, baseline=fft/zernike/pod）
- [ ] groupを減らし、デフォルト値はコード側へ移す
- [ ] `configs/_deprecated/` に移動し、一定期間後に削除する方針を docs に追記
- [ ] `docs/03_CONFIG_CONVENTIONS.md` を更新し、新ルールを固定

## Acceptance Criteria
- [ ] configs/ のファイル数が実測で大きく減る（目標: 30〜50%削減以上）
- [ ] 新しい「読む順」ドキュメント（docs/README）が更新される
- [ ] run.yaml 入口の例（examples/）が3〜5本で揃う
- [ ] deprecated群が隔離され、今後増やさないルールが明文化される

## Verification
- `git diff --stat` で configs/ が整理されている
- `python -m mode_decomp_ml.run --config examples/run_*.yaml` が最低3ケース動く
