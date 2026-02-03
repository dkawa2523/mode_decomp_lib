# Task: 398 RFC: Config/Output Simplification（run.yaml + manifest + output layout）

- Priority: P0
- Status: todo
- Depends on: (none)
- Unblocks: 399, 400, 401, 402

## Intent
YAML過多・出力階層の複雑化を解消するための **最終仕様（run.yaml + dataset manifest + 出力フラット化）** を確定し、
以降の実装タスクがブレないように “設計契約” として固定する。

## Context / Constraints
- 既存の比較可能性（seed/split/metrics/artifact契約）は維持する
- Hydra を完全に捨てる/残すの結論はこのRFCで決める（どちらでもよいがユーザーUX優先）
- “設定が真実” を保ちつつ、ユーザーが触る設定は **1枚**に縮約する
- 出力は `runs/<tag>/<run_id>/` に固定し、深い階層を廃止する（レビュー/現場利用優先）

## Plan
- [ ] `docs/20_CONFIG_OUTPUT_SIMPLIFICATION.md` を読み、現状コードとのギャップを洗い出す
- [ ] 決定事項（採用/却下/段階移行）を `docs/20_...` の末尾に追記する（Decision Log）
- [ ] 「互換性方針」：旧Hydra config をどこまで残すか、deprecate の期限と削除ルールを決める
- [ ] 「run.yaml仕様」：必須キーと省略時デフォルトを確定
- [ ] 「dataset manifest仕様」：必須キーと domain との対応表を確定
- [ ] 「出力仕様」：固定ファイル名と、artifact validator を更新する方針を確定

## Acceptance Criteria
- [ ] `docs/20_CONFIG_OUTPUT_SIMPLIFICATION.md` に Decision Log が追加され、今後の実装が参照できる
- [ ] “run.yaml の必須最小キー” が明文化されている
- [ ] dataset manifest の必須キー（domain/座標/field_kind 等）が明文化されている
- [ ] 出力レイアウト（固定名）が明文化され、Hydra出力の扱い（廃止/縮退）が決まっている

## Verification
- 目視で docs をレビューできること（第三者が読んで実装できる）
- `work/tasks/399...` 以降がこのRFCに従っていること

## Notes
- このタスクは **コード変更を伴わない**（仕様確定が目的）
