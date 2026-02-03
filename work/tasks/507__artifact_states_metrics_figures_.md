# TASK 507: 運用設計: 出力・artifactの単純化維持（states/metrics/figures固定）

## 目的
POD系追加で出力が複雑化しないよう、artifact契約を再確認し “フラット出力” を維持する。

## 作業内容
1. runs/<tag>/<run_id>/ の構造を維持
2. POD state を `states/` に格納（modes/eigvals/mean/weights_type等）
3. options の実行結果（選ばれたK、正規化有無等）を `manifest_run.json` / `metrics.json` に記録
4. figures は固定ファイル名で出す（docs/addons/32）
5. 余計な中間物（巨大npz等）を増やさない（必要ならcompress/サンプル数制限）

## 受け入れ条件
- POD系の出力がruns配下の既存構造に収まり、階層が増えていない
- 見れば分かる形で state/metrics/figures が揃っている

## 検証
- 1実行後の runs/<tag>/<run_id>/ を手で確認（ファイル名と中身）
