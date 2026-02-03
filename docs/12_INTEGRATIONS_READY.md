# Integrations Ready（将来統合の準備）

今は導入しないが、将来 ClearML 等に統合できるように “hooks” を用意する。

## ClearML を見越した設計
- Process開始時に `Task.init` できる位置を固定（process entrypoint）
- artifact（model/metrics/preds/viz）を run dir に必ず保存（docs/04）
- dataset hash / config snapshot を必ず保存（比較可能性）
- run の step 履歴（inputs/outputs 付き）を `manifest_run.json` に保存

## 今は入れない（方針）
- 依存追加・認証設定で詰まりやすいので P1 以降
- ただし `tracking.enabled` の config と logging hook は用意する（Task 120）

## Step 記録（将来の Task 化）
- `manifest_run.json` の `steps` に、順序付きの step 履歴を保存
- 各 step は `inputs` / `outputs` の artifact 参照と最小限の `meta` を持つ
- ClearML など外部実行基盤に移す際、この `steps` を Task/Stage の元情報として利用する
