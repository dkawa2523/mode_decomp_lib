# 共通coreとSDKの境界

modeはtabularと同列のcore依存packageです。通常の利用・開発にSDK、ClearML、基盤checkoutは不要です。`uv sync --locked` は [固定core 0.2.1 artifact](../vendor/wheels/core.json) を解決します。

## coreの再構築

通常のmode buildは `scripts/build_wheels.py` を使います。coreの変更が必要なmaintainerだけ、次を実行します。

```powershell
uv run --no-project python integration/build_core.py
uv lock --refresh-package ml-platform-core
```

基盤commit `91aa29a235527b9ecd6fc7f6d17101beb0cfe9a0` に [core patch](ml_platform_core.patch) を適用し、別の一時checkoutでbuildします。`--platform-source` はGit URLまたはlocal Git repository、`--out-dir` はartifact出力先です。元checkoutを変更しません。

`core.json` はwheel・patchのSHA256と基盤revisionを記録します。core 0.2.1はhash/明示来歴だけを確認する `validate_artifact_integrity` を追加し、既存 `validate_model_integrity` の厳密runtime方針は維持します。開始時に提供済みのconfig snapshotはfinishで再生成しません。リリース済みwheelを同名で差し替えず、新しいcodeは新versionで配布します。

## SDK統合は後続工程

旧0.5時点のSDK／core patchと過去の受入logは現行buildで使わないため整理しました。現在のmode sourceと旧SDK設定を混在させるprepare scriptも提供しません。

SDK再開時は対象基盤revisionを定め、inline入力・自動取込・両側の入力表現・field schema 4・較正モデル・追加成果物・local/外部Stage分離を一体で反映します。単体packageの外部Stageはsource場・係数・表現モデルを含むsnapshotに対応済みです。SDK側のadapterと実ClearML環境で、この受渡しを確認する工程が残っています。

基盤ブランチへのmerge、実ClearML/Agent/Registry/Draftの受入は未実施です。単体wheelの受入結果と区別します。[現在地](../docs/IMPLEMENTATION_STATUS.md)を参照してください。
