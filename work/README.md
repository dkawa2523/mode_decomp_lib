# 作業用の出力

現行仕様は[設定ガイド](../docs/CONFIGURATION.md)、[場と係数の対応学習](../docs/FIELD_MAPPING.md)、[現在の実装と次の工程](../docs/IMPLEMENTATION_STATUS.md)を参照してください。配布先の導入は[配布ガイド](../docs/DISTRIBUTION.md)、検証方法は[品質検査](../docs/07_TESTING_CI.md)にまとめています。

`verification/`、`analysis_reports/`、`_logs/`等は任意の作業出力です。通常の利用・buildの前提にせず、Git管理から除外します。既存の生成物、venv、検証用dataは今回の文書整理では変更していません。

過去の計画・生成監査JSON・log・patch・検討RFCを現行仕様として参照する経路は削除しました。旧実装の追跡にはGit履歴を使い、保存モデルの互換性に必要な実fixtureはtests側に維持します。
