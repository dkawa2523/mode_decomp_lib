# 検証記録と背景資料

現行仕様は[利用guide](../docs/CONFIGURATION.md)、配布先の導入は[配布用ガイド](../docs/DISTRIBUTION.md)、現在地は[実装状況](../docs/IMPLEMENTATION_STATUS.md)を参照してください。**最新の受入正本は[0.10.4単体配布記録](reviews/package_distribution_validation_20261025.json)です。** [統合実装計画](reviews/package_implementation_plan_20261025.json)のI00〜I12とI13ローカル配布受入、O・B・F・D各継続改良を完了しました。配布ZIP単独のstdlib venv/pip導入とfresh推論を確認済みです。remote CI・公開registry・実用途の認定は別です。以前の受入は履歴として保持します。

[内部構造・配置の統合検討](reviews/package_structural_synthesis_20261025.json)と`reviews/implementation_plan_20261025/`は変更前の根拠・設計付録です。現仕様は上記guideと受入正本へ集約します。数値手法・公開facade・自己完結YAML・実旧bundleを維持し、配置だけの全面変更や汎用frameworkは導入していません。

mode 0.9.0以前の配布・数値・容量受入は[0.9.0検証記録](reviews/package_capacity_validation_20261025.json)等へ履歴として保持します。[設計・科学解析の監査](reviews/package_reviewer_audit_20261025.json)と`verification/structural_review_20261025/`には変更前の所見・再現script・結果を保持しています。

[以前の統合改良検証](reviews/integration_improvement_validation_20261024.json)と`reviews/platform_integration_20261024/`の品質検査logは履歴です。後者はmode 0.5時点のSDK受入で、現行版のSDK統合完了を意味しません。

元の[2026-09-25レビュー](reviews/2026-09-25_architecture_review.md)、[移行前のユーザー差分](reviews/pre_mode_decomp_implementation.patch)、[Hodge/Poisson検討資料](rfcs/RFC_hodge_poisson_mask.md)は保持しています。

過去の比較解析例は `analysis_reports/current/` と `analysis_reports/fixture/`、0.7.1時点の配布受入成果物とlogは `verification/wheel_0_7_1/` に保持します。散布点・予測帯・Hodge境界の追加確認は `verification/package_acceptance_20261024/` です。`verification/wheel_0_7/` の旧truth・予測NPZ等は独立再計算の入力として保持します。これらの生成物はGit管理しません。保存した証拠や再現入力は通常利用・buildの前提ではありません。
