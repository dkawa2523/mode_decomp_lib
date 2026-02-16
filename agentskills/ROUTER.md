# ROUTER（タスク種別 → 参照スキル）

このファイルは、Codex にタスクを渡すときに「どのスキルカードを参照すべきか」を決めるための対応表です。

| タスク種別 | 参照スキル |
|---|---|
| repo棚卸し / 現状把握 | S00_repo_onboarding |
| Hydra導入 / config追加 | S10_config_hydra |
| データI/O / split / domain | S20_data_io, S44_domain_geometry |
| データセット設定（npy_dir/csv_fields/manifest） | S21_dataset_config_playbook |
| 前処理 | S30_preprocess |
| モード分解（Zernike/FFT/Bessel/RBF/Wavelet/Graph/Laplace…） | S42_decomposition |
| 手法設定（decompose/codec/coeff_post/model） | S45_method_config_playbook |
| dataset+手法の実行設定（run.yaml/Hydra） | S46_pipeline_config_assembly |
| 係数後処理（PCA/ICA/normalize/complex→実数など） | S43_coeff_post |
| 学習モデル（sklearn/torch） | S50_models, S60_training_hpo |
| 評価/指標/可視化/比較表 | S70_eval_metrics |
| テスト/doctor/CI | S95_tests_ci |
| セキュリティ/秘密情報 | S98_security |
| 将来統合（ClearML等） | S90_ops_integrations |
