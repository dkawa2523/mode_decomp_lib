# グラフ出力一覧

| 出力ファイル例 | 処理 | 内容 | 目的 |
| --- | --- | --- | --- |
| field_true_mean.png | decomposition | 真値の平均分布 | 空間分布の代表形状 |
| field_recon_mean.png | decomposition | 再構成平均分布 | 再構成精度確認 |
| field_error_rmse.png | decomposition | 再構成誤差分布 | 誤差局所の把握 |
| field_true_mean_stream.png | decomposition | ベクトル流線 | ベクトル傾向 |
| field_true_mean_quiver.png | decomposition | ベクトル矢印 | ベクトル方向 |
| cond_coeff_corr_topk.png | decomposition | 条件×係数相関（上位） | 重要係数の把握 |
| cond_coeff_corr_all.png | decomposition | 条件×係数相関（全） | 全体傾向 |
| coeff_hist.png | preprocessing | 係数分布 | 外れ・偏り確認 |
| coeff_spectrum.png | preprocessing | スペクトル | 次元削減判断 |
| pred_vs_true_scatter.png | train | 予測 vs 真値 | モデル性能 |
| optuna_history.png | inference(optimize) | Loss 履歴 | 最適化の推移 |
| feature_importance.png | inference(optimize) | 重要度 | 条件影響把握 |
