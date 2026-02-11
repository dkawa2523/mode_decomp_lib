# グラフ出力（plots）

このページは「どの処理が、何を把握するために、どのファイル名で図を出すか」を一覧化します。

## decomposition

| ファイル | 目的 | 何を見るか |
|---|---|---|
| `plots/key_decomp_dashboard.png` | 重要診断の集約 | R^2 vs K / scatter / error map / per-pixel R^2 |
| `plots/mode_r2_vs_k.png` | R^2 vs K | K増加での改善、飽和 |
| `plots/coeff_spectrum.png` | 係数エネルギー | 低次優勢か、分散しているか |
| `plots/field_scatter_true_vs_recon_*.png` | 散布図 | バイアス、外れ値、非線形歪み |
| `plots/per_pixel_r2_map_*.png` | 空間R^2 | 空間的に弱い領域（境界/穴/局所構造） |
| `plots/mask_fraction_hist.png` | mask統計 | 可変maskの観測密度（欠損が原因かを切り分け） |

## train

| ファイル | 目的 | 何を見るか |
|---|---|---|
| `train/plots/field_eval/field_scatter_true_vs_pred_*.png` | field比較 | バイアス/外れ値 |
| `train/plots/field_eval/per_pixel_r2_map_*.png` | 空間偏り | 弱い領域の検知 |
| `train/plots/val_residual_hist.png` | 残差分布 | 係数空間の外れ値・スケール |

## inference

| ファイル | 目的 | 何を見るか |
|---|---|---|
| `plots/field_value_hist_*.png` | 分布健全性 | 外れ/飽和 |
| `plots/coeff_std_hist.png` | 不確かさ分布 | `predict_with_std` があるモデルのみ |

生成箇所（source-of-truth）:

- decomposition の plots: `src/processes/decomposition.py` と `src/mode_decomp_ml/viz/`
- train の plots: `src/processes/train.py`
- inference の plots: `src/processes/inference.py`
- ベンチ集計レポート: `tools/bench/report/`
