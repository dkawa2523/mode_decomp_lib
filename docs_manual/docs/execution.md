# 実行体の説明（Local実行）

## 入口コマンド

Hydra（推奨）:
```bash
PYTHONPATH=src python3 -m mode_decomp_ml.cli.run task=pipeline
```

run.yaml（非Hydra）:
```bash
PYTHONPATH=src python3 -m mode_decomp_ml.run --config examples/run_scalar_rect_fft2_ridge.yaml --dry-run
PYTHONPATH=src python3 -m mode_decomp_ml.run --config examples/run_scalar_rect_fft2_ridge.yaml
```

## Local 実行フロー（テンプレ）

1. dataset 準備（`dataset.root` / `mask_policy` / `manifest.json`）
2. decomposition（モード分解 + coeff生成）
3. preprocessing（coeff_post など）
4. train（学習）
5. inference（推論・復元・可視化）

## 例（examples）

- `examples/run_scalar_rect_fft2_ridge.yaml`
- `examples/run_scalar_disk_zernike.yaml`
- `examples/run_scalar_mask_pod_ridge.yaml`

## 生成物（artifacts）

各 stage は `run_dir/` 配下に一貫したレイアウトで成果物を残します（比較・再現のため）。

参照（canonical）:
- artifacts と versioning: `docs/04_ARTIFACTS_AND_VERSIONING.md`

| 生成物 | どの stage | 何が入るか |
|---|---|---|
| `outputs/coeffs.npz` | decomposition | `coeff(a)`, `cond`, `sample_ids` |
| `outputs/preds.npz` | inference | `coeff_pred`, `field_hat` など（設定により変動） |
| `outputs/metrics.json` | decomposition/train/inference | 指標 |
| `plots/` | decomposition/train/inference | 可視化 |
| `outputs/states/**/state.pkl` | 各stage | state |

### run_dir の目安

```text
run_dir/
  configuration/run.yaml        # 実行設定のスナップショット
  outputs/                      # npz/json/pkl 等（機械可読）
  plots/                        # png 等（人間向け）
  tables/                       # csv/md 等（集計）
  manifest_run.json             # step 履歴（inputs/outputs の参照）
```

実装の入口:

- decomposition: `src/processes/decomposition.py`
- preprocessing: `src/processes/preprocessing.py`
- train: `src/processes/train.py`
- inference: `src/processes/inference.py`
- pipeline: `src/processes/pipeline.py`
