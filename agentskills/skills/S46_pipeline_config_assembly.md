# S46: dataset+手法の実行設定統合（run.yaml / Hydra / ドライラン）

## 目的
- `S21`（dataset設定）と `S45`（手法設定）を 1 実行可能な設定に統合する。
- 別環境 Copilot が「設定作成 → 検証 → 最小実行」まで一気通貫で回せる状態にする。

## このスキルを使う場面
- dataset 設定と手法設定は決まったが、実行設定の組み立てで止まっている。
- `run.yaml` と Hydra override のどちらで渡すべきか迷っている。
- 設定不整合を本実行前に確実に検知したい。

## 手順
1. 入口を選ぶ。
- 単発・共有しやすさ優先: `run.yaml`。
- sweep/多条件比較: Hydra override（`mode_decomp_ml.cli.run`）。

2. 共通最低項目を埋める。
- `dataset`, `task`, `pipeline`（run.yaml）または `decompose/codec/model`（Hydra）。
- `output.root/name` を毎回変える（run上書きを防ぐ）。
- `seed` を固定する。

3. run.yaml を組み立てる（推奨ひな形）。
```yaml
seed: 123

dataset:
  name: npy_dir
  root: data/my_dataset
  mask_policy: allow_none

task: pipeline

pipeline:
  decomposer: dct2
  codec: auto
  coeff_post: none
  model: ridge

output:
  root: runs
  name: my_project_v1

params:
  domain:
    name: rectangle
    x_range: [-1.0, 1.0]
    y_range: [-1.0, 1.0]
  offset_split:
    enabled: auto
    f_offset: 5.0
    max_samples: 128
```

4. dry-run で解決後設定を確認する。
- `PYTHONPATH=src python -m mode_decomp_ml.run --config <run.yaml> --dry-run`
- 確認点: `task`, `dataset`, `decompose`, `codec`, `run_dir`。

5. decomposition 単体で smoke 実行する。
- `PYTHONPATH=src python -m mode_decomp_ml.cli.run task=decomposition ...`
- ここで `domain`/`decompose` 不整合や codec 問題を先に潰す。

6. 問題なければ pipeline 実行に上げる。
- `task=pipeline` で decomposition→preprocessing→train を通す。

## Copilot に渡す指示テンプレ（そのまま使える）
- 「`S21_dataset_config_playbook` に沿って dataset+domain を作り、`--dry-run` まで実行して。」
- 「次に `S45_method_config_playbook` で domain に合う decompose/codec を選び、decomposition の smoke を回して。」
- 「最後に `S46_pipeline_config_assembly` で run.yaml を完成させ、pipeline 実行コマンドを出して。」

## 失敗時の切り分け順
1. `dataset` 失敗: `mask_policy`, `grid`, ファイル存在を確認。
2. `domain` 失敗: manifest と手動 `params.domain` の矛盾を確認。
3. `decompose` 失敗: domain 互換と必須パラメータ（例: `disk_policy`）を確認。
4. `codec` 失敗: `codec=auto` に戻して再実行。
5. optional 依存失敗: 手法を fallback（例: autoencoder→pod_svd）。

## 実装の根拠（読む場所）
- `src/mode_decomp_ml/run.py`
- `src/mode_decomp_ml/cli/run.py`
- `src/processes/pipeline.py`
- `src/processes/decomposition.py`
