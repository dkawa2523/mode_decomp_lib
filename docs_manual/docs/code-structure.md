# コード構成

## ディレクトリ構成（要点）

| ディレクトリ | 役割 |
|---|---|
| `src/` | 本体ライブラリ + processes |
| `configs/` | Hydra 設定（task/decompose/codec/model/...） |
| `examples/` | run.yaml 例（非Hydra入口） |
| `tools/bench/` | ベンチ実行 + 集計/レポート生成 |
| `tests/` | pytest |
| `data/` | 参照データ（小）・生成データ（大はGit管理しない） |

除外（このドキュメントでは扱わない）:
- `codex/`, `agentskills/`, `work/`, `tools/_legacy/`, `.venv*`, `runs/`, `outputs/`

## 各ディレクトリの中身（ガイド）

まず第三者が迷いやすい “入口” を固定します。

- 実行（Hydra）: `src/mode_decomp_ml/cli/run.py`
- 実行（run.yaml）: `src/mode_decomp_ml/run.py`
- 各 stage の実体: `src/processes/`（decomposition/preprocessing/train/inference/pipeline）
- 拡張ポイント（手法追加）: `src/mode_decomp_ml/plugins/` と `src/mode_decomp_ml/preprocess/`

主要ディレクトリの概観（抜粋）:

```text
src/
  processes/                    # stage 実装（入出力 artifacts の中心）
  mode_decomp_ml/
    cli/                        # Hydra 入口
    run.py                      # run.yaml 入口
    data/                       # dataset 読み込み（npy_dir, csv_fields 等）
    domain/                     # domain spec（mask/weights/coords）
    preprocess/                 # field 前処理（互換性のため現状はここに置く）
    plugins/                    # registry + decomposers/codecs/coeff_post/models
    evaluate/                   # metrics（field/coeff）
    viz/                        # plots（decomposition/train/inference/bench が利用）
    pipeline/                   # 共通の組み立て（domain 解決、ビルド、artifact）
configs/
  task/                         # pipeline/decomposition/train/inference 等
  decompose/                    # モード分解手法（analytic/data_driven 等）
  codec/                        # raw_coeff <-> coeff(a)
  coeff_post/                   # coeff(a) <-> coeff(z)（例: PCA）
  model/                        # 学習モデル
examples/
  run_*.yaml                    # run.yaml 入口の最小例
tools/
  bench/                        # ベンチ実行、集計、レポート生成
tests/
  test_*.py
data/
  mode_decomp_eval_dataset_v1/  # 小さめの eval データ（npy_dir 推奨）
```

## ワークフロー（Mermaid）

以下は “標準ワークフロー” の概念図です。

<div class="mermaid">
flowchart TD
  A[Dataset (cond, field, mask)] --> B[Field preprocess]
  B --> C[Build DomainSpec]
  C --> D[Decomposer fit]
  D --> E[Transform => raw_coeff]
  E --> F[Codec encode => coeff_a]
  F --> G[CoeffPost transform => coeff_z]
  G --> H[Train model: cond => coeff]
  H --> I[Inference: predict coeff]
  I --> J[CoeffPost inverse => coeff_a]
  J --> K[Codec decode => raw_coeff]
  K --> L[Inverse transform => field_hat]
  L --> M[Evaluate and plot]
</div>

## 拡張/修正ポイント早見表

| 変更したいこと | 触る場所（主） |
|---|---|
| 新しいモード分解手法 | `src/mode_decomp_ml/plugins/decomposers/` |
| 新しい codec | `src/mode_decomp_ml/plugins/codecs/` |
| 前処理（field）追加 | `src/mode_decomp_ml/preprocess/` |
| 係数後処理追加 | `src/mode_decomp_ml/plugins/coeff_post/` |
| 学習モデル追加 | `src/mode_decomp_ml/models/` |
| 評価指標追加 | `src/mode_decomp_ml/evaluate/` |
| 可視化追加 | `src/mode_decomp_ml/viz/` / `src/processes/*` |
