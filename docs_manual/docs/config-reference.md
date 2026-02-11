# 設定ファイルの説明（YAML）

## 方針

- Hydra 設定（`configs/`）と run.yaml（`examples/`）の両方を扱います。
- まずは「どこを触ると何が変わるか」をテーブルで固定し、詳細は段階的に埋めます。

## 設定の2系統（Hydra と run.yaml）

- Hydra 入口（推奨）: `PYTHONPATH=src python3 -m mode_decomp_ml.cli.run task=pipeline ...`
  - `configs/` 配下の group を `defaults:` で組み立てます。
  - sweep（複数条件）や benchmark にはこちらが向きます。
- run.yaml 入口: `PYTHONPATH=src python3 -m mode_decomp_ml.run --config examples/run_*.yaml`
  - 1回の実行を確実に回す用途（配布・再現）が向きます。
  - `pipeline.decomposer/model/...` に group 名を入れて、内部で `configs/` を読み込む形です。

## run.yaml（examples）キー

| キー | 型 | 意味 | デフォルト | Tips |
|---|---|---|---|---|
| `dataset.root` | str | npy_dir dataset root | なし | `manifest.json` 推奨 |
| `dataset.mask_policy` | enum | require/allow_none/forbid | なし | npy_dir では必須 |
| `task` | str | stage 名 | なし | `pipeline` がまとめて回る |
| `pipeline.decomposer` | str | decompose group 名 | （runごと） | `configs/decompose/**`（例: `analytic/dct2`） |
| `pipeline.codec` | str | codec group 名 | `auto_codec_v1` 推奨 | 複素系や構造化 raw_coeff は codec が必須 |
| `pipeline.coeff_post` | str | coeff_post group 名 | `none` | PCA 等（`configs/coeff_post/**`） |
| `pipeline.model` | str | model group 名 | `ridge` | `configs/model/**` |
| `output.root/name` | str | 出力先 | `runs`/`default` | 既存 run を上書きしない命名を推奨 |
| `params.*` | map | 各モジュール override | なし | 例: `params.decompose.n_max=8` のように上書き |

### `dataset.mask_policy`（npy_dir）

| 値 | 意味 |
|---|---|
| `require` | dataset が `mask.npy` を必ず持つ（mask前提の手法/評価向け） |
| `allow_none` | `mask.npy` が無くても良い（全点有効扱い） |
| `forbid` | mask を許可しない（maskがあるとエラー） |

## Hydra（configs）キー（どこを見れば良いか）

Hydra の場合は「group の場所」がそのまま “設定の索引” になります。

- `configs/task/` : 実行する stage（decomposition/train/inference/pipeline 等）
- `configs/decompose/` : モード分解（解析・データ駆動・近似）
- `configs/codec/` : raw_coeff <-> coeff(a)
- `configs/coeff_post/` : coeff(a) <-> coeff(z)（例: PCA）
- `configs/model/` : 学習モデル（ridge/gpr/mtgp 等）
- `configs/clearml/` : tracking（デフォルト無効）

### Override の基本形（Hydra）

```bash
# 例: disk の zernike を n_max=8 に変更して decomposition だけ回す
PYTHONPATH=src python3 -m mode_decomp_ml.cli.run \
  task=decomposition \
  dataset.root=data/mode_decomp_eval_dataset_v1/scalar_disk \
  dataset.mask_policy=require \
  decompose=analytic/zernike \
  decompose.n_max=8
```

## Hydra（configs）キー（テンプレ）

本マニュアルでは “全キーの列挙” は `docs/` 側（canonical）に委ねます。

- 手法一覧: `docs/20_METHOD_CATALOG.md`
- plugin registry: `docs/11_PLUGIN_REGISTRY.md`
- codec 契約: `docs/21_CODEC_LAYER_SPEC.md`
- coeff_meta 契約: `docs/28_COEFF_META_CONTRACT.md`
