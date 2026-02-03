# 20. Config & Output Simplification（YAML過多・出力複雑化の解消）

## 背景
現状の課題:
- 入力yamlが多すぎる / 配置が複雑で現場利用が困難
- 出力ディレクトリが深く、結果確認・レビュー・再現がしにくい
- 手法が増えるほど config 群が爆発し、比較可能性が壊れやすい

本ドキュメントは **“比較可能性” と “拡張性” を維持しつつ、UI/運用を単純化**するための不変方針です。

---

## 目標（不変）
1. **ユーザーが触る設定は原則1枚**（例: `run.yaml`）
2. domain の定義は **データセット側 manifest を真実**として、yaml指定を最小化
3. 出力は **フラットで固定名**（深い階層を廃止）
4. 既存 Hydra 運用（sweep等）は維持しつつ、非DSユーザーには “run.yaml” で隠蔽する

---

## 提案: 「run.yaml」単一エントリ

### 最小例
```yaml
dataset: data/<dataset_name>          # データセットディレクトリ（manifest必須）
task: train                           # train/predict/reconstruct/eval/viz/bench/doctor
pipeline:
  decomposer: zernike                 # registry key
  codec: zernike_pack_v1              # registry key
  coeff_post: pca                     # registry key
  model: ridge                        # registry key
output:
  root: runs
  tag: wafer_demo
```

### 重要な思想
- 手法ごとの細かなパラメータは、まず **コード側の safe default** とする
- 追加の微調整が必要なときのみ、run.yaml の下に `params:` を追加
- ユーザーが普段触るのは `task / dataset / pipeline / output` のみ

---

## dataset manifest（domain 設定を消す）

### データセット構造（提案）
```
data/<dataset_name>/
  manifest.json            # domain/座標/field_kind 等を記述
  cond.(csv|parquet|npy)
  field.npy                # (N,H,W,C) または fields/ 配下
  mask.npy                 # 任意（無い場合は全True）
```

### manifest.json（例）
```json
{
  "field_kind": "scalar",
  "grid": {"H": 64, "W": 64, "x_range":[0,1], "y_range":[0,1]},
  "domain": {"type":"disk", "center":[0.5,0.5], "radius":0.45}
}
```

### ルール
- domain は `type` で必ず指定（rectangle/disk/annulus/arbitrary_mask/sphere_grid/mesh）
- domain固有パラメータは domain 内へ（radius 等）
- 将来の ClearML を見越して **manifest は dataset artifact として保存**する

---

## 出力のフラット化（runs/<tag>/<run_id>/固定）

### 望ましい出力レイアウト
```
runs/<tag>/<run_id>/
  run.yaml
  manifest_run.json
  metrics.json
  preds.npz
  model/
  states/
  figures/
  tables/
  logs.txt
```

### 不変ルール
- “どの task でも” 同じ階層・同じファイル名に揃える
- 追加成果物が必要なときは `tables/` `figures/` `states/` のいずれかへ
- `outputs/` の深い Hydra run dir は原則廃止（どうしても必要なら 1段にする）

---

## 互換性戦略（既存Hydraの維持）
- Hydra sweep は強力なので **内部実装は Hydra のまま**でもよい
- ただし非DSユーザーには「run.yaml」を入口にし、
  run.yaml → Hydra override へ変換して実行する（Adapter層）
- 既存の config 群は段階的に **deprecated** 扱いとし、cleanup task で削除する

---

## 受け入れ基準（この方針が守れている状態）
- `run.yaml` だけで train/predict/eval/viz の一連が回る
- dataset root を変えるだけで domain 指定無しに domain が合う（manifest依存）
- 出力ディレクトリを見れば第三者が結果を追える（固定名）
