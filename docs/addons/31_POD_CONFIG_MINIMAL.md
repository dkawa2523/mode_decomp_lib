# Addon after Task 492: POD suite — Config & Options (run.yaml minimal)

このドキュメントは、POD系追加で「YAML増殖」を起こさないための **最小設定**を定義します。

## 1. run.yamlは1枚（ユーザーが触るのはこれだけ）

例（最小）：

```yaml
dataset: data/<dataset_name>        # dataset manifest が真実
task: pipeline                      # decomposition/preprocessing/train/inference/pipeline
pipeline:
  decomposer: pod                   # pod / gappy_pod / fft2 / zernike ...
  coeff_post: pca                   # none/pca/quantile/power (必要なら)
  model: ridge                      # ridge/gpr/xgb/...
options:
  rank_select: { enable: true, method: energy, energy: 0.99, max_modes: 128 }
  coeff_normalize: { enable: false, method: quantile }   # quantile|power
  mode_weight: { enable: false, method: eigval_scale }   # eigval_scale|none
output:
  root: runs
  name: pod_suite
```

TODO(Task493): 現行実装の decomposer 名は `pod_svd` のみ（`configs/decompose/data_driven/pod_svd.yaml`）。
`decomposer: pod` / `decomposer: gappy_pod` の alias は未実装（Task496以降で追加予定）。

## 2. decomposer個別の細かいパラメータは「コードデフォルト」優先
- run.yamlに増やさない（必要時だけoverride）
- 詳細パラメータは `--set key=value` 的にCLIで指定できる設計が望ましい（Hydra/OmegaConfどちらでも良い）

TODO(Task493): run.yaml adapter は `params` / `task_params` を deep update する方式のみ（`--set` 相当は未整備）。

## 3. dataset manifestを真実にする
domainやgridをconfigで持たず、dataset側 `manifest.json` に寄せる。
- domain.type: rectangle/disk/arbitrary_mask/mesh
- grid: H,W,x_range,y_range
- field_kind: scalar/vector
- optional: mask policy 等

## 4. 追加機能は options でON/OFF（ClearML Task化を見越す）
- optionsは将来、別Taskとして切り出せる粒度（rank_select / normalize / mode_weight）
- ただし現在は同一プロセス内でON/OFFしてよい

## 5. 互換性
既存の configs/ が残っている場合でも、**run.yaml 1枚**で回る経路を優先し、
古い設定経路は段階的に deprecated にする（削除は cleanup タスクで行う）。

---

## 6. Naming lock（POD suite v1）
- `decomposer=pod`（PODDecomposer）
- `decomposer=gappy_pod`（GappyPODDecomposer）
- `options.rank_select.*`
- `options.coeff_normalize.*`
- `options.mode_weight.*`
