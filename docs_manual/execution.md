# 実行体の説明（Local 実行）

## 基本コマンド
```bash
python -m mode_decomp_ml.run --config run.yaml
```

## 例: pipeline 実行
```yaml
# run.yaml
 dataset:
   conditions_csv: data/mode_decomp_eval_dataset_v1/scalar_rect/conditions.csv
   fields_dir: data/mode_decomp_eval_dataset_v1/scalar_rect/fields
   id_column: id
   grid:
     H: 64
     W: 64
     x_range: [0.0, 1.0]
     y_range: [0.0, 1.0]
 task: pipeline
 pipeline:
   decomposer: pod_svd
   coeff_post: none
   model: ridge
 output:
   root: runs
   name: scalar_rect_pod
 params:
   decompose:
     name: pod_svd
     mask_policy: ignore_masked_points
   model:
     target_space: z
```

## 実行フロー（概要）
- dataset から CSV を読み込み
- モード分解 → 係数
- 前処理 → 学習 → 推論（task に応じて）
- outputs / model / plots / logs に保存

## 実行時の注意
- 既存の runs/<name>/<process> は再実行時に上書き
- ベクトル場は `*_fx.csv` / `*_fy.csv` を使用
