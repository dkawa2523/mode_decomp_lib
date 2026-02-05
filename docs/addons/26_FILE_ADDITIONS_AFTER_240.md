# After 240: 追加ファイル地図（P1 tail / P2）

この文書は「どのタスクで、どのファイルが増えるか」の目安です。
実際の配置はプロジェクトの既存構成に合わせて調整します（無理に合わせない）。

## P1 tail
### Task 250: uncertainty
- add: `src/mode_decomp_ml/viz/uncertainty.py`
- update: decomposition/inference process（stdがある場合の保存/描画）
- add: `configs/uncertainty/gpr_mc.yaml`（MCサンプル数、対象サンプル数など）

### Task 260: ElasticNet / MultiTask
- add: `src/mode_decomp_ml/models/elasticnet.py`
- add: `configs/model/elasticnet.yaml`
- add: `configs/model/multitask_elasticnet.yaml`（任意）
- update: model registry

### Task 270: docs update
- update: `docs/20_METHOD_CATALOG.md`（Implementedマーク）
- add: `configs/examples/*.yaml`（最短実行例）

### Task 290: cleanup
- update/delete: 未使用ファイルの削除、重複統合
- update: docs整合

## P2（概要）
- add: `src/mode_decomp_ml/plugins/decomposers/graph_fourier.py`
- add: `src/mode_decomp_ml/plugins/decomposers/laplace_beltrami.py`, `src/mode_decomp_ml/domain/mesh.py`
- add: `src/mode_decomp_ml/plugins/decomposers/autoencoder.py`（torch）
- add: `src/mode_decomp_ml/plugins/decomposers/dict_learning.py`（疎表現）
- update: `src/mode_decomp_ml/coeff_post/__init__.py`（dict_learningのCoeffPost追加）
- add: `src/mode_decomp_ml/plugins/decomposers/helmholtz.py`（div/curl）
- add: `src/mode_decomp_ml/tracking/clearml.py`
