# Task: 404 Refactor: plugins/ディレクトリ構造を整理（移動+shim、迷子防止）

- Priority: P1
- Status: done
- Depends on: 398
- Unblocks: 405, 410

## Intent
プラグインが増えても迷子にならないように、`src/.../plugins/` 配下の構造を整理し、
decomposer/codec/coeff_post/model を “同じ場所・同じ命名規約” で管理できるようにする。

## Context / Constraints
- 移動/リネームは破壊的になりやすいので、段階的に行う（Adapter/alias許可）
- 既存 import が壊れないように、最初は re-export で互換を保つ
- “コード量を減らす” が目的。整理しても処理が増えたら失敗。

## Plan
- [ ] 現状の src/ 配下のモジュールを棚卸し（どこに何があるか）
- [ ] 目標構造（plugins/decomposers/grid|zernike|eigen|sphere 等）を決める
- [ ] 主要モジュールを移動し、旧パスは薄い shim で残す（deprecated警告）
- [ ] registry を1箇所に集約し、プラグイン追加手順を短くする

## Acceptance Criteria
- [ ] plugins 配下のカテゴリが整い、追加手法の配置場所が明確
- [ ] 旧パスでの import が当面動く（deprecatedメッセージ付き）
- [ ] docs/11（plugin registry）に新配置が反映される

## Verification
- `python -m mode_decomp_ml.run --config examples/run_scalar_rect_fft_ridge.yaml` が動く
- `python -c "import ..."` で旧パスが壊れていない（最低限）

## Review Map
- 変更ファイル一覧: `src/mode_decomp_ml/plugins/registry.py`, `src/mode_decomp_ml/plugins/__init__.py`, `src/mode_decomp_ml/plugins/decomposers/**`, `src/mode_decomp_ml/plugins/coeff_post/**`, `src/mode_decomp_ml/plugins/models/**`, `src/mode_decomp_ml/decompose/__init__.py`, `src/mode_decomp_ml/decompose/*.py`, `src/mode_decomp_ml/coeff_post/__init__.py`, `src/mode_decomp_ml/models/__init__.py`, `src/processes/train.py`, `src/processes/eval.py`, `src/processes/reconstruct.py`, `src/processes/predict.py`, `src/processes/viz.py`, `tests/test_*`, `tests/test_plugin_shims.py`, `docs/11_PLUGIN_REGISTRY.md`, `examples/run_scalar_rect_fft_ridge.yaml`, `work/queue.json`
- 重要な関数/クラス: `mode_decomp_ml.plugins.registry` (register/build/list), `mode_decomp_ml.plugins.decomposers.BaseDecomposer`, `mode_decomp_ml.plugins.coeff_post.BaseCoeffPost`, `mode_decomp_ml.plugins.models.BaseRegressor`, `mode_decomp_ml.plugins.decomposers.__init__` (登録のimport集約)
- 設計判断: レジストリを `plugins/registry.py` に集約し、実装は `plugins/decomposers|coeff_post|models` に移動。旧パスは shim + DeprecationWarning で互換維持し、内部参照は新パスへ移行。
- リスク/注意点: 旧パスの DeprecationWarning はデフォルトで非表示。非公開ヘルパーへ依存していた外部コードは shim の `import *` で届かない可能性あり。
- 検証コマンドと結果: `python3 -m mode_decomp_ml.run --config examples/run_scalar_rect_fft_ridge.yaml` → OmegaConfの `Unsupported interpolation type now` で失敗（設定側の問題）。`python3 - <<'PY' ...` で旧パス import 成功、警告出力確認。`pytest -q tests/test_plugin_shims.py` → 4 passed.
- 削除一覧: `src/mode_decomp_ml/plugins/coeff_post/._basic.py`, `src/mode_decomp_ml/plugins/models/._sklearn.py`
