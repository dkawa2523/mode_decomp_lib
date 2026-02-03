# Task: 399 Implement: dataset manifest.json 読み取り + domain自動推定

- Priority: P0
- Status: done
- Depends on: 398
- Unblocks: 400, 401, 402

## Intent
dataset root 配下の `manifest.json` を読み取り、domain設定・座標系・field_kind を **データ側を真実**として扱えるようにする。
これにより “domain用yaml群” を減らし、run.yaml を最小化する。

## Context / Constraints
- 既存の dataset loader（cond/field/mask）と互換を維持する（manifestが無い場合は従来通り）
- mask が無い場合は全Trueを生成し、評価と分解が破綻しないようにする
- 失敗は silent にせず、doctor で明確なエラーにする

## Plan
- [ ] dataset root に `manifest.json` があれば読む（schema検証）
- [ ] manifest から `domain_spec` を生成し、DomainFactory へ渡せる形にする
- [ ] `field_kind`（scalar/vector）と `grid`（H,W,x_range,y_range）を dataset schema に反映
- [ ] `doctor` を拡張し、manifest欠落/不整合/shape不一致を検出
- [ ] unit test: manifest読み取り（disk/rectangle/arbitrary_mask）の最小ケース

## Acceptance Criteria
- [ ] `dataset.root=<dir>` だけで domain が自動決定できる（manifestあり）
- [ ] manifestが無い dataset でも従来通り動く（後方互換）
- [ ] 不正manifestは doctor で明確に失敗し、理由がログに出る
- [ ] unit tests が追加され、CI/ローカルで通る

## Verification
- Command:
  - `python -m mode_decomp_ml.cli.run task=doctor dataset.root=data/mode_decomp_eval_dataset_v1/scalar_disk`
- Expected:
  - domain が disk として認識され、mask/shape が妥当と判定される

## Review Map
- 変更ファイル一覧:
  - 追加: `src/mode_decomp_ml/data/manifest.py`, `tests/test_manifest.py`, `data/mode_decomp_eval_dataset_v1/scalar_disk/manifest.json`
  - 更新: `src/mode_decomp_ml/data/datasets.py`, `src/mode_decomp_ml/pipeline/utils.py`, `src/mode_decomp_ml/pipeline/__init__.py`, `configs/dataset/synthetic.yaml`
  - 更新: `src/processes/doctor.py`, `src/processes/train.py`, `src/processes/predict.py`, `src/processes/eval.py`, `src/processes/viz.py`, `src/processes/benchmark.py`
- 重要な関数/クラス:
  - `src/mode_decomp_ml/data/manifest.py`: `load_manifest`, `manifest_domain_cfg`, `validate_field_against_manifest`
  - `src/mode_decomp_ml/pipeline/utils.py`: `resolve_domain_cfg`, `build_dataset_meta`
  - `src/mode_decomp_ml/data/datasets.py`: `NpyDirDataset.__init__`（manifest検証・full True mask生成）
- 設計判断:
  - manifest があれば domain を常に上書きし、config domain より dataset を真実にした
  - manifest なしは従来どおり domain を使用し後方互換を保持
  - mask 欠落時は manifest ありの場合のみ全Trueを生成し、mask生成有無を dataset_meta に記録
- リスク/注意点:
  - decomposer 側で mask を許容しない設定の場合、全True mask があると挙動が変わる可能性がある
  - manifest の field_kind/ grid と実データ shape が合わない場合は即時エラー
- 検証コマンドと結果:
  - `python3 -m pytest tests/test_manifest.py`（3 passed）
  - `PYTHONPATH=src python3 -m mode_decomp_ml.cli.run task=doctor dataset.root=data/mode_decomp_eval_dataset_v1/scalar_disk`（dataset=npy_dir, field/mask shape logged）
