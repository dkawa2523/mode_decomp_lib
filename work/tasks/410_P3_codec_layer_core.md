# Task: 410 Core: CoeffCodec層導入（registry + pipeline wiring + meta保存）

- Priority: P1
- Status: todo
- Depends on: 404
- Unblocks: 411, 412, 413, 414, 415, 416, 417, 420, 421, 430

## Intent
CoeffCodec 層（raw_coeff <-> vector_coeff）を導入し、係数表現の多様化（複素/階層/構造化）を
パイプラインから隔離する。以後の新規手法（Wavelet/SH/Slepian/PSWF）を安全に追加できる土台を作る。

## Context / Constraints
- Decomposer は raw_coeff を返すだけにする（型は自由）
- 学習/後処理/保存は vector_coeff（float32 1D）に統一する
- lossless/losssy を明示し、artifactに meta を保存する
- docs/21 を設計契約とする

## Plan
- [ ] `CoeffCodec` interface と registry を追加（plugins/codecs）
- [ ] pipeline に codec を組み込み: decomposer.transform -> codec.encode -> coeff_post -> model
- [ ] inverse側も組み込み: model -> coeff_post.inverse -> codec.decode -> decomposer.inverse
- [ ] 既存の「係数flatten」処理があれば codec に移動し削除
- [ ] テスト: ダミーcodec(lossless)で roundtrip を保証
- [ ] docs/21 を参照し、artifactに `states/coeff_meta.json` 等を保存する

## Acceptance Criteria
- [ ] pipeline が codec を必須コンポーネントとして扱う（none も可だが明示）
- [ ] lossless codec の roundtrip テストが通る
- [ ] 既存処理とバッティングする flatten/logics が削除または移設される
- [ ] 保存物に codec/meta が含まれる

## Verification
- Command:
  - `python -m mode_decomp_ml.run --config examples/run_scalar_rect_fft_ridge.yaml`
- Expected:
  - `states/coeff_meta.json` が生成される
  - reconstruct が通る

## Review Map
- 変更ファイル一覧: `src/mode_decomp_ml/plugins/codecs/basic.py`, `src/mode_decomp_ml/plugins/codecs/__init__.py`, `src/mode_decomp_ml/plugins/registry.py`, `src/mode_decomp_ml/plugins/__init__.py`, `src/mode_decomp_ml/pipeline/utils.py`, `src/mode_decomp_ml/pipeline/__init__.py`, `src/mode_decomp_ml/run.py`, `src/processes/train.py`, `src/processes/reconstruct.py`, `src/processes/eval.py`, `src/processes/viz.py`, `configs/codec/none.yaml`, `configs/config.yaml`, `tests/test_codec_roundtrip.py`, `tests/test_processes_e2e.py`
- 重要な関数/クラス: `BaseCoeffCodec`/`NoOpCoeffCodec`（`src/mode_decomp_ml/plugins/codecs/basic.py`）, `build_coeff_codec`（`src/mode_decomp_ml/plugins/registry.py`）, `load_coeff_meta`（`src/mode_decomp_ml/pipeline/utils.py`）, `train.main`/`reconstruct.main`/`eval.main`/`viz.main`（各process）
- 設計判断: codecはlosslessなnoneを既定として追加し、raw_coeffのflatten/float32化をcodecに集約。`states/coeff_meta.json`にdecomposerのmeta＋codec情報を保存しつつ既存の`states/decomposer/coeff_meta.json`も維持。復元系はtrain保存のcodec stateとcoeff_metaを参照してdecodeを挟む。
- リスク/注意点: 既存decomposerのflattenは残しており、本格的なraw構造の扱いはタスク411で整理予定。過去runにはcodec stateがないため新実装のreconstruct/eval/vizでは失敗する可能性あり。
- 検証コマンドと結果: `pytest tests/test_codec_roundtrip.py` ✅ / `pytest tests/test_processes_e2e.py` ✅ / `python -m mode_decomp_ml.run --config examples/run_scalar_rect_fft_ridge.yaml` は `python` 不在、`python3` 実行で `UnsupportedInterpolationType now` により失敗。
- 削除一覧: なし
