# 2D空間分布のモード分解 + 係数学習 基盤（Greenfield Devkit）

このzipは、既存リポジトリの大規模改修ではなく、**0から目的仕様に沿って作る**ための開発基盤です。
VSCode + Codex CLI で `autopilot.sh` を回し、`work/queue.json` のP0タスクを順番に完了させていきます。

## 目的（P0で達成する最小完成形）
- 2Dスカラー場/ベクトル場（2ch）を扱えるデータスキーマ
- domain（境界）: rectangle / disk をP0で対応
- decomposer（一次分解）: FFT2, Zernike（P0）
- coeff_post（係数後処理）: none, standardize, PCA（train-only fit, inverse）
- model（回帰）: Ridge（多出力）
- process（CLI）: decomposition / preprocessing / train / inference / pipeline / leaderboard / doctor
- artifact契約に沿った保存（config/meta/metrics/preds/model）
- 比較可能な評価（coeff誤差 + field再構成誤差）

## 使い方（最短）
1) 本zipを任意の新フォルダに展開
2) 依存をインストール（どちらでも）
   - `pip install -r requirements.txt`
   - もしくは `pip install -e .`
2.1) Optional dependencies（必要な場合のみ）
   - Wavelet2D: `pip install pywt`
   - Autoencoder / deep models: `pip install torch torchvision`
   - Spherical harmonics (sphere_grid): `pip install pyshtools`
   - GBDT models (xgb/lgbm/catboost): `pip install xgboost lightgbm catboost`
3) 最短実行例（`pip install -e .` を使わない場合は `PYTHONPATH=src` を付ける）
   - `PYTHONPATH=src python -m mode_decomp_ml.cli.run task=doctor`
   - `PYTHONPATH=src python -m mode_decomp_ml.cli.run task=pipeline`
4) examples（run.yaml）
   - `PYTHONPATH=src python -m mode_decomp_ml.run --config examples/run_scalar_rect_fft2_ridge.yaml`
   - `PYTHONPATH=src python -m mode_decomp_ml.run --config examples/run_scalar_disk_zernike.yaml`
   - `PYTHONPATH=src python -m mode_decomp_ml.run --config examples/run_scalar_mask_pod_ridge.yaml`
5) （推奨）git repo化
   - `git init && git add -A && git commit -m "init"`
6) Autopilot実行
   - `chmod +x autopilot.sh doctor.sh tools/autopilot.sh`
   - `LIVE_TEE=1 PYTHON_BIN=/usr/bin/python3 ./doctor.sh`
   - `LIVE_TEE=1 PYTHON_BIN=/usr/bin/python3 ./autopilot.sh 30`

## 重要：タスクが増え続けないためのルール
- Autopilot実行中に **queueへ新規タスクIDを追加することは禁止**（Task creep guard 既定ON）
- 新しい拡張は `docs/17_EXTENSION_PLAYBOOK.md` に従い、別途 `work/queue_p1.json` / `work/queue_p2.json` に起票します

## ドキュメント
- `docs/README.md` から読むのが最短です
- ベンチ/レポート: `docs/30_BENCHMARKING.md`
- コード全体の見取り図: `docs/31_CODE_TOUR.md`
- 追加の拡張計画は `work/queue_p1.json`, `work/queue_p2.json` にあります（P0完了後に実施）
