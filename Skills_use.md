# Skills_use.md

## 目的
- 新しいデータセット/ドメインに対して、モード分解評価・学習評価・推論評価を再現可能に回す。
- `agentskills` を使う前提で、Copilot への指示をテンプレート化する。

## 対象 Skills（本用途で主に使うもの）
- `S21_dataset_config_playbook`: dataset/domain 設定（npy_dir/csv_fields/manifest/mask_policy）
- `S45_method_config_playbook`: decompose/codec/coeff_post/model の整合設定
- `S46_pipeline_config_assembly`: run.yaml/Hydra 統合、dry-run→smoke→pipeline の実行組み立て
- `S44_domain_geometry`: domain 座標/重み/mask 整合の詳細確認
- `S70_eval_metrics`: 指標確認・比較観点の固定
- `S95_tests_ci`: 最低限のテスト/検証実行

## 推奨ワークフロー（順番固定）
1. `S21` で dataset 設定を確定
2. `S45` で手法設定を確定
3. `S46` で run.yaml を組み立て
4. dry-run 実行
5. decomposition 単体実行（smoke）
6. pipeline 実行（decomposition→preprocessing→train）
7. inference 実行
8. leaderboard/レポートで比較

## 最低限の実行コマンド
```bash
# 1) run.yaml 解決確認
PYTHONPATH=src python -m mode_decomp_ml.run --config examples/run_realdata_npy_template.yaml --dry-run

# 2) pipeline 実行（run.yaml エントリ）
PYTHONPATH=src python -m mode_decomp_ml.run --config examples/run_realdata_npy_template.yaml

# 3) inference 実行（Hydra エントリ例）
PYTHONPATH=src python -m mode_decomp_ml.cli.run \
  task=inference \
  output.name=realdata_infer_v1 \
  task.decomposition_run_dir=runs/realdata_trial_v1/decomposition \
  task.preprocessing_run_dir=runs/realdata_trial_v1/preprocessing \
  task.train_run_dir=runs/realdata_trial_v1/train \
  inference.mode=batch \
  inference.source=random \
  inference.n_samples=16

# 4) leaderboard 集約
PYTHONPATH=src python -m mode_decomp_ml.cli.run \
  task=leaderboard \
  output.name=realdata_board_v1 \
  task.runs='["runs/**/outputs/metrics.json"]' \
  task.sort_by=field_rmse
```

## Copilot プロンプトテンプレート

### Template A: 初期セットアップ（S21+S45+S46）
```text
このリポジトリの agentskills を使って、新規データセット評価の初期設定を行ってください。
順番は S21_dataset_config_playbook → S45_method_config_playbook → S46_pipeline_config_assembly。

目的:
- dataset と domain をコード契約どおりに設定
- decompose/codec/model を整合する設定にする
- run.yaml を 1 本作成し、dry-run が通る状態にする

入力:
- dataset root: <PATH>
- 想定 domain: <rectangle|disk|annulus|arbitrary_mask|sphere_grid|mesh>
- field type: <scalar|vector>
- 優先手法: <任意>

成果物:
1) 作成/更新ファイル一覧
2) run.yaml（完全版）
3) 実行コマンド（dry-run, decomposition smoke, pipeline）
4) 失敗時の切り分けポイント
```

### Template B: モード分解評価に集中（S45+S70）
```text
S45_method_config_playbook と S70_eval_metrics を使って、
このデータセットで分解性能比較を行う設定を作ってください。

要件:
- domain と整合する decomposer 候補を 3 つ提示
- 各候補の codec を明示
- 評価指標は field_rmse, field_r2, energy_cumsum を有効化
- decomposition 単体実行コマンドを出す

出力形式:
- 候補表（method / codec / 理由 / 注意点）
- params.decompose の具体 YAML
- 実行コマンド
```

### Template C: 学習評価に集中（S45+S46+S70）
```text
S45_method_config_playbook, S46_pipeline_config_assembly, S70_eval_metrics を使って、
cond->coeff->field の学習評価を実行したいです。

要件:
- model は ridge を baseline にし、追加で 1 手法比較
- train.eval, train.cv, train.tuning の ON/OFF を明示
- val_rmse/val_r2 と val_field_rmse/val_field_r2 を必ず記録
- 生成する run 名を衝突しない命名規則にする

成果物:
- run.yaml 修正案
- 実行コマンド
- 期待される outputs/metrics.json の確認ポイント
```

### Template D: 推論評価に集中（S46+S70）
```text
既存の decomposition/preprocessing/train の成果物を使って、
推論評価を行う設定を作ってください。
S46_pipeline_config_assembly と S70_eval_metrics を使ってください。

入力:
- decomposition_run_dir: <PATH>
- preprocessing_run_dir: <PATH>
- train_run_dir: <PATH>
- inference mode: <single|batch|optimize>

要件:
- inference 設定（values または grid/ranges）を作成
- 実行コマンドを提示
- 出力される preds と metrics の読み方を簡潔に説明
```

### Template E: エラー時の再指示
```text
前回の実行で失敗しました。以下のエラーをもとに、skills を使って修正してください。
優先順位は S21 → S45 → S46 です。必要なら S44 を使って domain 整合も確認してください。

error:
<貼り付け>

制約:
- 既存設定を全破棄しない
- 変更差分を最小化
- 修正後に dry-run コマンドまで提示
```

## 使い方のコツ（Copilot 指示品質）
- 必ず「入力条件」を固定する。
- 必ず「成果物形式」を指定する（ファイル名・コマンド・確認点）。
- 必ず「順番」を指定する（S21→S45→S46）。
- 推論だけ行う場合も、上流 run_dir を明示する。
- 失敗時はエラーメッセージを全文貼る。

## 用途別の開始点
- 新規実データ（npy）: `examples/run_realdata_npy_template.yaml`
- 新規実データ（csv）: `examples/run_realdata_csv_template.yaml`
- スキル索引: `agentskills/ROUTER.md`
- スキル本体: `agentskills/skills/`

## 完了チェックリスト
- dry-run が通る
- decomposition の `outputs/metrics.json` が生成される
- train の `val_*` と `val_field_*` が生成される
- inference の `outputs/preds.npz` と `outputs/metrics.json` が生成される
- 必要なら leaderboard を更新できる
