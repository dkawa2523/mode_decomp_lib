# Task 059 (P1): データ駆動分解：POD/SVD（静的2Dデータ集合の基底学習）

## 目的
時間変化なしの多数サンプル（2D場）の集合から、データ駆動で最適基底（POD=PCA/SVD）を学習し、
各サンプルを係数 `a` に射影して分解する `Decomposer` を追加する。

固定基底（Zernike/FFT等）と比較し、データ分布に適応した基底で再構成誤差や回帰精度が改善する可能性がある。

## 背景 / 根拠
- docs/00_INVARIANTS.md（skew禁止：fitはtrainのみ）
- docs/11_PLUGIN_REGISTRY.md（Decomposer規約）
- docs/09_EVALUATION_PROTOCOL.md（比較可能性）

## 依存関係
- depends_on: work/tasks/030_data_domain_io.md
- depends_on: work/tasks/050_decomposer_registry.md

## スコープ
- `pod_svd` decomposer を追加（fit/transform/inverse）
- fitは **train split の field** のみ
- maskがある場合の扱いを固定（例：mask内ベクトル化 or mask外0埋め）
- 逆変換で `field_hat` を生成できる

## 実装方針（推奨）
- sklearn PCA / TruncatedSVD を使用（randomized可）
- `n_modes` と `energy_threshold` を設定化（どちらかを使用）
- 学習した basis を artifact として保存し、推論で再利用する

## Acceptance Criteria（完了条件）
- [ ] `pod_svd` decomposer が registry に登録される
- [ ] `fit(train_fields)` が train-only で実行される（test漏洩しない）
- [ ] `transform`/`inverse` が動作し round-trip が成立する
- [ ] 再構成誤差が `n_modes` 増加で改善することが確認できる

## Verification（検証手順）
- [ ] tiny dataset で train split のみでfitされることをログで確認
- [ ] `decompose=pod_svd` を指定して reconstruct が完走する
- [ ] `n_modes` を 2→8→32 と変え、field_rmse が改善することを確認

## Autopilotルール（重要）
**DO NOT CONTINUE**: 上の Acceptance Criteria を全て満たすまで、queue上でこのタスクを `done` にしない。
不明点があっても質問で止まらず、TODOを残して安全側（train-only、保存優先）で前進する。
