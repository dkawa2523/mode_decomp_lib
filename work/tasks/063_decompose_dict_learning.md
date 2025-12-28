# Task 063 (P2): データ駆動分解：Dictionary Learning（疎表現）

## 目的
PODよりも疎な係数で再構成できる可能性がある Dictionary Learning を Decomposer として追加し、
再構成誤差と疎性のトレードオフを比較できるようにする。

## 依存関係
- depends_on: work/tasks/059_decompose_pod_svd.md

## ライブラリ候補
- scikit-learn DictionaryLearning / sparse_encode

## Acceptance Criteria（完了条件）
- [ ] `dict_learning` decomposer が registry に登録される
- [ ] fitはtrainのみ（skew禁止）
- [ ] transform/inverse が動作する
- [ ] 疎性（非ゼロ率）と誤差が評価ログに残る

## Verification（検証手順）
- [ ] tiny dataset で round-trip が動く
- [ ] alpha 等を変え、疎性が変化することを確認

## Autopilotルール（重要）
**DO NOT CONTINUE**: 受け入れ条件を満たすまで `done` にしない。
