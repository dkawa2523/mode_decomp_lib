# Task 116 (P2): 解釈性レポート（スペクトル/逐次再構成/相関）

## 目的
精度だけでなく「物理的解釈性」を比較できるよう、以下を run dir に保存するレポートを生成する。
- 係数エネルギースペクトル（次数/周波数）
- 低次モードからの逐次再構成（k=1,2,4,...）
- 予測係数 vs 真値係数の相関

## 依存関係
- depends_on: work/tasks/085_metrics_module.md
- depends_on: work/tasks/110_benchmark_runner.md

## Acceptance Criteria（完了条件）
- [ ] レポートが run dir に保存される（PNG/CSVでOK）
- [ ] 少なくとも上の3種が出力される
- [ ] 手法間で差が見える（例：低次の寄与率や高周波ノイズ）

## Verification（検証手順）
- [ ] 2つ以上の手法でレポートを生成して比較できることを確認

## Autopilotルール（重要）
**DO NOT CONTINUE**: レポート生成が動くまで `done` にしない。
