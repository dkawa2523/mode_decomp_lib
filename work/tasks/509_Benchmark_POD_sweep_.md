# TASK 509: Benchmark: POD系を含む最小sweep追加（組合せ爆発回避、速度指標も保存）

## 目的
benchmark sweep に POD系（Weighted/Gappy/Randomized/Incremental）と周辺技術の組合せを最小追加し、比較基盤として回せるようにします。

## 作業内容
1. `scripts/bench` または既存の sweep 実装を更新
2. 組合せ爆発を避けるため、まずは “最小行列” を定義
   - pod (sklearn full)
   - pod (sklearn randomized)
   - pod (modred weighted) ※modredがある場合のみ
   - gappy_pod（mask domainのみ）
   - options: normalize on/off を少数
3. 速度指標（fit_time_sec等）も leaderboard に載せる
4. sweep結果の集計（leaderboard）で確認できるようにする

## 受け入れ条件
- sweepが途中で爆発せずに完走できる（小さいmatrix）
- 結果が比較可能な形で保存される（leaderboard更新）

## 検証
- 小さなsweepを1回回し、runsが複数生成され、leaderboardが更新される
