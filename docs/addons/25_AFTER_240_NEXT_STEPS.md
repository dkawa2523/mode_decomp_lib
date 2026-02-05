# After 240: 次にやること（P1の残り → P2）

あなたは以下まで完了しています:
- P0 完了（000〜120）
- P1: domain(arbitrary_mask), Fourier–Bessel, POD/SVD, GPR, pipeline sweep

## P1で残っている必須（最短で“完成”にする）
### 250: 不確かさ（GPR）を artifact に残し、可視化する
- 係数の予測だけだと「どれくらい信じて良いか」が判断できない
- 最小で良いので **coeff_std** を保存し、代表サンプルで **uncertainty map** を出す

### 260: 比較用のモデル追加（ElasticNet / MultiTask）
- “モデル拡張が頻繁” の要件を満たす実証タスク
- leaderboardで ridge / gpr と比較できる状態に

### 270: docs + examples を更新
- 実装とドキュメントのズレを解消
- “どう実行するか” を短い例で固定

### 290: P1 cleanup
- 不要コード/未使用ファイル/試行コードを残さない（後からの大規模リファクタ回避）

## P2の入り口
- まず 299 (P2 Preflight) を入れて “依存・計算コスト・範囲合意” を固定
- その後、Graph Laplacian → Laplace–Beltrami → AE/VAE → DictLearn → Vector(Helmholtz) → ClearML の順

## 重要：P1の残りは “最小実装で終わらせる”
- Monte Carlo uncertainty など、重いことはやり過ぎない
- “将来のための抽象化” は禁止（docs/14）
