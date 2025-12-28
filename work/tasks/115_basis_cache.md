# Task 115 (P2): 基底キャッシュ（domain×method×params hash）

## 目的
Zernike/Bessel/Graph/Laplace系の基底前計算をキャッシュし、benchmark を現実的な時間で回せるようにする。
cacheはrun成果物（artifact）と混同しない。

## 依存関係
- depends_on: work/tasks/050_decomposer_registry.md
- depends_on: work/tasks/110_benchmark_runner.md

## Acceptance Criteria（完了条件）
- [ ] 同一条件の2回目以降で cache hit になる（ログで確認可能）
- [ ] key に shape/method/params/domain/version が含まれ、誤使用が起きない
- [ ] cacheの保存先が不変として docs に明記される

## Verification（検証手順）
- [ ] 2回連続で同じ分解を走らせ、2回目が速い/ログでhitを確認
- [ ] paramsを変えるとmissになることを確認

## Autopilotルール（重要）
**DO NOT CONTINUE**: hit/miss が確認できるまで `done` にしない。
