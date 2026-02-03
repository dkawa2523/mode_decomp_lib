# S60: Training / HPO

## 目的
- 学習を安定化し、HPOで性能を伸ばせる形にする

## 手順
- optimizer/lr/epochs/batch/seed を config 化
- early stopping / LR scheduler 等はオプション化
- HPOは “比較基盤（artifact/leaderboard）” が整ってから

## 事故りやすい点
- seed未固定で比較不能
