# TASK 494: decomposer配置整理（analytic vs data_driven の分離、互換alias維持）

## 目的
既存の解析基底 decomposer と、これから入れる POD系（データ駆動） decomposer を **同じ階層に混在させず**、レビューしやすい配置に整理します。
ただし、大規模移動で壊さないため **互換レイヤー（re-export / alias）** を残します。

## 作業内容
1. 現在の decomposer 実装ファイル群を棚卸し
   - 解析基底（FFT/Zernike/Bessel/…）
   - データ駆動（現状あるなら POD/SVD 等）
2. `src/**/decompose/data_driven/`（または現行の命名規約に沿う相当パス）を新設
3. POD/SVD系が既にある場合：
   - 原則、`data_driven/` に移す
   - 旧モジュールパスには **薄い互換ファイル** を残す（importとconfig aliasを壊さない）
4. Decomposer registry を更新して、`pod` / `gappy_pod` を登録する“場所”を確保
5. 解析基底とデータ駆動で `__init__.py` や registry の記述が衝突しないよう整理

## 受け入れ条件
- 解析基底 decomposer と data-driven decomposer がディレクトリで区別できる
- 既存の decomposer 名（configで使っていた名前）が壊れない（alias or re-export維持）
- registry の依存方向が単純（core→plugins で、plugins間の相互依存を増やさない）

## 検証
- 既存の最小実行（train/eval/predictのいずれか1つ）が動く（Task492時点で動いていたコマンド）
- `python -m compileall src` が通る
