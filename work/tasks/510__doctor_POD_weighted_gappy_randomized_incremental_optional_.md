# TASK 510: テスト/doctor: POD系（weighted/gappy/randomized/incremental）を固める、optional依存整理

## 目的
POD系追加で壊れやすい箇所（重み・欠損・乱択・増分）をテストで固め、doctorにも反映します。

## 作業内容
1. 単体テスト：
   - POD fit→inverse の roundtrip がK増で改善
   - Weighted POD が mask/disk で動く（重みshape検証）
   - Randomized がseedで再現
   - Incremental が動く（誤差が極端に悪化しない）
   - Gappy POD が観測領域で整合し、欠損領域も復元できる
2. optional dependency（modred）の扱い
   - 無い場合は skip し、doctorで案内
3. doctor を更新し、必要依存の不足を明確に出す

## 受け入れ条件
- pytest（または既存テスト枠）が通る
- modred無しでもテストが壊れない（skip/xfailで整理）
- doctorが “何が足りないか” を正しく表示する

## 検証
- `python -m pytest -q`（既存のテスト実行方法に従う）
- doctor実行
