# S95: Tests / CI

## 目的
- 反復開発で壊れない

## 手順
- smoke test: 最小データで一連が動く
- contract test: artifact契約（必須キー/列）を検査
- doctor をCIで回す（任意）

## DoD
- 変更に対応するテストがある
