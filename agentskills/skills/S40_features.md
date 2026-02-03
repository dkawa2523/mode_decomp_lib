# S40: Features

## 目的
- 特徴量を複数方式で差し替え可能にする

## 手順
- features registry（name→constructor）で追加
- stateがある場合は保存/ロードを必須化
- 追加後は contract test と smoke test

## 事故りやすい点
- 特徴量の状態（辞書/語彙/正規化）が保存されない
