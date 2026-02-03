# S30: Preprocess（fitが必要な変換）

## 目的
- train/infer skew を防ぐ

## 手順
- fitする変換は state を保存
- predictは state をロードして同一変換を適用

## 事故りやすい点
- train時にfitしたスケーラをpredictでfitし直す
