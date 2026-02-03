# S50: Models

## 目的
- モデル追加を「新規クラス + registry登録 + config追加」で完結させる

## 手順
- src/models/<name>.py を追加
- registryに登録
- configs/model/<name>.yaml 追加
- smoke test（最小データで train→eval）

## 事故りやすい点
- モデル保存/ロードが曖昧でpredictできない
