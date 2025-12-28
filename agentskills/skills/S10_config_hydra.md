# S10: Config / Hydra

## 目的
- configで挙動が変わる状態にし、再現/比較可能にする。

## 手順
- Process別入口config（configs/process/*）を作る
- groupを揃える（dataset/split/features/model/train/eval）
- run dir に config.yaml を保存する

## 事故りやすい点
- デフォルト値が散在して再現できない
- splitやseedが保存されない
