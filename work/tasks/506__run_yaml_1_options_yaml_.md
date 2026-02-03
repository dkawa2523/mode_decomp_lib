# TASK 506: 運用設計: run.yaml 1枚運用の維持（options切替、yaml増殖禁止）

## 目的
POD系と周辺技術を追加しても「YAMLが増える」「入口が増える」を起こさないよう、run.yaml 1枚運用を強制する。

## 作業内容
1. `docs/addons/31_POD_CONFIG_MINIMAL.md` の最小run.yamlが実際に動くように調整
2. 新規のconfigファイル追加は原則禁止（どうしても必要なら docs に理由とTODO）
3. run.yaml の options で以下が切り替えできること
   - rank_select ON/OFF
   - coeff_normalize ON/OFF（quantile/power）
   - mode_weight ON/OFF
4. 既存の古いconfig群が残っている場合、deprecated経路を明確化（削除は cleanup タスクで）

## 受け入れ条件
- run.yaml 1枚で POD系の train/eval/viz が実行できる
- options の有効/無効で挙動が切り替わる
- 追加で増えたyamlファイルが無い（例外は要説明）

## 検証
- `task=train` と `task=viz` を run.yamlだけで実行し、runs配下に成果物が出る
