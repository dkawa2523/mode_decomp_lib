# 改良方法ガイドライン

## 改良対象と変更箇所
| 改良内容 | 対象ファイル | 変更内容 |
| --- | --- | --- |
| 前処理追加 | `src/mode_decomp_ml/preprocess/` | 新しい前処理クラス追加 |
| モデル追加 | `configs/model/*.yaml`, `plugins/` | モデル定義・登録 |
| 分解法追加 | `configs/decompose/*.yaml`, `plugins/decomposers/` | 分解器実装 |
| 可視化追加 | `src/mode_decomp_ml/viz/__init__.py` | 新規プロット関数 |
| パイプライン変更 | `src/processes/pipeline.py` | フロー分岐の追加 |

## 詳細方針
- ワークフローは維持し、内部の実装を差し替える
- 変更時は outputs の構造を壊さない
- データセット仕様（CSV/NPY）は厳守
