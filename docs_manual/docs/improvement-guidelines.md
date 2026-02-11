# 改良方法ガイドライン

このページは「ワークフローは維持したまま、どこを変えるか」を明確にします。

## 変更対象とファイル早見表（テンプレ）

| 変更内容 | 対象ファイル/ディレクトリ | 変更の要点 |
|---|---|---|
| 新しいモード分解手法 | `src/mode_decomp_ml/plugins/decomposers/` | `@register_decomposer` + coeff_meta 契約 |
| 新しい codec | `src/mode_decomp_ml/plugins/codecs/` | encode/decode + raw_meta |
| coeff_post の追加 | `src/mode_decomp_ml/plugins/coeff_post/` | forward/inverse（近似なら明記） |
| 可視化（plot）追加 | `src/mode_decomp_ml/viz/` / `src/processes/*` | 生成物命名を固定 |
| ベンチ集計の追加 | `tools/bench/report/` | summary生成契約を維持 |

## 追加手順（テンプレ）

まず読む（canonical）:

- 追加手順の詳細: `docs/17_EXTENSION_PLAYBOOK.md`
- plugin 互換・登録: `docs/11_PLUGIN_REGISTRY.md`
- coeff_meta 契約: `docs/28_COEFF_META_CONTRACT.md`
- codec 契約: `docs/21_CODEC_LAYER_SPEC.md`

### 新しいモード分解手法（decomposer）を追加する最短手順

1. `src/mode_decomp_ml/plugins/decomposers/<name>.py` を追加
2. `@register_decomposer("<name>")` を付ける
3. `coeff_meta()` で必要キーを埋める（contractに従う）
4. `configs/decompose/analytic/<name>.yaml` を追加（最低限のデフォルト）
5. `configs/decompose/<name>.yaml` の shim を追加（config coverage 用）
6. `tests/test_decompose_<name>.py` を追加（roundtrip と meta）

### plots を追加する時の注意

- ファイル名は “契約” になるので、既存の命名規則（`mode_r2_vs_k.png` 等）に寄せる
- ベンチ集計（`tools/bench/report/`）が link を集めている場合は、リンク候補を更新する
