# パッケージ改良計画の現在地

2026-10-25、**mode 0.10.4 / core 0.2.1**。単体パッケージの実装・改良・配布先単独のローカル受入を完了し、配布用ZIPを作成しました。最新の正本は[0.10.4配布受入記録](../work/reviews/package_distribution_validation_20261025.json)、計画は[統合実装計画](../work/reviews/package_implementation_plan_20261025.json)です。

## 配布物

| 成果物 | 用途 |
| --- | --- |
| `dist/ml_platform_mode_decomp-0.10.4.zip` | 利用先へ渡す一式。590,521 bytes。2wheel、14YAML、5生成例、9利用guide、README、source archive、core来歴、SHA256SUMS |
| `dist/ml_platform_mode_decomp-0.10.4-py3-none-any.whl` | modeの実行package |
| `dist/ml_platform_core-0.2.1-py3-none-any.whl` | 同梱する共通依存package |
| `dist/ml_platform_mode_decomp-0.10.4.tar.gz` | modeの製品source archive。これからwheelをbuild済み |
| `dist/SHA256SUMS` | 現行配布物の照合用。通常利用で実行を必須にしない |

導入・最初の実行は[配布用ガイド](DISTRIBUTION.md)に集約しました。利用先では標準venvとpipを使い、Git・uv・基盤checkout・SDKを要求しません。NumPy等の依存はpipが取得します。対象OS/Pythonの全依存wheelを含むoffline wheelhouseではありません。

## 計画の現在地

| 段階 | 現在地 |
| --- | --- |
| I00〜I12 | 完了。入力・数値責務・layout/rank・評価目的・物理成分・実行/出力の分離と不要処理の整理 |
| I13 | 単体配布物とローカル受入完了。remote CI実行・外部registry公開は未実施 |
| O1〜O5 | 完了。設定再実行、評価measure、reader、OLS block化、Gappy固定domain/batch solve |
| B1〜B4 | 完了。較正予測batch、CV baseline copy削減、帯の評価/保存の全件保持削減 |
| F1〜F4 | 完了。入力identity/snapshot、Joint/EMの隠れたbasis保持、解析/モード保持と時間計測 |
| D1〜D4 | 完了。配布metadata、clean sdist→wheel、自己完結ZIP、stdlib venv/pip consumer、guideとCI定義 |

0.10.4では数値source105filesを0.10.3から変更していません。単一YAML/配列API、独立成分Kc/rank0、grid/observations/hidden、固定support較正、Hodge/Helmholtzの意味を維持しています。

## 配布で追加した整理

- 直接importするpandas・PyYAMLを直接runtime依存へ宣言。coreの推移依存へ偶然頼る構成を解消。
- package discoveryを製品namespaceへ限定。sdistからtests、build、cache、bytecodeを除外。
- 開発用tests/fixtures、work/履歴、integration、入力data、cacheを配布物へ混入させない。
- cleanなsource archiveからwheelを構築し、checkoutにしかないfileで欠落を隠さない。
- 同梱YAML・生成script・相互linkの揃った利用guideをZIPへ追加。利用例のuv前提と壊れたanchorを整理。
- distの旧0.10.3 wheelを不変比較基準へ保持した後に削除。現行4artifactとSHA256SUMSへ整理。

ライセンスはrepositoryに根拠がないため新しい権利許諾を設定していません。公開registryの選定・公開操作は実施していません。

## 受入結果

- Python3.12.13開発環境：**609 passed / 3 skipped**。NumPy2.4.2 / SciPy1.17.0 / sklearn1.8.0、wavelet・pod・torch・MTGP追加済み。
- 新規標準venv・通常pip導入：**579 passed / 33 skipped**。NumPy2.5.4 / SciPy1.18.1 / sklearn1.9.1。skipはoptional19、旧producerの厳密な数値runtime相違14。Python minorによるskipではない。
- 旧14bundleはproducer互換の開発環境で成功。新環境へ無理に互換を主張せず、理由付きskipを維持。
- 純正consumer確認はpytest追加前に実施し、pip check成功。同じenvへpytestだけを追加して数値試験し、数値依存は変更していない。
- ZIP内の例・設定だけで、installed console、配列DCT、単一YAML、学習/保存、訓練入力を隠した後の別process推論、NPY reader、ID/条件列整列が成功。
- ZIP34files、33checksum一致、全Markdown相対link有効。wheel110members、sdist131members、105sourceのbytes一致。
- 同じinstalled環境で3解析用途・固定/CV・NPZ/NPY・較正/散布点/Hodgeも成功。
- Ruff・format（211files）、Pyrefly0errors、依存方向2規則、lock78packages整合。原入力67NPYのhash不変。

以前のwheel検査はcheckoutのYAML/examplesを参照していました。今回は配布ZIP単独のconsumer検査を追加し、この違いを明示しました。CIもZIP生成→stdlib pip consumer→同envでwheel受入へ更新済みですが、remote実行は未実施です。

新規consumer venvは一つだけ保持しています。以前に削除を拒否された`backend/baseline_source`と旧wheel_envは再削除せず、開発workspace内に保持しました。どちらも配布物へ含めていません。

## 次の工程

1. 配布先が決まれば、このZIPを渡す。公開する場合は更新したremote CIを実行し、registry/channelと権利許諾を確定する。
2. 実観測・将来truth・実際の物理境界で用途別受入を行う。合成データやpackage受入を、その用途の有効性認定とは扱わない。
3. SVD/EM/GP・共有CV truth・厳密pointwise・配列返却の容量は実測で支配的な部分だけ改良する。SDK再統合、任意点較正帯、依存sample向け較正、任意PDE、3D/移動geometry等は別仕様で進める。

以前の実装・容量結果は[0.10.3改良](../work/reviews/package_refinement_validation_20261025.json)、[0.10.2較正/CV/帯出力](../work/reviews/package_streaming_validation_20261025.json)、[0.10.1運用](../work/reviews/package_operational_validation_20261025.json)、[0.10.0実装](../work/reviews/package_implementation_validation_20261025.json)に履歴として保持しています。
