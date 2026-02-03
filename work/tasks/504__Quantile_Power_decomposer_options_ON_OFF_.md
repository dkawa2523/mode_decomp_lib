# TASK 504: 周辺技術: 係数分布正規化（Quantile/Power、全decomposer共通、optionsでON/OFF）

## 目的
係数の分布正規化（Quantile/Power）を、POD系だけでなく **特殊関数系decomposerでも使える**共通機能として追加/統合します。

## 作業内容
1. `coeff_post` に以下を追加（既にあるなら統合して整理）
   - QuantileTransformer（normal/uniform）
   - PowerTransformer（Yeo-Johnson）
2. train-only fit を強制（fitはtrain係数のみ）
3. `options.coeff_normalize.enable` でON/OFFできるようにする
   - enable=false の場合は no-op
4. 保存：
   - fitted transformer を `states/coeff_post_*.pkl` 等に保存
5. 既存PCA/standardizeと組み合わせても設定爆発しないよう、pipeline構築を単純化
   - 例：mode_weight → normalize → pca の固定順（docsに明記）

## 受け入れ条件
- Quantile/Power が任意decomposerで適用可能（POD/FFT/Zernikeなど）
- 逆変換（inverse）が可能な場合は再構成が壊れない
- optionsでON/OFFでき、run.yamlを増やさない

## 検証
- normalize ON/OFFで係数分布（coeff_hist.png）が変化
- 同一モデルで学習したとき、極端な劣化がない（ケース依存なので定性的でOK）
