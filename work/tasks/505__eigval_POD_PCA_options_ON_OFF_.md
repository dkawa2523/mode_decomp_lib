# TASK 505: 周辺技術: モード重み付け（eigvalスケール、POD/PCA優先、optionsでON/OFF）

## 目的
モード毎の重み付け（eigvalでスケール）を共通機能として導入し、高次モードのノイズ影響を抑える。

## 作業内容
1. `options.mode_weight.enable` でON/OFF
2. method=`eigval_scale`:
   - POD/PCAの eigvals を使って係数をスケール（例：whitening相当）
3. eigvalsが無い decomposer（FFT等）の場合：
   - `enable=true` のとき警告して no-op（またはエラー）を選べるようにする（v1はno-op推奨）
4. 変換は coeff_post でも良いが、責務を増やさないため “options step” として pipeline で実施し、state/metricsに記録

## 受け入れ条件
- POD/PCAで mode_weight が機能し、係数スケールが変化する
- eigvals無しdecomposerでスパゲッティ分岐が増えていない

## 検証
- mode_weight ON/OFFで recon_error の傾向が変わることを確認（例：高次ノイズの抑制）
