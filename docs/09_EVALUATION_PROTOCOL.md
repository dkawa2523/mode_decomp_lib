# Evaluation Protocol（評価プロトコル）

## 1. 何を比較するか
- **最終的な場の誤差**（field-space error）
- **係数の誤差**（coeff-space error）
- **エネルギー/寄与率**（decomposition quality）
- （必要なら）**物理解釈性**：低次モードがどういう形状か、対称性がどうか

---

## 2. 指標（推奨）
### Field-space
- RMSE（mask内）
- 相対L2誤差（mask内）
- 最大誤差（max abs error）
- SSIM（画像として比較したい場合）

### Coeff-space
- 係数RMSE（a / z）
- energy_cumsum（累積エネルギー比）
- エネルギー捕捉率（上位Kで何%説明できるか）
- 回転不変化を使う場合：不変量の誤差

---

## 3. split / seed
- split方式（random / group / condition-bucket）を config で固定
- seed を固定して比較する（再現性）

---

## 4. 比較のルール
- decomposerが違う場合でも field-space metric は比較可能
- coeff-space metric は同じ decomposer/同じcoeff_meta の場合のみ比較する
- coeff_post が違う場合、`z` の次元が違い得るので `z` の比較は原則しない（最終再構成で比較）

---

## 5. ベースライン
- “何もしない” baseline（平均場、最近傍条件など）
- 分解 + 線形回帰 baseline
- 分解 + PCA + 線形回帰 baseline

---

## 6. Uncertainty（任意）
- uncertainty 出力（coeff_std / field_std）は任意で、比較/leaderboardには含めない
- 比較は mean 予測の field-space metric を使い、std/区間は別レポートとする
- GPR など coeff_std を返せるモデルでのみ有効（MC近似は可視化目的）

---

## 7. 可視化の読み方（簡易）
以下は **plots/** に出力される診断図のうち、非DS向けに見るポイントをまとめたものです。
対象の分解/係数表現を使わない場合は出力されません。

- `coeff_hist.png`: 係数の大きさ分布。極端な裾や外れ値は前処理/尺度の歪みを示唆。
- `coeff_topk_energy.png`: 上位モードのエネルギー比。少数モードに偏り過ぎると表現力不足の可能性。
- `fft_magnitude_spectrum.png`: FFTの周波数分布。低周波優位か高周波優位かを確認。
- `wavelet_band_energy.png`: Waveletのlevel別エネルギー。粗い構造/細かい構造のどちらが強いか。
- `sh_l_energy.png`: Spherical Harmonicsのl別エネルギー。角周波数成分の偏りを確認。
- `slepian_concentration.png`: Slepianの集中度。ROI内にどれだけ局在しているか。
