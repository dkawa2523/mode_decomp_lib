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
- 係数RMSE（必要なら低次だけ）
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
