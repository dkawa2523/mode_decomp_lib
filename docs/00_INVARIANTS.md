# Invariants（不変条件）

この文書は、開発中に「破ってはいけない契約」です。
破る場合は必ず `work/rfcs/` を起票し、影響範囲・移行手順・比較不能になる点を明記してください。

---

## 1. 設定が真実（Single Source of Truth）
- すべての可変パラメータは **設定（Hydra config）** に置く。
- コード内に「暗黙のデフォルト」を増やさない。増やす場合は config に追加し、保存された config が実験条件の真実になるようにする。

## 2. Process単位で実行できる
- `Process` は単独で実行でき、入出力（I/O）が明確であること。
- 例：`preprocess`, `decompose.fit`, `decompose.transform`, `coeff_post.fit`, `train`, `eval`, `predict`, `reconstruct`, `viz`, `leaderboard`, `doctor`
- Process間の受け渡しは **artifact契約**（docs/04）に従う。

## 3. 比較可能性（Comparability）
- 2つの結果を比較する時、次が一致している必要がある：
  - dataset版（ハッシュ/バージョン）、split、seed
  - domain定義（座標系、格子、マスク、境界条件）
  - decomposer種別と次数（Zernikeの(n,m)上限など）
  - coeff_post種別（PCA次元など）
  - metrics定義（docs/09）
- 一致しない場合は同じleaderboardに混ぜない（または明確にラベル分け）。

## 4. train/serve skew 禁止
- 学習時に行った前処理・分解・係数後処理は、推論でも同一に適用する。
- `fit` が必要な処理（PCA等）は **学習データでfitした状態** を保存し、推論は `transform` のみ行う。

## 5. 再現性（Reproducibility）
- 同じ入力 + 同じ config + 同じ seed で、同じ出力（係数、学習結果、評価）になること。
- ランダム性がある処理（split、初期値、最適化）は seed に支配させる。

## 6. “特徴量化”の定義は固定
- 本プロジェクトでの特徴量化は **「一次モード分解後の係数 a を PCA/ICA 等で再表現する Stage」** を指す。
- 分解前の派生量（勾配/ラプラシアン等）を特徴量として増やすのは、原則 **デフォルトでは行わない**。
  - 必要なら `aux_features/` として別枠にし、比較可能性が崩れないようラベル化する。

## 7. 境界・座標系は明示する
- 「円/矩形/任意マスク/メッシュ」等の domain は config で明示。
- ZernikeやFourier–Bessel等、**境界条件で直交性が変わる** 手法は boundary_condition を必ず記録する。

## 8. 係数の可逆性（可能な範囲で）
- `decompose.transform` で得た係数 `a` は、`decompose.inverse_transform` で元場を再構成できる（少なくとも近似できる）こと。
- `coeff_post` も同様に `inverse_transform` を提供し、`z -> a` が可能であること（PCA等）。

## 9. 失敗は早く・明確に
- 前提が満たされない時（shape不一致、mask不正、model未保存等）は、黙って進まずエラーで止める。
- “silent failure”（欠損を勝手に0埋め等）は禁止。必ずログ/設定で選べるようにする。

---

## 変更の進め方
- 実装は `work/tasks/*.md` 単位で行う
- 破壊的変更は `work/rfcs/`（RFC）を起票してから着手する
