# 用途ごとのモデルの説明

このページは「用途」から逆引きで、モード分解・前処理・学習・評価を選ぶための整理です。

## 用途一覧（テンプレ）

| 用途 | 目的 | 主に見る指標 |
|---|---|---|
| 係数学習（圧縮重視） | `cond -> coeff` の精度・次元 | `val_r2`, `val_field_r2`, `k_req_r2_0.95` |
| field再構成重視 | `field_hat` の再現性 | `field_rmse`, `field_r2`, per-pixel R^2 |
| 可変マスク対応 | 欠損/観測領域の変動 | mask統計 + gappy系手法の安定性 |
| ベクトル場 | 物理量（div/curl等）含む | `div_rmse`, `curl_rmse`（有効時） |

## まとめ表（モード分解/前処理/学習/評価）

代表構成（まず動く・比較に使える）:

| 用途 | 推奨モード分解 | 推奨 model | 推奨評価 |
|---|---|---|---|
| baseline（rectangle） | `dct2` | `ridge` | `field_r2_topk_k64`, `val_field_r2` |
| disk（解析基底） | `zernike` / `pseudo_zernike` | `ridge` | `mode_r2_vs_k`, per-pixel R^2 |
| 可変mask | `gappy_graph_fourier` / `pod_em` | `ridge` | mask統計 + field指標 |
| ベクトル場（u,v） | `pod_joint_em`（または `pod_joint`） | `ridge` / `mtgp` | `val_field_r2` +（必要なら）div/curl |

| 区分 | 名前 | 入力 | 出力 | 使い所 |
|---|---|---|---|---|
| モード分解 | `dct2` / `zernike` / `pod_svd` | field | raw_coeff | domain に合う “低次で表現しやすい” 基底を選ぶ |
| codec | `auto_codec_v1` | raw_coeff | coeff(a) | raw_coeff の多様性（複素/辞書/構造化）を吸収してベクトル化 |
| coeff_post | `none` / `pca` | coeff(a) | coeff(z) | 次元削減やノイズ低減（学習の安定化） |
| model | `ridge` / `gpr` / `mtgp` | cond | coeff(a) または coeff(z) | 回帰（baseline→不確かさ→多出力） |
| metrics | `field_r2_topk_k64`, `val_field_r2` | field/coeff | scalar | 手法間比較は field 指標を主にする |
