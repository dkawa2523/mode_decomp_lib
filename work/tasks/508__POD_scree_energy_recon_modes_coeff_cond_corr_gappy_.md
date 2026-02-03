# TASK 508: 可視化: POD標準図（scree/energy/recon/modes/coeff/cond_corr + gappy図）

## 目的
POD系の理解と比較のための標準可視化を追加します（レビュー必須）。

## 作業内容
1. docs/addons/32 の必須図を `viz` プロセス（またはeval内）で生成
2. 図は固定名で保存（scree/energy_cum/recon_error_vs_k/modes_gallery/coeff_hist/cond_coeff_corr）
3. Gappy POD 用の比較図も追加
4. 可視化のサンプル数を制限（例：最大8サンプル）し出力肥大を防ぐ
5. 既存の解析基底 decomposer でも出せる図は共通化する（可能な範囲で）

## 受け入れ条件
- PODの代表的な図が必ず生成され、比較・説明が可能
- 文字だけではなく画像で判断できる（レビューが楽）

## 検証
- `task=viz` の実行で figures/ に必須図が生成される
