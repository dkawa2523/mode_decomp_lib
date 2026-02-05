# 利用方法ガイドライン

## 用途別の実行例
- モード分解のみ
```bash
python -m mode_decomp_ml.run --config run_decomposition.yaml
```

- モード分解 + 学習（pipeline）
```bash
python -m mode_decomp_ml.run --config run_pipeline.yaml
```

- 推論（単一）
```bash
python -m mode_decomp_ml.run --config run_inference.yaml
```

## 精度向上のための Tips
- 分解モード数や基底パラメータ（n_max / l_max）を調整
- 手法依存の前処理ルールを尊重（POD で PCA を無効化など）
- マスク領域の取り扱いを明確化（mask_policy）
- ベクトル場は streamplot / quiver で方向性を確認
