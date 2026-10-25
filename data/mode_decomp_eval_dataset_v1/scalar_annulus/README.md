# scalar_annulus

固定パターンの生成係数を条件とする旧形式のsynthetic入力です。`cond.npy`、`field.npy`、`mask.npy`と`manifest.json`は原データとして保持しています。

現行の読込には別Dataset rootへ明示変換します。配列の値は変更しません。

```powershell
uv run python scripts/import_legacy_dataset.py data/mode_decomp_eval_dataset_v1/scalar_annulus data/normalized/scalar_annulus
```

条件名x1等は位置名で、物理量名ではありません。元のgeometry情報はmanifest.json、現行入力仕様は[DATA_FORMAT](../../../docs/DATA_FORMAT.md)を参照してください。旧Hydra設定とCLIは廃止しました。
