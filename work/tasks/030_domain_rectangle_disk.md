# Domain: rectangle/disk（座標生成・正規化・互換チェック）

**ID:** 030  
**Priority:** P0  
**Status:** done  
**Depends on:** 020  
**Unblocks:** 040, 050  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
boundary（領域）ごとに必要な座標系を提供する DomainAdapter を実装する。

P0 domain:
- rectangle: (x,y) 正規化座標
- disk: (r,theta) を生成し、mask を円板にする（r∈[0,1]）

また domain×decomposer の互換チェックを導入する:
- Zernike は disk を要求（rectangleなら明示エラー）
- FFT/DCT は rectangle を推奨（diskなら “mask外0埋め” か “エラー” を選べる）

## Acceptance Criteria
- [x] rectangle/disk の両方で coords が生成される（xy/rt）
- [x] decomposer互換チェックがあり、誤適用が起きない
- [x] disk は r の最大が1になる

## Verification
- [x] 合成データを disk domain に投げ、r,theta のshapeと範囲が確認できる

## Review Map（必須）
- 変更ファイル一覧（追加/変更/削除）: `src/mode_decomp_ml/domain/__init__.py`（追加）, `tests/test_domain.py`（追加）, `configs/domain/rectangle.yaml`, `configs/domain/disk.yaml`, `configs/decompose/fft2.yaml`, `configs/decompose/dct2.yaml`
- 重要な関数/クラス: `DomainSpec`, `build_domain_spec`, `validate_decomposer_compatibility`
- 設計判断: domainの座標生成はx/yを共通化し、diskはr/thetaと境界maskを生成してr<=1を保証。FFT/DCTのdisk適用は明示policyで遮断し、configを真実にする。
- リスク/注意点: diskの重みはrを含める近似であり、厳密な積分スキームは将来調整が必要。`x_range`/`y_range`はconfig必須。
- 検証コマンドと結果: `pytest tests/test_domain.py`（pass）
