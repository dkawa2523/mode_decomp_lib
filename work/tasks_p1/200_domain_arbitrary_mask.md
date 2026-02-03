# P1: Domain arbitrary_mask（任意形状mask）

**ID:** 200  
**Priority:** P1  
**Status:** done  
**Depends on:** 030  
**Unblocks:** 210  

> **⚠️ DO NOT CONTINUE**: このタスクの Acceptance Criteria / Verification をすべて満たすまで、次のタスクに進まないこと。

## 目的
- 任意形状マスクの domain を config から明示的に構築できるようにする
- mask と weights の扱いを統一し、比較可能性を確保する

## 背景 / 根拠（必読）
- docs/00_INVARIANTS.md
- docs/02_DOMAIN_MODEL.md
- docs/03_CONFIG_CONVENTIONS.md
- docs/10_PROCESS_CATALOG.md

## コンパクト実装ポリシー（必須）
- docs/14_COMPACT_CODE_POLICY.md に従う

## スコープ
### In
- `arbitrary_mask` の domain spec を生成（mask_source / mask_path / inline mask）
- config と unit test の更新

### Out
- decomposer の追加・変更
- dataset / process の挙動変更

## 変更計画（Plan）
- `src/mode_decomp_ml/domain/__init__.py` に arbitrary_mask 実装と mask 読み込み補助を追加
- `configs/domain/arbitrary_mask.yaml` を更新
- `tests/test_domain.py` に unit test を追加

## 実装メモ（任意）
- `mask_source=dataset` の場合は domain mask を持たず、サンプル mask に委ねる

## 削除・整理（必須）
- なし

## Acceptance Criteria（完了条件）
- [x] `arbitrary_mask` の domain spec を build できる
- [x] mask_path/inline mask から mask/weights を生成できる
- [x] unit test が追加される

## Verification（検証手順）
- [x] `pytest tests/test_domain.py`

## Review Map（必須：レビュワー向け）
- 変更ファイル: `src/mode_decomp_ml/domain/__init__.py`, `configs/domain/arbitrary_mask.yaml`, `tests/test_domain.py`
- 重要箇所: `build_domain_spec` の arbitrary_mask 分岐、`_load_mask_path`
- 設計判断: mask_source を明示し、mask外は weights=0 に固定
- リスク/注意点: FFT/DCT は mask_zero_fill 以外の mask を許可しないため、任意maskでは別手法を想定
- 検証結果: `pytest tests/test_domain.py`（pass）
