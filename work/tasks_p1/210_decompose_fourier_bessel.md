# P1: Decomposer Fourier–Bessel（円板）

**ID:** 210  
**Priority:** P1  
**Status:** done  
**Depends on:** 200  
**Unblocks:** 220  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Review Map
- **変更ファイル一覧**
  - 追加: `src/mode_decomp_ml/decompose/fourier_bessel.py`
  - 変更: `src/mode_decomp_ml/decompose/__init__.py`, `src/mode_decomp_ml/domain/__init__.py`,
    `configs/decompose/fourier_bessel.yaml`, `tests/test_domain.py`
  - 追加: `tests/test_decompose_fourier_bessel.py`
  - 削除: なし
- **重要な関数/クラス**
  - `src/mode_decomp_ml/decompose/fourier_bessel.py`: `FourierBesselDecomposer`（transform/inverse/coeff_meta）
  - `src/mode_decomp_ml/domain/__init__.py`: `validate_decomposer_compatibility`（disk限定チェック）
- **設計判断**
  - disk専用のFourier–Bessel基底を実装し、重み付き最小二乗で係数推定（maskは重み0で無視）
  - 係数順序は `m_then_n` 固定、境界条件/正規化/マスク方針はconfigで明示
  - 正規化は解析式ではなく離散重み付きL2で行い、coeff_metaに基準を記録
- **リスク/注意点**
  - モード数が有効サンプル数を超えるとエラー（maskが細い場合に注意）
  - 係数順序/境界条件/正規化を変えると比較不能になるためcoeff_meta依存
- **検証コマンドと結果**
  - `pytest tests/test_decompose_fourier_bessel.py tests/test_domain.py`
  - 結果: 6 passed
