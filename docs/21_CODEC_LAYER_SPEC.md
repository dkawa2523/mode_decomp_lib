# 21. CoeffCodec Layer Spec（係数表現統一層）

## 背景
decomposer が増えると係数の表現が多様化します:
- FFT: 複素係数
- Wavelet: 階層（list/tuple）係数
- SH/Slepian: (ℓ,m) 構造・集中度などメタを伴う
- Graph/LB: 固有値順序や符号の不定性

この多様性をパイプラインに直接持ち込むと分岐が増え、スパゲッティになります。

---

## 解決: raw_coeff と vector_coeff を分離
- Decomposer は **raw_coeff** を返す（型は自由: complex ndarray / dict / list）
- 学習器や後処理は **vector_coeff (float32, shape=[L])** だけ扱う
- raw_coeff <-> vector_coeff の変換を **CoeffCodec** に閉じ込める

---

## インターフェース（仕様）
### RawCoeffMeta
- raw_coeff の構造を完全に復元できるメタ情報
- 例: FFTの周波数並び、Zernikeの(n,m)順序、Waveletの(level,band)とshape、SHの(ℓ,m)

### CoeffCodec
- `encode(raw_coeff, raw_meta) -> vector_coeff`
- `decode(vector_coeff, raw_meta) -> raw_coeff`
- `is_lossless: bool`（完全可逆か）
- `dtype_policy`: float32固定（保存・学習の標準化）

---

## 例
### FFT 複素係数統一
- mode:
  - `real_imag`（lossless）
  - `mag_phase`（phaseが不安定になりやすいので注意）
  - `logmag_phase`（ダイナミックレンジ改善）
- encode/decode は codec で完結（decomposerは複素を返すだけ）

### Wavelet
- wavedec2 の係数（list構造）を flatten
- meta に（各レベルのshape、band順）を保存して復元

### SH / Slepian
- (ℓ,m) の固定順で flatten
- Slepianは集中度（eigenvalues）も meta に保存（可視化・解釈用）

---

## 利点
- 分解手法が増えてもパイプラインは一定（vector_coeffだけを扱う）
- FFT複素の表現統一など “表現の工夫” が codec に集約される
- 係数保存が統一され、比較・レビューが容易

---

## テスト要件（必須）
- lossless codec は `decode(encode(x)) == x`（許容誤差内）を保証
- lossy codec は情報損失を明示し、評価指標（再構成誤差）に影響を記録
- meta を保存/復元できること（artifact: `outputs/states/coeff_meta.json` 等）
