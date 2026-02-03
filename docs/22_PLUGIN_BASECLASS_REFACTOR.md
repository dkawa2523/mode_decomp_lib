# 22. Plugin Baseclass Refactor（共通化でコード増殖を止める）

## 背景
追加手法が増えると「似た処理のコピペ」が増えます:
- 座標生成、重み付き内積、mask処理、係数meta作成
- fit/transform/inverse の I/O 契約
- state 保存/ロード
- vector場のチャンネル処理

これを放置すると、スパゲッティ化・レビュー困難・バグ増殖につながります。

---

## 方針: 3系統の共通ベースで統一

### A) GridDecomposerBase（規則格子向け）
対象: rectangle/disk を格子上で扱う系
- FFT2 / DCT2 / Wavelet2D /（研究）PSWF2D tensor
共通化:
- field shape検証、mask適用、(x,y)座標提供
- roundtrip テスト共通テンプレ
- raw_coeff_meta の標準フォーマット

### B) ZernikeFamilyBase（Zernike系）
対象: Zernike / Annular Zernike /（将来）写像Zernike
共通化:
- (n,m)列挙、正規化、角成分、積分重み
差分:
- radial polynomial の実装だけを差し替え

### C) EigenBasisDecomposerBase（固有基底系）
対象: Graph Laplacian / Laplace–Beltrami / Slepian /（将来）厳密PSWF
共通化:
- `fit()` で basis を構築（eigs）
- basis cache / state保存（eigenvalues/eigenvectors）
- sign ambiguity 対策（基準点で符号固定など）
- transform/inverse は basis の射影・線形結合

---

## vector場（C=2など）の共通化
- 原則: scalar decomposer をチャンネルごとに適用する `ChannelwiseAdapter` を提供
- 例外: Helmholtz など “ベクトル固有” の分解のみ専用Decomposerにする
- これにより、手法追加が scalar 実装に集中し、コード量が増えにくい

---

## 受け入れ基準
- 既存 decomposer の重複コードが減り、ベースクラスに集約されている
- 新規手法は **ベースクラス + 小さな差分**で追加できる
- plugin の I/O 契約（raw_coeff / meta / state）が統一されている
