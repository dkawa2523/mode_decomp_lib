# Task 071 (P1): 学習モデル：PyTorch MLP回帰（拡張しやすい最小NN）

## 目的
手法改良が頻繁に起きる前提で、PyTorchベースの最小MLP回帰モデルを導入し、
将来のCNN/Transformer等への拡張の足場を作る。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/070_models_sklearn_baseline.md

## スコープ
### In
- `torch_mlp` regressor を実装（multi-output）
- train loop（epochs, batch, lr, optimizer, scheduler）を config 化
- checkpoint保存（best model）と再現性（seed, cudnn）を担保
- 推論は保存モデルをロードして実行できる

### Out
- 画像を直接出力するCNN（P2）
- HPOの本格導入（P2）

## 実装方針（Codex向け）
### 1) 最小設計
- 入力：condition（D）
- 出力：Y（K2：latent か K：coeff）
- loss：MSE（必要ならL1/Huber）
- normalize：入力と出力の標準化は `coeff_post` / `condition_encoder` で行う（skew防止）

### 2) 実装
- `torch.nn.Sequential` で MLP を構築（hidden dims は config）
- train loop は `processes/train.py` から呼ぶが、ロジックは `models/torch_trainer.py` に寄せる

### 3) 保存
- `model.pth`（state_dict）
- `trainer_state.json`（epochs, best_metric, seed, device, etc）

### 4) 再現性
- `core/seed.py` を使い、torchのdeterministic設定も config に

## ライブラリ候補
- PyTorch
- numpy
- tqdm（任意）

## Acceptance Criteria（完了条件）
- [ ] torch_mlp が registry に登録される
- [ ] 学習が完走し、best checkpoint が保存される
- [ ] predictで保存モデルから推論できる

## Verification（検証手順）
- [ ] tinyデータで数epoch学習し、lossが下がることを確認
- [ ] CPU実行で deterministic が効くことを確認
