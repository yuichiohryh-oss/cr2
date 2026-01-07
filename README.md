# Clash Royale AI Advisor (cr2)

Clash Royaleのプレイ画像を解析し、最適な手（いつ、どこに、何を出すか）を提案するAIアシスタントを作成するプロジェクトです。

## プロジェクト構成

### フェーズ
1. **Model 1 (行動認識)**: 画面から「どのユニットがどこに出されたか」を認識する。[完了]
2. **Model 2 (戦略提案)**: 上手いプレイヤーの動画から学習し、次の一手を提案する。[次回フェーズ]

### ディレクトリ構造
- `src/`
  - `data_collection/`: データ収集用スクリプト (`recorder.py`)
  - `processing/`: データ加工・ラベル付けツール (`labeler.py`, `manual_labeler.py`, `cluster_cards.py`)
  - `training/`: モデル学習用スクリプト (`dataset.py`, `train_model1.py`)
  - `utils/`: ユーティリティ (`window_manager.py`)
- `dataset/`: 収集・加工済みデータ
  - `raw_actions.csv`: 自動検出されたアクションログ
  - `labeled_actions.csv`: 手動ラベル付け済みのアクションログ (学習用)
  - `card_images/`: 切り抜かれたカード画像
- `images/`: 録画されたスクリーンショット

## セットアップ

### 必要要件
- Windows
- Python 3.10+
- NVIDIA GPU (RTX 5060 Ti 推奨) + CUDA 12.8 (PyTorch Nightlyを使用)
- Scrcpy (Android画面ミラーリング)

### インストール
```powershell
# 仮想環境の作成
python -m venv .venv
.\.venv\Scripts\activate

# 依存関係のインストール (PyTorch Nightly for CUDA 12.8)
pip install -r requirements.txt
pip uninstall -y torch torchvision
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

## 使い方 (Model 1 開発フロー)

### 1. データの録画
ScrcpyでClash Royaleの画面を表示した状態で実行します。
```powershell
python src/data_collection/recorder.py
```
- 終了: `ESC` キー

### 2. アクションの抽出
録画データ(`log.csv`, `images/`)からカード選択・配置アクションを抽出します。
```powershell
python src/processing/labeler.py
```
- 新しいデータが `dataset/raw_actions.csv` に追記されます。

### 3. ラベル付け (アノテーション)
抽出されたアクションに対して、正しいユニットIDを割り振ります。
```powershell
python src/processing/manual_labeler.py
```
- GUIが表示されます。画像を見て、対応するユニットの数字キー(0-9)を押してください。
- 未処理のデータから自動的に開始されます。
- 作業終了時に「SAVE」ボタンを押してください。

### 4. モデルの学習
ラベル付きデータ(`dataset/labeled_actions.csv`)を使ってモデルを学習します。
```powershell
python src/training/train_model1.py
```
- 成果物: `model1.pth` (学習済みモデル)

## ユニットID一覧
| ID | Unit Name |
|----|-----------|
| 0  | Skeleton |
| 1  | Evo Skeleton |
| 2  | Ice Spirit |
| 3  | Ice Golem |
| 4  | Hog Rider |
| 5  | Musketeer |
| 6  | Fireball |
| 7  | Cannon |
| 8  | Evo Cannon |
| 9  | The Log |
