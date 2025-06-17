import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from PokemonDataset import PokemonDataset

# モデルの定義
class PokemonClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PokemonClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def main():
    # --- 1. ハイパーパラメータとデバイス設定 ---
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 20
    INPUT_SIZE = 6  # hp, attack, defense, special_attack, special_defense, speed
    NUM_CLASSES = 18 # 18種類のタイプ
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {DEVICE} device.")

    # --- 2. データセットとデータローダーの準備 ---
    print("Loading dataset...")
    # 種族値とタイプをデータとして取得
    dataset = PokemonDataset(data_type=['base_stats', 'types'], download=True)

    # データを訓練用と検証用に分割
    train_dataset, val_dataset = dataset.random_split(rate=0.8, seed=42)
    print(f"Training data: {len(train_dataset)} samples")
    print(f"Validation data: {len(val_dataset)} samples")

    # データローダーの作成
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. モデル、損失関数、最適化手法の定義 ---
    model = PokemonClassifier(input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(DEVICE)
    # 多ラベル分類なのでBCEWithLogitsLossを使用
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. 学習ループ ---
    print("Start training...")
    for epoch in range(EPOCHS):
        # 訓練
        model.train()
        train_loss = 0.0
        for stats, types in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            stats = stats.to(DEVICE).float()
            types = types.to(DEVICE).float()

            # 順伝播
            outputs = model(stats)
            loss = criterion(outputs, types)

            # 逆伝播と最適化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 検証
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for stats, types in val_loader:
                stats = stats.to(DEVICE).float()
                types = types.to(DEVICE).float()
                outputs = model(stats)
                loss = criterion(outputs, types)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    print("Training finished.")

if __name__ == "__main__":
    main()
