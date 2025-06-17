# PokemonDataset for PyTorch

`PokemonDataset` は、[PokeAPI](https://pokeapi.co/) から取得したポケモンのデータを、PyTorchを使った機械学習プロジェクトで手軽に利用できるようにするためのライブラリです。

画像のダウンロードからデータの前処理、データセットの作成までを自動化し、数行のコードでポケモンの画像、タイプ、種族値などのデータセットを準備できます。

## 主な機能
- **データ自動取得**: 必要なデータを自動でダウンロードし、整形します。
- **柔軟なデータ選択**: 画像（通常色・色違い）、タイプ、種族値など、必要なデータを自由に組み合わせてデータセットを構築できます。
- **PyTorchとの連携**: PyTorchの`Dataset`クラスを継承しており、`DataLoader`とシームレスに連携します。
- **便利なユーティリティ**: データセットのフィルタリング (`query`)、ランダムサンプリング (`sample`)、訓練/検証データへの分割 (`random_split`) といった便利なメソッドを提供します。

## 動作要件
- Python 3.8+
- PyTorch
- torchvision
- pandas
- NumPy
- Pillow
- tqdm
- requests

## インストール
必要なライブラリをインストールします。
```
pip install torch torchvision pandas numpy pillow tqdm requests
```

その後、このリポジトリをクローンしてプロジェクトにインポートします。
```
git clone https://github.com/SabaCan0141/PokemonDataset.git
```

## クイックスタート
以下のコードは、ポケモンの通常色画像と色違い画像をペアで取得し、最初の10ペアを表示する簡単なサンプルです。
```Python
import matplotlib.pyplot as plt
from PokemonDataset import PokemonDataset

# データセットを準備 (初回実行時はデータが自動でダウンロードされます)
# data_typeに'image:default'と'image:shiny'を指定
pokemon_dataset = PokemonDataset(
    data_type=["image:default", "image:shiny"], 
    download=True
)

# 10個のサンプルを取得
sampled_pokemon = pokemon_dataset.sample(10)

# 画像を表示
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Default vs Shiny", fontsize=16)

for i in range(5):
    # 1段目: 通常色
    img_default, _ = sampled_pokemon[i]
    axes[0, i].imshow(img_default.permute(1, 2, 0))
    axes[0, i].set_title(f"#{sampled_pokemon.df.loc[i, 'pokedex_id']}")
    axes[0, i].axis('off')

    # 2段目: 色違い
    _, img_shiny = sampled_pokemon[i]
    axes[1, i].imshow(img_shiny.permute(1, 2, 0))
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()
```

## APIリファレンス
### `PokemonDataset` クラス
```Python
PokemonDataset(
    data_type: Iterable[DataType] = ["image:default", "types"],
    transform: dict[DataType, Transform] = None,
    image_mode: ImageMode = "RGB",
    download: bool = False
)
```

#### 引数
- `data_type` (list): 取得したいデータの種類を指定する文字列のリスト。指定可能な値は後述します。
- `transform` (dict, optional): データ型ごとに適用する変換処理（例: `transforms.ToTensor()`）を辞書形式で指定します。画像の変換に便利です。
- `image_mode` (str, optional): `Pillow`で画像を開く際のカラーモードを指定します。デフォルトは`"RGB"`です。
- `download` (bool, optional): ローカルにデータが存在しない場合に、データを自動でダウンロードするかどうかを指定します。デフォルトは`False`です。

#### `data_type` に指定可能な値
| カテゴリ | 値 | 説明 |
|:--|:--|:--|
| 画像 | `image:default` | 通常色の公式アートワーク画像 |
|  | `image:shiny` | 色違いの公式アートワーク画像 |
|  | `image:all` | 通常色と色違いの両方（`image:default`, `image:shiny`とは同時に使えない） |
| 基本情報 | `pokedex_id` | 全国図鑑ID |
|  | `name` | ポケモンの英語名 |
|  | `height` | 高さ |
|  | `weight` | 重さ |
| タイプ | `types` | 全18タイプをone-hotエンコーディングしたNumPy配列 |
|  | `type:normal`, `type:fire`, ... | 指定したタイプを持つかどうかのフラグ (0 or 1) |
| 種族値 | `base_stats` | 全6種の種族値をまとめたNumPy配列 |
|  | `base_stat:hp`, `base_stat:attack`, ... | 指定した能力の種族値 |

#### メソッド
##### `.query(query: str)`
特定の条件でデータセットをフィルタリングします。pandasの`DataFrame.query`と同じ形式のクエリ文字列を渡します。
```Python
# 炎タイプまたはドラゴンタイプのポケモンだけを抽出
fire_or_dragon_pokemon = pokemon_dataset.query("fire == 1 or dragon == 1")
```

##### `.sample(n: int)`
データセットから指定した数だけランダムに非復元抽出します。
```Python
# 5匹のポケモンをランダムに取得
random_5 = pokemon_dataset.sample(5)
```

##### `.random_split(rate: float, seed: int = None)`
データセットを2つに分割します。訓練データと検証データの作成に便利です。
```Python
# 80%を訓練データ、20%を検証データに分割
train_set, val_set = pokemon_dataset.random_split(0.8)

print(f"訓練データ数: {len(train_set)}")
print(f"検証データ数: {len(val_set)}")
```

## 実践例：種族値からタイプを予測する
このライブラリを使って、ポケモンの種族値（HP, 攻撃, 防御など）からそのタイプを予測する簡単な多ラベル分類モデルを学習させるコードです。
```Python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PokemonDataset import PokemonDataset

# 1. データセットの準備
dataset = PokemonDataset(data_type=['base_stats', 'types'], download=True)
train_dataset, val_dataset = dataset.random_split(rate=0.8)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32)

# 2. モデルの定義
class PokemonClassifier(nn.Module):
    def __init__(self, input_size=6, num_classes=18):
        super(PokemonClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = PokemonClassifier()

# 3. 学習ループ
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  # 10エポック学習
    model.train()
    for stats, types in train_loader:
        stats = stats.float()
        types = types.float()

        outputs = model(stats)
        loss = criterion(outputs, types)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Training finished.")

```

## ライセンス
このプロジェクトは **MIT License** のもとで公開されています。詳細は`LICENSE`ファイルをご覧ください。
