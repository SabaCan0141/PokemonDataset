import os
import sys
import zipfile
from typing import Literal, Iterable, Callable
from copy import deepcopy

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .data._get_data import get_data


_base_dir = os.path.dirname(os.path.abspath(__file__))

DATA_CSV_PATH = os.path.join(_base_dir, "data", "data.csv")
IMAGES_PATH = os.path.join(_base_dir, "data", "images")


TYPES = {"normal":0, "fire":1, "water":2, "grass":3, "electric":4, "ice":5, "fighting":6, "poison":7, "ground":8, "flying":9, "psychic":10, "bug":11, "rock":12, "ghost":13, "dragon":14, "dark":15, "steel":16,"fairy":17}
BASE_STATS = {"hp":0, "attack":1, "defense":2, "special_attack":3, "special_defense":4, "speed":5}


DataType = Literal[
    "image:all", "image:default", "image:shiny",
    "pokedex_id", "name",
    "types", "type:normal", "type:fire", "type:water", "type:grass", "type:electric", "type:ice", "type:fighting", "type:poison", "type:ground", "type:flying", "type:psychic", "type:bug", "type:rock", "type:ghost", "type:dragon", "type:dark", "type:steel","type:fairy",
    "base_stats", "base_stat:hp", "base_stat:attack", "base_stat:defense", "base_stat:special_attack", "base_stat:special_defense", "base_stat:speed",
    "weight", "height"
]
Transform = Callable[[Image.Image | torch.Tensor], Image.Image | torch.Tensor]
ImageMode = Literal["1", "L", "P", "RGB", "RGBA", "CMYK", "YCbCr", "HSV", "RGBa", "LA", "I", "F"]


class PokemonDataset(Dataset):
    def __init__(
            self,
            data_type: Iterable[DataType] = ["image:default", "types"],
            transform: dict[DataType, Transform] = None,
            image_mode: ImageMode = "RGB",
            download: bool = False
        ):
        self.data_type = data_type
        self.data_type_ = [dt.split(":")[-1] for dt in data_type]

        if download and not os.path.exists(DATA_CSV_PATH):
            get_data()

        df = pd.read_csv(DATA_CSV_PATH)
        columns = [col for col in df.columns if not col.endswith('_exist')]
        if "image:all" in data_type:
            assert "image:default" not in data_type and "image:shiny" not in data_type
            df_default = df[df["default_image_exist"] == 1][columns]
            df_default["image:all"] = "default"
            df_shiny = df[df["shiny_image_exist"] == 1][columns]
            df_shiny["image:all"] = "shiny"
            self.df = pd.concat([df_default, df_shiny], ignore_index=True)
        elif "image:default" in data_type and "image:shiny" in data_type:
            self.df = df[(df["default_image_exist"] == 1) & (df["shiny_image_exist"] == 1)][columns]
        elif "image:default" in data_type and "image:shiny" not in data_type:
            self.df = df[df["default_image_exist"] == 1][columns]
        elif "image:default" not in data_type and "image:shiny" in data_type:
            self.df = df[df["shiny_image_exist"] == 1][columns]
        else:
            self.df = df[columns]

        self.df = self.df.dropna().reset_index(drop=True)

        self.transform = {
            "image:all": transforms.ToTensor(),
            "image:default": transforms.ToTensor(),
            "image:shiny": transforms.ToTensor(),
        } | (transform or {})

        self.image_mode = image_mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        result = []
        for dt,col in zip(self.data_type, self.data_type_):
            if dt == "image:all":
                tmp = Image.open(os.path.join(
                    IMAGES_PATH,
                    self.df.loc[idx, "image:all"],
                    f"{self.df.loc[idx, 'id']}.png"
                )).convert(self.image_mode)
            elif dt == "image:default":
                tmp = Image.open(os.path.join(
                    IMAGES_PATH,
                    "default",
                    f"{self.df.loc[idx, 'id']}.png"
                )).convert(self.image_mode)
            elif dt == "image:shiny":
                tmp = Image.open(os.path.join(
                    IMAGES_PATH,
                    "shiny",
                    f"{self.df.loc[idx, 'id']}.png"
                )).convert(self.image_mode)
            elif dt == "types":
                tmp = self.df.loc[idx, list(TYPES)].to_numpy(dtype=int)
            elif dt == "base_stats":
                tmp = self.df.loc[idx, list(BASE_STATS)].to_numpy(dtype=int)
            else:
                tmp = self.df.loc[idx, col]
            result.append(self.transform[dt](tmp) if self.transform.get(dt, None) else tmp)

        if len(result) > 1:
            return tuple(result)
        else:
            return result[0]

    def query(self, query: str):
        new = deepcopy(self)
        new.df = new.df.query(query).reset_index(drop=True)

    def sample(self, n: int):
        new = deepcopy(self)
        samples = np.random.choice(range(len(new.df)), size=n, replace=False)
        new.df = new.df.iloc[samples, :].reset_index(drop=True)
        return new

    def random_split(self, rate: float = 0.8, seed: int = None):
        rng = np.random.default_rng(seed)
        N = len(self.df)
        n = int(N * rate)
        samples_1 = rng.choice(range(N), size=n, replace=False)
        samples_2 = np.setdiff1d(np.arange(N), samples_1)

        new_1 = deepcopy(self)
        new_1.df = new_1.df.iloc[samples_1, :].reset_index(drop=True)
        new_2 = deepcopy(self)
        new_2.df = new_2.df.iloc[samples_2, :].reset_index(drop=True)

        return new_1, new_2



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt


    # pokemon = PokemonDataset(["image:all", "types", "base_stats"])
    # loader= DataLoader(pokemon, batch_size=64, shuffle=True)

    # for image,types,base_stats in loader:
    #     print("Types:     ",types[0])
    #     print("Base stats:",base_stats[0])
    #     plt.imshow(image[0].permute(1,2,0))
    #     plt.show()

    pokemon = PokemonDataset(["image:default", "image:shiny"]).sample(100)

    fig,axes = plt.subplots(4, 5, figsize=(10, 8))
    for i in range(10):
        img_d,img_s = pokemon[i]
        axes.flat[i].imshow(img_d.permute(1,2,0))
        axes.flat[i].axis('off')
        axes.flat[10+i].imshow(img_s.permute(1,2,0))
        axes.flat[10+i].axis('off')
    plt.show()
