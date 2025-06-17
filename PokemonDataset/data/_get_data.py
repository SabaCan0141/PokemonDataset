import os
import requests
import csv

import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm

_base_dir = os.path.dirname(os.path.abspath(__file__))


TYPES = {"normal":0, "fire":1, "water":2, "grass":3, "electric":4, "ice":5, "fighting":6, "poison":7, "ground":8, "flying":9, "psychic":10, "bug":11, "rock":12, "ghost":13, "dragon":14, "dark":15, "steel":16,"fairy":17}
OFFICIAL_ARTWORK_SIZE = (475, 475)

def download_image(url, save_path, TARGET_SIZE):
    if os.path.exists(save_path): return
    # 画像をダウンロード
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")

    img = Image.open(BytesIO(response.content)).convert("RGBA")  # アルファ付きで読み込み
    img_w, img_h = img.size

    background = Image.new("RGBA", TARGET_SIZE, (0, 0, 0, 0))

    # オフセットを計算して中央に貼り付け
    offset_x = (TARGET_SIZE[0] - img_w) // 2
    offset_y = (TARGET_SIZE[1] - img_h) // 2
    background.paste(img, (offset_x, offset_y), mask=img)

    # NumPyで\alpha=0のピクセルのRGBを(0,0,0)に修正
    arr = np.array(background)
    alpha_mask = (arr[:, :, 3] == 0)
    arr[alpha_mask, 0:3] = 0

    # 保存
    cleaned_img = Image.fromarray(arr, mode="RGBA")
    cleaned_img.save(save_path)


def get_data():
    os.makedirs(os.path.join(_base_dir, "images", "default"), exist_ok=True)
    os.makedirs(os.path.join(_base_dir, "images", "shiny"), exist_ok=True)

    response = requests.get("https://pokeapi.co/api/v2/pokemon?limit=10000")

    with open(os.path.join(_base_dir, "data.csv"), "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "pokedex_id", "name", "normal", "fire", "water", "grass", "electric", "ice", "fighting", "poison", "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy", "hp", "attack", "defense", "special_attack", "special_defense", "speed", "height", "weight", "default_image_exist", "shiny_image_exist"])

        for entry in tqdm(response.json()["results"]):
            pokemon_info = requests.get(entry["url"]).json()

            id_ = pokemon_info["id"]
            pokedex_id = int(pokemon_info["species"]["url"].split("/")[-2])
            name = pokemon_info["name"]
            types = [0] * len(TYPES)
            for type_ in pokemon_info["types"]:
                types[TYPES[type_["type"]["name"]]] = 1
            base_stats = {s["stat"]["name"]: s["base_stat"] for s in pokemon_info["stats"]}
            height = int(pokemon_info["height"])
            weight = int(pokemon_info["weight"])

            sprite = pokemon_info["sprites"]["other"]["official-artwork"]
            sprite_default = sprite.get("front_default", None)
            if sprite_default:
                download_image(
                    sprite_default,
                    os.path.join(_base_dir, "images", "default", f"{id_}.png"),
                    OFFICIAL_ARTWORK_SIZE
                )
            sprite_shiny = sprite.get("front_shiny", None)
            if sprite_shiny:
                download_image(
                    sprite_shiny,
                    os.path.join(_base_dir, "images", "shiny", f"{id_}.png"),
                    OFFICIAL_ARTWORK_SIZE
                )

            writer.writerow([
                id_, pokedex_id, name, *types,
                base_stats["hp"],
                base_stats["attack"],
                base_stats["defense"],
                base_stats["special-attack"],
                base_stats["special-defense"],
                base_stats["speed"],
                height, weight,
                1 if sprite_default else 0,
                1 if sprite_shiny else 0,
            ])


if __name__ == "__main__":
    get_data()
