import matplotlib.pyplot as plt

from PokemonDataset import PokemonDataset



pokemon = PokemonDataset(["image:default", "image:shiny"]).sample(100)

fig,axes = plt.subplots(4, 5, figsize=(10, 8))
for i in range(10):
    img_d,img_s = pokemon[i]
    axes.flat[i].imshow(img_d.permute(1,2,0))
    axes.flat[i].axis('off')
    axes.flat[10+i].imshow(img_s.permute(1,2,0))
    axes.flat[10+i].axis('off')
plt.show()
