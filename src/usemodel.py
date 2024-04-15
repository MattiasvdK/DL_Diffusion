import torch
import numpy as np

import PIL
import matplotlib.pyplot as plt

from sampler import DiffusionSampler
from unet import UNet
import torchvision.transforms.v2 as tfv2

def main():

    model = UNet(out_channels=3)
    model.load_state_dict(torch.load("C:\\Users\\matti\\OneDrive\\Documents\\Universiteit\\Ms\\Y1\\Q3\\DL\\Project\\DL_Diffusion\\models\\model.pt"))
    model.eval()
    sampler = DiffusionSampler(model, timesteps=1000)

    imgs = sampler(shape=(1, 3, 32, 32), classes=100)

    fig, ax = plt.subplots(10, 10)


    transforms = tfv2.Compose([
        tfv2.Lambda(lambda x: x.clamp(-1, 1)),
        tfv2.Lambda(lambda x: (x + 1) / 2),
        tfv2.Lambda(lambda x: x * 255),
        tfv2.Lambda(lambda x: x.type(torch.uint8)),
    ])

    print(imgs[0][0])

    for row in range(10):
        for col in range(10):
            img = transforms(imgs[row * 10 + col][0]).transpose(1, 2, 0)
            ax[row, col].imshow(img)
            ax[row, col].axis('off')
    fig.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    main()