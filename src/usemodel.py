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
        tfv2.Lambda(lambda x: x * 0.5 + 0.5),
        tfv2.Lambda(lambda x: x * 255),
        tfv2.Lambda(lambda x: x.astype(np.uint8)),
        tfv2.ToPILImage()
    ])

    print(imgs[0][0])

    for row in range(10):
        for col in range(10):
            ax[row, col].imshow(imgs[row * 10 + col][0].transpose(1, 2, 0) * 0.5 + 0.5)
            ax[row, col].axis('off')
    fig.tight_layout()
    fig.suptitle("Diffusion samples", fontsize=16)
    plt.show(block=True)


if __name__ == "__main__":
    main()