import torch
import numpy as np

import PIL
import matplotlib.pyplot as plt

from sampler import DiffusionSampler
from unet import UNet

def main():

    model = UNet(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load("C:\\Users\\matti\\OneDrive\\Documents\\Universiteit\\Ms\\Y1\\Q3\\DL\\Project\\DL_Diffusion\\models\\model.pt"))
    model.eval()
    sampler = DiffusionSampler(model, timesteps=1000)

    imgs = sampler(shape=(1, 3, 256, 256))

    print(imgs[-1][0])

    arr = ((imgs[-1][0].transpose(1, 2, 0) * 255).astype(np.uint8))
    print(arr)

    PIL.Image.fromarray(arr).show()

    """
    for idx in range(4):
        plt.imshow(imgs[-1][idx].transpose(1, 2, 0))
        plt.show(block=True)
    """


if __name__ == "__main__":
    main()