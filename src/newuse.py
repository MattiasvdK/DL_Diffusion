import torch
import numpy as np

import PIL
import matplotlib.pyplot as plt

from sampler import DiffusionSampler
from unet import UNet
import torchvision.transforms.v2 as tfv2
from diffusion import Diffusion

device = "cuda"
model = UNet(out_channels=3).to(device)
ckpt = torch.load("../models/model_last.pt", map_location=torch.device('cuda'))
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=32, device=device)
# one hot encoded labels
labels = torch.tensor(np.eye(100)[np.random.randint(0, 100, 8)]).to(device)
print(labels)
x = diffusion.sample(model, 8, labels=labels)
print(x.shape)
plt.figure(figsize=(32, 32))
plt.imshow(torch.cat([
    torch.cat([i for i in x.cpu()], dim=-1),
], dim=-2).permute(1, 2, 0).cpu())
plt.show()