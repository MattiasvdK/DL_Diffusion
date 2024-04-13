import torch
import torch.utils.data as tud
import numpy as np
import torchvision as tv
import torchvision.transforms.v2 as tfv2
import os
from noise import CosineScheduler
import PIL

def get_data_loaders(train_dir, val_dir, test_dir, batch_size, timesteps=1000, shuffle=True):
    train_dataset = CocoDataset(train_dir, timesteps)
    val_dataset = CocoDataset(val_dir, timesteps)
    test_dataset = CocoDataset(test_dir, timesteps)

    train_loader = tud.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = tud.DataLoader(val_dataset, batch_size=1, shuffle=shuffle)
    test_loader = tud.DataLoader(test_dataset, batch_size=1, shuffle=shuffle)

    return train_loader, val_loader, test_loader

def get_data_loader(directory, batch_size, timesteps=1000, shuffle=True):
    dataset = CocoDataset(directory, timesteps)
    loader = tud.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

class CocoDataset(tud.Dataset):
    def __init__(self, directory, timesteps, img_size=256, labels=None):
        self.labels = labels
        self.dir = directory
        self.imgs = os.listdir(directory)
        self.timesteps = timesteps

        self.transform = tv.transforms.Compose([
            tfv2.Resize(img_size),
            tfv2.RandomCrop(img_size),
            tfv2.ToTensor(),
            tfv2.Lambda(lambda x: x * 2 - 1)
        ])
 
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = PIL.Image.open(self.dir + '/' + self.imgs[index])
        img = img.convert('RGB')
        # Transform the image and apply noise
        time = np.random.randint(0, self.timesteps)
        img = self.transform(img)
        return img, time

    def __iter__(self):
        return self
