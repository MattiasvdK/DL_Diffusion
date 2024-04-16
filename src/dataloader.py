import torch
import torch.utils.data as tud
import numpy as np
import torchvision as tv
import torchvision.transforms.v2 as tfv2
import os
import PIL

def get_data_loaders(train_dir, val_dir, test_dir, batch_size, timesteps=1000, shuffle=True):
    train_dataset = CocoDataset(train_dir, timesteps)
    val_dataset = CocoDataset(val_dir, timesteps)
    test_dataset = CocoDataset(test_dir, timesteps)
    
    # Determine the number of samples to use for training (60% of the total)
    train_size = int(0.1 * len(train_dataset))
    val_size = int(0.1 * len(val_dataset))

    # Split the train dataset into training and validation subsets
    train_subset, train_remaining_subset = tud.random_split(train_dataset, [train_size, len(train_dataset) - train_size])
    val_subset, val_remaining_subset = tud.random_split(val_dataset, [val_size, len(val_dataset) - val_size])

    train_loader = tud.DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle)
    val_loader = tud.DataLoader(val_subset, batch_size=1, shuffle=shuffle)
    test_loader = tud.DataLoader(test_dataset, batch_size=1, shuffle=shuffle)

    return train_loader, val_loader, test_loader

def get_data_loader(directory, batch_size, timesteps=500, shuffle=True):
    dataset = CocoDataset(directory, timesteps)
    loader = tud.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

class CocoDataset(tud.Dataset):
    def __init__(self, directory, timesteps, img_size=64, labels=None):
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

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

def get_cifar_loaders(batch_size, root='./data', shuffle=True):
    train_dataset = tv.datasets.CIFAR100(root=root, train=True, download=True, transform=tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x * 2 - 1)
    ]),
    target_transform=tv.transforms.Lambda(lambda x: one_hot_encode(x, 100))
    )
    val_dataset = tv.datasets.CIFAR100(root=root, train=False, download=True, transform=tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x * 2 - 1)
    ]),
    target_transform=tv.transforms.Lambda(lambda x: one_hot_encode(x, 100))
    )

    train_loader = tud.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = tud.DataLoader(val_dataset, batch_size=1, shuffle=shuffle)

    return train_loader, val_loader
