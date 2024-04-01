import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
import torch.optim as optim
from torch.optim import lr_scheduler

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.encoder11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.encoder12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.encoder32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pooling3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.encoder42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.encoder52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.decoder12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.decoder22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.decoder32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        xe11 = relu(self.encoder11(x))
        xe12 = relu(self.encoder12(xe11))
        xp1 = relu(self.pooling1(xe12))

        xe21 = relu(self.encoder21(xp1))
        xe22 = relu(self.encoder22(xe11))
        xp2 = relu(self.pooling2(xe22))

        xe31 = relu(self.encoder31(xp2))
        xe32 = relu(self.encoder32(xe31))
        xp3 = relu(self.pooling3(xe32))

        xe41 = relu(self.encoder41(xp3))
        xe42 = relu(self.encoder42(xe41))
        xp4 = relu(self.pooling4(xe42))

        xe51 = relu(self.encoder51(xp4))
        xe52 = relu(self.encoder52(xe51))

        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.decoder11(xu11))
        xd12 = relu(self.decoder12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.decoder21(xu22))
        xd22 = relu(self.decoder22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.decoder31(xu33))
        xd32 = relu(self.decoder32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.decoder41(xu44))
        xd42 = relu(self.decoder42(xd41))

        out = self.outconv(xd42)
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_class = 6
model = UNet(num_class).to(device)

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)


        







