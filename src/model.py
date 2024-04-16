import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
import torch.optim as optim
from torch.optim import lr_scheduler
import math

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

    def forward(self, x, timestep):
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

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3 
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])
        
        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)