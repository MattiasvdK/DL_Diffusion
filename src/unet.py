import torch
import torch.nn as nn
import torchvision.transforms.v2 as tfv2

"""
Still need to add the following:
-Conditional
-Time Stamp

Possible Additions:
-BatchNorm
-Self-Attention
-Different Activation Functions
"""

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
        )

    def forward(self, x):
        return self.conv(x)
    

class Downscale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downscale, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x
    
    
class Upscale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upscale, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2, img_size):
        x1 = tfv2.Resize(img_size)(x1)
        x2 = tfv2.Resize(img_size*2)(x2)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = tfv2.Resize(img_size*2)(x)
        return x

    

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
    
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UNet(nn.Module):
    def __init__(self, out_channels, img_size=256):
        super(UNet, self).__init__()
        self.img_size = img_size
        # Encoder
        self.layer1 = DoubleConv(3, 64)
        self.down1 = Downscale(64, 128)
        self.down2 = Downscale(128, 256)
        self.down3 = Downscale(256, 512)
        #self.down4 = Downscale(512, 1024)

        # Bottleneck
        #self.bottleneck = DoubleConv(512, 512)
        
        # Decoder
        #self.up1 = Upscale(1024, 512)
        self.up2 = Upscale(512, 256)
        self.up3 = Upscale(256, 128)
        self.up4 = Upscale(128, 64)

        # Output
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, timestep):
        # Encoder
        x1 = self.layer1(x)
        x2 = self.down1(x1)
        self.img_size //= 2
        x3 = self.down2(x2)
        self.img_size //= 2
        x = self.down3(x3)
        self.img_size //= 2
        #x5 = self.down4(x4)

        # Bottleneck
        #x = self.bottleneck(x4)

        # Decoder
        #x = self.up1(x, x4)
        x = self.up2(x, x3, self.img_size)
        self.img_size *= 2
        x = self.up3(x, x2, self.img_size)
        self.img_size *= 2
        x = self.up4(x, x1, self.img_size)

        # Output
        x = self.out(x)
        return x