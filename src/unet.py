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
    def __init__(self, in_channels, out_channels, groups=8):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
    

class Downscale(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super(Downscale, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, emb):
        x = self.pool(x)
        x = self.conv(x)
        emb = self.emb_layer(emb)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
    
class Upscale(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super(Upscale, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )
    
    def forward(self, x1, x2, emb):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(emb)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb



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
    def __init__(self, out_channels, embed_dim=256):
        super(UNet, self).__init__()
        self.embed_dim = embed_dim
        # Encoder
        self.layer1 = DoubleConv(3, 64)
        self.down1 = Downscale(64, 128, embed_dim)
        self.down2 = Downscale(128, 256, embed_dim)
        self.down3 = Downscale(256, 512, embed_dim)
        #self.down4 = Downscale(512, 1024)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 512)
        
        # Decoder
        #self.up1 = Upscale(1024, 512)
        self.up2 = Upscale(512, 256, embed_dim)
        self.up3 = Upscale(256, 128, embed_dim)
        self.up4 = Upscale(128, 64, embed_dim)

        # Output
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, label, timestep):

        enc_t = timestep.unsqueeze(-1).type(torch.float)
        enc_t = self.pos_encoding(enc_t, self.embed_dim)
        enc_l = self.label_encoding(label,self.embed_dim).to('cuda:0')
        encoding = enc_l + enc_t
    
        # Encoder
        x1 = self.layer1(x)
        x2 = self.down1(x1, encoding)
        x3 = self.down2(x2, encoding)
        x = self.down3(x3, encoding)
        #x5 = self.down4(x4)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        #x = self.up1(x, x4)
        x = self.up2(x, x3, encoding)
        x = self.up3(x, x2, encoding)
        x = self.up4(x, x1, encoding)

        # Output
        x = self.out(x)
        return x
    
    def pos_encoding(self, timestep, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device="cuda:0").float() / channels)
        )
        pos_enc_a = torch.sin(timestep.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(timestep.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def label_encoding(self, label, embed_dim):
        return torch.nn.functional.interpolate(
            label.unsqueeze(1).float(), size=embed_dim, mode='linear', align_corners=False
        ).squeeze(1)