"""
model.py
Architectures for KLA Semiconductor Joint Denoising & 2x Super-Resolution.

Includes:
  1. UNet (Aditya's enhanced baseline with PixelShuffle, ICNR init, Global Residuals)
  2. SymUNet (Symmetric Residual UNet with Channel Attention & Attention Gates)
  3. get_model factory for easy selection in training and inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================
# 1. ICNR WEIGHT INITIALIZATION (Sub-Pixel Anti-Checkerboard)
# ====================================================================
def icnr_init(tensor, upscale_factor=2):
    """
    ICNR initialization for PixelShuffle to prevent checkerboard artifacts.
    """
    out_channels, in_channels, k1, k2 = tensor.shape
    new_out_channels = out_channels // (upscale_factor ** 2)
    
    sub_tensor = torch.zeros(new_out_channels, in_channels, k1, k2)
    nn.init.kaiming_normal_(sub_tensor, mode='fan_out', nonlinearity='relu')
    sub_tensor = sub_tensor.repeat(upscale_factor ** 2, 1, 1, 1)
    
    with torch.no_grad():
        tensor.copy_(sub_tensor)


# ====================================================================
# 2. NAFNET (Nonlinear Activation Free Network) - SOTA FAST CNN
# ====================================================================
class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
    def forward(self, x):
        # x: B, C, H, W -> B, H, W, C -> norm -> B, C, H, W
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SimplifiedChannelAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 1, 1, 0),
        )
    def forward(self, x):
        att = self.conv(self.squeeze(x))
        return x * att

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.sca = SimplifiedChannelAttention(dw_channel // 2)
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        return y + x * self.gamma

class NAFNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        
        self.intro = nn.Conv2d(in_channels=in_channels, out_channels=base_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        
        # Encoders
        self.enc1 = nn.Sequential(*[NAFBlock(base_channels) for _ in range(2)])
        self.down1 = nn.Conv2d(base_channels, base_channels*2, 2, 2)
        
        self.enc2 = nn.Sequential(*[NAFBlock(base_channels*2) for _ in range(2)])
        self.down2 = nn.Conv2d(base_channels*2, base_channels*4, 2, 2)
        
        self.enc3 = nn.Sequential(*[NAFBlock(base_channels*4) for _ in range(2)])
        self.down3 = nn.Conv2d(base_channels*4, base_channels*8, 2, 2)
        
        # Bottleneck
        self.middle = nn.Sequential(*[NAFBlock(base_channels*8) for _ in range(4)])
        
        # Decoders
        self.up3 = nn.Sequential(nn.Conv2d(base_channels*8, base_channels*8*2, 1, 1), nn.PixelShuffle(2))
        self.dec3 = nn.Sequential(*[NAFBlock(base_channels*4) for _ in range(2)])
        
        self.up2 = nn.Sequential(nn.Conv2d(base_channels*4, base_channels*4*2, 1, 1), nn.PixelShuffle(2))
        self.dec2 = nn.Sequential(*[NAFBlock(base_channels*2) for _ in range(2)])
        
        self.up1 = nn.Sequential(nn.Conv2d(base_channels*2, base_channels*2*2, 1, 1), nn.PixelShuffle(2))
        self.dec1 = nn.Sequential(*[NAFBlock(base_channels) for _ in range(2)])
        
        # 2x Super-Resolution Block
        self.super_res = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        
        self.ending = nn.Conv2d(base_channels, out_channels, 3, 1, 1)

        # Apply ICNR to super_res pixel_shuffle
        icnr_init(self.super_res[0].weight, upscale_factor=2)

    def forward(self, inp):
        x = self.intro(inp)
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        
        mid = self.middle(self.down3(e3))
        
        d3 = self.dec3(self.up3(mid) + e3)
        d2 = self.dec2(self.up2(d3) + e2)
        d1 = self.dec1(self.up1(d2) + e1)
        
        # 2x Super-Resolution
        sr = self.super_res(d1)
        out = self.ending(sr)
        
        # Global Bilinear Shortcut
        base = F.interpolate(inp, scale_factor=2, mode='bilinear', align_corners=False)
        return out + base


# ====================================================================
# 3. MODEL FACTORY
# ====================================================================
def get_model(name="nafnet", in_channels=1, out_channels=1, base_channels=32, num_blocks=None):
    """
    Factory to retrieve the model.
    Dynamically imports legacy architectures to keep the main pipeline clean.
    """
    name = name.lower()
    
    if name == "symunet":
        from legacy_models import SymUNet
        return SymUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=64
        )
    elif name == "ultra_unet":
        from legacy_models import UltraUNet
        return UltraUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=96
        )
    
    # Default Champion
    return NAFNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels
    )
