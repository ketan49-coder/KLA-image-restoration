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
# 2. ATTENTION & RESIDUAL MODULES
# ====================================================================
class ChannelAttention(nn.Module):
    """
    Channel Attention (Squeeze-and-Excitation / RCAB mechanism)
    Dynamically amplifies semiconductor line structures while suppressing background noise.
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        reduced_ch = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_ch, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_ch, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """
    Residual Convolutional Block with BatchNorm, LeakyReLU, and Channel Attention.
    """
    def __init__(self, in_ch, out_ch):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch else nn.Identity()
        self.ca = ChannelAttention(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        res = self.conv(x)
        res = self.ca(res)
        return self.act(res + self.shortcut(x))


class AttentionGate(nn.Module):
    """
    Attention Gate to filter encoder skip features before concatenation with decoder.
    Suppresses noisy background regions while highlighting sharp edges.
    """
    def __init__(self, f_g, f_l, f_int):
        super(AttentionGate, self).__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ====================================================================
# 3. SYMUNET (Symmetric Residual UNet with Attention Gates)
# ====================================================================
class SymUNet(nn.Module):
    """
    Symmetric Residual UNet with Channel Attention and Attention Gates.
    Specifically optimized for SEM wafer denoising + 2x super-resolution.
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, use_attention_gates=True):
        super(SymUNet, self).__init__()
        bc = base_channels
        self.use_ag = use_attention_gates

        # Encoder (Residual Blocks + Channel Attention)
        self.enc1 = ResidualBlock(in_channels, bc)
        self.enc2 = ResidualBlock(bc, bc * 2)
        self.enc3 = ResidualBlock(bc * 2, bc * 4)
        self.enc4 = ResidualBlock(bc * 4, bc * 8)

        # Bottleneck
        self.bottleneck = ResidualBlock(bc * 8, bc * 16)

        # Attention Gates for Skip Connections
        if self.use_ag:
            self.ag4 = AttentionGate(f_g=bc * 8, f_l=bc * 8, f_int=bc * 4)
            self.ag3 = AttentionGate(f_g=bc * 4, f_l=bc * 4, f_int=bc * 2)
            self.ag2 = AttentionGate(f_g=bc * 2, f_l=bc * 2, f_int=bc)
            self.ag1 = AttentionGate(f_g=bc, f_l=bc, f_int=bc // 2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(bc * 16, bc * 8, 2, stride=2)
        self.dec4 = ResidualBlock(bc * 16, bc * 8)

        self.up3 = nn.ConvTranspose2d(bc * 8, bc * 4, 2, stride=2)
        self.dec3 = ResidualBlock(bc * 8, bc * 4)

        self.up2 = nn.ConvTranspose2d(bc * 4, bc * 2, 2, stride=2)
        self.dec2 = ResidualBlock(bc * 4, bc * 2)

        self.up1 = nn.ConvTranspose2d(bc * 2, bc, 2, stride=2)
        self.dec1 = ResidualBlock(bc * 2, bc)

        # 2x Super-Resolution Sub-Pixel Upscaler
        self.super_res_conv = nn.Conv2d(bc, out_channels * 4, kernel_size=3, padding=1)
        icnr_init(self.super_res_conv.weight, upscale_factor=2)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with Attention Gates
        d4 = self.up4(b)
        x_e4 = self.ag4(g=d4, x=e4) if self.use_ag else e4
        d4 = torch.cat([d4, x_e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        x_e3 = self.ag3(g=d3, x=e3) if self.use_ag else e3
        d3 = torch.cat([d3, x_e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        x_e2 = self.ag2(g=d2, x=e2) if self.use_ag else e2
        d2 = torch.cat([d2, x_e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        x_e1 = self.ag1(g=d1, x=e1) if self.use_ag else e1
        d1 = torch.cat([d1, x_e1], dim=1)
        d1 = self.dec1(d1)

        # Sub-Pixel 2x Reconstruction
        sr = self.super_res_conv(d1)
        sr = self.pixel_shuffle(sr)

        # Global Residual Shortcut
        base = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return sr + base


# ====================================================================
# 4. RESRESTORER (Full-Resolution Deep Residual Channel-Attention Network)
# ====================================================================
class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB)
    Core building block of state-of-the-art super-resolution (RCAN / EDSR).
    Maintains full spatial resolution (128x128) with residual scaling (0.1) for deep numerical stability.
    """
    def __init__(self, channels=64, reduction=16, res_scale=0.1):
        super(RCAB, self).__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
        )
        self.ca = ChannelAttention(channels, reduction=reduction)

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        return x + res * self.res_scale


class ResRestorer(nn.Module):
    """
    ResRestorer: Full-Resolution Deep Residual Network
    
    Architectural Advantages for SEM Wafer Nanostructures:
    1. Zero Downsampling / Zero Pooling: Never reduces spatial resolution to 8x8.
       Preserves 100% of fine nanometer edge features.
    2. Deep Residual Backbone: 16 RCAB blocks with Channel Attention & Residual Scaling.
    3. Global Long Residual: Input shallow features are added directly to deep features.
    4. ICNR Sub-Pixel Upsampling: 2x Super-Resolution via PixelShuffle with anti-checkerboard init.
    5. Global Bilinear Residual: Output learns the delta (residual) on top of the upscaled input.
    6. Ultra-Fast Inference: Simple residual convolutions without complex multi-scale routing.
    """
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_blocks=16):
        super(ResRestorer, self).__init__()
        
        # 1. Shallow Feature Extraction Head
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        
        # 2. Deep Residual Body (Full 128x128 resolution throughout)
        self.body = nn.Sequential(*[
            RCAB(channels=num_features, reduction=16, res_scale=0.1)
            for _ in range(num_blocks)
        ])
        self.body_tail = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
        # 3. Sub-Pixel 2x Upsampling Reconstruction Tail
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, out_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )
        icnr_init(self.upsample[0].weight, upscale_factor=2)
        
    def forward(self, x):
        # Shallow features
        f0 = self.head(x)
        
        # Deep residual feature extraction
        f_deep = self.body(f0)
        f_deep = self.body_tail(f_deep)
        
        # Long skip connection
        f_res = f0 + f_deep
        
        # 2x Super-Resolution Tail
        sr = self.upsample(f_res)
        
        # Global Bilinear Shortcut (learns high-frequency residual only)
        base = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return sr + base


# ====================================================================
# 5. MODEL FACTORY
# ====================================================================
def get_model(name="resrestorer", in_channels=1, out_channels=1, base_channels=64):
    """
    Factory to retrieve models cleanly:
      - 'resrestorer': Full-Resolution Deep Residual Network (No 4x pooling bottleneck, Peak PSNR)
      - 'symunet': Symmetric U-Net with Attention Gates
    """
    name = (name or "resrestorer").lower()
    if name in ["resrestorer", "edsr", "rcan", "residual_net"]:
        return ResRestorer(
            in_channels=in_channels,
            out_channels=out_channels,
            num_features=base_channels,
            num_blocks=16
        )
    else:
        return SymUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            use_attention_gates=True
        )


