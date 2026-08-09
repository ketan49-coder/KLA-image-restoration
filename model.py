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
# 5. RRDB (Residual in Residual Dense Block) — ESRGAN Architecture
# ====================================================================
class DenseLayer(nn.Module):
    """
    Single dense layer: Conv -> LeakyReLU.
    Output is concatenated with all previous inputs (dense connectivity).
    """
    def __init__(self, in_channels, growth_rate=32):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, 3, padding=1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB):
    5 densely-connected conv layers + residual scaling.
    Each layer receives ALL previous feature maps as input (DenseNet connectivity).
    
    Architecture:
        x -> Conv1 -> d1
        [x, d1] -> Conv2 -> d2
        [x, d1, d2] -> Conv3 -> d3
        [x, d1, d2, d3] -> Conv4 -> d4
        [x, d1, d2, d3, d4] -> Conv5 -> d5
        Output = x + beta * d5
    """
    def __init__(self, num_features=64, growth_rate=32, res_scale=0.2):
        super(ResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        
        # 5 dense layers with increasing input channels
        self.dense1 = DenseLayer(num_features, growth_rate)
        self.dense2 = DenseLayer(num_features + growth_rate, growth_rate)
        self.dense3 = DenseLayer(num_features + 2 * growth_rate, growth_rate)
        self.dense4 = DenseLayer(num_features + 3 * growth_rate, growth_rate)
        # Final layer maps back to num_features (no activation)
        self.dense5 = nn.Conv2d(num_features + 4 * growth_rate, num_features, 3, padding=1, bias=True)

    def forward(self, x):
        d1 = self.dense1(x)
        d2 = self.dense2(torch.cat([x, d1], dim=1))
        d3 = self.dense3(torch.cat([x, d1, d2], dim=1))
        d4 = self.dense4(torch.cat([x, d1, d2, d3], dim=1))
        d5 = self.dense5(torch.cat([x, d1, d2, d3, d4], dim=1))
        # Residual scaling (beta=0.2) for training stability in deep networks
        return x + d5 * self.res_scale


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block:
    Stacks 3 Residual Dense Blocks with an outer residual connection.
    
    This is the core building block of ESRGAN/Real-ESRGAN.
    The "residual in residual" design allows extremely deep networks (23+ blocks)
    to train stably without gradient vanishing.
    
    Architecture:
        x -> RDB1 -> RDB2 -> RDB3 -> out
        Output = x + beta * out
    """
    def __init__(self, num_features=64, growth_rate=32, res_scale=0.2):
        super(RRDB, self).__init__()
        self.res_scale = res_scale
        self.rdb1 = ResidualDenseBlock(num_features, growth_rate, res_scale)
        self.rdb2 = ResidualDenseBlock(num_features, growth_rate, res_scale)
        self.rdb3 = ResidualDenseBlock(num_features, growth_rate, res_scale)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.res_scale


class RRDBNet(nn.Module):
    """
    RRDBNet: Full RRDB-based Image Restoration Network (ESRGAN backbone).
    
    Architecture:
        Input (1, H, W)
          -> Shallow Feature Extraction (3x3 Conv)
          -> N x RRDB blocks (deep feature extraction with dense connections)
          -> Feature Fusion (3x3 Conv)
          -> Global Residual (shallow + deep)
          -> 2x Sub-Pixel Upsampling (PixelShuffle + ICNR init)
          -> Global Bilinear Residual Shortcut
        Output (1, 2H, 2W)
    
    Design Choices for SEM Wafer Restoration:
        - num_blocks=6 (default): Balanced speed vs quality. Each RRDB contains 3 RDBs
          with 5 dense layers each = 90 conv layers total. Fast enough for competition.
        - growth_rate=32: Standard DenseNet growth for 64-channel features.
        - res_scale=0.2: Prevents gradient explosion in deep dense networks.
        - No BatchNorm: ESRGAN paper showed removing BN improves quality and reduces
          computational cost. Dense connections provide implicit regularization.
    """
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_blocks=6, growth_rate=32):
        super(RRDBNet, self).__init__()
        
        # 1. Shallow Feature Extraction
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1, bias=True)
        
        # 2. Deep Feature Extraction: N x RRDB blocks
        self.body = nn.Sequential(*[
            RRDB(num_features=num_features, growth_rate=growth_rate, res_scale=0.2)
            for _ in range(num_blocks)
        ])
        self.body_tail = nn.Conv2d(num_features, num_features, 3, padding=1, bias=True)
        
        # 3. 2x Sub-Pixel Upsampling Reconstruction
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, out_channels * 4, 3, padding=1, bias=True),
            nn.PixelShuffle(2)
        )
        icnr_init(self.upsample[0].weight, upscale_factor=2)

    def forward(self, x):
        # Shallow features
        f0 = self.head(x)
        
        # Deep RRDB feature extraction
        f_deep = self.body(f0)
        f_deep = self.body_tail(f_deep)
        
        # Global residual learning (shallow + deep)
        f_res = f0 + f_deep
        
        # 2x Super-Resolution
        sr = self.upsample(f_res)
        
        # Global bilinear shortcut (model learns high-frequency residual only)
        base = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return sr + base


# ====================================================================
# 6. ULTRA-UNET (Double Depth + ASPP Bottleneck)
# ====================================================================
class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    Captures multi-scale contextual information by applying parallel dilated convolutions.
    Extremely effective for restoring large, sweeping structural lines in SEM imagery.
    """
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        # Parallel dilated convs
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=8, dilation=8, bias=False)
        
        # Batch norms
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Fusion conv to combine all 5 branches
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x1 = F.leaky_relu(self.bn1(self.conv1(x)), 0.2, inplace=True)
        x2 = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, inplace=True)
        x3 = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, inplace=True)
        x4 = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, inplace=True)
        
        # Global branch: average pool, 1x1 conv, then upsample back to original size
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate and fuse
        out = torch.cat((x1, x2, x3, x4, x5), dim=1)
        return self.fusion(out)


class UltraUNet(nn.Module):
    """
    UltraUNet: The ultimate fast U-Net architecture for 30+ dB PSNR.
    - Double the depth (2 ResidualBlocks per level instead of 1).
    - ASPP bottleneck for massive multi-scale receptive field.
    - Squeeze-and-Excitation Channel Attention in every block.
    - Attention Gates on skip connections to filter noise.
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=96):
        super(UltraUNet, self).__init__()
        bc = base_channels
        self.use_ag = True

        # Encoder (2x Residual Blocks + SE Attention per level)
        self.enc1a = ResidualBlock(in_channels, bc)
        self.enc1b = ResidualBlock(bc, bc)
        
        self.enc2a = ResidualBlock(bc, bc * 2)
        self.enc2b = ResidualBlock(bc * 2, bc * 2)
        
        self.enc3a = ResidualBlock(bc * 2, bc * 4)
        self.enc3b = ResidualBlock(bc * 4, bc * 4)
        
        self.enc4a = ResidualBlock(bc * 4, bc * 8)
        self.enc4b = ResidualBlock(bc * 8, bc * 8)

        # Bottleneck: ASPP (Atrous Spatial Pyramid Pooling)
        # Replaces the single residual block with massive multi-scale context
        self.bottleneck_aspp = ASPP(bc * 8, bc * 16)
        # Optional refinement block after ASPP
        self.bottleneck_refine = ResidualBlock(bc * 16, bc * 16)

        # Attention Gates for Skip Connections
        if self.use_ag:
            self.ag4 = AttentionGate(f_g=bc * 8, f_l=bc * 8, f_int=bc * 4)
            self.ag3 = AttentionGate(f_g=bc * 4, f_l=bc * 4, f_int=bc * 2)
            self.ag2 = AttentionGate(f_g=bc * 2, f_l=bc * 2, f_int=bc)
            self.ag1 = AttentionGate(f_g=bc, f_l=bc, f_int=bc // 2)

        # Decoder (2x Residual Blocks + SE Attention per level)
        self.up4 = nn.ConvTranspose2d(bc * 16, bc * 8, 2, stride=2)
        self.dec4a = ResidualBlock(bc * 16, bc * 8)
        self.dec4b = ResidualBlock(bc * 8, bc * 8)

        self.up3 = nn.ConvTranspose2d(bc * 8, bc * 4, 2, stride=2)
        self.dec3a = ResidualBlock(bc * 8, bc * 4)
        self.dec3b = ResidualBlock(bc * 4, bc * 4)

        self.up2 = nn.ConvTranspose2d(bc * 4, bc * 2, 2, stride=2)
        self.dec2a = ResidualBlock(bc * 4, bc * 2)
        self.dec2b = ResidualBlock(bc * 2, bc * 2)

        self.up1 = nn.ConvTranspose2d(bc * 2, bc, 2, stride=2)
        self.dec1a = ResidualBlock(bc * 2, bc)
        self.dec1b = ResidualBlock(bc, bc)

        # 2x Super-Resolution Sub-Pixel Upscaler
        self.super_res_conv = nn.Conv2d(bc, out_channels * 4, kernel_size=3, padding=1)
        icnr_init(self.super_res_conv.weight, upscale_factor=2)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1b(self.enc1a(x))
        e2 = self.enc2b(self.enc2a(self.pool(e1)))
        e3 = self.enc3b(self.enc3a(self.pool(e2)))
        e4 = self.enc4b(self.enc4a(self.pool(e3)))

        # Bottleneck (ASPP + Refine)
        b = self.bottleneck_aspp(self.pool(e4))
        b = self.bottleneck_refine(b)

        # Decoder with Attention Gates
        d4 = self.up4(b)
        x_e4 = self.ag4(g=d4, x=e4) if self.use_ag else e4
        d4 = torch.cat([d4, x_e4], dim=1)
        d4 = self.dec4b(self.dec4a(d4))

        d3 = self.up3(d4)
        x_e3 = self.ag3(g=d3, x=e3) if self.use_ag else e3
        d3 = torch.cat([d3, x_e3], dim=1)
        d3 = self.dec3b(self.dec3a(d3))

        d2 = self.up2(d3)
        x_e2 = self.ag2(g=d2, x=e2) if self.use_ag else e2
        d2 = torch.cat([d2, x_e2], dim=1)
        d2 = self.dec2b(self.dec2a(d2))

        d1 = self.up1(d2)
        x_e1 = self.ag1(g=d1, x=e1) if self.use_ag else e1
        d1 = torch.cat([d1, x_e1], dim=1)
        d1 = self.dec1b(self.dec1a(d1))

        # Sub-Pixel 2x Reconstruction
        sr = self.super_res_conv(d1)
        sr = self.pixel_shuffle(sr)

        # Global Bilinear Residual Shortcut
        base = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return sr + base


# ====================================================================
# 7. MODEL FACTORY
# ====================================================================
def get_model(name="symunet", in_channels=1, out_channels=1, base_channels=64, num_blocks=None):
    """
    Factory to retrieve models cleanly:
      - 'ultra_unet': Double depth + ASPP bottleneck for max performance (Target 30+ dB)
      - 'symunet': Symmetric U-Net with Attention Gates & Channel Attention (Fast & High Performance)
      - 'rrdb': RRDB Network (ESRGAN backbone — Dense connections, high capacity)
      - 'resrestorer': Full-Resolution Deep Residual Network
    """
    name = (name or "symunet").lower()
    if name in ["ultra_unet", "ultraunet", "ultra"]:
        return UltraUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels
        )
    elif name in ["rrdb", "rrdbnet", "esrgan"]:
        nb = num_blocks if num_blocks is not None else 6
        return RRDBNet(
            in_channels=in_channels,
            out_channels=out_channels,
            num_features=base_channels,
            num_blocks=nb,
            growth_rate=32
        )
    elif name in ["resrestorer", "edsr", "rcan", "residual_net"]:
        nb = num_blocks if num_blocks is not None else 16
        return ResRestorer(
            in_channels=in_channels,
            out_channels=out_channels,
            num_features=base_channels,
            num_blocks=nb
        )
    else:
        return SymUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            use_attention_gates=True
        )



