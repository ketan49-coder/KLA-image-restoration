import torch
import torch.nn as nn
import torch.nn.functional as F
from model import icnr_init

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


