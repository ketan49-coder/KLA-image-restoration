import torch
import torch.nn as nn

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

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super(UNet, self).__init__()

        bc = base_channels
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, bc)
        self.enc2 = self.conv_block(bc, bc*2)
        self.enc3 = self.conv_block(bc*2, bc*4)
        self.enc4 = self.conv_block(bc*4, bc*8)

        # Bottleneck
        self.bottleneck = self.conv_block(bc*8, bc*16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(bc*16, bc*8, 2, stride=2)
        self.dec4 = self.conv_block(bc*16, bc*8)

        self.up3 = nn.ConvTranspose2d(bc*8, bc*4, 2, stride=2)
        self.dec3 = self.conv_block(bc*8, bc*4)

        self.up2 = nn.ConvTranspose2d(bc*4, bc*2, 2, stride=2)
        self.dec2 = self.conv_block(bc*4, bc*2)

        self.up1 = nn.ConvTranspose2d(bc*2, bc, 2, stride=2)
        self.dec1 = self.conv_block(bc*2, bc)

        self.super_res_conv = nn.Conv2d(bc, out_channels * 4, kernel_size=3, padding=1)
        icnr_init(self.super_res_conv.weight, upscale_factor=2)
        
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.pool = nn.MaxPool2d(2, 2)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        sr = self.super_res_conv(d1)
        sr = self.pixel_shuffle(sr)
        
        # Global Residual Connection
        base = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return sr + base
