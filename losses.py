"""
losses.py
Loss functions for KLA Semiconductor Image Restoration.

Contains:
  1. Exact Reference Formulation from Zhao et al. (IEEE TCI 2017):
     "Loss Functions for Image Restoration with Neural Networks"
     - Multi-Scale Structural Similarity (MS-SSIM) with separated luminance at scale M
     - Gaussian-weighted L1 loss (G-L1)
     - Mix loss: alpha * L_MSSSIM + (1 - alpha) * L_G-L1
  2. Guided Frequency Loss (GFL via 2D FFT)
  3. Compound Loss (Zhao Mix + GFL)
  4. Modular Ablation Loss Suite
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================================================================
# 1. OPTIMIZED GAUSSIAN WINDOW CACHE
# ====================================================================
_WINDOW_CACHE = {}

def get_gaussian_window_2d(window_size=11, channel=1, sigma=1.5, device="cpu"):
    """
    Retrieves or generates a cached 2D Gaussian convolution kernel.
    """
    key = (window_size, channel, sigma, device)
    if key in _WINDOW_CACHE:
        return _WINDOW_CACHE[key]

    gauss = torch.tensor([
        -(x - window_size // 2) ** 2 / float(2 * sigma ** 2) for x in range(window_size)
    ], dtype=torch.float32, device=device).exp()
    gauss = (gauss / gauss.sum()).unsqueeze(1)

    _2d = gauss.mm(gauss.t()).unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channel, 1, window_size, window_size).contiguous()
    _WINDOW_CACHE[key] = window
    return window


# ====================================================================
# 2. EXACT ZHAO ET AL. (2017) MS-SSIM & GAUSSIAN-WEIGHTED L1
# ====================================================================
def compute_ssim_components(img1, img2, window_size=11, sigma=1.5):
    """
    Computes luminance (l) and contrast-structure (cs) maps separately.
    """
    window = get_gaussian_window_2d(window_size, img1.size(1), sigma=sigma, device=img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Luminance map
    l_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    # Contrast-structure map
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    return l_map.mean(), cs_map.mean()


class GaussianWeightedL1Loss(nn.Module):
    """
    Gaussian-Weighted L1 Loss as defined in Zhao et al. (2017) Section 3.2:
        L_G-L1 = G_sigma * |pred - target|
    Convolves the L1 absolute difference with the Gaussian window to weight
    patch centers smoothly, preventing boundary artifacts.
    """
    def __init__(self, window_size=11, sigma=1.5):
        super(GaussianWeightedL1Loss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, pred, target):
        window = get_gaussian_window_2d(self.window_size, pred.size(1), sigma=self.sigma, device=pred.device)
        diff = torch.abs(pred - target)
        weighted_diff = F.conv2d(diff, window, padding=self.window_size // 2, groups=pred.size(1))
        return weighted_diff.mean()


class ExactMSSSIMLoss(nn.Module):
    """
    Exact Multi-Scale SSIM according to Wang et al. (2003) & Zhao et al. (2017):
        MS-SSIM(x, y) = [l_M(x, y)]^alpha_M * prod_{j=1}^M [cs_j(x, y)]^beta_j
        L_MSSSIM = 1 - MS-SSIM
    """
    def __init__(self, weights=None, window_size=11, sigma=1.5):
        super(ExactMSSSIMLoss, self).__init__()
        # Standard Wang et al. weights for 5 scales
        self.weights = weights or [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, pred, target):
        levels = len(self.weights)
        weights = torch.tensor(self.weights, device=pred.device)
        cs_levels = []

        current_pred = pred
        current_target = target

        for i in range(levels):
            l_val, cs_val = compute_ssim_components(current_pred, current_target, self.window_size, self.sigma)
            cs_levels.append(torch.clamp(cs_val, min=1e-8, max=1.0))

            if i == levels - 1:
                # Luminance is evaluated ONLY at the coarsest / final scale M
                l_final = torch.clamp(l_val, min=1e-8, max=1.0)
            else:
                # Downsample by 2 for next scale
                current_pred = F.avg_pool2d(current_pred, kernel_size=2, stride=2)
                current_target = F.avg_pool2d(current_target, kernel_size=2, stride=2)

        # MS-SSIM = (l_M ^ alpha_M) * prod_{j=1}^M (cs_j ^ beta_j)
        cs_stacked = torch.stack(cs_levels)
        ms_ssim_val = (l_final ** weights[-1]) * torch.prod(cs_stacked ** weights)
        return 1.0 - ms_ssim_val


class ZhaoMixLoss(nn.Module):
    """
    Exact Mix Loss from Zhao et al. (2017):
        L_Mix = alpha * L_MSSSIM + (1 - alpha) * L_G-L1
    
    Default alpha in Zhao et al.: 0.025 (or 0.15 for high structure)
    """
    def __init__(self, alpha=0.025, window_size=11, sigma=1.5):
        super(ZhaoMixLoss, self).__init__()
        self.alpha = alpha
        self.msssim = ExactMSSSIMLoss(window_size=window_size, sigma=sigma)
        self.g_l1 = GaussianWeightedL1Loss(window_size=window_size, sigma=sigma)

    def forward(self, pred, target):
        loss_msssim = self.msssim(pred, target)
        loss_g_l1 = self.g_l1(pred, target)
        return self.alpha * loss_msssim + (1.0 - self.alpha) * loss_g_l1


# ====================================================================
# 3. GUIDED FREQUENCY LOSS (GFL via 2D FFT)
# ====================================================================
class GuidedFrequencyLoss(nn.Module):
    """
    Guided Frequency Loss (GFL) via 2D Real FFT:
    Penalizes spectral attenuation and forces high-frequency edge restoration.
    """
    def __init__(self, alpha=0.2):
        super(GuidedFrequencyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        weight = 1.0 + self.alpha * target_mag
        freq_diff = torch.abs(pred_mag - target_mag)

        return torch.mean(weight * freq_diff)


# ====================================================================
# 4. COMPOUND LOSS: ZHAO MIX + GUIDED FREQUENCY LOSS
# ====================================================================
class CompoundRestorationLoss(nn.Module):
    """
    Compound Loss combining:
      - Zhao et al. Mix Loss (L1 + MS-SSIM)
      - Guided Frequency Loss (2D FFT)
    """
    def __init__(self, alpha_zhao=0.15, w_gfl=0.15):
        super(CompoundRestorationLoss, self).__init__()
        self.w_gfl = w_gfl
        self.zhao_mix = ZhaoMixLoss(alpha=alpha_zhao)
        self.gfl = GuidedFrequencyLoss(alpha=0.2)

    def forward(self, pred, target):
        mix_loss = self.zhao_mix(pred, target)
        gfl_loss = self.gfl(pred, target)
        return (1.0 - self.w_gfl) * mix_loss + self.w_gfl * gfl_loss


# ====================================================================
# 5. ABLATION FACTORY
# ====================================================================
class CombinedLoss(nn.Module):
    """Stage 1 Baseline: L1 + 0.1 * MSE"""
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.l1(pred, target) + 0.1 * self.mse(pred, target)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size

    def forward(self, pred, target):
        l_val, cs_val = compute_ssim_components(pred, target, window_size=self.window_size)
        return 1.0 - (l_val * cs_val)


def get_loss_function(name="baseline"):
    """
    Factory function for ablation experiments.
    """
    name = name.lower()
    if name == "l1":
        return nn.L1Loss()
    elif name in ["l2", "mse"]:
        return nn.MSELoss()
    elif name == "g_l1":
        return GaussianWeightedL1Loss()
    elif name == "ssim":
        return SSIMLoss()
    elif name in ["msssim", "exact_msssim"]:
        return ExactMSSSIMLoss()
    elif name in ["zhao_paper", "zhao_0025"]:
        # Exact paper default (alpha=0.025, 97.5% L1 + 2.5% MS-SSIM)
        return ZhaoMixLoss(alpha=0.025)
    elif name in ["zhao_sem", "zhao_015", "l1_msssim"]:
        # Tuned for SEM edge contrast (alpha=0.15, 85% L1 + 15% MS-SSIM)
        return ZhaoMixLoss(alpha=0.15)
    elif name == "gfl":
        return GuidedFrequencyLoss(alpha=0.2)
    elif name in ["compound", "l1_msssim_gfl", "all"]:
        # Full compound: Zhao Mix + Guided Frequency Loss
        return CompoundRestorationLoss(alpha_zhao=0.15, w_gfl=0.15)
    elif name == "baseline":
        return CombinedLoss()
    else:
        raise ValueError(f"Unknown loss: '{name}'. Choose from: "
                         f"['l1', 'l2', 'g_l1', 'ssim', 'msssim', 'zhao_paper', 'zhao_sem', 'gfl', 'compound', 'baseline']")
