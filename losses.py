"""
losses.py
Optimal Loss Formulation for KLA Semiconductor Image Restoration.

Combines:
  1. Multi-Scale Structural Similarity (MS-SSIM, 5 scales) with separated luminance
  2. Gaussian-Weighted L1 (G-L1, sigma=1.5) for local intensity anchoring
  3. Guided Frequency Loss (GFL via orthonormal 2D Real FFT) for high-frequency harmonic restoration

Empirically selected as the Stage 2 benchmark winner (Run 12: 23.47 dB PSNR, 0.7109 SSIM, 0.3279 LPIPS).
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
# 2. EXACT MS-SSIM & GAUSSIAN-WEIGHTED L1 (Zhao et al. 2017)
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
    Gaussian-Weighted L1 Loss (Zhao et al. 2017):
        L_G-L1 = G_sigma * |pred - target|
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
    Exact Multi-Scale SSIM (5 scales):
        MS-SSIM(x, y) = [l_M(x, y)]^alpha_M * prod_{j=1}^M [cs_j(x, y)]^beta_j
        L_MSSSIM = 1 - MS-SSIM
    """
    def __init__(self, weights=None, window_size=11, sigma=1.5):
        super(ExactMSSSIMLoss, self).__init__()
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
                l_final = torch.clamp(l_val, min=1e-8, max=1.0)
            else:
                current_pred = F.avg_pool2d(current_pred, kernel_size=2, stride=2)
                current_target = F.avg_pool2d(current_target, kernel_size=2, stride=2)

        cs_stacked = torch.stack(cs_levels)
        ms_ssim_val = (l_final ** weights[-1]) * torch.prod(cs_stacked ** weights)
        return 1.0 - ms_ssim_val


class ZhaoMixLoss(nn.Module):
    """
    Mix Loss: alpha * L_MSSSIM + (1 - alpha) * L_G-L1
    """
    def __init__(self, alpha=0.90, window_size=11, sigma=1.5):
        super(ZhaoMixLoss, self).__init__()
        self.alpha = alpha
        self.msssim = ExactMSSSIMLoss(window_size=window_size, sigma=sigma)
        self.g_l1 = GaussianWeightedL1Loss(window_size=window_size, sigma=sigma)

    def forward(self, pred, target):
        return self.alpha * self.msssim(pred, target) + (1.0 - self.alpha) * self.g_l1(pred, target)


# ====================================================================
# 3. GUIDED FREQUENCY LOSS (2D FFT)
# ====================================================================
class GuidedFrequencyLoss(nn.Module):
    """
    Guided Frequency Loss (GFL) via 2D Real FFT with orthonormal normalization.
    """
    def __init__(self, alpha=0.2):
        super(GuidedFrequencyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")

        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        weight = 1.0 + self.alpha * target_mag
        freq_diff = torch.abs(pred_mag - target_mag)
        return torch.mean(weight * freq_diff)


# ====================================================================
# 4. WINNING FLAGSHIP COMPOUND RESTORATION LOSS
# ====================================================================
class CompoundRestorationLoss(nn.Module):
    """
    Production Compound Restoration Loss (Stage 2 Winner):
        L_final = (1 - w_gfl) * [alpha * L_MSSSIM + (1 - alpha) * L_G-L1] + w_gfl * L_GFL
    Default parameters:
        alpha = 0.90 (90% Multi-Scale Structural Similarity + 10% Gaussian-L1)
        w_gfl = 0.10 (10% Orthonormal 2D FFT Frequency Loss)
    """
    def __init__(self, alpha=0.90, w_gfl=0.10):
        super(CompoundRestorationLoss, self).__init__()
        self.w_gfl = w_gfl
        self.zhao_mix = ZhaoMixLoss(alpha=alpha)
        self.gfl = GuidedFrequencyLoss(alpha=0.2)

    def forward(self, pred, target):
        return (1.0 - self.w_gfl) * self.zhao_mix(pred, target) + self.w_gfl * self.gfl(pred, target)


# ====================================================================
# 4. CHARBONNIER LOSS (Direct PSNR Optimizer)
# ====================================================================
class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (differentiable smooth L1 variant):
        L(x, y) = sqrt((x - y)^2 + eps^2)
    Standard loss in NAFNet, Restormer, and modern Super-Resolution
    benchmarks for direct mathematical PSNR optimization.
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


# ====================================================================
# 5. LOSS FACTORY
# ====================================================================
def get_loss_function(name="compound", alpha=0.90, w_gfl=0.10, alpha_zhao=None, **kwargs):
    """
    Factory to retrieve loss functions cleanly for ablation experiments.
    """
    if alpha_zhao is not None:
        alpha = alpha_zhao
    name = (name or "compound").lower()
    if name in ["charbonnier", "charb", "psnr_loss"]:
        return CharbonnierLoss(eps=1e-3)
    elif name in ["compound", "compound_90", "optimal", "default"]:
        return CompoundRestorationLoss(alpha=alpha, w_gfl=w_gfl)
    elif name in ["l1", "mae"]:
        return nn.L1Loss()
    elif name in ["l2", "mse"]:
        return nn.MSELoss()
    elif name in ["msssim", "ms_ssim"]:
        return ExactMSSSIMLoss()
    elif name == "baseline":
        # Stage 1 baseline for comparison
        class BaselineLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.L1Loss()
                self.mse = nn.MSELoss()
            def forward(self, pred, target):
                return self.l1(pred, target) + 0.1 * self.mse(pred, target)
        return BaselineLoss()
    else:
        return CompoundRestorationLoss(alpha=alpha, w_gfl=w_gfl)
