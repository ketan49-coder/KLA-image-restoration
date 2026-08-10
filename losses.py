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


# ====================================================================
# 3. FOCAL FREQUENCY LOSS (2D FFT)
# ====================================================================
class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss (FFL) via 2D Real FFT.
    Fixes two critical issues for SEM imagery:
    1. Charbonnier Robustness: Frequency bins corrupted by sensor noise are 
       outliers. A raw L1 penalty overreacts to them. Charbonnier smoothly
       handles these outliers without dying gradients.
    2. Adaptive Focal Weighting: Dynamically weights the loss based on the 
       *current error* matrix rather than a static ground truth map. As 
       training progresses, the network automatically hunts down whichever 
       frequency bands it is still struggling to reconstruct.
    """
    def __init__(self, gamma=1.0, eps=1e-3):
        super(FocalFrequencyLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, target):
        # Compute orthonormal 2D FFT
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")

        # Extract magnitudes
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # 1. Charbonnier Robustness (replaces outlier-prone raw L1 difference)
        diff = pred_mag - target_mag
        charb_diff = torch.sqrt((diff * diff) + (self.eps * self.eps))

        # 2. Dynamic Focal Weighting (focuses on currently failing frequencies)
        # Detach to prevent gradients from flowing through the weighting factor itself
        focal_weight = (charb_diff.detach() ** self.gamma)

        # Final focused robust loss
        return torch.mean(focal_weight * charb_diff)



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
# 6. EDGE LOSS (Sobel Filters)
# ====================================================================
class EdgeLoss(nn.Module):
    """
    Penalizes differences in image gradients (edges) using Sobel filters.
    Forces the network to reconstruct sharp, high-frequency structures
    instead of producing soft/hazy outputs.
    """
    def __init__(self, eps=1e-3):
        super(EdgeLoss, self).__init__()
        self.eps = eps
        # Define Sobel kernels
        k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("weight_x", k_x)
        self.register_buffer("weight_y", k_y)

    def forward(self, pred, target):
        # We assume 1-channel grayscale images for this challenge
        pred_grad_x = F.conv2d(pred, self.weight_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.weight_y, padding=1)
        target_grad_x = F.conv2d(target, self.weight_x, padding=1)
        target_grad_y = F.conv2d(target, self.weight_y, padding=1)

        diff_x = pred_grad_x - target_grad_x
        diff_y = pred_grad_y - target_grad_y
        
        # Charbonnier penalty on gradients for robustness against noise spikes
        loss_x = torch.mean(torch.sqrt((diff_x * diff_x) + (self.eps * self.eps)))
        loss_y = torch.mean(torch.sqrt((diff_y * diff_y) + (self.eps * self.eps)))
        return loss_x + loss_y

# ====================================================================
# 7. QUAD-FIDELITY LOSS (Hackathon Maximizer)
# ====================================================================
class QuadFidelityLoss(nn.Module):
    """
    The ultimate 4-pillar loss function designed specifically for the KLA Hackathon:
        1. Charbonnier (0.45): Direct, noise-robust PSNR optimization.
        2. MS-SSIM (0.35): Multi-scale structural similarity (human perception).
        3. Focal Frequency Loss (0.10): Spectrum matching.
        4. Edge Loss (0.10): Explicit edge sharpness constraint.
    """
    def __init__(self, w_charb=0.45, w_msssim=0.35, w_gfl=0.10, w_edge=0.10, eps=1e-3):
        super(QuadFidelityLoss, self).__init__()
        self.w_charb = w_charb
        self.w_msssim = w_msssim
        self.w_gfl = w_gfl
        self.w_edge = w_edge

        self.charbonnier = CharbonnierLoss(eps=eps)
        self.ms_ssim = ExactMSSSIMLoss(window_size=3)
        self.gfl = FocalFrequencyLoss(gamma=1.0, eps=eps)
        self.edge = EdgeLoss(eps=eps)

    def forward(self, pred, target):
        loss_c = self.charbonnier(pred, target)
        loss_s = self.ms_ssim(pred, target)
        loss_g = self.gfl(pred, target)
        loss_e = self.edge(pred, target)
        return (self.w_charb * loss_c) + (self.w_msssim * loss_s) + (self.w_gfl * loss_g) + (self.w_edge * loss_e)


# ====================================================================
# 5. LOSS FACTORY
# ====================================================================
def get_loss_function(name="quad_fidelity", **kwargs):
    """
    Factory to retrieve the champion loss function cleanly.
    Hardcoded to QuadFidelityLoss for the final submission run.
    """
    return QuadFidelityLoss(w_charb=0.45, w_msssim=0.35, w_gfl=0.10, w_edge=0.10)

