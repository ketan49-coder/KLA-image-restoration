"""
metrics.py
Evaluation metrics for image restoration: PSNR, SSIM, and LPIPS.
Provides a unified RestorationMetrics class for batch evaluation on GPU/CPU.
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


class RestorationMetrics:
    def __init__(self, device="cpu", compute_lpips=True):
        """
        Unified Restoration Quality Metric Evaluator.
        Computes PSNR, SSIM, and LPIPS in a single pass.
        """
        self.device = torch.device(device)
        self.compute_lpips = compute_lpips and LPIPS_AVAILABLE
        
        if self.compute_lpips:
            try:
                self.lpips_model = lpips.LPIPS(net="alex").to(self.device)
                self.lpips_model.eval()
            except Exception as e:
                print(f"[WARNING] Could not load LPIPS model: {e}. LPIPS will be skipped.")
                self.compute_lpips = False
        else:
            self.lpips_model = None

    @torch.no_grad()
    def compute_batch(self, pred, target):
        """
        Compute metrics for a batch of images.
        pred, target: PyTorch Tensors [B, 1, H, W] in [0.0, 1.0] range.
        Returns: dict with 'psnr', 'ssim', 'lpips' (floats)
        """
        pred_c = torch.clamp(pred, 0.0, 1.0)
        target_c = torch.clamp(target, 0.0, 1.0)

        # 1. PSNR & SSIM via NumPy / Scikit-Image
        pred_np = pred_c.cpu().numpy()
        target_np = target_c.cpu().numpy()

        psnr_vals = []
        ssim_vals = []
        batch_size = pred_np.shape[0]

        for b in range(batch_size):
            p = pred_np[b, 0]
            t = target_np[b, 0]
            psnr_vals.append(psnr_fn(t, p, data_range=1.0))
            ssim_vals.append(ssim_fn(t, p, data_range=1.0))

        batch_psnr = float(np.mean(psnr_vals))
        batch_ssim = float(np.mean(ssim_vals))

        # 2. LPIPS (Expects 3-channel input scaled to [-1.0, 1.0])
        batch_lpips = 0.0
        if self.compute_lpips and self.lpips_model is not None:
            pred_3c = pred_c.repeat(1, 3, 1, 1) * 2.0 - 1.0
            target_3c = target_c.repeat(1, 3, 1, 1) * 2.0 - 1.0
            lp_val = self.lpips_model(pred_3c, target_3c)
            batch_lpips = float(lp_val.mean().item())

        return {
            "psnr": batch_psnr,
            "ssim": batch_ssim,
            "lpips": batch_lpips if self.compute_lpips else None
        }


# Standalone utility functions
def calculate_psnr(pred, gt, data_range=1.0):
    pred_c = np.clip(pred, 0.0, data_range)
    gt_c = np.clip(gt, 0.0, data_range)
    return float(psnr_fn(gt_c, pred_c, data_range=data_range))


def calculate_ssim(pred, gt, data_range=1.0):
    pred_c = np.clip(pred, 0.0, data_range)
    gt_c = np.clip(gt, 0.0, data_range)
    return float(ssim_fn(gt_c, pred_c, data_range=data_range))
