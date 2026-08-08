"""
eval.py
Evaluation script for the KLA Image Restoration model.

Computes:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)
    - LPIPS (Learned Perceptual Image Patch Similarity)
    - Inference speed (ms per image)

Usage:
    python eval.py --checkpoint checkpoints/baseline_unet_epoch5.pth --data_dir train/train
"""

import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("[WARNING] lpips package not installed. Run: pip install lpips")
    print("          LPIPS scores will be skipped.")

from model import SymUNet, get_model
from dataset import ImageRestorationDataset


def compute_metrics(pred, gt):
    """
    Compute PSNR and SSIM between a predicted and ground-truth image.
    Both pred and gt are expected as 2D numpy arrays, values in [0, 1].
    """
    pred = np.clip(pred, 0.0, 1.0)
    gt = np.clip(gt, 0.0, 1.0)

    psnr_val = psnr_fn(gt, pred, data_range=1.0)
    ssim_val = ssim_fn(gt, pred, data_range=1.0)

    return psnr_val, ssim_val


def evaluate(checkpoint_path, data_dir, device=None, batch_size=1, is_val=True, model_type="symunet", base_channels=64):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[INFO] Using device: {device} | Model: {model_type.upper()}")

    # ---- 1. Load model ----
    model = get_model(model_type, in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Loaded checkpoint from: {checkpoint_path}")

    # ---- 2. Load dataset ----
    val_dataset = ImageRestorationDataset(data_dir, split_ratio=0.9, is_val=is_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"[INFO] Evaluating on {len(val_dataset)} images...")

    # ---- 3. LPIPS Model ----
    if LPIPS_AVAILABLE:
        lpips_model = lpips.LPIPS(net="alex").to(device)
        lpips_model.eval()

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    inference_times = []

    with torch.no_grad():
        for i, (degraded, gt) in enumerate(val_loader):
            degraded, gt = degraded.to(device), gt.to(device)

            # Timed inference
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()

            output = model(degraded)

            # The model naturally outputs the 2x upscaled image via PixelShuffle.
            # No manual bilinear interpolation needed here anymore.

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.time() - start_time) * 1000
            inference_times.append(elapsed_ms)

            output_np = output.cpu().numpy()
            gt_np = gt.cpu().numpy()

            # Per-sample metrics
            for b in range(output_np.shape[0]):
                pred_img = output_np[b, 0]
                gt_img = gt_np[b, 0]

                psnr_val, ssim_val = compute_metrics(pred_img, gt_img)
                psnr_scores.append(psnr_val)
                ssim_scores.append(ssim_val)

            # LPIPS (expects 3 channels in [-1, 1] range)
            if LPIPS_AVAILABLE:
                pred_3c = output.repeat(1, 3, 1, 1) * 2.0 - 1.0
                gt_3c = gt.repeat(1, 3, 1, 1) * 2.0 - 1.0
                lp = lpips_model(pred_3c, gt_3c)
                lpips_scores.extend(lp.squeeze().cpu().tolist() if lp.numel() > 1 else [lp.item()])

    # ---- Summary ----
    print("\n" + "="*50)
    print("🏆 EVALUATION RESULTS")
    print("="*50)
    print(f"Samples evaluated:      {len(psnr_scores)}")
    print(f"Average PSNR:           {np.mean(psnr_scores):.4f} dB  (Higher is better)")
    print(f"Average SSIM:           {np.mean(ssim_scores):.4f}     (Higher is better, max 1.0)")
    if LPIPS_AVAILABLE and lpips_scores:
        print(f"Average LPIPS:          {np.mean(lpips_scores):.4f}    (Lower is better, min 0.0)")
    print(f"Average Inference Time: {np.mean(inference_times):.2f} ms/image")
    print(f"Min/Max Inference Time: {np.min(inference_times):.2f} / {np.max(inference_times):.2f} ms")
    print("="*50 + "\n")

    return {
        "psnr": np.mean(psnr_scores),
        "ssim": np.mean(ssim_scores),
        "lpips": np.mean(lpips_scores) if (LPIPS_AVAILABLE and lpips_scores) else None,
        "avg_inference_ms": np.mean(inference_times),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate KLA restoration model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--data_dir", type=str, default="train/train", help="Path to dataset root (containing GT/NoisyLR)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model", type=str, default="resrestorer", choices=["resrestorer", "symunet"], help="Model architecture (resrestorer / symunet)")
    parser.add_argument("--base_channels", type=int, default=64, help="Base channel count (default: 64)")
    args = parser.parse_args()

    evaluate(
        args.checkpoint,
        args.data_dir,
        device=args.device,
        batch_size=args.batch_size,
        model_type=args.model,
        base_channels=args.base_channels
    )
