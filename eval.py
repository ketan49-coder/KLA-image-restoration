"""
eval.py
Evaluation script for the KLA Image Restoration model.

Computes:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)
    - LPIPS (Learned Perceptual Image Patch Similarity)
    - Inference speed (ms per image)

Usage:
    python eval.py --checkpoint path/to/model.pth --data_dir path/to/val_split

Expects a model.py in the same folder defining a model class (see MODEL IMPORT below).
Adjust the import line once model.py is finalized to match the actual class name.
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

# ---- MODEL IMPORT ----
# Update this once model.py is finalized. Example:
# from model import RestorationUNet
try:
    from model import UNet as Model
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("[WARNING] Could not import model from model.py. "
          "Update the import line in eval.py to match the actual class name.")

from dataset import ImageRestorationDataset as RestorationDataset  # assumes dataset.py defines this class


def compute_metrics(pred, gt):
    """
    Compute PSNR and SSIM between a predicted and ground-truth image.
    Both pred and gt are expected as numpy arrays, single-channel, values in [0, 1].
    """
    pred = np.clip(pred, 0.0, 1.0)
    gt = np.clip(gt, 0.0, 1.0)

    psnr_val = psnr_fn(gt, pred, data_range=1.0)
    ssim_val = ssim_fn(gt, pred, data_range=1.0)

    return psnr_val, ssim_val


def evaluate(checkpoint_path, data_dir, device="cuda" if torch.cuda.is_available() else "cpu",
             batch_size=1, num_workers=2):

    if not MODEL_AVAILABLE:
        raise ImportError(
            "Model class could not be imported. Fix the import at the top of eval.py "
            "to match the actual class name in model.py."
        )

    print(f"[INFO] Using device: {device}")

    # ---- Load model ----
    model = Model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle both raw state_dict and dict-wrapped checkpoints
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ---- Load validation data ----
    val_dataset = RestorationDataset(data_dir, split="val")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ---- LPIPS model, if available ----
    if LPIPS_AVAILABLE:
        lpips_model = lpips.LPIPS(net="alex").to(device)
        lpips_model.eval()

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    inference_times = []

    print(f"[INFO] Evaluating on {len(val_dataset)} samples...")

    with torch.no_grad():
        for i, (degraded, gt) in enumerate(val_loader):
            degraded = degraded.to(device)
            gt_np = gt.numpy()

            # ---- Timed inference ----
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()

            output = model(degraded)

            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.time() - start_time) * 1000  # ms
            inference_times.append(elapsed)

            output_np = output.cpu().numpy()

            # ---- Per-image metrics (loop over batch) ----
            for b in range(output_np.shape[0]):
                pred_img = output_np[b, 0]  # assuming shape (B, 1, H, W)
                gt_img = gt_np[b, 0]

                psnr_val, ssim_val = compute_metrics(pred_img, gt_img)
                psnr_scores.append(psnr_val)
                ssim_scores.append(ssim_val)

            # ---- LPIPS (expects 3-channel, [-1, 1] range) ----
            if LPIPS_AVAILABLE:
                output_lpips = output.repeat(1, 3, 1, 1) * 2 - 1
                gt_lpips = gt.to(device).repeat(1, 3, 1, 1) * 2 - 1
                lp = lpips_model(output_lpips, gt_lpips)
                lpips_scores.extend(lp.squeeze().cpu().tolist() if lp.numel() > 1 else [lp.item()])

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(val_loader)} batches...")

    # ---- Summary ----
    print("\n===== EVALUATION RESULTS =====")
    print(f"Samples evaluated: {len(psnr_scores)}")
    print(f"Average PSNR:  {np.mean(psnr_scores):.4f} dB")
    print(f"Average SSIM:  {np.mean(ssim_scores):.4f}")
    if LPIPS_AVAILABLE:
        print(f"Average LPIPS: {np.mean(lpips_scores):.4f}  (lower is better)")
    print(f"Average inference time: {np.mean(inference_times):.2f} ms/image")
    print(f"Min/Max inference time: {np.min(inference_times):.2f} / {np.max(inference_times):.2f} ms")
    print("================================\n")

    return {
        "psnr": np.mean(psnr_scores),
        "ssim": np.mean(ssim_scores),
        "lpips": np.mean(lpips_scores) if LPIPS_AVAILABLE else None,
        "avg_inference_ms": np.mean(inference_times),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate KLA restoration model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root (containing GT/NoisyLR)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
