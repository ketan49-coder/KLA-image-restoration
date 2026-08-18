import os
import glob
import numpy as np
import argparse
import torch
from metrics import RestorationMetrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    args = parser.parse_args()
    
    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.npy")))
    
    if not pred_files:
        print("No .npy files found in pred_dir!")
        return
        
    metrics = RestorationMetrics(device="cpu", compute_lpips=False)
    
    psnrs, ssims = [], []
    print(f"Evaluating {len(pred_files)} images...")
    
    for p_file in pred_files:
        basename = os.path.basename(p_file)
        gt_file = os.path.join(args.gt_dir, basename)
        
        if not os.path.exists(gt_file):
            print(f"Skipping {basename} - GT not found")
            continue
            
        pred = np.load(p_file)
        gt = np.load(gt_file)
        
        # Add batch and channel dims for metrics.py
        pred_tensor = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)
        
        res = metrics.compute_batch(pred_tensor, gt_tensor)
        psnrs.append(res['psnr'])
        ssims.append(res['ssim'])
        
    print("\n" + "="*40)
    print("🏆 FINAL SCORECALC 🏆")
    print("="*40)
    print(f"Avg PSNR: {np.mean(psnrs):.4f} dB")
    print(f"Avg SSIM: {np.mean(ssims):.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
