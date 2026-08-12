import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from model import get_model
from dataset import ImageRestorationDataset
import skimage.metrics as metrics

def calculate_psnr(img1, img2):
    return metrics.peak_signal_noise_ratio(img1, img2, data_range=1.0)

def calculate_ssim(img1, img2):
    return metrics.structural_similarity(img1, img2, data_range=1.0)

def run_visual_check(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Using device: {device}")

    # Load dataset
    dataset = ImageRestorationDataset(
        data_dir=None if args.packed_data else args.data_dir, 
        split_ratio=0.9, 
        is_val=True, 
        packed_data_path=args.packed_data
    )
    loader = DataLoader(dataset, batch_size=args.num_samples, shuffle=True)
    
    noisy_samples, gt_samples = next(iter(loader))
    
    model = get_model('nafnet', in_channels=1, out_channels=1, base_channels=32).to(device)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        print(f"✅ Loaded weights from {args.checkpoint}")
    else:
        print("⚠ No checkpoint found. Using randomly initialized weights.")

    model.eval()
    with torch.no_grad():
        outputs = model(noisy_samples.to(device))

    if outputs.shape != gt_samples.shape:
        outputs = torch.nn.functional.interpolate(outputs, size=gt_samples.shape[2:], mode='bilinear', align_corners=False)

    outputs = outputs.clamp(0, 1).cpu().numpy()
    noisy_samples = noisy_samples.cpu().numpy()
    gt_samples = gt_samples.cpu().numpy()

    fig, axes = plt.subplots(args.num_samples, 3, figsize=(15, 5 * args.num_samples))
    if args.num_samples == 1:
        axes = [axes]
        
    for i in range(args.num_samples):
        noisy = noisy_samples[i, 0]
        out = outputs[i, 0]
        gt = gt_samples[i, 0]
        
        psnr_val = calculate_psnr(gt, out)
        ssim_val = calculate_ssim(gt, out)

        axes[i][0].imshow(noisy, cmap='gray')
        axes[i][0].set_title('Degraded Input', fontsize=14)
        axes[i][0].axis('off')
        
        axes[i][1].imshow(out, cmap='gray')
        axes[i][1].set_title(f'Model Output\nPSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}', fontsize=14, color='green')
        axes[i][1].axis('off')
        
        axes[i][2].imshow(gt, cmap='gray')
        axes[i][2].set_title('Ground Truth', fontsize=14, color='blue')
        axes[i][2].axis('off')
        
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"📸 Visual check saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--packed_data", type=str, default=None, help="Path to packed .pt dataset")
    parser.add_argument("--data_dir", type=str, default="train/train", help="Path to raw image folder")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of images to display")
    parser.add_argument("--output", type=str, default="visual_check_results.png", help="Output filename")
    args = parser.parse_args()
    
    run_visual_check(args)
