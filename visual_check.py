import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys

# Adjust imports based on the project structure
from model import get_model
from dataset import ImageRestorationDataset
from torch.utils.data import DataLoader

def run_visual_check(checkpoint_path=None, data_dir='train/train', output_path='visual_check.png'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataset = ImageRestorationDataset(data_dir=data_dir, split_ratio=1.0, is_val=False, preload_to_ram=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Get a single sample
    noisy_sample, gt_sample = next(iter(loader))
    
    # Initialize Model (Champion NAFNet)
    model = get_model('nafnet', in_channels=1, out_channels=1, base_channels=32).to(device)
    
    # Load weights if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print("No checkpoint found/provided. Using randomly initialized weights.")

    model.eval()
    with torch.no_grad():
        output = model(noisy_sample.to(device))

    # Resize output if dimensions mismatch (though NAFNet natively outputs 2x)
    if output.shape != gt_sample.shape:
        output = torch.nn.functional.interpolate(output, size=gt_sample.shape[2:], mode='bilinear', align_corners=False)

    # User's plotting snippet
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(noisy_sample.squeeze().cpu(), cmap='gray')
    axes[0].set_title('Degraded Input')
    axes[1].imshow(output.squeeze().cpu().clamp(0,1), cmap='gray')
    axes[1].set_title('Model Output')
    axes[2].imshow(gt_sample.squeeze().cpu(), cmap='gray')
    axes[2].set_title('Ground Truth')
    for ax in axes: ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Visual check saved to {output_path}")

if __name__ == "__main__":
    run_visual_check()
