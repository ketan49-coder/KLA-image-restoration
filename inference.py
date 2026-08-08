import os
import argparse
import glob
import numpy as np
import torch
from model import UNet

def normalize_image(img_tensor):
    min_val = img_tensor.min()
    max_val = img_tensor.max()
    norm_tensor = (img_tensor - min_val) / (max_val - min_val + 1e-8)
    return norm_tensor, min_val, max_val

def denormalize_image(norm_tensor, min_val, max_val):
    return (norm_tensor * (max_val - min_val + 1e-8)) + min_val

def run_inference(input_dir, output_dir, checkpoint_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[INFO] Using device: {device}")

    # Load model
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Loaded checkpoint from: {checkpoint_path}")

    # Setup directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .npy files
    search_pattern = os.path.join(input_dir, "**/*.npy")
    test_files = glob.glob(search_pattern, recursive=True)
    if not test_files:
        test_files = glob.glob(os.path.join(input_dir, "*.npy"))
        
    print(f"[INFO] Found {len(test_files)} images to process.")

    with torch.no_grad():
        for file_path in test_files:
            filename = os.path.basename(file_path)
            
            # Load and preprocess
            noisy = np.load(file_path)
            noisy_t = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
            
            noisy_norm, min_val, max_val = normalize_image(noisy_t)
            noisy_norm = noisy_norm.to(device)
            
            # Inference
            output = model(noisy_norm)
            
            # Denormalize and save
            output = output.cpu()
            restored = denormalize_image(output, min_val, max_val)
            restored_np = restored.squeeze().numpy()
            
            out_path = os.path.join(output_dir, filename)
            np.save(out_path, restored_np)
            
    print(f"[INFO] Inference complete. Saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Inference Script for KLA Restoration Model")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input test directory containing degraded images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save restored .npy outputs")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pth model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    run_inference(args.input_dir, args.output_dir, args.checkpoint, device=args.device)
