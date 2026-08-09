import os
import argparse
import glob
import numpy as np
import torch
from model import SymUNet, get_model

def normalize_image(img_tensor):
    """
    Robust Normalization using 1st and 99th percentiles to ignore speckle spikes.
    """
    # Flatten tensor to calculate quantiles
    flat = img_tensor.contiguous().view(-1)
    min_val = torch.quantile(flat, 0.01).item()
    max_val = torch.quantile(flat, 0.99).item()
    
    # Clip and normalize
    clipped = torch.clamp(img_tensor, min_val, max_val)
    norm_tensor = (clipped - min_val) / (max_val - min_val + 1e-8)
    return norm_tensor, min_val, max_val

def denormalize_image(norm_tensor, min_val, max_val):
    return (norm_tensor * (max_val - min_val + 1e-8)) + min_val

def run_inference(input_dir, output_dir, checkpoint_path, device=None, model_type="symunet", base_channels=64, batch_size=8, use_fp16=True, use_compile=False):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    is_cuda = device.type == "cuda"
    print(f"[INFO] Using device: {device} | Model: {model_type.upper()}")

    # ── 1. Load model to GPU ONCE ──────────────────────────────────
    model = get_model(model_type, in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()  # BatchNorm uses running stats, Dropout disabled
    print(f"[INFO] Loaded checkpoint from: {checkpoint_path}")

    # ── 2. torch.compile (PyTorch 2.0+ graph optimization) ────────
    if use_compile:
        try:
            model = torch.compile(model)
            print("[INFO] ⚡ torch.compile() applied — fused graph optimization enabled")
        except Exception as e:
            print(f"[INFO] torch.compile() skipped: {e}")

    # ── 3. Setup directories ──────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .npy files
    search_pattern = os.path.join(input_dir, "**/*.npy")
    test_files = sorted(glob.glob(search_pattern, recursive=True))
    if not test_files:
        test_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
        
    total = len(test_files)
    print(f"[INFO] Found {total} images to process (batch_size={batch_size}, fp16={use_fp16 and is_cuda})")

    # ── 4. Batched Inference with Mixed Precision ─────────────────
    processed = 0
    with torch.no_grad():  # Skip autograd graph entirely — free speed + memory win
        for batch_start in range(0, total, batch_size):
            batch_files = test_files[batch_start : batch_start + batch_size]
            
            # Load and normalize each image in this batch
            batch_tensors = []
            batch_params = []  # Store (min_val, max_val) for denormalization
            
            for file_path in batch_files:
                noisy = np.load(file_path)
                noisy_t = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                noisy_norm, min_val, max_val = normalize_image(noisy_t)
                batch_tensors.append(noisy_norm)
                batch_params.append((min_val, max_val))
            
            # Stack into a single batch tensor: (B, 1, H, W)
            batch_input = torch.cat(batch_tensors, dim=0).to(device)
            
            # Forward pass with FP16 mixed precision (1.5-2x speedup on CUDA)
            if use_fp16 and is_cuda:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    batch_output = model(batch_input)
                batch_output = batch_output.float()  # Cast back to FP32 for saving
            else:
                batch_output = model(batch_input)
            
            # Denormalize and save each image
            batch_output = batch_output.cpu()
            for i, file_path in enumerate(batch_files):
                filename = os.path.basename(file_path)
                min_val, max_val = batch_params[i]
                restored = denormalize_image(batch_output[i:i+1], min_val, max_val)
                restored_np = restored.squeeze().numpy()
                
                out_path = os.path.join(output_dir, filename)
                np.save(out_path, restored_np)
            
            processed += len(batch_files)
            if processed % max(batch_size * 4, 32) == 0 or processed == total:
                print(f"  [Progress] {processed}/{total} images processed")

    print(f"\n[INFO] ✅ Inference complete. {total} images saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Inference Script for KLA Restoration Model")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input test directory containing degraded images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save restored .npy outputs")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pth model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--model", type=str, default="symunet", choices=["symunet", "rrdb", "resrestorer"], help="Model architecture (symunet / rrdb / resrestorer)")
    parser.add_argument("--base_channels", type=int, default=64, help="Base channel count (default: 64)")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size (default: 8)")
    parser.add_argument("--no_fp16", action="store_true", help="Disable FP16 mixed precision inference")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile() graph optimization (PyTorch 2.0+)")
    
    args = parser.parse_args()
    
    run_inference(
        args.input_dir, args.output_dir, args.checkpoint,
        device=args.device, model_type=args.model, base_channels=args.base_channels,
        batch_size=args.batch_size, use_fp16=not args.no_fp16, use_compile=args.compile
    )
