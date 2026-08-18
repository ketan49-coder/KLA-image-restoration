import os
import argparse
import glob
import numpy as np
import torch
from model import get_model

def normalize_image(img_tensor, min_val=None, max_val=None):
    """
    Robust Normalization using 1st and 99th percentiles to ignore speckle spikes.
    If min_val and max_val are provided, uses those instead of computing them from the tensor.
    """
    if min_val is None or max_val is None:
        # Downsample by 4x4 for massive speedup on quantile calculation
        flat = img_tensor[..., ::4, ::4].contiguous().view(-1)
    if min_val is None:
        min_val = torch.quantile(flat, 0.01).item()
    if max_val is None:
        max_val = torch.quantile(flat, 0.99).item()
        
    clipped = torch.clamp(img_tensor, min_val, max_val)
    return (clipped - min_val) / (max_val - min_val + 1e-8), min_val, max_val

def denormalize_image(norm_tensor, min_val, max_val):
    return (norm_tensor * (max_val - min_val + 1e-8)) + min_val

def _forward_pass(model, x, use_fp16, is_cuda):
    if use_fp16 and is_cuda:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            return model(x).float()
    return model(x)

def forward_tta(model, x, use_fp16, is_cuda):
    """
    Batched 8x Test-Time Augmentation (flips & 90-degree rotations).
    All 8 augmentations are stacked into a SINGLE forward pass for maximum GPU throughput.
    """
    B, C, H, W = x.shape
    
    # Create all 8 augmented versions
    x_flip_h  = torch.flip(x, [3])
    x_flip_v  = torch.flip(x, [2])
    x_flip_hv = torch.flip(x, [2, 3])
    x_t       = torch.transpose(x, 2, 3)
    x_t_fh    = torch.flip(x_t, [3])
    x_t_fv    = torch.flip(x_t, [2])
    x_t_fhv   = torch.flip(x_t, [2, 3])
    
    # Stack all 8 into a single mega-batch: (8*B, C, H, W)
    mega_batch = torch.cat([x, x_flip_h, x_flip_v, x_flip_hv,
                            x_t, x_t_fh, x_t_fv, x_t_fhv], dim=0)
    
    # ONE forward pass for all 8 augmentations
    mega_output = _forward_pass(model, mega_batch, use_fp16, is_cuda)
    
    # Split back into 8 chunks of (B, C, H, W)
    o1, o2, o3, o4, o5, o6, o7, o8 = torch.chunk(mega_output, 8, dim=0)
    
    # Un-augment each chunk back to original orientation
    out1 = o1                                                    # Original
    out2 = torch.flip(o2, [3])                                   # Undo flip H
    out3 = torch.flip(o3, [2])                                   # Undo flip V
    out4 = torch.flip(o4, [2, 3])                                # Undo flip HV
    out5 = torch.transpose(o5, 2, 3)                             # Undo transpose
    out6 = torch.transpose(torch.flip(o6, [3]), 2, 3)            # Undo transpose + flip H
    out7 = torch.transpose(torch.flip(o7, [2]), 2, 3)            # Undo transpose + flip V
    out8 = torch.transpose(torch.flip(o8, [2, 3]), 2, 3)         # Undo transpose + flip HV
    
    return torch.mean(torch.stack([out1, out2, out3, out4, out5, out6, out7, out8]), dim=0)

def forward_shift_tta(model, x, use_fp16, is_cuda, shifts=None):
    """
    16x Shift Test-Time Augmentation using a 4x4 grid.
    Since the model is 2x Super Resolution, un-rolling the output requires 2x shift.
    """
    if shifts is None:
        shifts = [(dx, dy) for dx in range(4) for dy in range(4)]
        
    out_accum = 0
    for dx, dy in shifts:
        x_shifted = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
        out_shifted = _forward_pass(model, x_shifted, use_fp16, is_cuda)
        out_unshifted = torch.roll(out_shifted, shifts=(-2*dy, -2*dx), dims=(2, 3))
        out_accum = out_accum + out_unshifted
        
    return out_accum / len(shifts)

def forward_super_tta(model, x, use_fp16, is_cuda, shifts=None):
    """
    Super 128x TTA: Combines 16x Shift with 8x Dihedral Flips.
    Runs 8x Dihedral for EACH of the 16 shifts to avoid memory blowup.
    """
    if shifts is None:
        shifts = [(dx, dy) for dx in range(4) for dy in range(4)]
        
    out_accum = 0
    for dx, dy in shifts:
        x_shifted = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
        # Use batched Dihedral 8x TTA for this shift
        out_shifted_aug = forward_tta(model, x_shifted, use_fp16, is_cuda)
        # Unroll by 2x due to Super Resolution
        out_unshifted = torch.roll(out_shifted_aug, shifts=(-2*dy, -2*dx), dims=(2, 3))
        out_accum = out_accum + out_unshifted
        
    return out_accum / len(shifts)

def run_inference(input_dir, output_dir, checkpoint_path, device=None, base_channels=32, batch_size=8, use_fp16=True, use_compile=False, tta_mode="dihedral8x"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    is_cuda = device.type == "cuda"
    model_type = "nafnet"
    print(f"[INFO] Using device: {device} | Model: {model_type.upper()}")

    # ── 1. Load model to GPU ONCE ──────────────────────────────────
    model = get_model(model_type, in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()  # BatchNorm uses running stats, Dropout disabled
    print(f"[INFO] Loaded checkpoint from: {checkpoint_path}")

    # ── 2. torch.compile (PyTorch 2.0+ graph optimization) ────────
    if use_compile:
        try:
            model = torch.compile(model)
            print("[INFO] torch.compile() applied - fused graph optimization enabled")
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
    print(f"[INFO] Found {total} images to process (batch_size={batch_size}, fp16={use_fp16 and is_cuda}, TTA={tta_mode})")

    # ── 4. Batched Inference with GPU Normalization ───────────────
    processed = 0
    with torch.no_grad():  # Skip autograd graph entirely — free speed + memory win
        for batch_start in range(0, total, batch_size):
            batch_files = test_files[batch_start : batch_start + batch_size]
            
            # Load all images in this batch as numpy arrays
            raw_arrays = [np.load(fp) for fp in batch_files]
            
            # Stack into a single batch tensor and move to GPU immediately
            batch_np = np.stack([a[np.newaxis, np.newaxis, ...] for a in raw_arrays], axis=0)  # (B, 1, 1, H, W)
            batch_tensor = torch.from_numpy(batch_np.squeeze(axis=1)).float()  # (B, 1, H, W)
            batch_gpu = batch_tensor.to(device)
            
            # Normalize each image in the batch using GPU-accelerated quantiles
            batch_params = []  # Store (min_val, max_val) for denormalization
            for i in range(batch_gpu.shape[0]):
                img = batch_gpu[i:i+1]  # (1, 1, H, W) — keep dims
                flat = img[..., ::4, ::4].contiguous().view(-1)
                min_val = torch.quantile(flat, 0.01).item()
                max_val = torch.quantile(flat, 0.99).item()
                batch_gpu[i] = torch.clamp(batch_gpu[i], min_val, max_val)
                batch_gpu[i] = (batch_gpu[i] - min_val) / (max_val - min_val + 1e-8)
                batch_params.append((min_val, max_val))
            
            # Forward pass (with selected TTA mode)
            if tta_mode == "super128x":
                batch_output = forward_super_tta(model, batch_gpu, use_fp16, is_cuda)
            elif tta_mode == "shift16x":
                batch_output = forward_shift_tta(model, batch_gpu, use_fp16, is_cuda)
            elif tta_mode == "dihedral8x":
                batch_output = forward_tta(model, batch_gpu, use_fp16, is_cuda)
            else:
                batch_output = _forward_pass(model, batch_gpu, use_fp16, is_cuda)
            
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

    print(f"\n[INFO] Inference complete. {total} images saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Inference Script for KLA Restoration Model")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input test directory containing degraded images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save restored .npy outputs")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pth model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--base_channels", type=int, default=32, help="Base channel count for NAFNet (default: 32)")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size (default: 8)")
    parser.add_argument("--no_fp16", action="store_true", help="Disable FP16 mixed precision inference")
    parser.add_argument("--tta_mode", type=str, default="dihedral8x", choices=["none", "dihedral8x", "shift16x", "super128x"], help="TTA mode to use (default: dihedral8x)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile() graph optimization (PyTorch 2.0+)")
    
    args = parser.parse_args()
    
    run_inference(
        args.input_dir, args.output_dir, args.checkpoint,
        device=args.device, base_channels=args.base_channels,
        batch_size=args.batch_size, use_fp16=not args.no_fp16, use_compile=args.compile, tta_mode=args.tta_mode
    )
