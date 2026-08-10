"""
pack_dataset.py
Packs all .npy image pairs (GT + NoisyLR) into a single .pt file for instant loading.
Run this ONCE. After that, use --packed_data to load in seconds.

Usage (in Colab):
    !python pack_dataset.py --data_dir /content/train --output /content/drive/MyDrive/dataset_packed.pt
"""
import os
import glob
import argparse
import numpy as np
import torch
from tqdm import tqdm


def find_gt_dir(data_dir):
    """Auto-detect GT folder (same logic as dataset.py)."""
    if os.path.exists(os.path.join(data_dir, 'GT')):
        return os.path.join(data_dir, 'GT')
    candidates = [p for p in glob.glob(os.path.join(data_dir, '**/GT'), recursive=True) if '__MACOSX' not in p]
    if candidates:
        return candidates[0]
    return os.path.join(data_dir, 'GT')


def pack(data_dir, output_path):
    gt_dir = find_gt_dir(data_dir)
    noisy_dir = gt_dir.replace('GT', 'NoisyLR')

    print(f"[PACK] GT folder:      {gt_dir}")
    print(f"[PACK] NoisyLR folder: {noisy_dir}")

    files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.npy')])
    print(f"[PACK] Found {len(files)} image pairs")

    gt_list = []
    noisy_list = []

    for f in tqdm(files, desc="Packing"):
        gt_arr = np.load(os.path.join(gt_dir, f))
        noisy_arr = np.load(os.path.join(noisy_dir, f))
        gt_list.append(torch.from_numpy(gt_arr).float().unsqueeze(0))
        noisy_list.append(torch.from_numpy(noisy_arr).float().unsqueeze(0))

    packed = {
        'gt': gt_list,
        'noisy': noisy_list,
        'files': files,
    }

    print(f"[PACK] Saving to {output_path} ...")
    torch.save(packed, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"✓ Packed dataset saved! ({size_mb:.1f} MB, {len(files)} pairs)")
    print(f"\nTo use: !python trainn.py --packed_data {output_path} ...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pack .npy dataset into a single .pt file")
    parser.add_argument("--data_dir", type=str, required=True, help="Root data directory containing GT/NoisyLR")
    parser.add_argument("--output", type=str, default="dataset_packed.pt", help="Output .pt file path")
    args = parser.parse_args()
    pack(args.data_dir, args.output)
