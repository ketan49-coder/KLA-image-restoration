import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import random


def normalize_image(img_tensor, min_val=None, max_val=None):
    """
    Standard Min-Max normalization.
    If min_val and max_val are provided, uses those instead of computing them from the tensor.
    """
    if min_val is None:
        min_val = img_tensor.min()
    if max_val is None:
        max_val = img_tensor.max()
    return (img_tensor - min_val) / (max_val - min_val + 1e-8), min_val, max_val


def denormalize_image(norm_tensor, min_val, max_val):
    """
    Invert Min-Max normalization back to original dynamic range.
    """
    return (norm_tensor * (max_val - min_val + 1e-8)) + min_val


class ImageRestorationDataset(Dataset):
    def __init__(self, data_dir, split_ratio=0.9, is_val=False, split=None, preload_to_ram=False):
        """
        Dataset loader for KLA Image Restoration pairs (.npy).
        Supports train/val splitting, auto-discovery of GT/NoisyLR directories,
        and optional RAM preloading.
        """
        if split is not None:
            is_val = (split.lower() == 'val' or split.lower() == 'validation')
        self.is_val = is_val

        # Auto-detect GT folder
        possible_gt = glob.glob(os.path.join(data_dir, '**/GT'), recursive=True)
        self.gt_dir = possible_gt[0] if possible_gt else os.path.join(data_dir, 'GT')
        self.noisy_dir = self.gt_dir.replace('GT', 'NoisyLR')

        all_files = sorted([f for f in os.listdir(self.gt_dir) if f.endswith('.npy')])
        split_idx = int(len(all_files) * split_ratio)
        self.files = all_files[split_idx:] if is_val else all_files[:split_idx]

        self.preload_to_ram = preload_to_ram
        self.noisy_cache = []
        self.gt_cache = []

        if self.preload_to_ram:
            split_name = "Validation" if is_val else "Training"
            print(f"⚡ Preloading {len(self.files)} {split_name} images into RAM...")
            for f in self.files:
                gt_arr = np.load(os.path.join(self.gt_dir, f))
                noisy_arr = np.load(os.path.join(self.noisy_dir, f))

                gt_t = torch.from_numpy(gt_arr).float().unsqueeze(0)
                noisy_t = torch.from_numpy(noisy_arr).float().unsqueeze(0)

                # GT-Anchored Normalization
                gt_t, gt_min, gt_max = normalize_image(gt_t)
                noisy_t, _, _ = normalize_image(noisy_t, min_val=gt_min, max_val=gt_max)

                self.gt_cache.append(gt_t)
                self.noisy_cache.append(noisy_t)
            print(f"✓ {split_name} set cached in RAM!")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.preload_to_ram:
            noisy, gt = self.noisy_cache[idx], self.gt_cache[idx]
        else:
            filename = self.files[idx]
            gt = np.load(os.path.join(self.gt_dir, filename))
            noisy = np.load(os.path.join(self.noisy_dir, filename))

            # Add channel dimension: (1, H, W)
            gt = torch.from_numpy(gt).float().unsqueeze(0)
            noisy = torch.from_numpy(noisy).float().unsqueeze(0)

            # GT-Anchored Normalization
            gt, gt_min, gt_max = normalize_image(gt)
            noisy, _, _ = normalize_image(noisy, min_val=gt_min, max_val=gt_max)

        if not self.is_val:
            # Random horizontal flip
            if random.random() > 0.5:
                noisy = torch.flip(noisy, [2])
                gt = torch.flip(gt, [2])
            
            # Random vertical flip
            if random.random() > 0.5:
                noisy = torch.flip(noisy, [1])
                gt = torch.flip(gt, [1])

            # Random 90-degree rotation
            k = random.randint(0, 3)
            if k > 0:
                noisy = torch.rot90(noisy, k, [1, 2])
                gt = torch.rot90(gt, k, [1, 2])

        return noisy, gt

# Alias for backwards compatibility
RestorationDataset = ImageRestorationDataset