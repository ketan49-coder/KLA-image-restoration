import torch
from torch.utils.data import Dataset
import numpy as np
import os

class ImageRestorationDataset(Dataset):
    def __init__(self, data_dir):
        self.gt_dir = os.path.join(data_dir, 'GT')
        self.noisy_dir = os.path.join(data_dir, 'NoisyLR')
        self.files = sorted([f for f in os.listdir(self.gt_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        gt = np.load(os.path.join(self.gt_dir, filename))
        noisy = np.load(os.path.join(self.noisy_dir, filename))

        # Add channel dimension
        gt = torch.from_numpy(gt).float().unsqueeze(0)
        noisy = torch.from_numpy(noisy).float().unsqueeze(0)

        # Normalize
        gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
        noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min() + 1e-8)

        return noisy, gt
