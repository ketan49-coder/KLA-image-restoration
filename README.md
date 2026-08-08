# KLA Image Restoration — SEMICON India Hackathon 2026

**Track 1 (KLA-sponsored): AI-Based Restoration of Degraded Semiconductor Inspection Images**

**Team:** Ketan Shinde, Rikhil Vaswani, Aditya Jagtap  


---

## Problem Statement

KLA's semiconductor inspection tools face a fundamental trade-off between imaging speed and
image quality — faster scans produce noisier, lower-resolution images, which can hide real
defects. This project trains a deep learning model to reverse three types of image degradation
simultaneously:

1. **Speckle Noise** — grainy, multiplicative noise that can push pixel values beyond the
   true image range.
2. **Gaussian Noise / Blur** — softening of edges and fine structures.
3. **Spatial Resolution Reduction** — downsampling (512→256 or 256→128) requiring
   super-resolution to reverse.

The model must handle all three degradations jointly (a single image may combine any/all of
them), generalize to out-of-distribution test images, and run fast enough for practical
inspection-line throughput.

---

## Approach

We use a **U-Net convolutional neural network** with:
- Symmetric encoder-decoder structure with skip connections
- 4 encoder blocks with max pooling for downsampling
- Bottleneck layer at 1024 channels
- 4 decoder blocks with transposed convolutions for upsampling
- Skip connections concatenating encoder and decoder features
- Batch normalization and ReLU activations throughout

**Loss function**: Combined L1 + MSE loss for training stability and quality.

The U-Net architecture is well-suited for this image restoration task due to its proven effectiveness in medical imaging and ability to preserve spatial information through skip connections.

---

## Project Structure

```
.
├── dataset.py       # PyTorch Dataset/DataLoader for GT/NoisyLR .npy pairs
├── model.py         # U-Net architecture implementation
├── losses.py        # Combined L1 + MSE loss function
├── trainn.py        # Training loop with optimizer and checkpointing
├── eval.py          # Evaluation script: PSNR, SSIM, LPIPS, inference speed
├── requirements.txt # Python dependencies
└── README.md        # This file
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/ketan49-coder/KLA-image-restoration.git
cd KLA-image-restoration
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
The KLA-provided dataset is **not included in this repository** (too large for git).
Download it from [KLA's dataset link — TODO: add actual link] and place it so the
structure looks like:

```
train/
└── train/
    ├── GT/          # 3200 ground-truth (clean, full-resolution) .npy images
    └── NoisyLR/     # 3200 degraded (noisy, low-resolution) .npy images, matching filenames

Test_NoisyLR/
└── NoisyLR/         # Test set degraded images
```

### 4. Train the model
```bash
python trainn.py
```
This will:
- Train for 1 epoch on the dataset
- Save model weights to `checkpoints/unet_model.pth`
- Display loss progress every 100 batches

### 5. Evaluate
```bash
python eval.py --checkpoint checkpoints/unet_model.pth --data_dir ./train/train
```
This computes PSNR, SSIM, LPIPS metrics and inference speed.

---

## Results

*TODO: fill in once final training run is complete*

| Metric | Value |
|---|---|
| PSNR | TBD |
| SSIM | TBD |
| LPIPS | TBD |
| Avg. inference time | TBD ms/image |

---

## References

1. Patent US 12,511,720 — "Image denoising for examination of a semiconductor specimen"
2. Zhao et al. — "Loss Functions for Image Restoration with Neural Networks" (arXiv:1511.08861)
3. "Unleashing Degradation-Carrying Features in Symmetric U-Net: Simpler and Stronger
   Baselines for All-in-One Image Restoration" (arXiv, Dec 2025)
4. SAR image despeckling / ID-CNN — multiplicative noise handling via log-transform

---

## Team

- **Ketan Shinde** — 
- **Aditya Jagtap** — 
