# KLA Image Restoration — SEMICON India Hackathon 2026

**Track 1 (KLA-sponsored): AI-Based Restoration of Degraded Semiconductor Inspection Images**

Team: [Your Name], Rikhil [Last Name]
Institution: DES Pune

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

We use a **U-Net-style convolutional neural network** with:
- Symmetric encoder-decoder structure with skip connections (inspired by recent findings
  that well-designed U-Nets match or exceed complex all-in-one restoration frameworks —
  see [References](#references))
- **Residual/noise prediction**: the model predicts the degradation residual rather than
  the clean image directly, subtracting it from the (upsampled) input — a technique
  confirmed in industrial practice (see US Patent 12,511,720)
- An upsampling head (pixel-shuffle / transpose convolution) to handle the
  super-resolution component
- LeakyReLU activations throughout the encoder/decoder

**Loss function**: combined L1 + MS-SSIM loss, chosen based on empirical evidence that this
combination outperforms L1, L2, or SSIM alone across standard image quality metrics
(see [References](#references)).

We deliberately chose a task-specific CNN/U-Net over more complex "all-in-one" restoration
architectures (e.g., CoRE-UIR, QuReC) after evaluating the domain mismatch risk (those
architectures are trained on natural-photo/remote-sensing domains, not grayscale
semiconductor imagery) against our development timeline.

---

## Project Structure

```
.
├── dataset.py       # PyTorch Dataset/DataLoader for GT/NoisyLR .npy pairs
├── model.py         # U-Net architecture with residual prediction + upsampling head
├── train.py         # Training loop (loss function, optimizer, checkpointing)
├── eval.py          # Evaluation script: PSNR, SSIM, LPIPS, inference speed benchmarking
├── utils.py         # Shared helpers (speckle log-transform, normalization, visualization)
├── config.py        # Central hyperparameters and file paths
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
python train.py --config config.py
```

### 5. Evaluate
```bash
python eval.py --checkpoint path/to/checkpoint.pth --data_dir ./train/train
```

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

- [Your Name] — architecture, training
- Rikhil [Last Name] — data pipeline, model implementation
