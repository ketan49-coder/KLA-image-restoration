# KLA Image Restoration Hackathon

This repository contains the inference and training pipeline for our 2x Super-Resolution Image Restoration model.

## 🚀 1. Setup Instructions

The codebase is built on PyTorch and requires minimal dependencies.

### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/ketan49-coder/KLA-image-restoration.git
cd KLA-image-restoration
pip install -r requirements.txt
```

---

## ⚡ 2. Evaluation / Inference (Mandatory Requirement)

To run the standalone inference script exactly as required by the benchmarking team, use `run.py`.

The script accepts a folder of degraded `.npy` images, processes them through the model using **8x Test-Time Augmentation (TTA)** for maximum PSNR/SSIM, and saves the restored 256x256 `.npy` images to the output directory.

### Inference Command
```bash
python run.py <input-dir> <output-dir>
```

### Arguments:
- `<input-dir>`: Path to the directory containing degraded 128x128 images.
- `<output-dir>`: Path where the restored 256x256 images will be saved.

*(Note: The script automatically loads the pre-trained checkpoint from `models/final_model.pth` and applies 8x Test-Time Augmentation and FP16 Mixed Precision inference under the hood to maximize quality while staying under the 10-second hardware penalty rule).*

---

## 🧠 3. Model Architecture & Training Details

This repository implements the SOTA restoration architecture explicitly engineered to balance **Speed** (passing the 10-second penalty rule) and **Quality**:

1. **NAFNet (Nonlinear Activation Free Network):** Achieves Transformer-level PSNR while retaining CNN inference speeds. Our champion architecture.

### The Quad-Fidelity Loss Function
Our models were trained using a custom `QuadFidelityLoss` function designed specifically to eliminate the "high PSNR but blurry" failure mode. It combines four distinct penalties:
1. **Charbonnier Loss:** Robust L1 for handling severe Speckle noise outliers without exploding gradients.
2. **MS-SSIM Loss:** Enforces structural integrity and perceptually accurate textures.
3. **Focal Frequency Loss:** Forces the network to learn both low and high-frequency details.
4. **Edge Loss (Sobel):** Explicitly penalizes blurry boundaries.

Furthermore, checkpoints were evaluated and selected using a custom `CompositeScorer` that weights PSNR (40%), SSIM (35%), and LPIPS (25%), guaranteeing the submitted model is visually and mathematically elite.

### Reproducing the Training
To reproduce our training pipeline from scratch:
```bash
python trainn.py --data_dir /path/to/train --epochs 1600
```
