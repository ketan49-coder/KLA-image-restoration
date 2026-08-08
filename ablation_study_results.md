# 🧪 Stage 2 — Loss Function Ablation Study

## Experiment Scorecard

| Run # | Loss Function | Best PSNR (dB) | Best SSIM | Best LPIPS | Best Epoch | Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | Baseline (L1 + 0.1×MSE) | 22.56 | 0.6505 | 0.3770 | — | Stage 1 reference anchor |
| **2** | Pure L1 | 22.54 | 0.6635 | — | 2 | Stable, median-seeking, slight blur |
| **3** | Pure L2 (MSE) | **22.89** ⭐ | **0.6807** ⭐ | **0.3561** ⭐ | 6 | Directly optimizes PSNR formula; peak at Ep 6 |
| **4** | Pure SSIM Loss | — | — | — | — | *Pending* |
| **5** | Pure MS-SSIM Loss | — | — | — | — | *Pending* |
| **6** | Hybrid: L1 + SSIM | — | — | — | — | *Pending* |
| **7** | Zhao Paper (α=0.025) | — | — | — | — | *Pending* |
| **8** | Zhao SEM / L1+MS-SSIM | — | — | — | — | *Pending* |
| **9** | Pure GFL (2D FFT) | — | — | — | — | *Pending* |
| **10**| Full Compound (Zhao+GFL) | — | — | — | — | *Pending* |

---

## 🔬 Special Technical Analysis: Why the LR Scheduler Didn't Trigger in Run 3 (L2)

### The Observation
In Run 3 (L2), training ran for 7 epochs and LR stayed at `0.001000` throughout.

### Why It Didn't Cut LR (The "Reset" Rule):
- **Epoch 1**: Val PSNR = 20.62 dB $\rightarrow$ Baseline
- **Epoch 2**: Val PSNR = 21.42 dB $\rightarrow$ **Improvement** (`bad = 0`)
- **Epoch 3**: Val PSNR = 22.56 dB $\rightarrow$ **Improvement** (`bad = 0`)
- **Epoch 4**: Val PSNR = 22.54 dB $\rightarrow$ Bad epoch #1 (`bad = 1`)
- **Epoch 5**: Val PSNR = 22.68 dB $\rightarrow$ **Improvement!** (`22.68 > 22.56`) $\rightarrow$ `bad` counter reset to `0`!
- **Epoch 6**: Val PSNR = 22.89 dB $\rightarrow$ **New Best!** (`22.89 > 22.68`) $\rightarrow$ `bad` counter reset to `0`!
- **Epoch 7**: Val PSNR = 22.00 dB $\rightarrow$ Bad epoch #1 (`bad = 1`).

**Conclusion**: The scheduler did **not** fail — the model kept legitimately improving at Epoch 5 and 6, resetting the patience counter each time!

---

## Run 1 — Baseline: L1 + 0.1 × MSE

**Command:** *Ran during Stage 1 (prior session)*
**Result:** PSNR = 22.56 dB | SSIM = 0.6505 | LPIPS = 0.3770

### What This Loss Does
- Combines absolute pixel error (L1) with a small squared error penalty (0.1 × MSE)
- Formula: $\mathcal{L} = |pred - gt| + 0.1 \times (pred - gt)^2$

---

## Run 2 — Pure L1 (MAE)

**Command:**
```bash
!python trainn.py --stage stage_2 --loss l1 --run_number 1 --epochs 5 --use_drive \
    --data_dir "/content/train/train"
```

### Results Summary
**Best PSNR:** 22.54 dB (Epoch 2) | **Best SSIM:** 0.6635 (Epoch 4) | **Best Epoch:** 2

---

## Run 3 — Pure L2 (MSE)

**Command:**
```bash
!python trainn.py --stage stage_2 --loss l2 --run_number 2 --epochs 7 --use_drive \
    --data_dir "/content/train/train"
```

### Epoch-by-Epoch Results

| Epoch | Train Loss | Val Loss | Val PSNR (dB) | Val SSIM | Val LPIPS (↓) | LR | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.0173 | 0.0134 | 20.62 | 0.6164 | 0.4005 | 0.001 | |
| 2 | 0.0122 | 0.0112 | 21.42 | 0.6284 | 0.3779 | 0.001 | |
| 3 | 0.0108 | 0.0093 | 22.56 | 0.6706 | 0.3641 | 0.001 | 🌟 Best |
| 4 | 0.0106 | 0.0094 | 22.54 | 0.6653 | 0.3696 | 0.001 | (Bad #1) |
| 5 | 0.0104 | 0.0093 | 22.68 | 0.6688 | 0.3631 | 0.001 | 🌟 Best (Reset) |
| 6 | 0.0098 | **0.0089** | **22.89** ⭐ | **0.6807** ⭐ | **0.3561** ⭐ | 0.001 | 🌟 ALL-TIME BEST |
| 7 | 0.0096 | 0.0104 | 22.00 | 0.6221 | 0.3688 | 0.001 | Overfitting plunge |

### Key Observations & Insights
1. **Direct PSNR Optimization:** Because $\text{PSNR} = 10 \cdot \log_{10}(1/\text{MSE})$, optimizing pure MSE directly targets the PSNR metric formula, driving PSNR to **22.89 dB (+0.33 dB over baseline)**.
2. **Perceptual Improvement via LPIPS:** LPIPS dropped to **0.3561** (lower is better, beating baseline 0.3770).
3. **Severe Overfitting at Epoch 7:** 
   - Training loss dropped from 0.0098 to 0.0096.
   - But Validation loss surged from 0.0089 to 0.0104.
   - Validation SSIM plunged drastically from 0.6807 to 0.6221 (loss of 0.058 in structure in a single epoch!).
4. **Checkpoint Safety:** The training script successfully saved `checkpoints/stage_2_l2_run2_best.pth` at **Epoch 6**, shielding our final model from the Epoch 7 overfitting disaster.

---

## Run 4 — Pure SSIM Loss

*Results pending — Next Up!*

### What SSIM Loss Does
- Formula: $\mathcal{L}_{SSIM} = 1 - \text{SSIM}(pred, gt)$
- Directly optimizes local contrast, luminance, and covariance across 11×11 sliding patches.
- Unlike L1/L2 which treat pixels independently, SSIM explicitly penalizes edge blurring.

---

## Run 5 — Pure MS-SSIM Loss
*Pending*

## Run 6 — Hybrid: L1 + SSIM
*Pending*

## Run 7 — Zhao Paper Default (α=0.025)
*Pending*

## Run 8 — Zhao SEM-Tuned (α=0.15)
*Pending*

## Run 9 — Guided Frequency Loss (GFL)
*Pending*

## Run 10 — Full Compound Loss (Zhao Mix + GFL)
*Pending*
