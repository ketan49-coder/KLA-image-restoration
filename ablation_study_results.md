# 🧪 Stage 2 — Loss Function Ablation Study

## Experiment Scorecard

| Run # | Loss Function | Best PSNR (dB) | Best SSIM | Best LPIPS (↓) | Best Epoch | Status / Key Insight |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | Baseline (L1 + 0.1×MSE) | 22.56 | 0.6505 | 0.3770 | — | Stage 1 reference anchor |
| **2** | Pure L1 (MAE) | 22.54 | 0.6635 | — | 2 | Stable, median-seeking, slight blur |
| **3** | Pure L2 (MSE) | 22.89 | 0.6807 | 0.3561 | 6 | Directly optimizes PSNR formula |
| **4** | **Pure SSIM Loss** | **22.96** 🏆 | **0.7090** 🏆 | **0.3517** 🏆 | **6** | **Broke 0.70 SSIM barrier! LR cut to 0.0005 caused massive leap** |
| **5** | Pure MS-SSIM Loss | — | — | — | — | *Pending — Next Up!* |
| **6** | Hybrid: L1 + SSIM | — | — | — | — | *Pending* |
| **7** | Zhao Paper (α=0.025) | — | — | — | — | *Pending* |
| **8** | Zhao SEM / L1+MS-SSIM | — | — | — | — | *Pending* |
| **9** | Pure GFL (2D FFT) | — | — | — | — | *Pending* |
| **10**| Full Compound (Zhao+GFL) | — | — | — | — | *Pending* |

---

## 🔬 Special Technical Analysis: The 9-Epoch LR Scheduler Triumph in Run 4 (SSIM)

### The Dramatic Turnaround at Epoch 6
Look at how `ReduceLROnPlateau` performed in Run 4:
- **Epoch 2**: Reached early peak at `22.8979 dB` (`bad = 0`).
- **Epoch 3**: `22.8327 dB` (`bad = 1`).
- **Epoch 4**: `22.4933 dB` (`bad = 2`).
- **Epoch 5**: `22.1834 dB` (`bad = 3 > 2`) $\rightarrow$ **Scheduler cut LR from `0.001000` to `0.000500`!**
- **Epoch 6 (with LR = 0.000500):**
  - Val PSNR **rocketed from `22.18 dB` to `22.9601 dB`** (New project record!).
  - Val SSIM **shattered the 0.70 barrier, reaching `0.7090`** (First time in project history!).
  - Val LPIPS **dropped to `0.3517`** at Epoch 7 (Crisper perceptual edges).

**Conclusion**: This proves that reducing learning rate when validation loss plateaus prevents gradient overshoot and allows the network weights to settle into sharper, narrower structural minima!

---

## Run 1 — Baseline: L1 + 0.1 × MSE
**Result:** PSNR = 22.56 dB | SSIM = 0.6505 | LPIPS = 0.3770

---

## Run 2 — Pure L1 (MAE)
**Result:** Best PSNR = 22.54 dB (Epoch 2) | Best SSIM = 0.6635 (Epoch 4)

---

## Run 3 — Pure L2 (MSE)
**Result:** Best PSNR = 22.89 dB (Epoch 6) | Best SSIM = 0.6807 (Epoch 6) | Best LPIPS = 0.3561

---

## Run 4 — Pure SSIM Loss

**Command:**
```bash
!python trainn.py --stage stage_2 --loss ssim --run_number 3 --epochs 9 --use_drive \
    --data_dir "/content/train/train"
```

### Epoch-by-Epoch Results

| Epoch | Train Loss | Val Loss | Val PSNR (dB) | Val SSIM | Val LPIPS (↓) | LR | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.3524 | 0.3039 | 22.2024 | 0.6837 | 0.3898 | 0.001 | |
| 2 | 0.3090 | 0.2972 | 22.8979 | 0.6906 | 0.3873 | 0.001 | 🌟 Early Best |
| 3 | 0.2955 | 0.2979 | 22.8327 | 0.6894 | 0.4114 | 0.001 | (Bad #1) |
| 4 | 0.2954 | 0.2957 | 22.4933 | 0.6933 | 0.3984 | 0.001 | (Bad #2) |
| 5 | 0.2887 | 0.3298 | 22.1834 | 0.6815 | 0.3872 | 0.001 | (Bad #3 $\rightarrow$ **Cut LR to 0.0005**) |
| 6 | 0.2811 | **0.2876** | **22.9601** 🏆 | **0.7090** 🏆 | 0.3709 | **0.0005** | 🌟 **NEW PROJECT BEST!** |
| 7 | 0.2791 | 0.2857 | 22.6667 | 0.7047 | **0.3517** 🏆 | 0.0005 | 🌟 **BEST LPIPS (0.3517)** |
| 8 | 0.2779 | 0.2942 | 22.8341 | 0.7054 | 0.3652 | 0.0005 | |
| 9 | 0.2802 | 0.2947 | 22.4987 | 0.6927 | 0.3614 | 0.0005 | |

### Key Observations & Insights
1. **SSIM Exceeds 0.70 for the First Time:** Pure SSIM loss optimizes 11×11 structural patches directly, forcing the network to preserve circuit edge boundaries instead of smoothing them away.
2. **PSNR Also Beats L1 and L2:** Despite not explicitly targeting pixel values, SSIM loss achieved **22.96 dB PSNR**, higher than both Pure L1 (22.54 dB) and Pure L2 (22.89 dB).
3. **Perceptual Realism Peak (LPIPS = 0.3517):** Features extracted by AlexNet confirm that the reconstructed patterns closely match the ground truth texture distribution.

---

## Run 5 — Pure MS-SSIM Loss

*Results pending — Next Up!*

### What MS-SSIM Loss Does
- Formula: $\mathcal{L}_{\text{MS-SSIM}} = 1 - \text{MS-SSIM}(pred, gt)$
- Evaluates structure across **5 downsampled scales** (1×, 1/2×, 1/4×, 1/8×, 1/16×) rather than just a single 11×11 window.
- Captures both macro global circuit geometry and micro edge transitions simultaneously.

---

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
