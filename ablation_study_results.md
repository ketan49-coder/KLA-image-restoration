# 🧪 Stage 2 — Loss Function Ablation Study

## Experiment Scorecard

| Run # | Loss Function | Best PSNR (dB) | Best SSIM | Best LPIPS (↓) | Best Epoch | Status / Key Insight |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | Baseline (L1 + 0.1×MSE) | 22.56 | 0.6505 | 0.3770 | — | Stage 1 reference anchor |
| **2** | Pure L1 (MAE) | 22.54 | 0.6635 | — | 2 | Stable, median-seeking, slight blur |
| **3** | Pure L2 (MSE) | 22.89 | 0.6807 | 0.3561 | 6 | Direct PSNR optimization |
| **4** | Pure SSIM Loss | 22.96 | **0.7090** | 0.3517 | 6 | Broke 0.70 SSIM barrier |
| **5** | **Pure MS-SSIM Loss** | **23.33** 🏆 | 0.7063 | **0.3319** 🏆 | **6** | **CRUSHED 23 dB! (+0.77 dB over baseline)** |
| **6** | Hybrid: L1 + SSIM | 22.71 | 0.6928 | 0.3511 | 9 | Beats pure L1; single-scale limits ceiling |
| **7** | Zhao Paper (α=0.025) | — | — | — | — | *Pending — Next Up!* |
| **8** | Zhao SEM / L1+MS-SSIM | — | — | — | — | *Pending* |
| **9** | Pure GFL (2D FFT) | — | — | — | — | *Pending* |
| **10**| Full Compound (Zhao+GFL) | — | — | — | — | *Pending* |

---

## 🔬 Special Technical Analysis: Single-Scale vs Multi-Scale in Hybrids

### What Run 6 (L1 + SSIM) Taught Us:
1. **L1 + Single-Scale SSIM (22.71 dB, 0.6928 SSIM):**
   - Definitely superior to Pure L1 (22.54 dB, 0.6635 SSIM) and Baseline (22.56 dB, 0.6505 SSIM).
   - Proves that adding structural loss prevents pixel-level blurring.
2. **The Multi-Scale Gap:**
   - Because single-scale SSIM only evaluates an 11×11 window, the 80% L1 weighting pulls the network back toward median pixel regression on larger patterns.
   - Multi-Scale SSIM (Run 5) broke through to **23.33 dB** because its 5-scale pyramid penalizes errors across all spatial frequencies.

**Hypothesis for Next Runs**: Combining L1 with **Multi-Scale SSIM** (Zhao SEM) will provide the missing link — capturing multi-scale structure while maintaining photometric pixel accuracy!

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
**Result:** Best PSNR = 22.96 dB (Epoch 6) | Best SSIM = 0.7090 (Epoch 6) | Best LPIPS = 0.3517

---

## Run 5 — Pure MS-SSIM Loss
**Result:** Best PSNR = 23.33 dB (Epoch 6) | Best SSIM = 0.7063 (Epoch 6) | Best LPIPS = 0.3319

---

## Run 6 — Hybrid: L1 + SSIM

**Command:**
```bash
!python trainn.py --stage stage_2 --loss l1_ssim --run_number 5 --epochs 9 --use_drive \
    --data_dir "/content/train/train"
```

### Epoch-by-Epoch Results

| Epoch | Train Loss | Val Loss | Val PSNR (dB) | Val SSIM | Val LPIPS (↓) | LR | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.1496 | 0.1345 | 21.2201 | 0.6383 | 0.3915 | 0.0010 | |
| 2 | 0.1322 | 0.1199 | 22.4095 | 0.6693 | 0.3926 | 0.0010 | |
| 3 | 0.1254 | 0.1359 | 20.9337 | 0.6447 | 0.4243 | 0.0010 | |
| 4 | 0.1243 | 0.1322 | 21.7975 | 0.6470 | 0.3809 | 0.0010 | |
| 5 | 0.1213 | 0.1138 | 22.4837 | 0.6894 | 0.3675 | 0.0010 | |
| 6 | 0.1191 | 0.1210 | 22.1398 | 0.6745 | 0.3901 | 0.0010 | |
| 7 | 0.1194 | 0.1151 | 22.6733 | 0.6853 | **0.3511** 🏆 | 0.0010 | 🌟 Best LPIPS |
| 8 | 0.1174 | 0.1152 | 22.6917 | 0.6850 | 0.3770 | 0.0010 | |
| 9 | 0.1158 | **0.1129** | **22.7066** 🌟 | **0.6928** 🌟 | 0.3592 | 0.0010 | 🌟 Best PSNR/SSIM |

---

## Run 7 — Zhao Paper Default (α=0.025)

*Results pending — Next Up!*

### What Zhao Paper Loss Does
- Formula: $\mathcal{L} = 0.025 \cdot \mathcal{L}_{\text{MS-SSIM}} + 0.975 \cdot \mathcal{L}_{G\text{-}L1}$
- Exact formulation proposed in Zhao et al. (2017) using Gaussian-weighted L1.
- Evaluates if a subtle multi-scale boost (2.5%) combined with Gaussian-smoothed L1 outperforms standard L1.

---

## Run 8 — Zhao SEM-Tuned (α=0.15) / L1 + MS-SSIM
*Pending*

## Run 9 — Guided Frequency Loss (GFL)
*Pending*

## Run 10 — Full Compound Loss (Zhao Mix + GFL)
*Pending*
