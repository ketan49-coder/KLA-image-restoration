# 🧪 Stage 2 — Loss Function Ablation Study

## Experiment Scorecard

| Run # | Loss Function | Best PSNR (dB) | Best SSIM | Best LPIPS (↓) | Best Epoch | Status / Key Insight |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | Baseline (L1 + 0.1×MSE) | 22.56 | 0.6505 | 0.3770 | — | Stage 1 reference anchor |
| **2** | Pure L1 (MAE) | 22.54 | 0.6635 | — | 2 | Stable, median-seeking, slight blur |
| **3** | Pure L2 (MSE) | 22.89 | 0.6807 | 0.3561 | 6 | Direct PSNR optimization |
| **4** | Pure SSIM Loss | 22.96 | **0.7090** | 0.3517 | 6 | Broke 0.70 SSIM barrier |
| **5** | Pure MS-SSIM Loss | **23.33** 🏆 | 0.7063 | **0.3319** 🏆 | 6 | Multi-scale structural pyramid leader |
| **6** | Hybrid: L1 + SSIM | 22.71 | 0.6928 | 0.3511 | 9 | Beats pure L1; single-scale limits ceiling |
| **7** | Zhao Paper (α=0.025) | 23.15 | 0.6814 | 0.3475 | 6 | Good PSNR; 2.5% MS-SSIM too weak for SEM edges |
| **8** | **Zhao SEM / L1+MS-SSIM (α=0.15)** | — | — | — | — | *Pending — Next Up! (Boosted 15% MS-SSIM)* |
| **9** | Pure GFL (2D FFT) | — | — | — | — | *Pending* |
| **10**| Full Compound (Zhao+GFL) | — | — | — | — | *Pending* |

---

## 🔬 Special Technical Analysis: Why the Zhao Paper Ratio (2.5%) Underperformed on SEM

### The Discovery
Zhao et al. (2017) designed their original loss with **$\alpha = 0.025$** (97.5% Gaussian-L1 + 2.5% MS-SSIM) for standard RGB photography (Kodak dataset, natural portraits, landscapes).

### Why Natural Photos $\neq$ SEM Semiconductor Images:
1. **Natural Photos:** Have smooth color gradients, soft shadows, and gradual textures. A tiny 2.5% structural penalty is sufficient to prevent blur without disrupting color balance.
2. **Semiconductor SEM Images:** Contain **sharp, abrupt step-function edges** (metal tracks, contact vias, etched silicon borders) against flat dark backgrounds.
3. In Run 7, the 97.5% L1 dominance overpowered the tiny 2.5% MS-SSIM term, capping SSIM at **0.6814** (well below pure MS-SSIM's **0.7063**).

**The Solution in Run 8 (Zhao SEM):** Boost $\alpha$ to **$0.15$ (15% MS-SSIM + 85% Gaussian-L1)** to give circuit edge geometry 6× stronger gradient priority!

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
**Result:** Best PSNR = 22.71 dB (Epoch 9) | Best SSIM = 0.6928 (Epoch 9) | Best LPIPS = 0.3511

---

## Run 7 — Zhao Paper Default (α=0.025)

**Command:**
```bash
!python trainn.py --stage stage_2 --loss zhao_paper --run_number 6 --epochs 9 --use_drive \
    --data_dir "/content/train/train"
```

### Epoch-by-Epoch Results

| Epoch | Train Loss | Val Loss | Val PSNR (dB) | Val SSIM | Val LPIPS (↓) | LR | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.0976 | 0.0717 | 22.2021 | 0.6444 | 0.4148 | 0.0010 | |
| 2 | 0.0831 | 0.0662 | 22.7691 | 0.6697 | 0.3751 | 0.0010 | 🌟 Early Best |
| 3 | 0.0790 | 0.0858 | 20.7959 | 0.6049 | 0.3989 | 0.0010 | (Bad #1) |
| 4 | 0.0782 | 0.0678 | 22.6248 | 0.6568 | 0.3690 | 0.0010 | (Bad #2) |
| 5 | 0.0764 | 0.0677 | 22.6983 | 0.6800 | 0.3689 | 0.0010 | (Bad #3 $\rightarrow$ **Cut LR to 0.0005**) |
| 6 | 0.0714 | **0.0649** | **23.1517** 🌟 | 0.6780 | 0.3568 | **0.0005** | 🌟 **BEST PSNR (23.15 dB)** |
| 7 | 0.0705 | 0.0660 | 23.0514 | **0.6814** 🌟 | 0.3506 | 0.0005 | 🌟 **BEST SSIM (0.6814)** |
| 8 | 0.0705 | 0.0646 | 23.0771 | 0.6739 | 0.3547 | 0.0005 | |
| 9 | 0.0702 | 0.0661 | 22.9986 | 0.6794 | **0.3475** 🌟 | 0.0005 | 🌟 **BEST LPIPS (0.3475)** |

---

## Run 8 — Zhao SEM-Tuned (α=0.15) / L1 + MS-SSIM

*Results pending — Next Up!*

### What Zhao SEM-Tuned Does
- Formula: $\mathcal{L} = 0.15 \cdot \mathcal{L}_{\text{MS-SSIM}} + 0.85 \cdot \mathcal{L}_{G\text{-}L1}$
- 6× higher structural weight than standard RGB photography (15% vs 2.5%).
- Specifically tailored to recover sharp step-function semiconductor edges while maintaining Gaussian-L1 photometric stability.

---

## Run 9 — Guided Frequency Loss (GFL)
*Pending*

## Run 10 — Full Compound Loss (Zhao Mix + GFL)
*Pending*
