# 🧪 Stage 2 — Loss Function Ablation Study

## Experiment Scorecard (Matched to Colab Run Numbers)

| Colab Run # | Loss Function | Best PSNR (dB) | Best SSIM | Best LPIPS (↓) | Best Epoch | Status / Key Insight |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Anchor** | Baseline (L1 + 0.1×MSE) | 22.56 | 0.6505 | 0.3770 | — | Stage 1 reference anchor |
| **Run 1** | Pure L1 (MAE) | 22.54 | 0.6635 | — | 2 | Stable, median-seeking, slight blur |
| **Run 2** | Pure L2 (MSE) | 22.89 | 0.6807 | 0.3561 | 6 | Direct PSNR optimization |
| **Run 3** | Pure SSIM Loss | 22.96 | **0.7090** | 0.3517 | 6 | Broke 0.70 SSIM barrier |
| **Run 4** | Pure MS-SSIM Loss | 23.33 | 0.7063 | **0.3319** 🏆 | 6 | Multi-scale pyramid leader |
| **Run 5** | Hybrid: L1 + SSIM | 22.71 | 0.6928 | 0.3511 | 9 | Beats pure L1; single-scale limits ceiling |
| **Run 6** | Zhao Paper (α=0.025) | 23.15 | 0.6814 | 0.3475 | 6 | Good PSNR; 2.5% MS-SSIM too weak for SEM |
| **Run 7** | **Zhao SEM / L1+MS-SSIM (α=0.15)**| **23.34** 🏆 | 0.6936 | 0.3446 | **7** | **NEW ALL-TIME PSNR RECORD (23.34 dB)! (+0.78 dB over baseline)** |
| **Run 8** | Pure GFL (2D FFT) | — | — | — | — | *Pending — Next Up!* |
| **Run 9** | Structure-Dominant (85% MS-SSIM) | — | — | — | — | *Pending (Custom Ratio Analysis)* |
| **Run 10**| Full Compound (Zhao+GFL) | — | — | — | — | *Pending (Flagship Finale)* |

---

## 🔬 Special Technical Analysis: How Zhao SEM-Tuned (Run 7) Set the All-Time PSNR Record

### The Empirical Proof: 2.5% vs 15% MS-SSIM
- **Run 6 (Zhao Paper - 2.5% MS-SSIM):** 23.15 dB PSNR | 0.6814 SSIM | 0.3475 LPIPS
- **Run 7 (Zhao SEM - 15% MS-SSIM):** **23.34 dB PSNR** 🏆 | **0.6936 SSIM** | **0.3446 LPIPS** 🏆

By increasing the structural gradient weight from 2.5% to 15%:
1. **PSNR improved by +0.19 dB** (pushing to 23.34 dB).
2. **SSIM gained +0.0122** in structural edge fidelity.
3. **LPIPS dropped by -0.0029** towards sharper perceptual boundaries.
4. **LR Step Impact:** The learning rate step-down to `0.0005` at Epoch 6 was the exact mechanism that powered the jump from 22.66 dB $\rightarrow$ 23.06 dB $\rightarrow$ **23.34 dB at Epoch 7**!

---

## Anchor — Baseline: L1 + 0.1 × MSE
**Result:** PSNR = 22.56 dB | SSIM = 0.6505 | LPIPS = 0.3770

---

## Run 1 — Pure L1 (MAE)
**Result:** Best PSNR = 22.54 dB (Epoch 2) | Best SSIM = 0.6635 (Epoch 4)

---

## Run 2 — Pure L2 (MSE)
**Result:** Best PSNR = 22.89 dB (Epoch 6) | Best SSIM = 0.6807 (Epoch 6) | Best LPIPS = 0.3561

---

## Run 3 — Pure SSIM Loss
**Result:** Best PSNR = 22.96 dB (Epoch 6) | Best SSIM = 0.7090 (Epoch 6) | Best LPIPS = 0.3517

---

## Run 4 — Pure MS-SSIM Loss
**Result:** Best PSNR = 23.33 dB (Epoch 6) | Best SSIM = 0.7063 (Epoch 6) | Best LPIPS = 0.3319

---

## Run 5 — Hybrid: L1 + SSIM
**Result:** Best PSNR = 22.71 dB (Epoch 9) | Best SSIM = 0.6928 (Epoch 9) | Best LPIPS = 0.3511

---

## Run 6 — Zhao Paper Default (α=0.025)
**Result:** Best PSNR = 23.15 dB (Epoch 6) | Best SSIM = 0.6814 (Epoch 7) | Best LPIPS = 0.3475 (Epoch 9)

---

## Run 7 — Zhao SEM-Tuned (α=0.15) / L1 + MS-SSIM

**Command:**
```bash
!python trainn.py --stage stage_2 --loss l1_msssim --run_number 7 --epochs 9 --use_drive \
    --data_dir "/content/train/train"
```

### Epoch-by-Epoch Results

| Epoch | Train Loss | Val Loss | Val PSNR (dB) | Val SSIM | Val LPIPS (↓) | LR | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.1018 | 0.0803 | 21.8168 | 0.6638 | 0.3941 | 0.0010 | |
| 2 | 0.0873 | 0.0704 | 22.8114 | 0.6761 | 0.3746 | 0.0010 | 🌟 Early Best |
| 3 | 0.0825 | 0.0839 | 21.3786 | 0.6299 | 0.3861 | 0.0010 | (Bad #1) |
| 4 | 0.0808 | 0.0720 | 22.5150 | 0.6803 | 0.3705 | 0.0010 | (Bad #2) |
| 5 | 0.0796 | 0.0704 | 22.6687 | 0.6889 | 0.3519 | 0.0010 | (Bad #3 $\rightarrow$ **Cut LR to 0.0005**) |
| 6 | 0.0732 | 0.0686 | 23.0685 | 0.6867 | 0.3455 | **0.0005** | 🌟 Step-down surge |
| 7 | 0.0726 | **0.0664** | **23.3368** 🏆 | **0.6936** 🌟 | **0.3446** 🌟 | 0.0005 | 🌟 **ALL-TIME PSNR RECORD (23.34 dB)** |
| 8 | 0.0728 | 0.0680 | 23.0347 | 0.6917 | 0.3478 | 0.0005 | |
| 9 | 0.0720 | 0.0712 | 22.7575 | 0.6868 | 0.3510 | 0.0005 | |

---

## Run 8 — Pure Guided Frequency Loss (GFL via 2D FFT)

*Results pending — Next Up!*

### What GFL Loss Does
- Formula: $\mathcal{L}_{\text{GFL}} = \frac{1}{HW} \sum | \mathcal{F}(pred) - \mathcal{F}(gt) |$
- Computes the 2D Fast Fourier Transform into the frequency domain.
- Evaluates spatial frequency power, specifically penalizing high-frequency attenuation where subtle edge blur occurs.

---

## Run 9 — Structure-Dominant (85% MS-SSIM + 15% L1)
*Pending (Custom Ratio Analysis)*

## Run 10 — Full Compound Loss (Zhao Mix + GFL)
*Pending (Flagship Finale)*
