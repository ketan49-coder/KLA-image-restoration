# 🧪 Stage 2 — Loss Function Ablation Study

## Experiment Scorecard (Matched to Colab Run Numbers)

| Colab Run # | Loss Function | Best PSNR (dB) | Best SSIM | Best LPIPS (↓) | Best Epoch | Status / Key Insight |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Anchor** | Baseline (L1 + 0.1×MSE) | 22.56 | 0.6505 | 0.3770 | — | Stage 1 reference anchor |
| **Run 1** | Pure L1 (MAE) | 22.54 | 0.6635 | — | 2 | Stable, median-seeking, slight blur |
| **Run 2** | Pure L2 (MSE) | 22.89 | 0.6807 | 0.3561 | 6 | Direct PSNR optimization |
| **Run 3** | Pure SSIM Loss | 22.96 | **0.7090** | 0.3517 | 6 | Broke 0.70 SSIM barrier |
| **Run 4** | Pure MS-SSIM Loss | 23.33 | 0.7063 | 0.3319 | 6 | Multi-scale pyramid leader |
| **Run 5** | Hybrid: L1 + SSIM | 22.71 | 0.6928 | 0.3511 | 9 | Single-scale hybrid |
| **Run 6** | Zhao Paper (2.5% MS-SSIM) | 23.15 | 0.6814 | 0.3475 | 6 | 2.5% too weak for SEM |
| **Run 7** | Zhao SEM (15% MS-SSIM) | 23.34 | 0.6936 | 0.3446 | 7 | Photometric balance |
| **Run 8** | Structure-Dominant (85% MS-SSIM)| 23.32 | **0.7071** ⭐ | **0.3317** 🏆 | 8 | Structural & LPIPS Record |
| **Run 9** | Balanced 50/50 (50% MS-SSIM)| **23.35** 🏆 | 0.7013 | 0.3390 | 8 | All-time PSNR record |
| **Run 10**| **Pure GFL (2D FFT)** | 20.84 | 0.5583 | 0.4210 | 8 | Proves spectral loss needs spatial anchor! |
| **Run 11**| **Full Compound (Zhao + GFL)** | — | — | — | — | *Pending — Grand Finale!* |

---

## 🔬 Special Technical Analysis: Why Frequency Loss Cannot Stand Alone

### The Discovery in Run 10:
When trained **purely on 2D Fourier Frequency Loss (GFL)**:
- Val PSNR: **20.84 dB** | Val SSIM: **0.5583** | Val LPIPS: **0.4210**

### The Physics Behind This Result:
1. **Shift Invariance of Fourier Magnitude:** The Fourier amplitude $|\mathcal{F}(x)|$ measures *how much* high and low frequency exists, but Fourier magnitude alone does not penalize slight spatial coordinate offsets or local DC brightness drift in the image domain.
2. **Why GFL Must Be a Regularizer, NOT a Standalone Loss:**
   - In computer vision literature, frequency losses (e.g., Focal Frequency Loss, GFL) are **designed as auxiliary regularizers ($\approx 10\text{--}15\%$)** alongside a strong spatial loss.
   - The spatial loss (Zhao / MS-SSIM + Gaussian-L1) anchors exact pixel coordinates and luminance, while GFL sharpens high-frequency Fourier transitions at nanometer edges.
3. This creates the exact empirical justification for **Run 11: Full Compound Loss**!

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

## Run 7 — Zhao SEM-Tuned (α=0.15)
**Result:** Best PSNR = 23.34 dB (Epoch 7) | Best SSIM = 0.6936 (Epoch 7) | Best LPIPS = 0.3446 (Epoch 7)

---

## Run 8 — Structure-Dominant (85% MS-SSIM + 15% Gaussian-L1)
**Result:** Best PSNR = 23.32 dB (Epoch 8) | Best SSIM = 0.7071 (Epoch 8) | Best LPIPS = 0.3317 (Epoch 7)

---

## Run 9 — Balanced 50/50 (50% MS-SSIM + 50% Gaussian-L1)
**Result:** Best PSNR = 23.35 dB (Epoch 8) | Best SSIM = 0.7013 (Epoch 8) | Best LPIPS = 0.3390 (Epoch 9)

---

## Run 10 — Pure Guided Frequency Loss (GFL via 2D FFT)

**Command:**
```bash
!python trainn.py --stage stage_2 --loss gfl --run_number 10 --epochs 9 --use_drive \
    --data_dir "/content/train/train"
```

### Epoch-by-Epoch Results

| Epoch | Train Loss | Val Loss | Val PSNR (dB) | Val SSIM | Val LPIPS (↓) | LR | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1180.35 | 772.41 | 20.8304 | 0.5499 | 0.4858 | 0.0010 | |
| 2 | 968.78 | 899.84 | 19.8327 | 0.5054 | 0.4713 | 0.0010 | |
| 3 | 921.91 | 764.43 | 20.6003 | 0.5535 | 0.4588 | 0.0010 | |
| 4 | 902.75 | 918.85 | 20.0782 | **0.5583** 🌟 | 0.4718 | 0.0010 | 🌟 Best SSIM |
| 5 | 840.55 | 704.94 | 20.8269 | 0.5201 | 0.4354 | 0.0005 | LR cut to 0.0005 |
| 6 | 819.97 | 705.81 | 20.7786 | 0.5420 | 0.4268 | 0.0005 | |
| 7 | 812.46 | 761.07 | 20.5110 | 0.5448 | 0.4256 | 0.0005 | |
| 8 | 777.26 | **672.38** | **20.8416** 🌟 | 0.5436 | **0.4210** 🌟 | 0.00025 | LR cut to 0.00025 |
| 9 | 767.58 | 684.72 | 20.7766 | 0.5328 | 0.4217 | 0.00025 | |

---

## Run 11 — Full Compound Loss (Zhao Mix + Guided Frequency Loss)

*Results pending — The Flagship Grand Finale!*

### What Full Compound Loss Does:
$$\mathcal{L}_{\text{Compound}} = (1 - w_{\text{GFL}}) \cdot \mathcal{L}_{\text{Zhao}} + w_{\text{GFL}} \cdot \mathcal{L}_{\text{GFL}}$$
- Combines the best spatial multi-scale structural engine (Zhao Mix) with spectral frequency sharpening (2D FFT).
