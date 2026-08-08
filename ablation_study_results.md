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
| **Run 8** | Structure-Dominant (85% MS-SSIM)| 23.32 | **0.7071** ⭐ | 0.3317 | 8 | Structural leader |
| **Run 9** | Balanced 50/50 (50% MS-SSIM)| **23.35** 🏆 | 0.7013 | 0.3390 | 8 | **ALL-TIME PSNR RECORD (23.35 dB)** |
| **Run 10**| Pure GFL (2D FFT) | 20.84 | 0.5583 | 0.4210 | 8 | Spectral needs spatial anchor |
| **Run 11**| **Compound (Zhao-85 + 10% GFL)**| 23.33 | **0.7069** ⭐ | **0.3302** 🏆 | 6 & 8 | **ALL-TIME LPIPS PERCEPTUAL SHARPNESS RECORD (0.3302)!** |
| **Run 12**| Compound (Zhao-85 + 5% GFL) | — | — | — | — | *Optional GFL Sweep* |
| **Run 13**| Compound (Zhao-85 + 20% GFL)| — | — | — | — | *Optional GFL Sweep* |

---

## 🔬 Special Technical Analysis: The Multi-Domain Synergies

```
┌──────────────────────────────────────┬──────────────┬───────────────┬────────────────┬──────────────────────────────┐
│ Loss Configuration                   │ Best PSNR    │ Best SSIM     │ Best LPIPS (↓) │ Scientific Role              │
├──────────────────────────────────────┼──────────────┼───────────────┼────────────────┼──────────────────────────────┤
│ Anchor: Baseline (L1 + MSE)          │ 22.56 dB     │ 0.6505        │ 0.3770         │ Standard starting point      │
│ Run 8: Structure-Dominant (Zhao-85)  │ 23.32 dB     │ 0.7071 🏆     │ 0.3317         │ Spatial & Multi-scale Leader │
│ Run 9: Balanced 50/50                │ 23.35 dB 🏆  │ 0.7013        │ 0.3390         │ APEX PSNR Record             │
│ Run 11: Compound (Zhao-85 + 10% GFL) │ 23.33 dB     │ 0.7069 ⭐     │ 0.3302 🏆      │ 🌟 APEX PERCEPTUAL SHARPNESS │
└──────────────────────────────────────┴──────────────┴───────────────┴────────────────┴──────────────────────────────┘
```

### Key Scientific Conclusions:
1. **Perceptual Sharpness Breakthrough:** Adding **10% Orthonormal 2D Fast Fourier Transform (GFL)** to the **85% Structure-Dominant Zhao Mix** drove the **LPIPS score down to 0.3302** — the sharpest perceptual reconstruction in the entire project history.
2. **Frequency Regularization Effect:** The 2D FFT regularizer forces high-frequency restoration across nanometer circuit lines without sacrificing the 23.3+ dB PSNR plateau.

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
**Result:** Best PSNR = 20.84 dB (Epoch 8) | Best SSIM = 0.5583 (Epoch 4) | Best LPIPS = 0.4210 (Epoch 8)

---

## Run 11 — Flagship Compound Loss (Zhao-85 + 10% GFL)

**Command:**
```bash
!python trainn.py --stage stage_2 --loss compound_10 --run_number 11 --epochs 9 --use_drive \
    --data_dir "/content/train/train"
```

### Epoch-by-Epoch Results

| Epoch | Train Loss | Val Loss | Val PSNR (dB) | Val SSIM | Val LPIPS (↓) | LR | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.1104 | 0.0851 | 22.6321 | 0.6870 | 0.3584 | 0.0010 | |
| 2 | 0.0936 | 0.0805 | 23.2543 | 0.6942 | 0.3672 | 0.0010 | 🌟 Blazing early start (23.25 dB) |
| 3 | 0.0873 | 0.0792 | 23.1112 | 0.6957 | 0.3608 | 0.0010 | |
| 4 | 0.0851 | 0.0885 | 21.7055 | 0.6896 | 0.3649 | 0.0010 | |
| 5 | 0.0846 | 0.0776 | 22.9347 | 0.7008 | 0.3453 | 0.0010 | |
| 6 | 0.0791 | **0.0740** | **23.3281** 🏆 | **0.7069** 🌟 | 0.3360 | 0.0005 | 🌟 **Best PSNR (23.33 dB) & SSIM (0.7069)** |
| 7 | 0.0786 | 0.0747 | 23.0914 | 0.7020 | 0.3325 | 0.0005 | |
| 8 | 0.0781 | 0.0766 | 23.1889 | 0.7002 | **0.3302** 🏆 | 0.0005 | 🌟 **NEW ALL-TIME LPIPS RECORD (0.3302)!** |
| 9 | 0.0774 | 0.0824 | 22.9593 | 0.6912 | 0.3369 | 0.0005 | |

---

## Next Steps for the Project:
1. **Optional GFL Fine-Sweep:** Test `--loss compound_05` (5% GFL) or `--loss compound_20` (20% GFL).
2. **Stage 3: Learning Rate & Advanced Scheduler Tuning:** Test Cosine Annealing, warmups, and learning rates on our winning compound loss.
3. **Stage 6: Architectural Upgrades:** Integrate **SymUNet + RRDB / Attention Gates** using our winning Compound Loss.
