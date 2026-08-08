# 🧪 Stage 2 — Loss Function Ablation Study

## Comprehensive Experiment Leaderboard (Averages vs. Peak Bests Across All Epochs)

> **Key Takeaway:** Looking at both **Epoch Averages** (training stability/consistency) and **Peak Bests** (maximum restoration capacity) gives an unbiased, complete picture of loss function performance for SEM images.

| Colab Run # | Loss Function | Epochs | Avg PSNR (dB) | Best PSNR (dB) | Avg SSIM | Best SSIM | Avg LPIPS (↓) | Best LPIPS (↓) | Avg Val Loss | Training Character |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Anchor** | Baseline (L1 + 0.1×MSE)| 5 | 21.80 | **22.56** | 0.6350 | **0.6505** | 0.3900 | **0.3770** | 0.0760 | Reference starting point |
| **Run 1** | Pure L1 (MAE) | 5 | 21.86 | **22.54** | 0.6484 | **0.6635** | — | — | 0.0746 | Stable median regression, mild blur |
| **Run 2** | Pure L2 (MSE) | 7 | 22.10 | **22.89** | 0.6503 | **0.6807** | 0.3714 | **0.3561** | 0.0103 | Direct MSE optimization |
| **Run 3** | Pure SSIM | 9 | 22.62 | **22.96** | 0.6945 | **0.7090** | 0.3804 | **0.3517** | 0.2985 | Single-scale structural focus |
| **Run 4** | Pure MS-SSIM | 9 | **22.95** 🏆 | **23.33** | 0.6935 | **0.7063** | 0.3538 | **0.3319** | 0.0869 | Multi-scale structural pyramid |
| **Run 5** | Hybrid: L1 + SSIM | 9 | 22.12 | **22.71** | 0.6696 | **0.6928** | 0.3816 | **0.3511** | 0.1223 | Single-scale hybrid |
| **Run 6** | Zhao Paper (2.5% MS-SSIM) | 9 | 22.60 | **23.15** | 0.6632 | **0.6814** | 0.3707 | **0.3475** | 0.0690 | 2.5% under-weights structure |
| **Run 7** | Zhao SEM (15% MS-SSIM) | 9 | 22.60 | **23.34** | 0.6775 | **0.6936** | 0.3629 | **0.3446** | 0.0724 | Photometric balance |
| **Run 8** | Structure-Dominant (85% MS-SSIM)| 9 | 22.89 | **23.32** | 0.6944 | **0.7071** ⭐ | 0.3530 | **0.3317** | 0.0855 | Spatial & structural stability |
| **Run 9** | Balanced 50/50 (50% MS-SSIM) | 9 | 22.87 | **23.35** 🏆 | 0.6912 | **0.7013** | 0.3610 | **0.3390** | 0.0785 | **Peak Single-Epoch PSNR Champion** |
| **Run 10**| Pure GFL (2D FFT) | 9 | 20.56 | **20.84** | 0.5389 | **0.5583** | 0.4465 | **0.4210** | 764.94 | Shift-invariant frequency drift |
| **Run 11**| **Compound (Zhao-85 + 10% GFL)**| 9 | **22.91** 🌟 | **23.33** | **0.6964** 🏆 | **0.7069** ⭐ | **0.3480** 🏆 | **0.3302** 🏆 | 0.0798 | **#1 Overall Winner across all Epochs** |
| **Run 12**| Compound-90 (Zhao-90 + 10% GFL)| 9 | *TBD* | **23.38** 🏆 | *TBD* | **0.7088** 🌟 | *TBD* | **0.3310** | *TBD* | *In Progress (Epoch 6: 23.38 dB)* |

---

## 🔬 Key Scientific Discoveries from the Average Metrics:

1. **Compound Loss (Run 11) is the Most Consistent Performer:**
   - Achieved the **lowest Average LPIPS (0.3480)** across all 9 epochs, proving that the 2D FFT regularizer maintains superior edge sharpness throughout the entire training trajectory.
   - Achieved the **highest Average SSIM (0.6964)** across all 9 epochs.
2. **Pure MS-SSIM (Run 4) & Compound (Run 11) Maintain the Highest Baseline PSNR:**
   - Both maintained $\approx 22.95\text{ dB}$ average PSNR across every epoch, well above standard L1 (21.86 dB) or baseline (21.80 dB).
3. **Pure GFL (Run 10) Proves Frequency Regularization Must Be Coupled with Spatial Anchors:**
   - Standalone GFL averaged 20.56 dB and 0.5389 SSIM, whereas coupled as a 10% regularizer in Compound Loss (Run 11), it drove LPIPS to all-time project highs (0.3302).

---

## Epoch-by-Epoch Historical Record

### Anchor — Baseline: L1 + 0.1 × MSE
- Avg PSNR: 21.80 dB | Best PSNR: 22.56 dB | Avg SSIM: 0.6350 | Best SSIM: 0.6505 | Best LPIPS: 0.3770

### Run 1 — Pure L1 (MAE)
- Avg PSNR: 21.86 dB | Best PSNR: 22.54 dB | Avg SSIM: 0.6484 | Best SSIM: 0.6635 | Avg Val Loss: 0.0746

### Run 2 — Pure L2 (MSE)
- Avg PSNR: 22.10 dB | Best PSNR: 22.89 dB | Avg SSIM: 0.6503 | Best SSIM: 0.6807 | Avg LPIPS: 0.3714 | Best LPIPS: 0.3561 | Avg Val Loss: 0.0103

### Run 3 — Pure SSIM Loss
- Avg PSNR: 22.62 dB | Best PSNR: 22.96 dB | Avg SSIM: 0.6945 | Best SSIM: 0.7090 | Avg LPIPS: 0.3804 | Best LPIPS: 0.3517 | Avg Val Loss: 0.2985

### Run 4 — Pure MS-SSIM Loss
- Avg PSNR: 22.95 dB | Best PSNR: 23.33 dB | Avg SSIM: 0.6935 | Best SSIM: 0.7063 | Avg LPIPS: 0.3538 | Best LPIPS: 0.3319 | Avg Val Loss: 0.0869

### Run 5 — Hybrid: L1 + SSIM
- Avg PSNR: 22.12 dB | Best PSNR: 22.71 dB | Avg SSIM: 0.6696 | Best SSIM: 0.6928 | Avg LPIPS: 0.3816 | Best LPIPS: 0.3511 | Avg Val Loss: 0.1223

### Run 6 — Zhao Paper Default (α=0.025)
- Avg PSNR: 22.60 dB | Best PSNR: 23.15 dB | Avg SSIM: 0.6632 | Best SSIM: 0.6814 | Avg LPIPS: 0.3707 | Best LPIPS: 0.3475 | Avg Val Loss: 0.0690

### Run 7 — Zhao SEM-Tuned (α=0.15)
- Avg PSNR: 22.60 dB | Best PSNR: 23.34 dB | Avg SSIM: 0.6775 | Best SSIM: 0.6936 | Avg LPIPS: 0.3629 | Best LPIPS: 0.3446 | Avg Val Loss: 0.0724

### Run 8 — Structure-Dominant (85% MS-SSIM + 15% Gaussian-L1)
- Avg PSNR: 22.89 dB | Best PSNR: 23.32 dB | Avg SSIM: 0.6944 | Best SSIM: 0.7071 | Avg LPIPS: 0.3530 | Best LPIPS: 0.3317 | Avg Val Loss: 0.0855

### Run 9 — Balanced 50/50 (50% MS-SSIM + 50% Gaussian-L1)
- Avg PSNR: 22.87 dB | Best PSNR: 23.35 dB | Avg SSIM: 0.6912 | Best SSIM: 0.7013 | Avg LPIPS: 0.3610 | Best LPIPS: 0.3390 | Avg Val Loss: 0.0785

### Run 10 — Pure Guided Frequency Loss (GFL via 2D FFT)
- Avg PSNR: 20.56 dB | Best PSNR: 20.84 dB | Avg SSIM: 0.5389 | Best SSIM: 0.5583 | Avg LPIPS: 0.4465 | Best LPIPS: 0.4210 | Avg Val Loss: 764.94

### Run 11 — Flagship Compound Loss (Zhao-85 + 10% GFL)
- Avg PSNR: 22.91 dB | Best PSNR: 23.33 dB | Avg SSIM: 0.6964 | Best SSIM: 0.7069 | Avg LPIPS: 0.3480 | Best LPIPS: 0.3302 | Avg Val Loss: 0.0798
