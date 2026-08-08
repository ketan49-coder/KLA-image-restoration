# 🧪 Stage 2 — Loss Function Ablation Study

## Experiment Scorecard

| Run # | Loss Function | Best PSNR (dB) | Best SSIM | Best LPIPS (↓) | Best Epoch | Status / Key Insight |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | Baseline (L1 + 0.1×MSE) | 22.56 | 0.6505 | 0.3770 | — | Stage 1 reference anchor |
| **2** | Pure L1 (MAE) | 22.54 | 0.6635 | — | 2 | Stable, median-seeking, slight blur |
| **3** | Pure L2 (MSE) | 22.89 | 0.6807 | 0.3561 | 6 | Direct PSNR optimization |
| **4** | Pure SSIM Loss | 22.96 | **0.7090** | 0.3517 | 6 | Broke 0.70 SSIM barrier |
| **5** | **Pure MS-SSIM Loss** | **23.33** 🏆 | 0.7063 | **0.3319** 🏆 | **6** | **CRUSHED 23 dB! (+0.77 dB over baseline, LPIPS down to 0.33)** |
| **6** | Hybrid: L1 + SSIM | — | — | — | — | *Pending — Next Up!* |
| **7** | Zhao Paper (α=0.025) | — | — | — | — | *Pending* |
| **8** | Zhao SEM / L1+MS-SSIM | — | — | — | — | *Pending* |
| **9** | Pure GFL (2D FFT) | — | — | — | — | *Pending* |
| **10**| Full Compound (Zhao+GFL) | — | — | — | — | *Pending* |

---

## 🔬 Special Technical Analysis: The Multi-Scale Advantage in Run 5 (MS-SSIM)

### Why MS-SSIM Crushed the 23 dB Ceiling
Single-scale SSIM operates on a fixed 11×11 pixel window. In semiconductor SEM images:
- **Macro-scale structures** (large silicon tracks, contact pads) span 50–100+ pixels.
- **Micro-scale details** (edge transitions, line roughness) span 2–5 pixels.

**MS-SSIM computes structural similarity across 5 successive dyadic downsampling scales (1×, 1/2×, 1/4×, 1/8×, 1/16×)**.
This forces the gradients to optimize both broad circuit geometry and microscopic edge fidelity simultaneously!

### The Result:
1. **PSNR surged to 23.3303 dB (+0.77 dB over Stage 1 Baseline)**.
2. **LPIPS dropped to 0.3319** (Massive perceptual realism jump).
3. **LR Scheduler precision:** Once again, stepping down from `0.001` to `0.0005` at Epoch 6 was the catalyst that propelled the model into the 23.33 dB local minimum.

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

**Command:**
```bash
!python trainn.py --stage stage_2 --loss msssim --run_number 4 --epochs 9 --use_drive \
    --data_dir "/content/train/train"
```

### Epoch-by-Epoch Results

| Epoch | Train Loss | Val Loss | Val PSNR (dB) | Val SSIM | Val LPIPS (↓) | LR | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.1234 | 0.0934 | 22.8274 | 0.6869 | 0.3750 | 0.0010 | |
| 2 | 0.1011 | 0.0904 | 23.1736 | 0.6902 | 0.3752 | 0.0010 | 🌟 Early Best |
| 3 | 0.0948 | 0.0890 | 23.0575 | 0.6926 | 0.3686 | 0.0010 | (Bad #1) |
| 4 | 0.0921 | 0.0937 | 22.5492 | 0.6833 | 0.3699 | 0.0010 | (Bad #2) |
| 5 | 0.0905 | 0.0848 | 23.0782 | 0.6964 | 0.3506 | 0.0010 | (Bad #3 $\rightarrow$ **Cut LR to 0.0005**) |
| 6 | 0.0860 | **0.0808** | **23.3303** 🏆 | **0.7063** | 0.3412 | **0.0005** | 🌟 **NEW PROJECT RECORD (23.33 dB)** |
| 7 | 0.0850 | 0.0829 | 22.4819 | 0.6892 | 0.3333 | 0.0005 | |
| 8 | 0.0843 | 0.0821 | 23.0848 | 0.7031 | 0.3386 | 0.0005 | |
| 9 | 0.0833 | 0.0846 | 22.9771 | 0.6939 | **0.3319** 🏆 | 0.0005 | 🌟 **BEST LPIPS (0.3319)** |

---

## Run 6 — Hybrid: L1 + SSIM

*Results pending — Next Up!*

### What L1 + SSIM Does
- Formula: $\mathcal{L} = 0.8 \cdot \mathcal{L}_{L1} + 0.2 \cdot (1 - \text{SSIM})$
- Combines point-wise pixel absolute intensity accuracy (L1) with 11×11 sliding structural contrast (SSIM).
- Addresses pure SSIM's potential color/luminance drift by anchoring mean pixel values with L1.

---

## Run 7 — Zhao Paper Default (α=0.025)
*Pending*

## Run 8 — Zhao SEM-Tuned (α=0.15)
*Pending*

## Run 9 — Guided Frequency Loss (GFL)
*Pending*

## Run 10 — Full Compound Loss (Zhao Mix + GFL)
*Pending*
