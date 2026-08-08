# KLA Image Restoration: Ablation & Architecture Study

## 🚀 Stage 4: Overcoming the 26.9 dB Plateau (3-Pillar Benchmark Campaign)

### 🎯 The Critical Diagnosis:
While Stage 3 achieved a high mark of **26.9443 dB**, adding deep attention and residual complexity to a 4-level U-Net yielded only **+0.017 dB** over standard U-Net with GT Normalization (26.9272 dB). 
We identified the two fundamental architectural bottlenecks causing this plateau:
1. **The 4× Downsampling Penalty:** Squeezing $128 \times 128$ SEM inputs down to $8 \times 8$ bottlenecks destroys microscopic semiconductor nanometer line boundaries.
2. **The MS-SSIM Metric Tradeoff:** `Compound-90` optimizes structural perceptual correlation rather than direct mathematical pixel error ($10\log_{10}(1/\text{MSE})$), which caps raw PSNR.

### 🧪 Stage 4 Benchmark Execution Roadmap:
| Run ID | Strategy / Architecture | Loss Formulation | Epochs | Peak Val PSNR (dB) | Peak Val SSIM | Peak Val LPIPS (↓) | Status | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Run 13** | **SymUNet** | **Charbonnier** | 15 | **26.9754 dB** | **0.7224** | **0.2954** | 🟢 **COMPLETED** | Verified raw Charbonnier gradient behavior |
| **Run 13_Hybrid** | **SymUNet** | **Charb-Compound** | 15 | **27.1093 dB** 🚀 | **0.7276** 🚀 | **0.2816** 🚀 | 🟢 **COMPLETED** | **BROKE 27 dB BARRIER!** Tri-Fidelity loss wins across all metrics |
| **Run 14** | **ResRestorer** (Full-Res ResNet) | **Charb-Compound** | 15 | $\ge 28.5\text{ dB}$ | $\ge 0.74$ | $\le 0.28$ | 🟡 Pending | Eliminates all 4x downsampling; full $128 \times 128$ feature flow |
| **Run 15** | **ShallowUNet** (2-Stage Multi-Scale)| **Charb-Compound** | 15 | $\ge 28.0\text{ dB}$ | $\ge 0.73$ | $\le 0.28$ | 🟡 Pending | 1 downsampling level only + Dense Residual Blocks |
| **Run 16** | **Grand Winner + Hybrid Loss** | **Optimal Formulation**| 25 | $\ge 30.0\text{ dB}$ | $\ge 0.76$ | $\le 0.25$ | 🟡 Pending | Final hackathon flagship submission |

---

### 📊 Run 13_Hybrid: SymUNet + Tri-Fidelity Charb-Compound Loss (15-Epoch Progression)

| Epoch | Train Loss | Val Loss | Val PSNR (dB) | Val SSIM | Val LPIPS (↓) | Learning Rate | Checkpoint Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 0.2662 | 2.6108 | 20.8790 | 0.5790 | 0.3997 | 0.001000 | 🌟 New Best |
| **2** | 0.0756 | 0.2749 | 24.6783 | 0.6483 | 0.3517 | 0.001000 | 🌟 New Best |
| **3** | 0.0733 | 0.1346 | 22.4034 | 0.6033 | 0.3851 | 0.001000 | |
| **4** | 0.0657 | 0.0611 | 25.4139 | 0.6765 | 0.3346 | 0.001000 | 🌟 New Best |
| **5** | 0.0582 | 0.0671 | 24.6382 | 0.6664 | 0.3285 | 0.001000 | |
| **6** | 0.0561 | 0.0548 | 26.4310 | 0.7106 | 0.3084 | 0.001000 | 🌟 New Best |
| **7** | 0.0536 | 0.0557 | 26.4786 | 0.6983 | 0.3184 | 0.001000 | 🌟 New Best |
| **8** | 0.0522 | 0.0530 | 26.6406 | 0.7109 | 0.2959 | 0.001000 | 🌟 New Best |
| **9** | 0.0517 | 0.0593 | 26.2761 | 0.6943 | 0.3360 | 0.001000 | |
| **10** | 0.0502 | 0.0512 | 26.8550 | 0.7183 | 0.2933 | 0.001000 | 🌟 New Best |
| **11** | 0.0499 | 0.0584 | 26.2164 | 0.7040 | 0.3118 | 0.001000 | |
| **12** | 0.0497 | 0.0531 | 26.8730 | 0.7160 | 0.3068 | 0.001000 | 🌟 New Best |
| **13** | 0.0493 | 0.0548 | 26.3501 | 0.7112 | 0.3022 | 0.001000 | |
| **14** | **0.0486** | **0.0489** | **27.1093** | **0.7276** | **0.2816** | 0.001000 | 🚀 **NEW ALL-TIME RECORD (>27 dB!)** |
| **15** | 0.0484 | 0.0538 | 26.6650 | 0.7054 | 0.3069 | 0.001000 | |

---

### 📊 Run 13: SymUNet + Pure Charbonnier Loss (15-Epoch Full Progression)

| Epoch | Train Loss | Val Loss | Val PSNR (dB) | Val SSIM | Val LPIPS (↓) | Learning Rate | Checkpoint Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 0.2865 | 0.0638 | 24.1313 | 0.6107 | 0.3763 | 0.001000 | 🌟 New Best |
| **2** | 0.0535 | 0.0595 | 24.0253 | 0.6207 | 0.3578 | 0.001000 | |
| **3** | 0.0431 | 0.0403 | 26.1014 | 0.6822 | 0.3283 | 0.001000 | 🌟 New Best |
| **4** | 0.0381 | 0.0393 | 26.2301 | 0.6821 | 0.3336 | 0.001000 | 🌟 New Best |
| **5** | 0.0371 | 0.0435 | 25.5022 | 0.6577 | 0.3398 | 0.001000 | |
| **6** | 0.0363 | 0.0398 | 26.1098 | 0.7044 | 0.3130 | 0.001000 | |
| **7** | 0.0356 | 0.0378 | 26.7444 | 0.7044 | 0.3153 | 0.001000 | 🌟 New Best |
| **8** | 0.0352 | 0.0373 | 26.7891 | 0.7121 | 0.3025 | 0.001000 | 🌟 New Best |
| **9** | 0.0356 | 0.0389 | 26.2511 | 0.7000 | 0.2965 | 0.001000 | |
| **10** | 0.0352 | 0.0386 | 26.4121 | 0.7029 | 0.3253 | 0.001000 | |
| **11** | 0.0344 | 0.0370 | 26.9178 | 0.7139 | 0.3229 | 0.001000 | 🌟 New Best |
| **12** | **0.0339** | **0.0368** | **26.9754** | **0.7152** | **0.3326** | 0.001000 | 🌟 Best PSNR (Run 13) |
| **13** | 0.0354 | 0.0369 | 26.7348 | 0.7174 | 0.3166 | 0.001000 | |
| **14** | 0.0339 | 0.0372 | 26.9184 | 0.7184 | 0.3036 | 0.001000 | |
| **15** | **0.0336** | **0.0375** | **26.7249** | **0.7224** | **0.2954** | 0.001000 | 🌟 Peak SSIM & LPIPS (Run 13) |

---

## 🏆 Current Overall Leaderboard Scorecard (Baseline to Date)
| Model Architecture | Loss Function | Epochs | Peak Val PSNR (dB) | Peak Val SSIM | Peak Val LPIPS (↓) | Checkpoint Path |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Baseline U-Net | L1 + 0.1×MSE | 5 | 22.56 dB | 0.6505 | 0.3770 | `checkpoints/baseline.pth` |
| Enhanced U-Net | Compound-90 | 9 | 23.47 dB | 0.7109 | 0.3279 | `checkpoints/run12_compound90.pth` |
| SymUNet (Run 1) | Compound-90 | 15 | 26.9443 dB | 0.7243 | 0.2899 | `checkpoints/stage_2_compound_run1_best.pth` |
| SymUNet (Run 13) | Charbonnier | 15 | 26.9754 dB | 0.7224 | 0.2954 | `checkpoints/stage_4_charbonnier_run13_best.pth` |
| **SymUNet (Run 13_Hybrid)**| **Charb-Compound** | **15** | **27.1093 dB** 🏆 | **0.7276** 🏆 | **0.2816** 🏆 | `checkpoints/stage_4_charb_compound_run13_hybrid_best.pth` |

**Net Improvement over Baseline:**
- **+4.55 dB PSNR** improvement (from 22.56 dB $\rightarrow$ **27.11 dB**)
- **+0.0771 SSIM** improvement (from 0.6505 $\rightarrow$ **0.7276**)
- **-0.0954 LPIPS** perceptual error reduction (from 0.3770 $\rightarrow$ **0.2816**, new benchmark record!)

---

### 📊 SymUNet 15-Epoch Training Progression:

| Epoch | Train Loss | Val Loss | Val PSNR (dB) | Val SSIM | Val LPIPS (↓) | Learning Rate | Checkpoint Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 0.2350 | 0.2911 | 23.4687 | 0.6207 | 0.3723 | 0.001000 | 🌟 New Best |
| **2** | 0.0953 | 0.1207 | 24.8494 | 0.6551 | 0.3485 | 0.001000 | 🌟 New Best |
| **3** | 0.0856 | 0.0769 | 25.4207 | 0.6730 | 0.3245 | 0.001000 | 🌟 New Best |
| **4** | 0.0781 | 0.0732 | 26.4029 | 0.7004 | 0.3153 | 0.001000 | 🌟 New Best |
| **5** | 0.0737 | 0.0897 | 25.3013 | 0.6870 | 0.3356 | 0.001000 | |
| **6** | 0.0721 | 0.0748 | 26.2455 | 0.7058 | 0.3237 | 0.001000 | |
| **7** | 0.0708 | 0.0716 | 26.0807 | 0.7081 | 0.3141 | 0.001000 | |
| **8** | 0.0675 | 0.0676 | 26.5382 | 0.7142 | 0.2974 | 0.000500 | 🌟 New Best (LR Cut) |
| **9** | 0.0671 | 0.0681 | 26.6327 | 0.7155 | 0.3124 | 0.000500 | 🌟 New Best |
| **10** | 0.0660 | 0.0662 | 26.7036 | 0.7209 | 0.3021 | 0.000500 | 🌟 New Best |
| **11** | 0.0654 | 0.0698 | 26.6249 | 0.7096 | 0.3062 | 0.000500 | |
| **12** | 0.0646 | 0.0801 | 26.3361 | 0.7054 | 0.3321 | 0.000500 | |
| **13** | 0.0642 | 0.0652 | 26.8308 | 0.7206 | 0.3022 | 0.000500 | 🌟 New Best |
| **14** | 0.0630 | 0.0649 | 26.8463 | 0.7241 | 0.2956 | 0.000500 | 🌟 New Best |
| **15** | **0.0622** | **0.0647** | **26.9443** | **0.7243** | **0.2899** | 0.000500 | 🌟 **FINAL BEST (Peak Model)** |

---

### What Makes SymUNet So Effective?
1. **Local Residual Blocks + BatchNorm:** Prevents gradient vanishing across deep feature hierarchies and stabilizes high-order gradient backpropagation.
2. **Squeeze-and-Excitation (SE) Channel Attention:** Recalibrates channel-wise feature responses dynamically, giving higher weight to informative edge filters over flat background responses.
3. **Attention Gates (AGs) on Skip Connections:** Explicitly gates low-level encoder features using high-level decoder gating signals, suppressing noise in skip connections before spatial concatenation.
4. **Sub-Pixel Convolution (PixelShuffle with ICNR):** Eliminates checkerboard deconvolution artifacts and directly learns 2x spatial reconstruction kernels.
5. **Compound-90 Loss Synergy:** Simultaneously optimizes structural integrity (MS-SSIM), local intensity consistency (Gaussian-L1), and high-frequency Fourier harmonics (GFL).

---

## Stage 2: Loss Function Ablation Study (Legacy Baseline)

This document summarizes the quantitative comparison of loss formulations evaluated for semiconductor Scanning Electron Microscope (SEM) image restoration prior to the V2 architecture overhaul. 

| Run ID | Loss Function | Epochs | Mean PSNR (dB) | Peak PSNR (dB) | Mean SSIM | Peak SSIM | Mean LPIPS (↓) | Peak LPIPS (↓) | Mean Val Loss | Formulation Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Anchor** | Baseline (L1 + 0.1×MSE)| 5 | 21.80 | 22.56 | 0.6350 | 0.6505 | 0.3900 | 0.3770 | 0.0760 | Stage 1 Reference Baseline |
| **Run 1** | Pure L1 (MAE) | 5 | 21.86 | 22.54 | 0.6484 | 0.6635 | — | — | 0.0746 | Median regression baseline |
| **Run 2** | Pure L2 (MSE) | 7 | 22.10 | 22.89 | 0.6503 | 0.6807 | 0.3714 | 0.3561 | 0.0103 | Mean square error objective |
| **Run 3** | Pure SSIM | 9 | 22.62 | 22.96 | 0.6945 | 0.7090 | 0.3804 | 0.3517 | 0.2985 | Single-scale structural similarity |
| **Run 4** | Pure MS-SSIM | 9 | 22.95 | 23.33 | 0.6935 | 0.7063 | 0.3538 | 0.3319 | 0.0869 | Multi-scale structural pyramid (5 scales) |
| **Run 5** | Hybrid: L1 + SSIM | 9 | 22.12 | 22.71 | 0.6696 | 0.6928 | 0.3816 | 0.3511 | 0.1223 | Linear combination (0.8 L1 + 0.2 SSIM) |
| **Run 6** | Zhao Paper Default | 9 | 22.60 | 23.15 | 0.6632 | 0.6814 | 0.3707 | 0.3475 | 0.0690 | alpha = 0.025 (97.5% L1 + 2.5% MS-SSIM) |
| **Run 7** | Zhao SEM-Tuned | 9 | 22.60 | 23.34 | 0.6775 | 0.6936 | 0.3629 | 0.3446 | 0.0724 | alpha = 0.15 (85% L1 + 15% MS-SSIM) |
| **Run 8** | Structure-Dominant (85%) | 9 | 22.89 | 23.32 | 0.6944 | 0.7071 | 0.3530 | 0.3317 | 0.0855 | alpha = 0.85 (85% MS-SSIM + 15% Gaussian-L1) |
| **Run 9** | Balanced 50/50 | 9 | 22.87 | 23.35 | 0.6912 | 0.7013 | 0.3610 | 0.3390 | 0.0785 | alpha = 0.50 (50% MS-SSIM + 50% Gaussian-L1) |
| **Run 10**| Pure GFL (2D FFT) | 9 | 20.56 | 20.84 | 0.5389 | 0.5583 | 0.4465 | 0.4210 | 764.94 | Frequency magnitude loss without spatial anchor |
| **Run 11**| Compound-85 (Zhao-85 + 10% GFL)| 9 | 22.91 | 23.33 | 0.6964 | 0.7069 | 0.3480 | 0.3302 | 0.0798 | 90% Zhao-85 + 10% Orthonormal 2D FFT |
| **Run 12**| **Compound-90 (Zhao-90 + 10% GFL)**| 9 | **23.07** | **23.47** | **0.7002** | **0.7109** | **0.3450** | **0.3279** | **0.0785** | **Optimal Configuration (Rank 1 across all metrics)** |

---

## Technical Performance Summary: Run 12 (Compound-90)

```
========================================================================================
OPTIMAL LOSS CONFIGURATION SUMMARY: RUN 12 (COMPOUND-90)
========================================================================================
Metric                            Epoch Average (9 Epochs)    Peak Validation Result
----------------------------------------------------------------------------------------
Peak Signal-to-Noise Ratio (PSNR) 23.07 dB                   23.47 dB (+0.91 dB vs baseline)
Structural Similarity Index (SSIM)0.7002                     0.7109
Perceptual Loss (LPIPS, AlexNet)  0.3450                     0.3279 (-0.0491 vs baseline)
Mean Validation Loss              0.0785                     0.0724
========================================================================================
```

---

## Analytical Findings

### 1. Mathematical Formulation & Synergistic Mechanics
The optimal loss formulation across all spatial fidelity, structural similarity, and perceptual metrics is defined as:
$$\mathcal{L}_{\text{final}} = (1 - w_{\text{GFL}}) \cdot \left[ \alpha \cdot \mathcal{L}_{\text{MS-SSIM}} + (1 - \alpha) \cdot \mathcal{L}_{G\text{-}L1} \right] + w_{\text{GFL}} \cdot \mathcal{L}_{\text{GFL}}$$
where $\alpha = 0.90$ and $w_{\text{GFL}} = 0.10$.

- **Multi-Scale Structural Similarity ($\mathcal{L}_{\text{MS-SSIM}}$, $\alpha = 0.90$):**
  Unlike single-scale SSIM, MS-SSIM decomposes the image over 5 dyadic downsampling scales ($M=5$). Semiconductor SEM images exhibit defects and circuit lines across variable spatial frequencies (from broad substrate backgrounds down to nanometer etch boundaries). Evaluating contrast-structure ($cs_j$) across all 5 scales while evaluating luminance ($l_M$) exclusively at the coarsest scale ensures scale-invariant edge reconstruction and prevents gradient vanishing in fine line geometries.
  
- **Gaussian-Weighted L1 Loss ($\mathcal{L}_{G\text{-}L1}$, $1 - \alpha = 0.10$):**
  Standard pixel-wise L1 treats all pixels uniformly, making it vulnerable to high-frequency electron detector shot noise. By convolving the absolute residual $|pred - target|$ with an 11×11 Gaussian kernel ($\sigma=1.5$), the network is anchored to the local median intensity of clean silicon regions while remaining robust against isolated stochastic noise spikes.

- **Orthonormal Guided Frequency Loss ($\mathcal{L}_{\text{GFL}}$, $w_{\text{GFL}} = 0.10$):**
  Spatial losses inherently exhibit low-pass filtering characteristics, leading to oversmoothed, blurry line edges. By computing the 2D Real Fast Fourier Transform ($\text{rfft2}$) with unitary orthonormal scaling, $\mathcal{L}_{\text{GFL}}$ penalizes spectral energy attenuation in the high-frequency Fourier quadrant, driving the mean LPIPS perceptual error down to 0.3450 (and peak to 0.3279).

---

### 2. Comparative Analysis Across Ablation Categories

#### A. Spatial Baselines ($L_1$, $L_2$, Baseline)
- **Observations:** Pure $L_1$ and $L_2$ achieved moderate PSNR (22.54–22.89 dB) but suffered from poor structural fidelity (SSIM $\le 0.6807$) and pronounced perceptual blur (LPIPS $\ge 0.3561$).
- **Mechanism:** $L_2$ minimizes Mean Squared Error by predicting the conditional mean of all possible clean states, causing severe smoothing across sharp etched line boundaries.

#### B. The Structural Ratio Progression ($\alpha = 0.025 \rightarrow 0.15 \rightarrow 0.50 \rightarrow 0.85 \rightarrow 0.90$)
- **Observations:** The Zhao et al. paper default ($\alpha = 0.025$) underperformed in SEM restoration (SSIM 0.6814), as 97.5% $L_1$ dominance over-smoothed nanometer edges.
- **Scaling Trend:** Progressively increasing $\alpha$ from 0.025 to 0.90 yielded a monotonic increase in both mean and peak SSIM ($0.6814 \rightarrow 0.6936 \rightarrow 0.7013 \rightarrow 0.7071 \rightarrow 0.7109$).
- **Conclusion:** Semiconductor SEM structures require an ultra-structure-dominant loss regime ($\alpha \ge 0.85$) to force precise boundary alignment.

#### C. Spectral Regularization vs. Spatial Coordinate Anchoring (Pure GFL vs. Compound)
- **Observations:** Pure GFL (Run 10) failed to converge effectively (Mean PSNR 20.56 dB, SSIM 0.5389), whereas Compound-90 (Run 12) achieved the project benchmark record.
- **Physical Reason:** Fourier magnitude spectra are translation and shift-invariant. A network trained exclusively on frequency magnitude lacks coordinate-space spatial localization, resulting in structural phase drift. When coupled with the 90% MS-SSIM spatial anchor, GFL acts as a precise spectral regularizer without disturbing spatial alignment.

---

### 3. Key Conclusions & Downstream Model Design
1. **Loss Selection:** Compound-90 is established as the permanent loss criterion for all downstream stages.
2. **Architectural Synergies:** The combination of multi-scale spatial loss with Fourier spectral loss creates an ideal training signal for architectures with global/local residual paths (e.g., SymUNet, ResUNet, Attention UNet), allowing the network to focus purely on learning the residual high-frequency noise map $\mathcal{R}(x) = y - x$.

---

## Epoch Log: Run 12 (Compound-90)

**Execution Parameters:**
- Architecture: 4-Level Baseline UNet
- Image Dimensions: 128x128 patches
- Optimizer: Adam ($lr = 10^{-3}$, ReduceLROnPlateau)
- Epochs: 9

| Epoch | Train Loss | Val Loss | Val PSNR (dB) | Val SSIM | Val LPIPS (↓) | Learning Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.1152 | 0.0854 | 22.7424 | 0.6830 | 0.3605 | 1.0e-3 |
| 2 | 0.0936 | 0.0804 | 23.1449 | 0.6979 | 0.3530 | 1.0e-3 |
| 3 | 0.0873 | 0.0799 | 23.1429 | 0.6969 | 0.3617 | 1.0e-3 |
| 4 | 0.0844 | 0.0822 | 22.3437 | 0.6983 | 0.3604 | 1.0e-3 |
| 5 | 0.0834 | 0.0770 | 22.9043 | 0.7039 | 0.3418 | 1.0e-3 |
| 6 | 0.0785 | 0.0735 | 23.3846 | 0.7088 | 0.3310 | 5.0e-4 |
| 7 | 0.0777 | 0.0754 | 23.1906 | 0.7057 | 0.3368 | 5.0e-4 |
| 8 | 0.0768 | 0.0724 | 23.4673 | 0.7109 | 0.3279 | 5.0e-4 |
| 9 | 0.0764 | 0.0803 | 23.2721 | 0.6966 | 0.3317 | 5.0e-4 |
