# 🧪 Stage 2 — Loss Function Ablation Study

## Experiment Scorecard

| Run # | Loss Function | Best PSNR (dB) | Best SSIM | Best LPIPS | Best Epoch | Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | Baseline (L1 + 0.1×MSE) | 22.56 | 0.6505 | 0.3770 | — | Stage 1 reference anchor |
| **2** | Pure L1 | 22.54 | 0.6635 | — | 2 | Nearly identical to baseline |
| **3** | Pure L2 (MSE) | — | — | — | — | *Pending* |
| **4** | SSIM Loss | — | — | — | — | *Pending* |
| **5** | MS-SSIM Loss | — | — | — | — | *Pending* |
| **6** | Zhao Paper (α=0.025) | — | — | — | — | *Pending* |
| **7** | Zhao SEM (α=0.15) | — | — | — | — | *Pending* |
| **8** | GFL (Frequency) | — | — | — | — | *Pending* |
| **9** | Compound (Zhao + GFL) | — | — | — | — | *Pending* |

---

## 🔬 Special Technical Analysis: Why the LR Scheduler Didn't Cut LR in Run 2

### The Observation
In Run 2 (L1), the best PSNR was reached at **Epoch 2 (22.5399 dB)**.
- Epoch 3: 22.4484 dB (no improvement)
- Epoch 4: 21.6657 dB (no improvement)
- Epoch 5: 21.7537 dB (no improvement)

Yet throughout all 5 epochs, the logged learning rate stayed at `0.001000`.

### The Exact PyTorch Mathematics of `ReduceLROnPlateau`
In PyTorch's `ReduceLROnPlateau(mode='max', factor=0.5, patience=2)`:
1. PyTorch tracks an internal integer: `self.num_bad_epochs`.
2. A learning rate reduction is triggered **only when** `self.num_bad_epochs > self.patience` (strictly greater than).
3. Let's trace how the counter evolved in Run 2:
   - **Epoch 1**: Val PSNR = 20.8958 dB $\rightarrow$ Initialized `best = 20.8958`, `num_bad_epochs = 0`.
   - **Epoch 2**: Val PSNR = 22.5399 dB $\rightarrow$ **Improvement!** `best = 22.5399`, `num_bad_epochs = 0`.
   - **Epoch 3**: Val PSNR = 22.4484 dB $\rightarrow$ Bad epoch #1: `num_bad_epochs = 1`. Check: `1 > 2` is **False** $\rightarrow$ No LR change.
   - **Epoch 4**: Val PSNR = 21.6657 dB $\rightarrow$ Bad epoch #2: `num_bad_epochs = 2`. Check: `2 > 2` is **False** $\rightarrow$ No LR change.
   - **Epoch 5**: Val PSNR = 21.7537 dB $\rightarrow$ Trained with LR = `0.001000`. At the end of Epoch 5 evaluation: Bad epoch #3: `num_bad_epochs = 3`. Check: `3 > 2` is **True** $\rightarrow$ **LR was cut to `0.0005` at the very end of Epoch 5!**
   - Because the training script was set for `--epochs 5`, Epoch 6 never ran, so we didn't see the `0.0005` in the batch logs.

### Key Takeaway for Short (5-Epoch) Ablations
- `patience=2` is designed for longer runs (15–30+ epochs).
- For short 5-epoch ablation comparisons, `patience=1` would trigger the LR cut immediately at Epoch 4 (allowing Epoch 5 to train with the reduced LR).
- **The mode was correctly configured to `mode='max'`** tracking Val PSNR, and the scheduler was working as mathematically defined by PyTorch.

---

## Run 1 — Baseline: L1 + 0.1 × MSE

**Command:** *Ran during Stage 1 (prior session)*
**Result:** PSNR = 22.56 dB | SSIM = 0.6505 | LPIPS = 0.3770

### What This Loss Does
- Combines absolute pixel error (L1) with a small squared error penalty (0.1 × MSE)
- The MSE term adds extra punishment for large pixel errors
- Formula: $\mathcal{L} = |pred - gt| + 0.1 \times (pred - gt)^2$

### Key Takeaway
This is our **anchor point**. Every other loss must beat these numbers to be worth using. The SSIM of 0.6505 tells us the baseline model is losing significant structural detail — edges are being blurred.

---

## Run 2 — Pure L1 (MAE)

**Command:**
```bash
!python trainn.py --stage stage_2 --loss l1 --run_number 1 --epochs 5 --use_drive \
    --data_dir "/content/train/train"
```

### Epoch-by-Epoch Results

| Epoch | Train Loss | Val Loss | Val PSNR (dB) | Val SSIM | LR | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.0975 | 0.0827 | 20.90 | 0.6395 | 0.001000 | |
| 2 | 0.0841 | **0.0691** | **22.54** ⭐ | 0.6327 | 0.001000 | 🌟 BEST |
| 3 | 0.0831 | 0.0702 | 22.45 | 0.6449 | 0.001000 | (Bad #1) |
| 4 | 0.0804 | 0.0764 | 21.67 | **0.6635** | 0.001000 | (Bad #2) |
| 5 | 0.0756 | 0.0745 | 21.75 | 0.6612 | 0.001000 | (Bad #3 -> cut to 0.0005) |

**Best PSNR:** 22.54 dB (Epoch 2) | **Best SSIM:** 0.6635 (Epoch 4)

### What L1 Loss Does
- Formula: $\mathcal{L}_{L1} = \frac{1}{N} \sum |pred_i - gt_i|$
- Simply averages the absolute pixel-by-pixel difference
- Treats every pixel independently — no awareness of edges, patterns, or structure
- Each pixel is penalized equally regardless of whether it's on an edge or flat background

### Why L1 Causes Blur
When a pixel could plausibly be 0.3 or 0.7 (different training examples disagree), L1's optimal output is the **median** value (≈0.5). Across the whole image, this averaging effect turns sharp edges into smooth gradual ramps — that's blur.

### What We Observed
1. **Overfitting detected:** Train loss kept dropping (0.0975 → 0.0756) but Val loss rose after epoch 2 (0.0691 → 0.0764). The model started memorizing training images.
2. **PSNR and SSIM disagreed:** PSNR peaked at epoch 2, but SSIM peaked at epoch 4. As training continued, the model got worse at pixel matching but slightly better at structure — a known L1 behavior.
3. **vs. Baseline:** Pure L1 matched baseline PSNR (22.54 ≈ 22.56) but slightly improved SSIM (0.6635 > 0.6505). Removing the MSE component didn't hurt pixels but marginally helped structure.

### Verdict
L1 is a **stable but mediocre** loss. It doesn't blow up, but it has zero structural awareness. It's the "safe but boring" choice — good enough to not fail, not smart enough to excel. The blur it produces is exactly why we need structural losses (SSIM, MS-SSIM) to push further.

---

## Run 3 — Pure L2 / MSE

*Results pending — will be filled after execution*

### What L2 Loss Does
- Formula: $\mathcal{L}_{L2} = \frac{1}{N} \sum (pred_i - gt_i)^2$
- Squares each pixel error before averaging
- Punishes **large errors much more** than small ones (an error of 0.5 is penalized 25× more than an error of 0.1)
- The optimal output is the **mean** (not median like L1), which tends to produce even more blur than L1

### What We Expect to See
- Possibly **lower PSNR** than L1 (MSE optimization doesn't directly maximize PSNR well)
- Likely **lower SSIM** than L1 (more aggressive blur from mean-regression)
- This would confirm L2 is worse than L1 for restoration tasks — a well-known result in the literature

---

## Run 4 — SSIM Loss

*Results pending*

---

## Run 5 — MS-SSIM Loss

*Results pending*

---

## Run 6 — Zhao Paper Default (α=0.025)

*Results pending*

---

## Run 7 — Zhao SEM-Tuned (α=0.15)

*Results pending*

---

## Run 8 — Guided Frequency Loss (GFL)

*Results pending*

---

## Run 9 — Compound Loss (Zhao Mix + GFL)

*Results pending*
