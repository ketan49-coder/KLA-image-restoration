# SymU-Net + RRDB Training Plan
## KLA Image Restoration — SEMICON India Hackathon 2026

**Team:** Ketan Shinde, Rikhil Vaswani, Aditya Jagtap  
**Institution:** DES Pune  
**Date:** August 5, 2026

---

## Executive Summary

This document outlines a systematic 7-stage training plan for our AI-based semiconductor image restoration model. We will progressively build from a baseline SymU-Net architecture to an advanced SymU-Net+RRDB configuration, locking in optimal hyperparameters at each stage.

**Total Training Sessions:** 19-22 runs  
**Estimated Time:** 8-11 hours of active work  
**Timeline:** Completable within 1 week

---

## Project Overview

### Problem Statement
Restore degraded semiconductor inspection images affected by:
1. **Speckle Noise** — multiplicative grainy noise
2. **Gaussian Noise/Blur** — edge softening
3. **Spatial Resolution Reduction** — downsampling requiring super-resolution

### Architecture Strategy
- **Baseline:** SymU-Net (Symmetric U-Net) with plain convolutional blocks
- **Advanced:** SymU-Net + RRDB (Residual-in-Residual Dense Blocks)
- **Loss Functions:** Progressive from L1 to L1+MS-SSIM+GFL (Guided Frequency Loss)

---

## Training Sessions Summary

| Stage | Description | Runs | Time | Key Decision |
|-------|-------------|------|------|--------------|
| 1 | Baseline SymU-Net | 2 | 30-40 min | Pipeline validation |
| 2 | Loss function experiments | 5 | 75-100 min | Lock optimal loss |
| 3 | LR/scheduler tuning | 3-4 | 45-80 min | Lock learning rate |
| 4 | Data augmentation | 1 | 15-20 min | Enable/disable |
| 5 | Speckle preprocessing | 1 | 15-20 min | Enable/disable |
| 6 | SymU-Net + RRDB | 5-6 | 90-120 min | Architecture upgrade |
| 7 | Final optimization | 2 | 40-60 min | Submission ready |
| **TOTAL** | **Complete pipeline** | **19-22** | **~8-11 hrs** | **Final model** |


## STAGE 1: Baseline SymU-Net (Safety Net)

**Objective:** Confirm end-to-end pipeline works before adding complexity

**Configuration:**
- Architecture: SymU-Net with plain conv blocks
- Loss: L1 only
- Optimizer: Adam, lr=0.001
- Batch size: 4
- Epochs: 1 (sanity check), then 5-10 (baseline)
- Augmentation: None
- Preprocessing: Standard normalization only

**Runs:**
1. **Sanity Check** (1-2 epochs) — verify data loads, model trains, no crashes
2. **Baseline** (5-10 epochs) — train to reasonable convergence

**Success Criteria:**
- No errors in data loading
- Loss decreases steadily
- Model saves checkpoint successfully
- Inference runs on test images

**Expected Time:** 30-40 minutes (2 runs × 15-20 min)

**What to Record:**
- Final training loss
- Validation PSNR, SSIM
- Inference time (ms/image)
- Any issues encountered

**GitHub Commit:** "Stage 1: Baseline SymU-Net with L1 loss"

---

## STAGE 2: Loss Function Experiments

**Objective:** Find optimal loss function for our multi-degradation task

**Fixed from Stage 1:**
- Architecture: SymU-Net (plain conv blocks)
- Optimizer: Adam, lr=0.001
- Batch size: 4
- Epochs: 5-10 per experiment
- Augmentation: None

**Experiments (5 runs):**

1. **L2 Loss (MSE)**
   - Good for: Pixel-level accuracy
   - Weakness: Can blur edges

2. **SSIM Loss**
   - Good for: Structural similarity
   - Weakness: May miss fine details

3. **L1 + SSIM** (0.5 weight each)
   - Good for: Balance pixel accuracy + structure

4. **L1 + MS-SSIM** (Multi-Scale SSIM)
   - Good for: Multi-resolution structure preservation
   - Better than single-scale SSIM

5. **L1 + MS-SSIM + GFL** (Guided Frequency Loss)
   - Good for: Pixel + structure + high-frequency details
   - Best for blur/resolution restoration
   - Suggested weights: 0.5 L1, 0.3 MS-SSIM, 0.2 GFL

**Evaluation Metrics:**
- PSNR (higher is better)
- SSIM (higher is better)
- LPIPS (lower is better)
- Visual quality inspection

**Expected Time:** 75-100 minutes (5 runs)

**Winner Selection:**
- Highest PSNR + SSIM combination
- Best visual quality on sample test images
- Lock this loss for all subsequent stages

**GitHub Commit:** "Stage 2: Loss experiments - [Winner] selected"

---

## STAGE 3: Learning Rate & Scheduler

**Objective:** Optimize training convergence speed and stability

**Fixed from Stages 1-2:**
- Architecture: SymU-Net (plain conv blocks)
- Loss: [Winner from Stage 2]
- Batch size: 4
- Epochs: 10-15 per experiment
- Augmentation: None

**Experiments (3-4 runs):**

1. **LR = 0.0001** (conservative)
   - Slower but more stable

2. **LR = 0.0005** (moderate)
   - Middle ground

3. **LR = 0.001 + ReduceLROnPlateau**
   - Start aggressive, reduce when loss plateaus
   - patience=3, factor=0.5

4. **LR = 0.001 + Cosine Annealing** (optional)
   - Cyclical learning rate
   - Can escape local minima

**Expected Time:** 45-80 minutes (3-4 runs)

**Winner Selection:**
- Fastest convergence to best final loss
- Most stable training curve

**GitHub Commit:** "Stage 3: LR/scheduler optimization - [Winner] selected"


## STAGE 4: Data Augmentation

**Objective:** Improve generalization to out-of-distribution test images

**Fixed from Stages 1-3:**
- Architecture: SymU-Net (plain conv blocks)
- Loss: [Winner from Stage 2]
- LR/Scheduler: [Winner from Stage 3]
- Batch size: 4
- Epochs: 10-15

**Experiment (1 run):**

**With Augmentation:**
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Random 90° rotation (p=0.5)
- NO color jitter (grayscale images)
- NO elastic deformation (can distort defects)

**Compare Against:** Stage 3 best result (no augmentation)

**Expected Time:** 15-20 minutes

**Decision:**
- Keep augmentation if validation OR test quality improves
- Keep if metrics stay same (better generalization)
- Only skip if metrics significantly degrade

**GitHub Commit:** "Stage 4: Augmentation [enabled/disabled]"

---

## STAGE 5: Speckle Log-Transform Preprocessing

**Objective:** Handle multiplicative speckle noise more effectively

**Fixed from Stages 1-4:**
- Architecture: SymU-Net (plain conv blocks)
- Loss: [Winner from Stage 2]
- LR/Scheduler: [Winner from Stage 3]
- Augmentation: [Decision from Stage 4]

**Experiment (1 run):**

**Log-Transform Preprocessing:**
- Convert multiplicative noise to additive noise
- Train in log space: log(noisy) → log(clean)
- Convert back at inference

**Rationale:**
- Speckle noise is multiplicative: I_degraded = I_clean × noise
- Log transforms to additive: log(I_degraded) = log(I_clean) + log(noise)
- CNNs handle additive noise better

**Expected Time:** 15-20 minutes

**Decision:**
- Keep if metrics improve on speckle-degraded images
- Skip if no improvement or introduces artifacts

**GitHub Commit:** "Stage 5: Speckle log-transform [enabled/disabled]"

---

## STAGE 6: Architecture Upgrade — SymU-Net + RRDB

**Objective:** Replace plain conv blocks with RRDB for denser feature learning

**Architecture Changes:**

**Before (Plain Conv Block):**
```
Conv(in, out, 3×3) → BatchNorm → ReLU
Conv(out, out, 3×3) → BatchNorm → ReLU
```

**After (RRDB Block):**
```
Residual-in-Residual Dense Block:
  - 3 Dense blocks (each with 5 conv layers)
  - Dense connections within each block
  - Residual scaling (β=0.2)
  - LeakyReLU activations
```

**Experiments (5-6 runs):**

1. **RRDB Sanity Check** (1-2 epochs)
   - Verify GPU memory fit
   - Check training stability

2. **RRDB Baseline** (10 epochs)
   - Use all locked settings from Stages 2-5
   - Compare against Stage 5 best

3. **RRDB Loss Weight Tuning** (2 runs)
   - Try adjusting GFL weight: 0.1, 0.3
   - RRDB may need different loss balance

4. **RRDB Learning Rate Tuning** (2 runs)
   - Try LR = 0.0005, 0.0001
   - Heavier model may need lower LR

**Expected Time:** 90-120 minutes (5-6 runs, RRDB trains slower)

**Success Criteria:**
- PSNR improvement ≥ 1-2 dB over plain SymU-Net
- SSIM improvement ≥ 0.02-0.05
- Visual quality clearly better

**If RRDB Fails:**
- Fall back to Stage 5 best (plain SymU-Net)
- Stage 1-5 results are still submission-worthy

**GitHub Commit:** "Stage 6: SymU-Net+RRDB - [results summary]"

---

## STAGE 7: Final Optimization & Submission

**Objective:** Production-ready model with speed optimization

**Experiments (2 runs):**

1. **FP16 Mixed Precision** (1 run)
   - Enable automatic mixed precision (AMP)
   - Faster training + inference
   - Check quality stays same

2. **Final Full-Convergence Training** (1 run)
   - Train 20-30 epochs until plateau
   - Save best checkpoint
   - Generate all test outputs
   - Record final metrics

**Expected Time:** 40-60 minutes

**Deliverables:**
- `checkpoints/final_model.pth`
- `outputs/` restored test images
- `results.json` final metrics
- Updated README

**GitHub Commit:** "Stage 7: Final model - ready for submission"

---

## Timeline & Milestones

### Day 1 (Aug 5) — Baseline
- Stage 1: Baseline SymU-Net (2 runs)
- Confirm pipeline works

### Day 2 (Aug 6) — Loss & LR
- Stage 2: Loss experiments (5 runs)
- Stage 3: LR/scheduler (3-4 runs)
- Lock optimal settings

### Day 3 (Aug 7) — Data Pipeline
- Stage 4: Augmentation (1 run)
- Stage 5: Preprocessing (1 run)

### Day 4-5 (Aug 8-9) — RRDB
- Stage 6: Architecture upgrade (5-6 runs)

### Day 6 (Aug 10) — Final
- Stage 7: Optimization + submission (2 runs)

### Buffer (Aug 11) — Contingency
- Fix issues, polish docs

---

## Success Metrics

### Baseline (Stage 1)
- PSNR: 26-28 dB
- SSIM: 0.75-0.82
- Inference: <50 ms/image

### After Loss Optimization (Stage 2)
- PSNR: +1.5-2.5 dB improvement
- SSIM: +0.03-0.05 improvement

### Final SymU-Net+RRDB (Stage 6-7)
- PSNR: 32-35 dB (competitive)
- SSIM: 0.88-0.92 (high quality)
- LPIPS: <0.10 (perceptually good)
- Inference: <100 ms/image

### Stretch Goals
- PSNR: >35 dB (excellent)
- SSIM: >0.92 (near-perfect)

---

## Team Responsibilities

### Ketan Shinde (Project Lead)
- Stage 1-2 execution
- Loss function analysis
- Final architecture decisions
- GitHub management

### Rikhil Vaswani (Infrastructure)
- Colab setup & GPU management
- Dataset handling
- Stage 3-4 execution
- Checkpoint backups

### Aditya Jagtap (Optimization)
- Stage 5-6 execution
- RRDB implementation
- Hyperparameter tuning
- Metrics tracking

---

## Risk Mitigation

**Risk 1: RRDB Implementation Issues**
- Mitigation: Stage 1-5 is fallback
- Action: Submit plain SymU-Net if needed

**Risk 2: GPU Quota Exhaustion**
- Mitigation: 20 runs = 6.7 hours < 30hr limit
- Action: Prioritize Stages 1-5

**Risk 3: Colab Disconnects**
- Mitigation: Save to Drive after each run
- Action: Implement checkpoint resume

**Risk 4: Time Overruns**
- Mitigation: 6-day plan + 1 buffer day
- Action: Skip optional experiments if needed

---

## Submission Checklist

### Code & Model
- [ ] model.py — SymU-Net + RRDB
- [ ] dataset.py — data loading
- [ ] losses.py — L1+MS-SSIM+GFL
- [ ] trainn.py — training script
- [ ] eval.py — evaluation (standalone)
- [ ] requirements.txt — dependencies
- [ ] checkpoints/final_model.pth — weights

### Results
- [ ] outputs/ — restored test images
- [ ] experiments.csv — all runs logged
- [ ] results.json — final metrics

### Documentation
- [ ] README.md — complete instructions
- [ ] Results table filled
- [ ] Team info complete

### GitHub
- [ ] All code pushed
- [ ] Clean commit history
- [ ] Repository public

---

## Technical References

### Loss Functions

**L1 Loss:**
```
L1 = (1/N) Σ |pred - gt|
```

**MS-SSIM Loss:**
```
MS-SSIM = Π [SSIM(pred, gt, scale_i)]^weight_i
Loss = 1 - MS-SSIM
```

**Guided Frequency Loss (GFL):**
```
GFL = ||FFT(pred) - FFT(gt)||₁ × (1 + α×|FFT(gt)|)
α = guidance weight (0.1-0.5)
```

**Combined (Candidate):**
```
L_total = 0.5×L1 + 0.3×(1-MS-SSIM) + 0.2×GFL
```

---

## Hardware Requirements

**Colab Free Tier:**
- GPU: T4 (16GB VRAM)
- RAM: 12GB
- 30 hours/week GPU quota

**Dataset Size:**
- Train: ~1.6GB (3200 images)
- Test: ~400MB (800 images)
- Total: ~2GB

---

**Document Version:** 1.0  
**Last Updated:** August 5, 2026  
**Status:** Ready for execution

---

**Team:** Ketan Shinde, Rikhil Vaswani, Aditya Jagtap  
**GitHub:** https://github.com/ketan49-coder/KLA-image-restoration  
**Institution:** DES Pune  
**Hackathon:** SEMICON India 2026 — KLA Track 1

