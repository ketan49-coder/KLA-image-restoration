# Ketan's Task List
**Total: 55 tasks | 29-37 hours**

---

## IMPLEMENTATION (Before Training)

### Loss Functions ✓ KETAN
- [ ] Implement MS-SSIM loss
- [ ] Implement Guided Frequency Loss (GFL)
- [ ] Create combined loss function
- [ ] Test loss implementations
- [ ] Add loss switching logic in trainn.py

### RRDB Architecture ✓ KETAN
- [ ] Design RRDB block architecture
- [ ] Implement Dense Block (5 conv layers)
- [ ] Implement RRDB (3 Dense Blocks)
- [ ] Replace encoder conv blocks with RRDB
- [ ] Replace decoder conv blocks with RRDB
- [ ] Keep skip connections intact
- [ ] Test RRDB forward pass
- [ ] Verify gradient flow
- [ ] Add RRDB toggle in model.py

---

## STAGE 1: Baseline ✓ KETAN
- [ ] Run sanity check (1-2 epochs)
- [ ] Run baseline training (5-10 epochs)
- [ ] Record metrics (PSNR, SSIM, time)
- [ ] Verify pipeline works
- [ ] Update experiments.csv
- [ ] Git commit Stage 1

---

## STAGE 2: Loss Experiments ✓ KETAN
- [ ] Run L2 loss training
- [ ] Run SSIM loss training
- [ ] Run L1+SSIM training
- [ ] Run L1+MS-SSIM training
- [ ] Run L1+MS-SSIM+GFL training
- [ ] Compare all results
- [ ] Select winner loss
- [ ] Update experiments.csv
- [ ] Git commit Stage 2

---

## STAGE 3: Learning Rate ✓ KETAN
- [ ] Run LR=0.0001 training
- [ ] Run LR=0.0005 training
- [ ] Run LR=0.001 + ReduceLROnPlateau
- [ ] Run Cosine Annealing (optional)
- [ ] Compare convergence curves
- [ ] Select winner LR/scheduler
- [ ] Update experiments.csv
- [ ] Git commit Stage 3

---

## STAGE 4: Augmentation ✓ KETAN
- [ ] Run training WITH augmentation
- [ ] Compare vs Stage 3 best
- [ ] Evaluate validation metrics
- [ ] Test on sample images
- [ ] Decide enable/disable
- [ ] Update experiments.csv
- [ ] Git commit Stage 4

---

## STAGE 5: Log-Transform ✓ KETAN
- [ ] Run training WITH log-transform
- [ ] Compare vs Stage 4 best
- [ ] Evaluate on speckle images
- [ ] Check for artifacts
- [ ] Decide enable/disable
- [ ] Update experiments.csv
- [ ] Git commit Stage 5

---

## STAGE 6: RRDB Upgrade ✓ KETAN
- [ ] Run RRDB sanity check
- [ ] Verify GPU memory fit
- [ ] Run RRDB baseline (10 epochs)
- [ ] Run RRDB with GFL=0.1
- [ ] Run RRDB with GFL=0.3
- [ ] Run RRDB with LR=0.0005
- [ ] Run RRDB with LR=0.0001
- [ ] Compare all RRDB results
- [ ] Compare vs Stage 5 best
- [ ] Select final architecture
- [ ] Update experiments.csv
- [ ] Git commit Stage 6

---

## STAGE 7: Final Training ✓ KETAN
- [ ] Run FP16 test
- [ ] Verify FP16 quality
- [ ] Run final training (20-30 epochs)
- [ ] Monitor until plateau
- [ ] Save best checkpoint
- [ ] Run inference on test set
- [ ] Generate outputs
- [ ] Calculate final metrics
- [ ] Create results.json
- [ ] Update README results
- [ ] Download final model
- [ ] Git commit Stage 7

---

## ONGOING ✓ KETAN
- [ ] Update experiments.csv after each run
- [ ] Git commit after each stage
- [ ] Monitor GPU quota
- [ ] Download checkpoints immediately
- [ ] Daily team communication
- [ ] Make go/no-go decisions
- [ ] Review code changes

---

## FINAL SUBMISSION ✓ KETAN
- [ ] Test README instructions
- [ ] Write final results summary
- [ ] Upload model weights
- [ ] Add download link
- [ ] Create submission package
- [ ] Submit to portal
- [ ] Final git push

