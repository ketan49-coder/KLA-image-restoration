# Team Relay Handoff Document (Round 1)

**To: Aditya**  
**From: Ketan (via Antigravity)**  

We have completely finalized the training infrastructure, fixed the major roadblocks, and prepared everything for the 1600-epoch relay race. Here is what you need to know for your shift.

---

## 1. The Resolution Bug is Dead
Yesterday, the model was stuck learning "blur" because the dataset loading was masking a dimension mismatch. We fixed this by rewriting the dataset loader and relying purely on the architecture's native `PixelShuffle` layer for 2x upsampling. 
**Your Action:** You do not need to do anything. The data pipeline is now fully RAM-cached (loads in 26 seconds) and dimensionally perfect.

## 2. The Quad-Fidelity Loss
Raw PSNR is dangerous because it encourages the network to produce blurry, artifact-filled images to minimize pixel error. To prevent this, we implemented the `QuadFidelityLoss`. It uses:
1. **Charbonnier Loss** (Robust L1 for speckle noise)
2. **MS-SSIM** (For structural perception)
3. **Focal Frequency Loss** (For sharp high-frequency edges)
4. **Sobel Edge Loss** (For object boundaries)

## 3. The `CompositeScorer` (Our Secret Weapon)
You do not need to guess if a checkpoint is "better" than the last. We wired a `CompositeScorer` into `trainn.py` that weights PSNR (40%), SSIM (35%), and LPIPS (25%).
- A checkpoint is ONLY saved as `best.pth` if it improves on the Master Score.
- **Why this matters for the Relay:** When you resume a run, the script remembers the exact high score from my shift, ensuring we never accept a visually degraded model just because the PSNR ticked up 0.01 dB.

## 4. Atomic Saves (Crash-Proofing)
Colab drops Google Drive connections randomly. `trainn.py` now uses "Atomic Saves". It saves the `.pth` file to a temporary file first, and only renames it to `.pth` if the save is 100% successful. We will never lose a checkpoint to corruption.

## 5. Test-Time Augmentation (TTA)
Our `inference.py` script now has hardcoded 8x TTA. This means on Submission Day, it will mathematically rotate and flip the image 8 times, predict them all, and average the result. This grants a massive "free" PSNR boost on the test set while still easily passing the 10-second rule.

---

## 🎯 Your Mission (The Architecture Race)

Before we commit to the 1600-epoch run, we need to pick the winning architecture. 

**I have already run SymUNet for 25 epochs.** It scored:
- PSNR: 27.73 dB
- SSIM: ~0.71

**Your task:**
Run the other two architectures for 25 epochs to see if they can beat SymUNet's composite score.

1. **Run UltraUNet:**
```bash
python trainn.py --model ultra_unet --loss quad_fidelity --epochs 25
```
2. **Run NAFNet:**
*(Note: NAFNet learns slowly, so it gets a 2x learning rate boost in this race)*
```bash
python trainn.py --model nafnet --loss quad_fidelity --epochs 25 --lr 0.002
```

Whichever of these three models (SymUNet, UltraUNet, or NAFNet) has the highest `CompositeScore` at the end of 25 epochs is crowned the Winner. **That is the model we will pass back and forth for 1600 epochs.**

Good luck!
