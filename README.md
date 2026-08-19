# KLA Image Restoration Hackathon 2026: 28.04 dB PSNR Submission

This repository contains our team's inference and training pipeline for the AI-Based Restoration of Degraded Images problem statement. 

Through aggressive architectural ablation, custom loss curriculums, and a highly optimized inference engine, our final model achieves a validated **28.04 dB PSNR** and **0.7844 SSIM** while remaining extremely fast (sub-100ms per image), easily passing the 10-second hardware evaluation limit.

---

## ⚡ 1. Evaluation (KLA Benchmark Standard)

Our solution is fully compliant with the hackathon submission guidelines. The `run.py` script automatically loads our final 32-channel NAFNet weights (`models/model.pth`), activates FP16 acceleration, and deploys a massive **128x Test-Time Augmentation (TTA)** engine.

### Inference Command
```bash
python run.py <input-dir> <output-dir>
```

**Example:**
```bash
python run.py Test_NoisyLR final_submission_outputs
```
*Note: The output `.npy` arrays are guaranteed to be exactly `(256, 256)` grayscale, perfectly clipped to `[0, 1]`, and scrubbed of any NaN/Inf values.*

---

## 🧠 2. The Engineering Journey: From U-Net to NAFNet

This submission was not built by simply downloading a pre-trained model and pressing "train." We went through several major architectural iterations and overcame significant engineering bottlenecks to break the 27.0 dB physical ceiling.

### Phase 1: The U-Net Bottlenecks
We initially hypothesized that U-Net architectures would be ideal for this task. We implemented and ablated multiple variants:
*   **Base U-Net:** Hit a hard PSNR ceiling around 26 dB.
*   **SymUNet (Symmetric U-Net):** Attempted to balance the encoder/decoder paths better for high-frequency SEM noise.
*   **UltraUNet:** Added deep residual connections and attention blocks.

**The Problem:** While UltraUNet pushed PSNR higher, the inference time skyrocketed, bringing us dangerously close to the 10-second penalty limit. Furthermore, standard ReLU activations in these networks struggled to accurately map the continuous, delicate high-frequency gradients of SEM silicon traces. 

### Phase 2: The NAFNet Pivot
To solve the Speed vs. Quality tradeoff, we pivoted to **NAFNet (Nonlinear Activation Free Network)**. 
By replacing standard non-linearities (like ReLU/GELU) with a mathematical operation called **SimpleGate** ($X_1 \odot X_2$), NAFNet allowed us to achieve Transformer-level PSNR while retaining the raw speed of a lightweight CNN. We optimized the network width to exactly 32 channels—the perfect sweet spot that maximized mathematical depth without sacrificing speed.

### Phase 3: Solving the "Blur" Problem (Quad-Fidelity Loss)
Training on Mean-Squared Error (MSE) taught the model to "play it safe," resulting in high PSNR but blurry edges. To force the network to hallucinate sharp silicon boundaries, we engineered a custom `QuadFidelityLoss` combining four distinct penalties:
1.  **Charbonnier Loss:** A robust L1 loss that handles severe Speckle noise outliers without causing exploding gradients.
2.  **MS-SSIM Loss:** Enforces structural integrity and perceptually accurate textures.
3.  **Focal Frequency Loss (2D FFT):** Forces the network into the Fourier domain, ensuring it learns both low-level structure and high-frequency noise patterns.
4.  **Sobel Edge Loss:** Explicitly penalizes blurry boundaries.

### Phase 4: Overcoming Convergence Shock (The Phase-Switched Curriculum)
During training, we realized the model was getting trapped in local minima early on. 
*   We implemented **Warm Restarts** (`CosineAnnealingWarmRestarts`) which acted like a sledgehammer, aggressively kicking the model out of shallow valleys. 
*   However, past Epoch 400, these violent restarts were destroying the delicate fine-tuning required to breach 27 dB.
*   **The Solution:** We engineered a custom *Phase-Switched Alternating Curriculum*. We used Warm Restarts early, then automatically transitioned to a smooth `CosineAnnealingLR` for final convergence. Crucially, we wrote PyTorch overrides to physically **wipe the AdamW momentum buffers** during the transition, preventing gradient shock and allowing the model to coast perfectly to 28.04 dB.

### Phase 5: The 128x TTA Engine
Because we restricted NAFNet to 32 channels, our raw inference was blazingly fast. We realized we had massive spare overhead before hitting KLA's 10-second limit. 
We built a highly optimized **128x Test-Time Augmentation (TTA)** engine inside `inference.py` (combining 16 geometric shifts with 8 dihedral flips). This mathematical brute-force trick took our base 26.9 dB weights and artificially elevated them to **28.04 dB**, scoring us a massive +1.1 dB "free" boost during evaluation.

---

## 🛠️ 3. Reproducing the Training

If you wish to re-run our 1,600-epoch curriculum from scratch:
```bash
pip install -r requirements.txt
python trainn.py --data_dir /path/to/train --epochs 1600
```
