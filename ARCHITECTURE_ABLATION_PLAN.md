# Architecture Ablation Showdown: The 25-Epoch Race

Before we commit to the massive 600-epoch grind, we need to prove scientifically which architecture actually has the highest PSNR ceiling and the best speed. We will race the 3 titans against each other for 25 epochs each.

## The Contenders

| Model | Architecture Type | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- |
| **1. SymUNet** | CNN | Insanely fast, proven baseline (27.72 dB). | Small receptive field, low capacity ceiling. |
| **2. UltraUNet** | CNN + ASPP | High capacity, global context via ASPP. | Heavy footprint, slower inference. |
| **3. NAFNet** | Non-Linear Activation Free | SOTA performance, Transformer-level PSNR, ultra-fast CNN inference. | Math explosion risks, slow convergence. |

## Implementation Plan

### 1. Implement NAFNet with Safety Counter-Measures
I will code the `NAFNet` architecture directly into your `model.py` file, incorporating the following specific safety protocols to fix its weaknesses:
- **ICNR Initialization:** Will be applied to the PixelShuffle layer to completely prevent checkerboard upsampling artifacts.
- **Gradient Clipping (Max Norm = 1.0):** I will update `trainn.py` to clip gradients. This acts as a physical speed limiter to prevent the math from exploding when the network hits a massive Speckle noise outlier.
- **Higher Starting Learning Rate:** I will adjust the training command to use `--lr 0.002` (double the standard). This forces NAFNet to learn faster and prevents it from being a "late bloomer" in the 25-epoch race.
- **FP16 Mixed Precision:** Will be enabled by default to cut NAFNet's massive RAM usage in half, preventing Colab Out-Of-Memory crashes.

### 2. Update the Pipeline
I will update the `get_model` factory in `model.py` and the argument parser in `trainn.py` to accept `--model nafnet`.

### 3. The Racing Protocol
Once the code is pushed, you will run these three commands sequentially in Colab. 

**Runner 1: UltraUNet (The Heavyweight)**
```bash
!python trainn.py --packed_data /content/dataset_packed.pt --model ultraunet --loss quad_fidelity --scheduler cosine --epochs 25 --stage stage_5_ablation --run_number 21_ultra --use_drive --base_channels 64
```

**Runner 2: NAFNet (The SOTA Challenger)**
*(Note the higher `--lr 0.002` to force fast convergence)*
```bash
!python trainn.py --packed_data /content/dataset_packed.pt --model nafnet --loss quad_fidelity --scheduler cosine --lr 0.002 --epochs 25 --stage stage_5_ablation --run_number 22_nafnet --use_drive --base_channels 32
```

*(Note: We already have the SymUNet 25-epoch baseline from Run 19, so you only have to run these two new ones!).*

## Open Questions
- Please click **Proceed** to approve this plan, and I will immediately write the NAFNet architecture and safety features into your codebase!
