import os
import sys
from inference import run_inference

def main():
    # Strict KLA requirements: script must run exactly as: python run.py <input-dir> <output-dir>
    # Check if correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Path to the model is explicitly requested to be in models/
    checkpoint_path = os.path.join("models", "model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Final model not found at {checkpoint_path}!")
        print("Please ensure your .pth file is renamed to 'model.pth' and placed inside the 'models' folder.")
        sys.exit(1)
        
    print(f"[KLA BENCHMARK LAUNCH] Input: {input_dir} | Output: {output_dir}")
    
    # Run the inference core with the absolute maximum TTA available
    # to squeeze out the highest mathematical PSNR.
    try:
        run_inference(
            input_dir=input_dir,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            device=None,             # auto-detect GPU
            base_channels=32,
            batch_size=8,
            use_fp16=True,           # Fast fp16 inference on modern GPUs
            use_compile=False,       # Disabled to avoid PyTorch 2.0+ compatibility crash risks
            tta_mode="super128x"     # MAXIMUM 128x TTA
        )
    except Exception as e:
        print(f"[CRITICAL ERROR] Inference failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
