import time
import torch
from model import get_model  # your model.py

# ----------------------------------------------------------------------
# Config — adjust to match your real input size (128x128 per your dataset)
# ----------------------------------------------------------------------
INPUT_SIZE = 128          # degraded input resolution (H=W)
IN_CHANNELS = 1
BATCH_SIZE = 1            # 1 = realistic "one test image at a time" scenario
N_WARMUP = 10              # untimed runs to let CUDA/cuDNN autotune kick in
N_RUNS = 50                # timed runs, averaged for a stable estimate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_TO_TEST = ["symunet", "rrdb", "resrestorer", "ultra_unet"]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_model(name, use_amp=False):
    model = get_model(name).to(DEVICE)
    model.eval()

    dummy_input = torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE, device=DEVICE)

    # Warmup — first few CUDA calls include kernel selection/compilation overhead,
    # which would otherwise badly skew the average if included in the timed runs.
    with torch.no_grad():
        for _ in range(N_WARMUP):
            if use_amp and DEVICE.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    with torch.no_grad():
        for _ in range(N_RUNS):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            if use_amp and DEVICE.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - start) * 1000)

    times_ms.sort()
    mean_ms = sum(times_ms) / len(times_ms)
    median_ms = times_ms[len(times_ms) // 2]
    std_ms = (sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)) ** 0.5

    n_params = count_params(model)
    images_per_10s = 10_000 / mean_ms  # how many single-image inferences fit in the 10s budget

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "name": name,
        "params_M": n_params / 1e6,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "std_ms": std_ms,
        "images_per_10s": images_per_10s,
    }


def main():
    print(f"Device: {DEVICE} | Input: {BATCH_SIZE}x{IN_CHANNELS}x{INPUT_SIZE}x{INPUT_SIZE} | "
          f"Warmup: {N_WARMUP} | Timed runs: {N_RUNS}\n")

    results = []
    for name in MODELS_TO_TEST:
        try:
            r_fp32 = benchmark_model(name, use_amp=False)
            results.append(r_fp32)
            print(f"[{name:12s}] FP32  | {r_fp32['params_M']:6.2f}M params | "
                  f"{r_fp32['mean_ms']:7.2f} ms/img (median {r_fp32['median_ms']:.2f}, "
                  f"std {r_fp32['std_ms']:.2f}) | ~{r_fp32['images_per_10s']:.0f} imgs / 10s budget")

            if DEVICE.type == "cuda":
                r_fp16 = benchmark_model(name, use_amp=True)
                print(f"[{name:12s}] FP16  | {'':6s}          | "
                      f"{r_fp16['mean_ms']:7.2f} ms/img (median {r_fp16['median_ms']:.2f}, "
                      f"std {r_fp16['std_ms']:.2f}) | ~{r_fp16['images_per_10s']:.0f} imgs / 10s budget "
                      f"({r_fp32['mean_ms'] / r_fp16['mean_ms']:.2f}x speedup)\n")
        except Exception as e:
            print(f"[{name:12s}] FAILED: {e}\n")

    print("-" * 90)
    print("Well under the '10 seconds per image' bar for all of these unless mean_ms is enormous.")
    print("Watch relative differences between architectures, and re-run this after adding RRDB")
    print("blocks to SymUNet, or after any architecture change, since speed can shift a lot.")


if __name__ == "__main__":
    main()
