import numpy as np
import matplotlib.pyplot as plt
import os

gt_dir = r"C:\Users\shind\OneDrive\Desktop\ketan\studies\hackhathon project\train\train\GT"
noisy_dir = r"C:\Users\shind\OneDrive\Desktop\ketan\studies\hackhathon project\train\train\NoisyLR"

# Get one filename to test with
filenames = [f for f in os.listdir(gt_dir) if f.endswith('.npy')]
print("Total GT files:", len(filenames))
print("Example filename:", filenames[0])

sample_name = filenames[0]

gt_img = np.load(os.path.join(gt_dir, sample_name))
noisy_img = np.load(os.path.join(noisy_dir, sample_name))

print("GT shape:", gt_img.shape, "dtype:", gt_img.dtype, "min/max:", gt_img.min(), gt_img.max())
print("Noisy shape:", noisy_img.shape, "dtype:", noisy_img.dtype, "min/max:", noisy_img.min(), noisy_img.max())

fig, axes = plt.subplots(1, 2, figsize=(10,5))
axes[0].imshow(gt_img, cmap='gray')
axes[0].set_title("Ground Truth")
axes[0].axis('off')

axes[1].imshow(noisy_img, cmap='gray')
axes[1].set_title("Degraded (Noisy/LR)")
axes[1].axis('off')

plt.show()