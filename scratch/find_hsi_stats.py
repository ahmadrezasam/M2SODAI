import scipy.io as sio
import numpy as np
import os
from tqdm import tqdm

DATA_ROOT = "/home/ahmadreza/Downloads/Research/M2SODAI/data"
TRAIN_MAT_DIR = os.path.join(DATA_ROOT, "train")
VAL_MAT_DIR = os.path.join(DATA_ROOT, "val")

def find_robust_max(dirs):
    all_values = []
    print("Collecting samples for robust max calculation...")
    
    # We don't need all files if the dataset is huge, but let's take a good sample
    mat_files = []
    for d in dirs:
        mat_files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.mat')]
    
    # Take up to 500 random samples to avoid memory issues and speed up
    if len(mat_files) > 500:
        import random
        random.seed(42)
        mat_files = random.sample(mat_files, 500)
    
    for mat_path in tqdm(mat_files):
        try:
            hsi = sio.loadmat(mat_path)['data']
            # Bands 53, 32, 11 (0-indexed 52, 31, 10? No, let's check upscale script)
            # upscale_v3_bilinear.py: hsi_rgb = hsi[:, :, [53, 32, 11]].astype(np.float32)
            # It used 53, 32, 11 as indices.
            hsi_rgb = hsi[:, :, [53, 32, 11]].astype(np.float32)
            all_values.append(hsi_rgb.flatten())
        except Exception as e:
            print(f"Error reading {mat_path}: {e}")
    
    all_values = np.concatenate(all_values)
    robust_max = np.percentile(all_values, 99.9)
    actual_max = np.max(all_values)
    actual_min = np.min(all_values)
    
    print(f"\nResults:")
    print(f"Actual Min: {actual_min}")
    print(f"Actual Max: {actual_max}")
    print(f"99.9th Percentile (Robust Max): {robust_max}")
    
    return actual_min, robust_max

if __name__ == "__main__":
    find_robust_max([TRAIN_MAT_DIR, VAL_MAT_DIR])
