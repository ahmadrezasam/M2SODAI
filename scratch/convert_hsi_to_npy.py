import os
import scipy.io as sio
import numpy as np
from tqdm import tqdm
import glob

PROJECT_ROOT = "/home/ahmadreza/Downloads/Research/M2SODAI"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "data/hsi_npy")
NORM_PATH = os.path.join(DATA_ROOT, "pca_mean_std.mat")

def convert_hsi():
    # Load normalization stats
    norm_data = sio.loadmat(NORM_PATH)
    mean = norm_data['mean'].reshape(1, 1, 30)
    std = norm_data['std'].reshape(1, 1, 30)

    # Find all .mat files in *_pca directories
    mat_files = []
    for split in ["train_pca", "val_pca", "test_pca"]:
        mat_files.extend(glob.glob(os.path.join(DATA_ROOT, split, "*.mat")))

    print(f"Found {len(mat_files)} HSI files to convert...")

    for mat_path in tqdm(mat_files):
        try:
            # Load .mat
            data = sio.loadmat(mat_path)
            if 'data' not in data:
                continue
            
            hsi = data['data'] # Expected shape (H, W, 30)
            
            # Normalize
            hsi = (hsi - mean) / std
            
            # Convert to float32
            hsi = hsi.astype(np.float32)
            
            # Define output path
            rel_path = os.path.relpath(mat_path, DATA_ROOT)
            npy_path = os.path.join(OUTPUT_ROOT, rel_path.replace(".mat", ".npy"))
            
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            np.save(npy_path, hsi)
            
        except Exception as e:
            print(f"Error converting {mat_path}: {e}")

if __name__ == "__main__":
    convert_hsi()
