import scipy.io as sio
import cv2
import numpy as np
import os
from tqdm import tqdm

# Paths
PROJECT_ROOT = "/home/ahmadreza/Downloads/Research/M2SODAI"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
MAT_DIR = os.path.join(DATA_ROOT, "train")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "baseline_official", "train_hsi_unlabeled_v3_bilinear")

def prepare_train_hsi_unlabeled():
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)
    
    # GLOBAL ROBUST MAX (matching v3)
    GLOBAL_MAX = 0.30
    GLOBAL_MIN = 0.0
    
    # Get all .mat files in train directory
    mat_files = [f for f in os.listdir(MAT_DIR) if f.endswith('.mat')]
    
    print(f"Generating Unlabeled HSI Training Dataset (v3-Bilinear)...")
    for mat_file in tqdm(mat_files):
        mat_path = os.path.join(MAT_DIR, mat_file)
        basename = mat_file.replace('.mat', '.jpg')
        
        try:
            mat_data = sio.loadmat(mat_path)
            # Find the actual data key, usually 'data' or similar
            data_key = next(key for key in mat_data.keys() if not key.startswith('__'))
            hsi = mat_data[data_key]
            
            # Use channels 53, 32, 11 for RGB mapping as done in v3
            if hsi.shape[2] > 53:
                hsi_rgb = hsi[:, :, [53, 32, 11]].astype(np.float32)
            else:
                # Fallback if fewer channels (e.g. if some images are different)
                hsi_rgb = hsi[:, :, :3].astype(np.float32)
            
            # 1. UPSCALE in float32 using BILINEAR
            hsi_rgb_640 = cv2.resize(hsi_rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
            
            # 2. GLOBAL Normalization
            hsi_rgb_640 = (hsi_rgb_640 - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)
            
            # 3. Final uint8 conversion
            final_img = np.clip(hsi_rgb_640 * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(OUTPUT_DIR, "images", basename), final_img)
            
        except Exception as e:
            print(f"Error processing {mat_file}: {e}")

    print(f"Created Unlabeled HSI dataset at {OUTPUT_DIR}")

if __name__ == "__main__":
    prepare_train_hsi_unlabeled()
