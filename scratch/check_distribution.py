import scipy.io as sio
import numpy as np
import os
from tqdm import tqdm

DATA_ROOT = "/home/ahmadreza/Downloads/Research/M2SODAI/data"
TRAIN_MAT_DIR = os.path.join(DATA_ROOT, "train")

def check_distribution(dirs):
    all_values = []
    mat_files = []
    for d in dirs:
        mat_files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.mat')]
    
    import random
    random.seed(42)
    mat_files = random.sample(mat_files, 200)
    
    for mat_path in tqdm(mat_files):
        hsi = sio.loadmat(mat_path)['data']
        hsi_rgb = hsi[:, :, [53, 32, 11]].astype(np.float32)
        all_values.append(hsi_rgb.flatten())
    
    all_values = np.concatenate(all_values)
    
    percentiles = [90, 95, 99, 99.9, 99.99, 100]
    results = {p: np.percentile(all_values, p) for p in percentiles}
    
    print("\nPercentiles for Training Set (RGB bands):")
    for p, val in results.items():
        print(f"{p}%: {val}")

if __name__ == "__main__":
    check_distribution([TRAIN_MAT_DIR])
