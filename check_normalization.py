import numpy as np
import cv2
import os
from tqdm import tqdm

def get_stats(image_dir):
    means = []
    stds = []
    for img_name in tqdm(os.listdir(image_dir)[:200]):
        if not img_name.endswith(('.jpg', '.png')): continue
        img = cv2.imread(os.path.join(image_dir, img_name))
        if img is None: continue
        img = img.astype(np.float32) / 255.0
        means.append(np.mean(img, axis=(0, 1)))
        stds.append(np.std(img, axis=(0, 1)))
    return np.mean(means, axis=0), np.mean(stds, axis=0)

train_dir = "/home/ahmadreza/Downloads/Research/M2SODAI/baseline_official/fold_1/train/images"
test_dir = "/home/ahmadreza/Downloads/Research/M2SODAI/data/yolo_hsi/test/images"

print("Analyzing Training Set (Optical)...")
train_mean, train_std = get_stats(train_dir)
print(f"Train Mean: {train_mean}, Std: {train_std}")

print("\nAnalyzing Test Set (HSI-RGB)...")
test_mean, test_std = get_stats(test_dir)
print(f"Test Mean: {test_mean}, Std: {test_std}")
