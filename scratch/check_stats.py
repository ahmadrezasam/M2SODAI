import cv2
import numpy as np
import os

path = 'baseline_official/test_hsi_rgb_640_v3_bilinear/images'
images = os.listdir(path)[:10]

means = []
stds = []

for img_name in images:
    img = cv2.imread(os.path.join(path, img_name))
    if img is None: continue
    means.append(np.mean(img, axis=(0,1)))
    stds.append(np.std(img, axis=(0,1)))

print("HSI Test Set Stats (BGR):")
print(f"Mean: {np.mean(means, axis=0)}")
print(f"Std:  {np.mean(stds, axis=0)}")

# Check Training set as well
train_path = 'baseline_official/fold_1/train/images'
train_images = os.listdir(train_path)[:10]
t_means = []
t_stds = []
for img_name in train_images:
    img = cv2.imread(os.path.join(train_path, img_name))
    if img is None: continue
    t_means.append(np.mean(img, axis=(0,1)))
    t_stds.append(np.std(img, axis=(0,1)))

print("\nRGB Training Set Stats (BGR):")
print(f"Mean: {np.mean(t_means, axis=0)}")
print(f"Std:  {np.mean(t_stds, axis=0)}")
