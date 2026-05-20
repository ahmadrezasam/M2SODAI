import scipy.io as sio
import cv2
import numpy as np
import os
from tqdm import tqdm
import json

# Paths
PROJECT_ROOT = "/home/ahmadreza/Downloads/Research/M2SODAI"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
MAT_DIR = os.path.join(DATA_ROOT, "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "baseline_official", "test_hsi_rgb_640_v3_bilinear")
YAML_PATH = os.path.join(PROJECT_ROOT, "baseline_official", "test_hsi_rgb_640_v3_bilinear.yaml")

def upscale_v3_bilinear():
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)
    
    with open(os.path.join(DATA_ROOT, "test_coco", "annotations.json")) as f:
        test_data = json.load(f)
    with open(os.path.join(DATA_ROOT, "test_coco", "annotations_HSI.json")) as f:
        hsi_data = json.load(f)
        
    hsi_img_anns = {}
    for ann in hsi_data['annotations']:
        if ann['category_id'] == 0:
            iid = ann['image_id']
            if iid not in hsi_img_anns:
                hsi_img_anns[iid] = []
            hsi_img_anns[iid].append(ann)

    # GLOBAL ROBUST MAX (matching v3)
    GLOBAL_MAX = 0.30
    GLOBAL_MIN = 0.0
    
    print(f"Generating 'v3-Bilinear' Dataset (Bilinear + Global Normalization)...")
    for img_info in tqdm(test_data['images']):
        img_id = img_info['id']
        basename = os.path.basename(img_info['file_name'])
        mat_path = os.path.join(MAT_DIR, basename.replace('.jpg', '.mat'))
        
        if not os.path.exists(mat_path):
            continue
            
        hsi = sio.loadmat(mat_path)['data']
        hsi_rgb = hsi[:, :, [53, 32, 11]].astype(np.float32)
        
        # 1. UPSCALE in float32 using BILINEAR
        hsi_rgb_640 = cv2.resize(hsi_rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        
        # 2. GLOBAL Normalization
        hsi_rgb_640 = (hsi_rgb_640 - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)
        
        # 3. Final uint8 conversion
        final_img = np.clip(hsi_rgb_640 * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "images", basename), final_img)
        
        # 4. Labels
        lbl_path = os.path.join(OUTPUT_DIR, "labels", basename.replace('.jpg', '.txt'))
        with open(lbl_path, 'w') as f:
            for ann in hsi_img_anns.get(img_id, []):
                x, y, w, h = ann['bbox']
                f.write(f"0 {(x + w/2)/224.0:.6f} {(y + h/2)/224.0:.6f} {w/224.0:.6f} {h/224.0:.6f}\n")

    with open(YAML_PATH, 'w') as f:
        f.write(f"path: {os.path.abspath(OUTPUT_DIR)}\ntrain: images\nval: images\nnc: 1\nnames: ['ship']\n")
    print(f"Created v3-Bilinear test set at {OUTPUT_DIR}")

if __name__ == "__main__":
    upscale_v3_bilinear()
