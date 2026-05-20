import scipy.io as sio
import cv2
import numpy as np
import os
from tqdm import tqdm
import json

# Paths
PROJECT_ROOT = "/home/ahmadreza/Downloads/Research/M2SODAI"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "baseline_official", "hsi_rgb_640_v3_bilinear_obb")

# Global Stats (matching v3 contrast)
GLOBAL_MAX = 0.30
GLOBAL_MIN = 0.0

def process_split(split_name):
    print(f"\nProcessing {split_name} split...")
    
    # Setup directories
    img_dir = os.path.join(OUTPUT_ROOT, "images", split_name)
    lbl_dir = os.path.join(OUTPUT_ROOT, "labels", split_name)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    
    # Load annotations
    ann_path = os.path.join(DATA_ROOT, f"{split_name}_coco", "annotations_HSI.json")
    if not os.path.exists(ann_path):
        print(f"Annotations not found at {ann_path}")
        return
        
    with open(ann_path) as f:
        coco_data = json.load(f)
        
    # Group annotations by image_id
    img_id_to_anns = {}
    for ann in coco_data['annotations']:
        if ann['category_id'] == 0: # ship
            iid = ann['image_id']
            if iid not in img_id_to_anns:
                img_id_to_anns[iid] = []
            img_id_to_anns[iid].append(ann)
            
    # Process images
    mat_dir = os.path.join(DATA_ROOT, split_name)
    for img_info in tqdm(coco_data['images']):
        img_id = img_info['id']
        basename = os.path.basename(img_info['file_name']) # usually something.mat
        mat_path = os.path.join(mat_dir, basename)
        
        if not os.path.exists(mat_path):
            continue
            
        try:
            # 1. Load HSI and extract RGB
            mat_data = sio.loadmat(mat_path)
            # Some files might use different keys, but 'data' is standard
            hsi = mat_data.get('data')
            if hsi is None:
                data_key = next(key for key in mat_data.keys() if not key.startswith('__'))
                hsi = mat_data[data_key]
                
            # Bands 53, 32, 11
            hsi_rgb = hsi[:, :, [53, 32, 11]].astype(np.float32)
            
            # 2. UPSCALE to 640x640 using Bilinear
            hsi_rgb_640 = cv2.resize(hsi_rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
            
            # 3. GLOBAL Normalization
            hsi_rgb_640 = (hsi_rgb_640 - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)
            
            # 4. Save as uint8 JPG
            final_img = np.clip(hsi_rgb_640 * 255.0, 0, 255).astype(np.uint8)
            out_img_name = basename.replace('.mat', '.jpg')
            cv2.imwrite(os.path.join(img_dir, out_img_name), final_img)
            
            # 5. Generate OBB labels
            lbl_path = os.path.join(lbl_dir, out_img_name.replace('.jpg', '.txt'))
            with open(lbl_path, 'w') as f:
                anns = img_id_to_anns.get(img_id, [])
                for ann in anns:
                    if 'segmentation' in ann and ann['segmentation']:
                        # Normalize by 1600.0 as verified from existing test set script
                        poly = np.array(ann['segmentation'][0]).reshape(-1, 2) / 1600.0
                        obb_str = " ".join([f"{v:.6f}" for v in poly.flatten()])
                        f.write(f"0 {obb_str}\n")
                        
        except Exception as e:
            print(f"Error processing {basename}: {e}")

def create_yaml():
    yaml_path = os.path.join(PROJECT_ROOT, "baseline_official", "hsi_rgb_640_v3_bilinear_obb.yaml")
    content = f"""path: {os.path.abspath(OUTPUT_ROOT)}
train: images/train
val: images/val
test: images/val  # Using val as test for training convenience

nc: 1
names: ['ship']
"""
    with open(yaml_path, 'w') as f:
        f.write(content)
    print(f"\nCreated YOLO YAML at {yaml_path}")

if __name__ == "__main__":
    process_split("train")
    process_split("val")
    create_yaml()
