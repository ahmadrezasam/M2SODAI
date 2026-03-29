import json
import os
import numpy as np
import scipy.io as sio
import cv2
from tqdm import tqdm

def project_hsi_to_rgb(gt_json_path, hsi_dir, output_dir, r_idx=53, g_idx=32, b_idx=11):
    """
    Projects 127-band HSI .mat files to 3-channel pseudo-RGB JPEGs.
    """
    with open(gt_json_path, 'r') as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Projecting {len(data['images'])} images to {output_dir}...")
    
    for img_info in tqdm(data['images']):
        # Mapping from JPEGImages/0.jpg to data/test/0.mat
        basename = os.path.basename(img_info['file_name'])
        mat_name = basename.replace('.jpg', '.mat')
        mat_path = os.path.join(hsi_dir, mat_name)
        
        if not os.path.exists(mat_path):
            # Try recursive find if not in top level
            continue
            
        # Load HSI
        try:
            hsi = sio.loadmat(mat_path)['data']
        except Exception as e:
            print(f"Error loading {mat_path}: {e}")
            continue
            
        # Select bands (B, G, R order for OpenCV)
        indices = [b_idx, g_idx, r_idx]
        rgb = hsi[:, :, indices].astype(np.float32)
        
        # Min-Max Scaling per channel to [0, 255]
        for c in range(3):
            c_min, c_max = rgb[:,:,c].min(), rgb[:,:,c].max()
            if c_max > c_min:
                rgb[:,:,c] = (rgb[:,:,c] - c_min) / (c_max - c_min) * 255.0
            else:
                rgb[:,:,c] = rgb[:,:,c] * 255.0
                
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        
        # Resize to 1600x1600 (original annot space)
        rgb_resized = cv2.resize(rgb, (1600, 1600), interpolation=cv2.INTER_CUBIC)
        
        # Save as JPEG
        out_path = os.path.join(output_dir, basename)
        cv2.imwrite(out_path, rgb_resized)

if __name__ == "__main__":
    gt = "data/test_coco/annotations.json"
    hsi_src = "data/test"
    output = "data/yolo_hsi/test/images"
    
    project_hsi_to_rgb(gt, hsi_src, output)
    print(f"Successfully projected HSI images to {output}")
