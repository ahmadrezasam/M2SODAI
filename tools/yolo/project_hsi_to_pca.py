import json
import os
import numpy as np
import scipy.io as sio
import cv2
from tqdm import tqdm
from sklearn.decomposition import PCA

def project_hsi_to_pca(gt_json_path, hsi_dir, output_dir, n_components=3):
    """
    Fits PCA on the HSI test set and projects 127 bands to top n components.
    """
    with open(gt_json_path, 'r') as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Collect pixels for PCA fitting
    print("Collecting pixels for PCA fitting...")
    all_pixels = []
    
    # Sample from each image to build the fitting set
    for img_info in tqdm(data['images'][:50]): # Sample first 50 images for speed
        basename = os.path.basename(img_info['file_name'])
        mat_name = basename.replace('.jpg', '.mat')
        mat_path = os.path.join(hsi_dir, mat_name)
        
        if os.path.exists(mat_path):
            hsi = sio.loadmat(mat_path)['data'] # (H, W, 127)
            hsi_flat = hsi.reshape(-1, 127)
            # Sample 1000 pixels per image
            idx = np.random.choice(hsi_flat.shape[0], 1000, replace=False)
            all_pixels.append(hsi_flat[idx])
            
    X = np.concatenate(all_pixels, axis=0)
    
    # 2. Fit PCA
    print(f"Fitting PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    # 3. Project and Save
    print(f"Projecting {len(data['images'])} images to {output_dir}...")
    for img_info in tqdm(data['images']):
        basename = os.path.basename(img_info['file_name'])
        mat_name = basename.replace('.jpg', '.mat')
        mat_path = os.path.join(hsi_dir, mat_name)
        
        if not os.path.exists(mat_path):
            continue
            
        hsi = sio.loadmat(mat_path)['data']
        h, w, c = hsi.shape
        hsi_flat = hsi.reshape(-1, c)
        
        # Project
        pca_img = pca.transform(hsi_flat).reshape(h, w, n_components)
        
        # Normalize to [0, 255]
        for i in range(n_components):
            c_min, c_max = pca_img[:,:,i].min(), pca_img[:,:,i].max()
            if c_max > c_min:
                pca_img[:,:,i] = (pca_img[:,:,i] - c_min) / (c_max - c_min) * 255.0
            else:
                pca_img[:,:,i] = pca_img[:,:,i] * 255.0
                
        pca_img = np.clip(pca_img, 0, 255).astype(np.uint8)
        
        # Resize to 1600x1600
        pca_resized = cv2.resize(pca_img, (1600, 1600), interpolation=cv2.INTER_CUBIC)
        
        out_path = os.path.join(output_dir, basename)
        cv2.imwrite(out_path, pca_resized)

if __name__ == "__main__":
    gt = "data/test_coco/annotations.json"
    hsi_src = "data/test"
    output = "data/yolo_pca_eval/test/images"
    
    project_hsi_to_pca(gt, hsi_src, output)
    print(f"Successfully projected PCA images to {output}")
