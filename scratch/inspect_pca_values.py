import os
import sys
import numpy as np

PROJECT_ROOT = "/home/ahmadreza/Downloads/Research/M2SODAI"

def inspect():
    images_dir = os.path.join(PROJECT_ROOT, "baseline_official/hsi_pca_30ch_eval/val/images")
    labels_dir = os.path.join(PROJECT_ROOT, "baseline_official/hsi_pca_30ch_eval/val/labels")
    
    npy_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".npy")])
    for filename in npy_files:
        npy_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, filename.replace(".npy", ".txt"))
        
        if not os.path.exists(label_path):
            continue
            
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        if len(lines) == 0:
            continue
            
        # Let's inspect the first image that has a ship
        print(f"\nInspecting {filename} with {len(lines)} ground truth OBBs")
        im = np.load(npy_path) # Shape: (224, 224, 30)
        
        # Let's look at the first OBB
        # Format of OBB: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized coordinates)
        parts = list(map(float, lines[0].strip().split()))
        cls_id = int(parts[0])
        coords = np.array(parts[1:]).reshape(4, 2)
        
        # Convert coords to pixel space
        coords_px = coords * 224
        print("OBB pixel coordinates:\n", coords_px)
        
        # Let's get the center of the ship
        center_x = int(np.mean(coords_px[:, 0]))
        center_y = int(np.mean(coords_px[:, 1]))
        print(f"Center pixel: x={center_x}, y={center_y}")
        
        # Print PCA values around center of ship (3x3 region)
        cx = np.clip(center_x, 1, 222)
        cy = np.clip(center_y, 1, 222)
        
        # Ship PCA values
        ship_region = im[cy-1:cy+2, cx-1:cx+2]
        
        # Water/background PCA values (top-left corner of the image)
        water_region = im[0:5, 0:5]
        
        print("\n--- Mean values for first 5 PCA components ---")
        for c in range(5):
            ship_mean = ship_region[:, :, c].mean()
            water_mean = water_region[:, :, c].mean()
            print(f"PC{c+1} | Ship: {ship_mean:.4f} | Water: {water_mean:.4f} | Diff: {ship_mean - water_mean:.4f}")
            
        break

if __name__ == "__main__":
    inspect()
