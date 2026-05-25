import os
import sys
import numpy as np
import torch
from ultralytics import YOLO

# Monkey-patch torch.load for PyTorch 2.6+ compatibility
original_torch_load = torch.load
def custom_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = custom_torch_load

def debug_inference():
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    model_path = os.path.join(project_root, "baseline_official/runs_yolo26s_obb/yolo26s_fold2_obb/weights/best.pt")
    
    # Let's find a .npy file and its corresponding label
    images_dir = os.path.join(project_root, "baseline_official/hsi_pca_30ch_eval/val/images")
    labels_dir = os.path.join(project_root, "baseline_official/hsi_pca_30ch_eval/val/labels")
    
    npy_files = [f for f in os.listdir(images_dir) if f.endswith(".npy")]
    if not npy_files:
        print("No .npy files found in validation images dir!")
        return
        
    print(f"Found {len(npy_files)} .npy files.")
    
    # Load the first 3 npy files and run inference
    model = YOLO(model_path)
    
    for filename in npy_files[:3]:
        npy_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, filename.replace(".npy", ".txt"))
        
        print(f"\n--- Debugging file: {filename} ---")
        
        # Load ground truth labels
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                print(f"Ground Truth boxes count: {len(lines)}")
                for line in lines[:3]:
                    print("GT Box:", line.strip())
        else:
            print("No ground truth labels found!")
            
        # Load PCA .npy image (Shape: H, W, 30)
        im = np.load(npy_path)
        print("PCA array shape:", im.shape, "dtype:", im.dtype)
        
        # Extract PC1 (channel 0)
        pc1 = im[:, :, 0]
        
        # Apply min-max normalization to [0.0, 1.0]
        pc1_min = pc1.min()
        pc1_max = pc1.max()
        pc1_norm = (pc1 - pc1_min) / (pc1_max - pc1_min + 1e-8)
        
        # Replicate to 3 channels (Shape: H, W, 3)
        pseudo_rgb = np.stack([pc1_norm, pc1_norm, pc1_norm], axis=2) # Shape: (H, W, 3)
        
        print("Running inference with float32 [0.0, 1.0]...")
        results_float = model.predict(pseudo_rgb, imgsz=640, conf=0.001, device=0)
        
        obb = results_float[0].obb
        if obb is not None:
            print(f"Float32 inference detected OBB boxes count: {len(obb)}")
            for i in range(min(len(obb), 5)):
                cls_val = obb.cls[i].item()
                conf_val = obb.conf[i].item()
                xywhr_val = obb.xywhr[i].tolist()
                print(f"  OBB {i} - cls: {cls_val}, conf: {conf_val:.4f}, xywhr: {xywhr_val}")
        else:
            print("Float32 inference: obb is None!")
            
        print("\nRunning inference with uint8 [0, 255]...")
        pseudo_rgb_uint8 = (pseudo_rgb * 255.0).astype(np.uint8)
        results_uint8 = model.predict(pseudo_rgb_uint8, imgsz=640, conf=0.001, device=0)
        
        obb_u8 = results_uint8[0].obb
        if obb_u8 is not None:
            print(f"Uint8 inference detected OBB boxes count: {len(obb_u8)}")
            for i in range(min(len(obb_u8), 5)):
                cls_val = obb_u8.cls[i].item()
                conf_val = obb_u8.conf[i].item()
                xywhr_val = obb_u8.xywhr[i].tolist()
                print(f"  OBB {i} - cls: {cls_val}, conf: {conf_val:.4f}, xywhr: {xywhr_val}")
        else:
            print("Uint8 inference: obb is None!")

if __name__ == "__main__":
    debug_inference()
