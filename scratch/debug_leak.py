import os
import sys
import torch
import numpy as np
import psutil

# Add project root to path
sys.path.append("/home/ahmadreza/Downloads/Research/M2SODAI")

from train_dual_branch_mt_integrated import DualBranchMTTrainer, Spectral3DCNN
from hsi_dataset import HSIDataset

def print_memory(label):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[{label}] Resident Memory: {mem_mb:.2f} MB")

def test_leak():
    print("Initializing test...")
    print_memory("Start")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build a dummy target batch (batch of 4, HSI 30 channels, 224x224 resized to 640x640 eventually)
    # We will mimic the trainer's forward pass
    img = torch.rand((4, 30, 224, 224), dtype=torch.float32).to(device)
    target_batch = {"img": img}

    # Load starting model weights
    best_rgb_weights = "/home/ahmadreza/Downloads/Research/M2SODAI/baseline_official/runs_yolo26s_obb/yolo26s_fold2_obb/weights/best.pt"
    
    print("Loading model...")
    from ultralytics import YOLO
    yolo = YOLO(best_rgb_weights)
    model = yolo.model.to(device)
    model.spectral_adaptor = Spectral3DCNN(in_channels=30, out_channels=3).to(device)
    
    # Instantiate trainer and mock the necessary attributes
    print("Mocking trainer...")
    trainer = DualBranchMTTrainer(
        overrides={
            "model": best_rgb_weights,
            "data": "/home/ahmadreza/Downloads/Research/M2SODAI/baseline_official/yolo26_fold2_obb.yaml",
            "imgsz": 640,
            "batch": 4,
            "device": 0,
            "workers": 0,
        },
        target_data="/home/ahmadreza/Downloads/Research/M2SODAI/baseline_official/yolo26_pca30_full.yaml",
        val_data="/home/ahmadreza/Downloads/Research/M2SODAI/baseline_official/yolo26_pca30_val.yaml",
        lambda_pseudo=0.2,
        conf_start=0.3,
        conf_end=0.25,
        lambda_warmup_frac=0.0,
        burn_in_epochs=0,
    )
    
    # Bind necessary fields
    trainer.device = device
    trainer.ema = type('EMA', (object,), {'ema': model})() # Mock EMA
    trainer.amp = True
    
    print_memory("Before loop")
    
    # Run loop to observe memory leak
    for step in range(50):
        # Run forward pass through spectral adaptor first
        with torch.no_grad():
            adapted_batch = target_batch.copy()
            adapted_img = model.spectral_adaptor(target_batch["img"])
            if adapted_img.shape[-2:] != (640, 640):
                adapted_img = torch.nn.functional.interpolate(adapted_img, size=(640, 640), mode='bilinear', align_corners=False)
            # Standardize scale [0, 255] as returned by dataloader
            adapted_batch["img"] = (adapted_img * 255.0).clamp(0, 255)
            
            pseudo_batch, n_pls, mean_conf = trainer._generate_pseudo_labels(adapted_batch, conf_thresh=0.30)
            
        if step % 5 == 0:
            print_memory(f"Iteration {step} (PLs generated: {n_pls})")

    print_memory("After loop")

if __name__ == "__main__":
    test_leak()
