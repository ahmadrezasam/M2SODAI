import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import cv2
import torch
import torch.multiprocessing as mp

# Disable OpenCV multithreading to prevent memory leaks in dataloader workers
cv2.setNumThreads(0)

# Set sharing strategy to 'file_system' to prevent shared memory (/dev/shm) exhaustion crashes when using workers > 0
mp.set_sharing_strategy('file_system')

# Set number of PyTorch CPU threads to 1 to prevent CPU core over-subscription and thrashing inside worker processes
torch.set_num_threads(1)

from uda.mt_obb_trainer import MeanTeacherOBBTrainer

def run_fold(fold_num):
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"

    print("\n" + "="*80)
    print(f"STARTING ONLINE MEAN TEACHER UDA TRAINING FOR FOLD {fold_num}")
    print("="*80 + "\n")

    # Source: RGB Fold (labeled)
    source_data = os.path.join(project_root, f"baseline_official/yolo26_fold{fold_num}_obb.yaml")

    # Target: HSI (unlabeled bilinear upscaled)
    target_data = os.path.join(project_root, "baseline_official/hsi_rgb_640_v3_bilinear_obb.yaml")

    # Starting weights: Converged RGB model for this fold
    best_rgb_weights = os.path.join(
        project_root,
        f"baseline_official/runs_yolo26s_obb/yolo26s_fold{fold_num}_obb/weights/best.pt",
    )

    overrides = {
        "model": best_rgb_weights,
        "data": source_data,        # Source domain (supervised)
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,                  # Dual forward pass batch size
        "task": "obb",
        "device": 0,
        "single_cls": True,
        "project": os.path.join(project_root, "runs/mt_integrated"),
        "name": f"yolo26s_mt_fold{fold_num}_adaptive",
        "exist_ok": True,
        "patience": 0,             # Disable early stopping to train full duration
        "save_period": 10,
        "plots": True,
        "cos_lr": True,
        "lr0": 0.0001,               # Stable learning rate
        "optimizer": "AdamW",        # Explicitly set to AdamW
        "workers": 0,                # Set to 0 to run everything in the main process, eliminating multiprocessing OOM leaks
    }

    trainer = MeanTeacherOBBTrainer(
        overrides=overrides,
        target_data=target_data,
        val_data=target_data,
        lambda_pseudo=0.20,      # Increased for sufficient target signal
        conf_start=0.50,         # Start at 0.50 to capture early predictions
        conf_end=0.35,           # Decay to 0.35 protected by Fix 1
        lambda_warmup_frac=0.2,  # 20% warmup fraction (20 epochs)
    )

    trainer.train()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        fold = int(sys.argv[1])
        run_fold(fold)
    else:
        # Default: run both Fold 2 and Fold 3
        run_fold(1)
        run_fold(2)
        run_fold(3)
