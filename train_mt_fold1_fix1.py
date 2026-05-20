"""
Launch script for online Mean Teacher Unsupervised Domain Adaptation (UDA) on Fold 1.
Includes Fix 1 (Confidence-Weighted Pseudo-Label Loss) and the NMS Dimension Correction.
"""

import os
from uda.mt_obb_trainer import MeanTeacherOBBTrainer

def main():
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"

    # Source: RGB Fold 1 (labeled)
    source_data = os.path.join(project_root, "baseline_official/yolo26_fold1_obb.yaml")

    # Target: HSI (unlabeled bilinear upscaled)
    target_data = os.path.join(project_root, "baseline_official/hsi_rgb_640_v3_bilinear_obb.yaml")

    # Starting weights: Converged RGB Fold 1 model
    best_rgb_weights = os.path.join(
        project_root,
        "baseline_official/runs_yolo26s_obb/yolo26s_fold1_obb/weights/best.pt",
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
        "name": "yolo26s_mt_fold1_adaptive",
        "exist_ok": True,
        "patience": 0,             # Disable early stopping to train full duration
        "save_period": 10,
        "plots": True,
        "cos_lr": True,
        "lr0": 0.0001,               # Stable learning rate
        "optimizer": "AdamW",        # Explicitly set to AdamW
        "workers": 4,                # Optimize worker thread count to prevent CPU/IO deadlocks
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

    print("\n" + "="*80)
    print("STARTING ONLINE MEAN TEACHER UDA TRAINING FOR FOLD 1 WITH FIX 1")
    print("="*80 + "\n")
    trainer.train()

if __name__ == "__main__":
    main()
