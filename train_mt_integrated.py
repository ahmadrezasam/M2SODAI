"""
Launch script for Mean Teacher UDA training integrated into YOLO's OBBTrainer.

Usage:
    python train_mt_integrated.py
"""

import os
from uda.mt_obb_trainer import MeanTeacherOBBTrainer

def main():
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"

    # Source: RGB fold 2 (labeled)
    source_data = os.path.join(project_root, "baseline_official/yolo26_fold2_obb.yaml")

    # Target: HSI (unlabeled — labels exist but are not used in training, only for validation)
    target_data = os.path.join(project_root, "baseline_official/hsi_rgb_640_v3_bilinear_obb.yaml")

    # Starting weights: Converged RGB fold 2 model
    best_rgb_weights = os.path.join(
        project_root,
        "baseline_official/runs_yolo26s_obb/yolo26s_fold2_obb/weights/best.pt",
    )

    overrides = {
        "model": best_rgb_weights,
        "data": source_data,        # Source domain (supervised)
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,                  # Reduced for dual forward pass (src + tgt)
        "task": "obb",
        "device": 0,
        "single_cls": True,
        "project": os.path.join(project_root, "runs/mt_integrated"),
        "name": "yolo26s_mt_v9_static_conf",
        "exist_ok": True,
        "patience": 0,             # Disable early stopping entirely
        "save_period": 10,
        "plots": True,
        "cos_lr": True,
        "lr0": 0.0001,               # Lower LR for stable adaptation
        "optimizer": "AdamW",        # Explicitly set to respect custom lr0
    }

    trainer = MeanTeacherOBBTrainer(
        overrides=overrides,
        target_data=target_data,
        val_data=target_data,
        lambda_pseudo=0.05,      # Even lower weight for more stability
        conf_start=0.7,
        conf_end=0.7,            # Static high threshold to avoid noise drift
        lambda_warmup_frac=0.2,
    )

    trainer.train()


if __name__ == "__main__":
    main()
