"""
Cross-validation training for Mean Teacher UDA (Folds 1 and 3).
Uses the stable settings from v9_static_conf.
"""

import os
from uda.mt_obb_trainer import MeanTeacherOBBTrainer

def run_fold(fold_num):
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    
    print(f"\n{'='*60}\nSTARTING MEAN TEACHER UDA - FOLD {fold_num}\n{'='*60}\n")

    # Source: RGB Fold (labeled)
    source_data = os.path.join(project_root, f"baseline_official/yolo26_fold{fold_num}_obb.yaml")

    # Target: HSI (unlabeled validation split)
    target_data = os.path.join(project_root, "baseline_official/hsi_rgb_640_v3_bilinear_obb.yaml")

    # Starting weights: Converged RGB model for this fold
    best_weights = os.path.join(
        project_root,
        f"baseline_official/runs_yolo26s_obb/yolo26s_fold{fold_num}_obb/weights/best.pt",
    )

    overrides = {
        "model": best_weights,
        "data": source_data,
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "task": "obb",
        "device": 0,
        "single_cls": True,
        "project": os.path.join(project_root, "runs/mt_integrated"),
        "name": f"yolo26s_mt_fold{fold_num}_stable",
        "exist_ok": True,
        "patience": 0,               # No early stopping
        "save_period": 10,
        "plots": True,
        "cos_lr": True,
        "lr0": 0.0001,               # Stable LR
        "optimizer": "AdamW",        # Fixed optimizer
    }

    trainer = MeanTeacherOBBTrainer(
        overrides=overrides,
        target_data=target_data,
        val_data=target_data,
        lambda_pseudo=0.05,          # Stable weight
        conf_start=0.7,
        conf_end=0.7,                # Static high confidence
        lambda_warmup_frac=0.2,
    )

    trainer.train()

if __name__ == "__main__":
    # Run Fold 1
    run_fold(1)
    
    # Run Fold 3
    run_fold(3)
