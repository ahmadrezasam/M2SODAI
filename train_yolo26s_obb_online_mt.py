import os
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.utils import DEFAULT_CFG
from hsi_dataset import HSIDataset
from uda.online_mean_teacher import OnlineMeanTeacherTrainer
from ultralytics.data.utils import check_det_dataset
from torch.utils.data import DataLoader

def train_mt():
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    source_data_path = os.path.join(project_root, "baseline_official/yolo26_fold2_obb.yaml")
    target_data_path = os.path.join(project_root, "baseline_official/hsi_rgb_640_v3_bilinear_obb.yaml")
    
    source_data = check_det_dataset(source_data_path)
    target_data = check_det_dataset(target_data_path)

    # 1. Models
    # Use the CONVERGED Fold 2 RGB weights as starting point
    best_rgb_weights = os.path.join(project_root, 'baseline_official', 'runs_yolo26s_obb', 'yolo26s_fold2_obb', 'weights', 'best.pt')
    
    overrides = {
        "model": best_rgb_weights,
        "data": source_data_path,
        "imgsz": 640,
        "batch": 8,
        "task": "obb",
        "mode": "train",
        "epochs": 100,
    }
    from copy import deepcopy
    trainer = OBBTrainer(overrides=overrides)
    trainer._setup_train() 
    student = trainer.model
    
    # Initialize teacher from the same converged weights
    teacher = deepcopy(student)
    
    # 2. Datasets & Loaders
    source_set = HSIDataset(
        img_path=source_data['train'],
        imgsz=640,
        augment=True,
        task='obb',
        data=source_data
    )
    source_loader = DataLoader(source_set, batch_size=8, shuffle=True, num_workers=4, collate_fn=source_set.collate_fn)
    
    target_set = HSIDataset(
        img_path=target_data['train'],
        imgsz=640,
        augment=False, 
        task='obb',
        data=target_data
    )
    target_loader = DataLoader(target_set, batch_size=8, shuffle=True, num_workers=4, collate_fn=target_set.collate_fn)
    
    # 3. Trainer
    cfg = {
        "epochs": 100,
        "burnin_epochs": 5,  # Short burn-in since we start from converged weights
        "lr": 5e-5,          # OneCycleLR peak LR — ramps from ~5e-7 up to this
        "ema_decay": 0.999,
        "lambda_pseudo": 1.0,
        "imgsz": 640,
        "save_dir": "runs/online_mt_yolo26s_v3_fixed"
    }
    
    mt_trainer = OnlineMeanTeacherTrainer(
        student_model=student,
        teacher_model=teacher,
        source_loader=source_loader,
        target_loader=target_loader,
        cfg=cfg,
        device='cuda' # Use cpu for stability first, then move to cuda if available
    )
    
    mt_trainer.train()

if __name__ == "__main__":
    train_mt()
