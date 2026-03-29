import os
import sys
import torch
import yaml
from torch.utils.data import DataLoader

# Add project root and LMW-YOLO to path
project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
lmw_yolo_dir = os.path.join(project_root, "mmdet/models/LMW-YOLO")
if lmw_yolo_dir not in sys.path:
    sys.path.insert(0, lmw_yolo_dir)

from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import select_device
import ultralytics.nn.tasks as tasks
from ultralytics.modifiednn.modules.block import LKCA, MSDP

from uda.trainer import UDATrainer

def setup_dataloaders(data_src_yaml, data_tgt_yaml, batch_size=8, imgsz=640):
    # Load configs
    with open(data_src_yaml, 'r') as f:
        src_cfg = yaml.safe_load(f)
    with open(data_tgt_yaml, 'r') as f:
        tgt_cfg = yaml.safe_load(f)

    device = select_device(0)
    
    # Setup ultralytics args (hacky but necessary for build_yolo_dataset)
    from types import SimpleNamespace
    args = SimpleNamespace(**DEFAULT_CFG.__dict__)
    args.imgsz = imgsz
    args.batch = batch_size
    args.rect = False
    args.single_cls = False
    args.task = 'detect'
    args.workers = 2
    args.augment = True
    args.mosaic = 1.0
    args.mixup = 0.0
    args.copy_paste = 0.0
    args.degrees = 0.0
    args.translate = 0.1
    args.scale = 0.5
    args.shear = 0.0
    args.perspective = 0.0
    args.flipud = 0.0
    args.fliplr = 0.5
    args.bgr = 0.0
    args.hsv_h = 0.015
    args.hsv_s = 0.7
    args.hsv_v = 0.4
    args.overlap_mask = True
    args.mask_ratio = 4
    args.fraction = 1.0

    # Build datasets
    src_dataset = build_yolo_dataset(args, src_cfg['train'], batch_size, src_cfg, mode='train', stride=32)
    tgt_dataset = build_yolo_dataset(args, tgt_cfg['test'], batch_size, tgt_cfg, mode='train', stride=32)
    val_dataset = build_yolo_dataset(args, tgt_cfg['test'], batch_size, tgt_cfg, mode='val', stride=32)

    # Build dataloaders
    src_loader = build_dataloader(src_dataset, batch=batch_size, workers=2, shuffle=True, rank=-1)
    tgt_loader = build_dataloader(tgt_dataset, batch=batch_size, workers=2, shuffle=True, rank=-1)
    val_loader = build_dataloader(val_dataset, batch=batch_size, workers=4, shuffle=False, rank=-1)

    return src_loader, tgt_loader, val_loader

def main():
    # 1. Inject modules
    setattr(tasks, 'LKCA', LKCA)
    setattr(tasks, 'MSDP', MSDP)

    # 2. Paths
    data_src = os.path.join(project_root, "configs/yolo/dataset_rgb.yaml")
    data_tgt = os.path.join(project_root, "data/yolo_hsi/test/dataset.yaml") # using test as unlabeled for now
    weights = os.path.join(project_root, "yolo11n_lmw_init.pt")

    # 3. Model
    model = YOLO(weights)
    student = model.model
    device = select_device(0)
    student.to(device)

    # 4. Dataloaders
    src_loader, tgt_loader, val_loader = setup_dataloaders(data_src, data_tgt, batch_size=4)

    # 5. Trainer
    trainer = UDATrainer(
        student_model=student,
        source_loader=src_loader,
        target_loader=tgt_loader,
        val_loader=val_loader,
        cfg={
            "epochs": 30,
            "conf_thresh": 0.7,
            "lambda_pseudo": 0.5,
            "save_dir": "runs/detect/lmw_yolo_uda_simple"
        }
    )

    # 6. Start training
    print("Starting Simplified UDA Training...")
    trainer.train()

if __name__ == "__main__":
    main()
