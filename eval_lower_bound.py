import os
import sys
import torch
import torch.nn as nn
import torch.multiprocessing as mp

# Set sharing strategy to 'file_system' to prevent multiprocessing shared memory exhaustion
mp.set_sharing_strategy('file_system')

# Set PyTorch threads to 1 to prevent CPU core over-subscription inside dataloader
torch.set_num_threads(1)

# Monkey-patch torch.load for safe weights loading in PyTorch 2.6+
original_torch_load = torch.load
def custom_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = custom_torch_load

from hsi_dataset import HSIDataset

# Monkey-patch Ultralytics to recognize .npy files
import ultralytics.data.utils as data_utils
if 'npy' not in data_utils.IMG_FORMATS:
    data_utils.IMG_FORMATS.add('npy')

from PIL import Image
original_image_open = Image.open

class DummyImage:
    def __init__(self):
        self.format = 'PNG'
        self.size = (640, 640)
    def verify(self):
        pass
    def getexif(self):
        return None

def custom_image_open(fp, *args, **kwargs):
    if str(fp).endswith('.npy'):
        return DummyImage()
    return original_image_open(fp, *args, **kwargs)

Image.open = custom_image_open

# Global Monkey-patching for build_yolo_dataset to return HSIDataset for evaluation
import ultralytics.data.build as build_module
import ultralytics.data as data_module
import ultralytics.models.yolo.detect.val as detect_val
import ultralytics.models.yolo.obb.val as obb_val

original_build_yolo_dataset = build_module.build_yolo_dataset

def custom_build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    if "hsi" in str(img_path).lower() or "pca" in str(img_path).lower():
        print(f"[Dataset] Building HSIDataset for baseline evaluation path: {img_path} at 224x224 native resolution")
        return HSIDataset(
            img_path=img_path,
            imgsz=224,               # Load at native 224x224
            batch_size=batch,
            augment=mode == 'train',
            hyp=cfg,
            rect=rect,
            cache=cfg.cache,
            single_cls=cfg.single_cls,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            data=data,
            classes=cfg.classes,
            fraction=cfg.fraction if mode == "train" else 1.0,
            task=cfg.task,
        )
    return original_build_yolo_dataset(cfg, img_path, batch, data, mode=mode, rect=rect, stride=stride, multi_modal=multi_modal)

build_module.build_yolo_dataset = custom_build_yolo_dataset
data_module.build_yolo_dataset = custom_build_yolo_dataset
detect_val.build_yolo_dataset = custom_build_yolo_dataset
if hasattr(obb_val, "build_yolo_dataset"):
    obb_val.build_yolo_dataset = custom_build_yolo_dataset

# ==========================================================
# OBBModel Monkey-patching: Bypasses adaptor & slices PC1-3
# ==========================================================
from ultralytics.nn.tasks import OBBModel

if not hasattr(OBBModel, "_original_forward"):
    OBBModel._original_forward = OBBModel.forward

def baseline_forward(self_model, x, *args, **kwargs):
    if isinstance(x, dict):
        img = x["img"]
    else:
        img = x
        
    is_hsi = img.size(1) == 30

    if is_hsi:
        # Lower bound: Select first 3 channels (PC1, PC2, PC3) and normalize to [0.0, 1.0]
        img = img[:, :3]
        B, C, H, W = img.shape
        img_flat = img.view(B, -1)
        im_min = img_flat.min(dim=1, keepdim=True)[0]
        im_max = img_flat.max(dim=1, keepdim=True)[0]
        img = (img - im_min.view(B, 1, 1, 1)) / (im_max.view(B, 1, 1, 1) - im_min.view(B, 1, 1, 1) + 1e-8)
        
        # GPU-side resize to 640x640 for spatial feature transfer compatibility
        if img.shape[-2:] != (640, 640):
            img = torch.nn.functional.interpolate(img, size=(640, 640), mode='bilinear', align_corners=False)
        
    if isinstance(x, dict):
        x = x.copy()
        x["img"] = img
    else:
        x = img

    main_dtype = torch.float32
    for name, param in self_model.named_parameters():
        if "spectral_adaptor" not in name:
            main_dtype = param.dtype
            break

    if isinstance(x, dict):
        x["img"] = x["img"].to(main_dtype)
    else:
        x = x.to(main_dtype)

    return self_model._original_forward(x, *args, **kwargs)

OBBModel.forward = baseline_forward

# =========================================================================
# OBBValidator Preprocessor Monkey-Patching: Slices PC1-3 and normalizes
# =========================================================================
from ultralytics.models.yolo.obb import OBBValidator

original_obb_validator_call = OBBValidator.__call__
def custom_obb_validator_call(self_val, *args, **kwargs):
    trainer = kwargs.get("trainer", None)
    if len(args) > 0:
        trainer = args[0]
    model = kwargs.get("model", None)
    if len(args) > 1:
        model = args[1]
        
    if trainer is not None:
        self_val.model = getattr(trainer, "ema", None) and trainer.ema.ema or trainer.model
    elif model is not None:
        self_val.model = model
        
    return original_obb_validator_call(self_val, *args, **kwargs)
OBBValidator.__call__ = custom_obb_validator_call

original_obb_validator_preprocess = OBBValidator.preprocess
def custom_obb_validator_preprocess(self_val, batch):
    batch = original_obb_validator_preprocess(self_val, batch)
    img = batch["img"]
    if img.size(1) == 30:
        # Lower bound: Select first 3 channels (PC1, PC2, PC3) and normalize to [0.0, 1.0]
        img = img[:, :3]
        B, C, H, W = img.shape
        img_flat = img.view(B, -1)
        im_min = img_flat.min(dim=1, keepdim=True)[0]
        im_max = img_flat.max(dim=1, keepdim=True)[0]
        img = (img - im_min.view(B, 1, 1, 1)) / (im_max.view(B, 1, 1, 1) - im_min.view(B, 1, 1, 1) + 1e-8)
        
        # GPU-side resize to 640x640 (saves 90% memory and aligns coordinates perfectly!)
        if img.shape[-2:] != (640, 640):
            img = torch.nn.functional.interpolate(
                img, size=(640, 640), mode='bilinear', align_corners=False
            )
        batch["img"] = img
    return batch
OBBValidator.preprocess = custom_obb_validator_preprocess

# ==========================================
# Run Test Evaluation
# ==========================================
from ultralytics import YOLO

def main():
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    test_yaml = os.path.join(project_root, "baseline_official/yolo26_pca30_test.yaml")
    
    # Use specified pre-trained RGB weights, default to the converged Fold 2 supervised RGB weights
    if len(sys.argv) > 1:
        weights_path = sys.argv[1]
    else:
        weights_path = os.path.join(
            project_root, 
            "baseline_official/runs_yolo26s_obb/yolo26s_fold2_obb/weights/best.pt"
        )
        
    print("\n" + "="*80)
    print("EVALUATING PCA LOWER-BOUND BASELINE ON PCA30 TEST SET")
    print(f"Weights Path: {weights_path}")
    print(f"Test YAML:    {test_yaml}")
    print("="*80 + "\n")
    
    if not os.path.exists(weights_path):
        print(f"ERROR: Weights checkpoint not found at: {weights_path}!")
        return

    # Load pre-trained spatial weights
    model = YOLO(weights_path)
    
    # Run OBB validation/evaluation
    results = model.val(
        data=test_yaml,
        imgsz=640,
        batch=16,
        device=0,
        single_cls=True,
        plots=True,
        save_json=False,
        workers=0,
        project=os.path.join(project_root, "runs/supervised_30ch/eval_pca_baseline")
    )
    
    metrics = results.results_dict
    precision = metrics.get('metrics/precision(B)', 0)
    recall = metrics.get('metrics/recall(B)', 0)
    map50 = metrics.get('metrics/mAP50(B)', 0)
    map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
    
    print("\n" + "="*80)
    print("PCA LOWER-BOUND BASELINE EVALUATION SUMMARY")
    print("="*80)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"mAP50:     {map50:.4f}")
    print(f"mAP50-95:  {map50_95:.4f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
