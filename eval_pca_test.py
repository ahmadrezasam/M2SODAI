import os
import sys
import torch
import torch.nn as nn
import torch.multiprocessing as mp

# Set sharing strategy to 'file_system' to prevent multiprocessing shared memory exhaustion
mp.set_sharing_strategy('file_system')

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

# Global Monkey-patching for build_yolo_dataset to return HSIDataset for target/validation data
import ultralytics.data.build as build_module
import ultralytics.data as data_module
import ultralytics.models.yolo.detect.val as detect_val
import ultralytics.models.yolo.obb.val as obb_val

original_build_yolo_dataset = build_module.build_yolo_dataset

def custom_build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    if "hsi" in str(img_path).lower() or "pca" in str(img_path).lower():
        print(f"[Dataset] Building HSIDataset for evaluation path: {img_path} at 224x224 native resolution")
        return HSIDataset(
            img_path=img_path,
            imgsz=224,               # Load at native 224x224 to eliminate CPU resize bottlenecks
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

# ==========================================
# OBBModel Monkey-patching
# ==========================================
from ultralytics.nn.tasks import OBBModel

if not hasattr(OBBModel, "_original_forward"):
    OBBModel._original_forward = OBBModel.forward

def new_forward(self_model, x, *args, **kwargs):
    if isinstance(x, dict):
        img = x["img"]
    else:
        img = x
        
    is_hsi = img.size(1) == 30
    has_adaptor = hasattr(self_model, "spectral_adaptor")

    if has_adaptor and is_hsi:
        if next(self_model.spectral_adaptor.parameters()).dtype != img.dtype:
            self_model.spectral_adaptor = self_model.spectral_adaptor.to(img.dtype)
        img = self_model.spectral_adaptor(img)
        
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

OBBModel.forward = new_forward

# ==========================================
# Spectral-Specific Adaptor (needed for model loading)
# ==========================================
class Spectral3DCNN(nn.Module):
    def __init__(self, in_channels=30, out_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 0, 0, 0] = 1.0
            self.conv.weight[1, 1, 0, 0] = 1.0
            self.conv.weight[2, 2, 0, 0] = 1.0
            self.bn.weight.fill_(1.0)
            self.bn.bias.fill_(0.0)

    def forward(self, x):
        x = self.conv(x)
        
        # Batch-vectorised min-max normalization per-image to [0.0, 1.0]
        B, C, H, W = x.shape
        x_flat = x.view(B, -1)
        im_min = x_flat.min(dim=1, keepdim=True)[0]
        im_max = x_flat.max(dim=1, keepdim=True)[0]
        x = (x - im_min.view(B, 1, 1, 1)) / (im_max.view(B, 1, 1, 1) - im_min.view(B, 1, 1, 1) + 1e-8)
        
        # Force BatchNorm to use evaluation mode (constant mean=0, var=1)
        # to prevent train-eval discrepancy and ensure perfect stability.
        self.bn.eval()
        x = self.bn(x)
        return x

# ==========================================
# OBBValidator Preprocessor Monkey-Patching for Safe Upscaling & Correct Coordinates
# ==========================================
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
    if img.size(1) == 30 and hasattr(self_val, "model") and hasattr(self_val.model, "spectral_adaptor"):
        adaptor = self_val.model.spectral_adaptor
        if next(adaptor.parameters()).dtype != img.dtype:
            self_val.model.spectral_adaptor = adaptor.to(img.dtype)
            adaptor = self_val.model.spectral_adaptor
            
        # 1. Project 30 channels -> 3 channels (Pseudo-RGB)
        img = adaptor(img)
        
        # 2. Resize 3 channels -> 640x640 (saves 90% memory and aligns coordinates perfectly!)
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
    
    # Use specified checkpoint from CLI, default to the newly trained supervised Fold 2 weights
    if len(sys.argv) > 1:
        weights_path = sys.argv[1]
    else:
        weights_path = os.path.join(
            project_root, 
            "runs/supervised_30ch/yolo26s_hsi_supervised_fold2/weights/best.pt"
        )
        
    print("\n" + "="*80)
    print("EVALUATING SPATIAL-SPECTRAL YOLO MODEL ON PCA30 TEST SET")
    print(f"Weights Path: {weights_path}")
    print(f"Test YAML:    {test_yaml}")
    print("="*80 + "\n")
    
    if not os.path.exists(weights_path):
        print(f"ERROR: Weights checkpoint not found at: {weights_path}!")
        return

    # Load custom model weights (automatically restores OBB spatial layers + Spectral3DCNN adaptor)
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
        project=os.path.join(project_root, "runs/supervised_30ch/eval_pca_test")
    )
    
    metrics = results.results_dict
    precision = metrics.get('metrics/precision(B)', 0)
    recall = metrics.get('metrics/recall(B)', 0)
    map50 = metrics.get('metrics/mAP50(B)', 0)
    map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
    
    print("\n" + "="*80)
    print("HSI PCA30 TEST SET EVALUATION SUMMARY")
    print("="*80)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"mAP50:     {map50:.4f}")
    print(f"mAP50-95:  {map50_95:.4f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
