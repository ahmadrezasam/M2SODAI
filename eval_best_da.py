import os
import sys
import torch
import torch.nn as nn
import torch.multiprocessing as mp

# Set sharing strategy to prevent multiprocessing shared memory exhaustion
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
            imgsz=224,
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
# 1. Correct Sigmoid Spectral Adaptor (For Unpickling Namespace)
# ==========================================================
class Spectral3DCNN(nn.Module):
    """
    Trained Spectral Adaptor: 1x1 Conv2d + BatchNorm + Sigmoid.
    Projects 30 HSI channels to 3 channels using Sigmoid activation.
    """
    def __init__(self, in_channels=30, out_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.zero_()
            self.conv.weight[0, 0, 0, 0] = 1.0
            self.conv.weight[1, 1, 0, 0] = 1.0
            self.conv.weight[2, 2, 0, 0] = 1.0

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return torch.sigmoid(x)


# ==========================================================
# 2. OBBModel forward patching with Float16/Float32 safety
# ==========================================================
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
        adaptor = self_model.spectral_adaptor
        # Run adaptor in float32 for BatchNorm numerical stability.
        # Do NOT cast the adaptor module permanently to fp16 — it breaks BN running stats.
        was_half = next(adaptor.parameters()).dtype == torch.float16
        if was_half:
            adaptor = adaptor.float()
        img = adaptor(img.float())
        if was_half:
            adaptor = adaptor.half()
            
        # Ensure spatial dimension is 640x640 for YOLO backbone compatibility.
        if img.shape[-2:] != (640, 640):
            img = torch.nn.functional.interpolate(img, size=(640, 640), mode='bilinear', align_corners=False)
            
        # Match output dtype to the backbone's dtype (e.g. half if the model is half)
        backbone_dtype = img.dtype
        for name, param in self_model.named_parameters():
            if "spectral_adaptor" not in name:
                backbone_dtype = param.dtype
                break
        img = img.to(backbone_dtype)

    if isinstance(x, dict):
        x = x.copy()
        x["img"] = img
    else:
        x = img

    return self_model._original_forward(x, *args, **kwargs)

OBBModel.forward = new_forward


# ==========================================================
# 3. OBBValidator Monkey-Patching for safe validation passes
# ==========================================================
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
    img = batch["img"]
    is_hsi = img.shape[1] == 30

    if is_hsi:
        # HSI PCA data is already float32 — do NOT divide by 255.
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self_val.device, non_blocking=self_val.device.type == "cuda")
        batch["img"] = (batch["img"].half() if self_val.args.half else batch["img"].float())

        if hasattr(self_val, "model") and hasattr(self_val.model, "spectral_adaptor"):
            adaptor = self_val.model.spectral_adaptor
            # Run adaptor in float32 for BN stability, even if validator set model to half
            was_half = next(adaptor.parameters()).dtype == torch.float16
            if was_half:
                adaptor = adaptor.float()
            img = adaptor(batch["img"].float())
            if was_half:
                adaptor = adaptor.half()
            if img.shape[-2:] != (640, 640):
                img = torch.nn.functional.interpolate(img, size=(640, 640), mode='bilinear', align_corners=False)
            
            # Determine the rest of the model's dtype (backbone dtype)
            backbone_dtype = img.dtype
            for name, param in self_val.model.named_parameters():
                if "spectral_adaptor" not in name:
                    backbone_dtype = param.dtype
                    break
            batch["img"] = img.to(backbone_dtype)
    else:
        # RGB data: standard YOLO validation preprocessing (includes /255)
        batch = original_obb_validator_preprocess(self_val, batch)
    return batch

OBBValidator.preprocess = custom_obb_validator_preprocess


# ==========================================
# 4. Evaluation Loop Execution
# ==========================================
from ultralytics import YOLO

def main():
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    test_yaml = os.path.join(project_root, "baseline_official/yolo26_pca30_test.yaml")
    
    # 1. Default to the absolute highest-performing UDA checkpoint (epoch70.pt)
    if len(sys.argv) > 1:
        weights_path = sys.argv[1]
    else:
        weights_path = os.path.join(
            project_root, 
            "runs/mt_dual_branch/dual_branch_mt_full/weights/epoch70.pt"
        )
        
    print("\n" + "="*80)
    print("EVALUATING BEST DOMAIN-ADAPTIVE YOLO MODEL ON PCA30 TEST SET")
    print(f"Weights Path: {weights_path}")
    print(f"Test YAML:    {test_yaml}")
    print("="*80 + "\n")
    
    if not os.path.exists(weights_path):
        print(f"ERROR: Weights checkpoint not found at: {weights_path}!")
        return

    # Load UDA model weights (automatically unpickles Sigmoid Spectral3DCNN adaptor)
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
        workers=0
    )
    
    metrics = results.results_dict
    precision = metrics.get('metrics/precision(B)', 0)
    recall = metrics.get('metrics/recall(B)', 0)
    map50 = metrics.get('metrics/mAP50(B)', 0)
    map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
    
    print("\n" + "="*80)
    print("BEST UDA MODEL TEST SET SUMMARY")
    print("="*80)
    print(f"Model File: {os.path.basename(weights_path)}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"mAP50:      {map50:.4f}")
    print(f"mAP50-95:   {map50_95:.4f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
