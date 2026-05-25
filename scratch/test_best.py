import os
import sys
project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
from ultralytics import YOLO

# Monkey-patch torch.load for PyTorch 2.6+ compatibility
original_torch_load = torch.load
def custom_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = custom_torch_load

# Monkey-patch Ultralytics to recognize .npy files
import ultralytics.data.utils as data_utils
if 'npy' not in data_utils.IMG_FORMATS:
    data_utils.IMG_FORMATS.add('npy')

# Monkey-patch build_yolo_dataset to return HSIDataset for target/validation data
import ultralytics.data.build as build_module
from hsi_dataset import HSIDataset
original_build_yolo_dataset = build_module.build_yolo_dataset

def custom_build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    if "hsi" in str(img_path).lower() or "pca" in str(img_path).lower():
        print(f"Building HSIDataset for validation: {img_path}")
        return HSIDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=False,
            hyp=cfg,
            rect=rect,
            cache=cfg.cache,
            single_cls=cfg.single_cls,
            stride=stride,
            pad=0.0,
            data=data,
            classes=cfg.classes,
            fraction=1.0,
        )
    return original_build_yolo_dataset(cfg, img_path, batch, data, mode=mode, rect=rect, stride=stride, multi_modal=multi_modal)

build_module.build_yolo_dataset = custom_build_yolo_dataset

# Define Spectral3DCNN and attach it to loaded model
class Spectral3DCNN(nn.Module):
    def __init__(self, in_channels=30, out_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, 0, 0, 0] = 1.0

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        x_flat = x.view(B, -1)
        im_min = x_flat.min(dim=1, keepdim=True)[0]
        im_max = x_flat.max(dim=1, keepdim=True)[0]
        x = (x - im_min.view(B, 1, 1, 1)) / (im_max.view(B, 1, 1, 1) - im_min.view(B, 1, 1, 1) + 1e-8)
        return x

def test():
    model_path = os.path.join(project_root, "runs/mt_dual_branch/dual_branch_mt_fold2/weights/best.pt")
    val_data = os.path.join(project_root, "baseline_official/yolo26_pca30_val.yaml")

    # Load YOLO wrapper
    yolo_model = YOLO(model_path)
    model = yolo_model.model
    
    print("\n--- Inspecting loaded model structure ---")
    print(f"Has spectral_adaptor attribute: {hasattr(model, 'spectral_adaptor')}")
    if hasattr(model, 'spectral_adaptor'):
        print(f"Spectral Conv Weight shape: {model.spectral_adaptor.conv.weight.shape}")
        print(f"PC1 weight component: {model.spectral_adaptor.conv.weight[:, 0, 0, 0]}")
        print(f"PC2 weight component: {model.spectral_adaptor.conv.weight[:, 1, 0, 0]}")

    # Explicitly monkey-patch OBBModel class's forward function in case it wasn't
    if not hasattr(model.__class__, "_original_forward"):
        model.__class__._original_forward = model.__class__.forward

    def new_forward(self_model, x, *args, **kwargs):
        if isinstance(x, dict):
            img = x["img"]
        else:
            img = x
            
        if hasattr(self_model, "spectral_adaptor") and img.size(1) == 30:
            if next(self_model.spectral_adaptor.parameters()).dtype != img.dtype:
                self_model.spectral_adaptor = self_model.spectral_adaptor.to(img.dtype)
            img = self_model.spectral_adaptor(img)
            
        if isinstance(x, dict):
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

        return self_model.__class__._original_forward(self_model, x, *args, **kwargs)

    model.__class__.forward = new_forward

    print("\n--- Running Validation ---")
    results = yolo_model.val(
        data=val_data,
        imgsz=640,
        device=0,
        task="obb",
        batch=2,
        workers=0
    )
    
    print("\nValidation results:")
    print(results.results_dict)

if __name__ == "__main__":
    test()
