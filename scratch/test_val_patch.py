import os
import sys
import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = "/home/ahmadreza/Downloads/Research/M2SODAI"
sys.path.append(PROJECT_ROOT)

# Monkey-patch torch.load
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

# Dummy Image for verify checks
class DummyImage:
    def __init__(self):
        self.format = 'PNG'
        self.size = (640, 640)
    def verify(self):
        pass
    def getexif(self):
        return None

original_image_open = Image.open
def custom_image_open(fp, *args, **kwargs):
    if str(fp).endswith('.npy'):
        return DummyImage()
    return original_image_open(fp, *args, **kwargs)
Image.open = custom_image_open

# Global Monkey-Patching for build_yolo_dataset
from hsi_dataset import HSIDataset
import ultralytics.data.build as build_module
import ultralytics.data as data_module
import ultralytics.models.yolo.detect.val as detect_val
import ultralytics.models.yolo.obb.val as obb_val

original_build_yolo_dataset = build_module.build_yolo_dataset

def custom_build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    if "hsi" in str(img_path).lower() or "pca" in str(img_path).lower():
        print(f"[GLOBAL PATCH] Building HSIDataset for target/validation: {img_path}")
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
            task=cfg.task,
        )
    return original_build_yolo_dataset(cfg, img_path, batch, data, mode=mode, rect=rect, stride=stride, multi_modal=multi_modal)

# Apply global monkey-patch
build_module.build_yolo_dataset = custom_build_yolo_dataset
data_module.build_yolo_dataset = custom_build_yolo_dataset
detect_val.build_yolo_dataset = custom_build_yolo_dataset
if hasattr(obb_val, "build_yolo_dataset"):
    obb_val.build_yolo_dataset = custom_build_yolo_dataset

class MultiChannelSpectralAdaptor(nn.Module):
    def __init__(self, in_channels=30, out_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            # Map PC1 -> R (channel 0)
            self.conv.weight[0, 0, 0, 0] = 1.0
            # Map PC2 -> G (channel 1)
            self.conv.weight[1, 1, 0, 0] = 1.0
            # Map PC3 -> B (channel 2)
            self.conv.weight[2, 2, 0, 0] = 1.0

    def forward(self, x):
        # x shape: (B, 30, H, W)
        x = self.conv(x)
        B, C, H, W = x.shape
        x_flat = x.view(B, -1)
        im_min = x_flat.min(dim=1, keepdim=True)[0]
        im_max = x_flat.max(dim=1, keepdim=True)[0]
        x = (x - im_min.view(B, 1, 1, 1)) / (im_max.view(B, 1, 1, 1) - im_min.view(B, 1, 1, 1) + 1e-8)
        return x

sys.modules['__main__'].MultiChannelSpectralAdaptor = MultiChannelSpectralAdaptor

# Monkey patch OBBModel's forward pass
from ultralytics.nn.tasks import OBBModel
if not hasattr(OBBModel, "_original_forward"):
    OBBModel._original_forward = OBBModel.forward

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

    return self_model._original_forward(x, *args, **kwargs)

OBBModel.forward = new_forward

def test():
    from ultralytics import YOLO
    
    # Load original pre-trained RGB model weights
    rgb_weights_path = os.path.join(PROJECT_ROOT, "baseline_official/runs_yolo26s_obb/yolo26s_fold2_obb/weights/best.pt")
    print(f"\nLoading pre-trained RGB model from: {rgb_weights_path}")
    
    yolo_model = YOLO(rgb_weights_path)
    model = yolo_model.model
    
    # Attach our MultiChannelSpectralAdaptor
    model.spectral_adaptor = MultiChannelSpectralAdaptor(in_channels=30, out_channels=3).to("cuda:0")
    
    val_yaml = os.path.join(PROJECT_ROOT, "baseline_official/yolo26_pca30_val.yaml")
    
    print("\n--- Running Validation with GLOBAL Patch ---")
    results = yolo_model.val(
        data=val_yaml,
        imgsz=640,
        device=0,
        task="obb",
        batch=2,
        workers=0
    )
    
    print("\n--- Validation Results ---")
    print("Metrics dict:", results.results_dict)

if __name__ == "__main__":
    test()
