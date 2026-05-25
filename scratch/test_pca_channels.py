import os
import sys
import torch
import torch.nn as nn

PROJECT_ROOT = "/home/ahmadreza/Downloads/Research/M2SODAI"
sys.path.append(PROJECT_ROOT)

# Monkey-patch torch.load
original_torch_load = torch.load
def custom_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = custom_torch_load

from hsi_dataset import HSIDataset
import ultralytics.data.utils as data_utils
if 'npy' not in data_utils.IMG_FORMATS:
    data_utils.IMG_FORMATS.add('npy')

from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.tasks import OBBModel

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

# Inject MultiChannelSpectralAdaptor into sys.modules['__main__']
sys.modules['__main__'].MultiChannelSpectralAdaptor = MultiChannelSpectralAdaptor

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
    device = select_device("0")
    print(f"Using device: {device}")

    # Load original RGB model
    ckpt_path = os.path.join(PROJECT_ROOT, "baseline_official/runs_yolo26s_obb/yolo26s_fold2_obb/weights/best.pt")
    print(f"Loading RGB model from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    model = ckpt.get("ema") or ckpt.get("model")
    model = model.to(device)
    model.eval()
    
    # Attach fixed Multi-Channel PCA mapping
    model.spectral_adaptor = MultiChannelSpectralAdaptor(in_channels=30, out_channels=3).to(device)
    
    # Load validation data
    val_yaml = os.path.join(PROJECT_ROOT, "baseline_official/yolo26_pca30_val.yaml")
    from ultralytics.data.utils import check_det_dataset
    val_data = check_det_dataset(val_yaml)
    
    # Ensure model.args is a SimpleNamespace/object with attributes
    from types import SimpleNamespace
    if isinstance(model.args, dict):
        model.args = SimpleNamespace(**model.args)
    
    val_dataset = build_yolo_dataset(
        model.args,
        val_data["val"],
        batch=2,
        data=val_data,
        mode="val",
        rect=True,
        stride=32
    )
    
    val_loader = build_dataloader(
        val_dataset,
        batch=2,
        workers=0,
        shuffle=False,
        rank=-1
    )
    
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Get first batch
    batch = next(iter(val_loader))
    imgs = batch["img"].to(device).float() / 255.0
    
    # Forward pass
    with torch.no_grad():
        preds = model(imgs)
        
    from ultralytics.utils.nms import non_max_suppression
    is_end2end = getattr(model, "end2end", False)
    
    detections = non_max_suppression(
        preds,
        conf_thres=0.001,
        iou_thres=0.45,
        nc=1,
        rotated=True,
        end2end=is_end2end
    )
    
    print("\n--- Multi-Channel PCA (PC1, PC2, PC3 -> R, G, B) ---")
    for idx, det in enumerate(detections):
        print(f"Image {idx}: {len(det)} detections found.")
        if len(det) > 0:
            print(f"  det max confidence: {det[:, 4].max().item():.4f}")
            print(f"  first 3 detections:\n{det[:3]}")

if __name__ == "__main__":
    test()
