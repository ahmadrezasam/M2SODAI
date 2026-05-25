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
from train_dual_branch_mt_integrated import Spectral3DCNN
# Inject Spectral3DCNN into sys.modules['__main__'] so pickle can find it
sys.modules['__main__'].Spectral3DCNN = Spectral3DCNN

import ultralytics.data.utils as data_utils
if 'npy' not in data_utils.IMG_FORMATS:
    data_utils.IMG_FORMATS.add('npy')

from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.tasks import OBBModel

# ------------------------------------------------------------------
# Define the new routing forward pass and monkey patch OBBModel.forward
# ------------------------------------------------------------------
if not hasattr(OBBModel, "_original_forward"):
    OBBModel._original_forward = OBBModel.forward

def new_forward(self_model, x, *args, **kwargs):
    # Ultralytics sometimes passes a dict (training), sometimes a tensor (inference)
    if isinstance(x, dict):
        img = x["img"]
    else:
        img = x
        
    print(f"Inside custom new_forward: input image shape = {img.shape}, dtype = {img.dtype}")
    
    # If the input is HSI (30 channels), route it through the Spectral adaptor
    if hasattr(self_model, "spectral_adaptor") and img.size(1) == 30:
        # Ensure the adaptor dtype matches the input dtype (for AMP/Half-precision compatibility)
        if next(self_model.spectral_adaptor.parameters()).dtype != img.dtype:
            self_model.spectral_adaptor = self_model.spectral_adaptor.to(img.dtype)
        
        print("Routing through spectral_adaptor...")
        img = self_model.spectral_adaptor(img)
        print(f"After spectral_adaptor: shape = {img.shape}, min = {img.min().item():.4f}, max = {img.max().item():.4f}")
        
    # Put the 3-channel pseudo-RGB (or original RGB) back
    if isinstance(x, dict):
        x["img"] = img
    else:
        x = img

    # Ensure the output tensor matches the main model's weight dtype
    main_dtype = torch.float32
    for name, param in self_model.named_parameters():
        if "spectral_adaptor" not in name:
            main_dtype = param.dtype
            break

    if isinstance(x, dict):
        x["img"] = x["img"].to(main_dtype)
    else:
        x = x.to(main_dtype)

    # Pass to standard YOLO
    return self_model._original_forward(x, *args, **kwargs)

OBBModel.forward = new_forward


def diagnose():
    device = select_device("0")
    print(f"Using device: {device}")

    # 1. Load the checkpoint
    ckpt_path = os.path.join(PROJECT_ROOT, "runs/mt_dual_branch/dual_branch_mt_fold2/weights/last.pt")
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    model = ckpt.get("ema") or ckpt.get("model")
    if model is None:
        raise ValueError("Both 'ema' and 'model' keys in checkpoint are None!")
        
    model = model.to(device)
    model.eval()
    
    print("\nModel information:")
    print(f"Model type: {type(model)}")
    print(f"Has spectral_adaptor: {hasattr(model, 'spectral_adaptor')}")
    if hasattr(model, 'spectral_adaptor'):
        print(f"Spectral adaptor conv weight shape: {model.spectral_adaptor.conv.weight.shape}")
    
    # 2. Load validation data
    val_yaml = os.path.join(PROJECT_ROOT, "baseline_official/yolo26_pca30_val.yaml")
    from ultralytics.data.utils import check_det_dataset
    val_data = check_det_dataset(val_yaml)
    
    print("\nBuilding validation dataset...")
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
    
    # 3. Get first batch
    batch = next(iter(val_loader))
    imgs = batch["img"].to(device).float()
    print(f"\nRaw batch images shape: {imgs.shape}")
    
    # Apply preprocessing division by 255
    imgs_preprocessed = imgs / 255.0
    print(f"Preprocessed batch images shape: {imgs_preprocessed.shape}")
    
    # 4. Forward pass
    print("\nRunning model forward...")
    with torch.no_grad():
        preds = model(imgs_preprocessed)
        
    print(f"Predictions type: {type(preds)}")
    if isinstance(preds, (list, tuple)):
        print(f"Predictions len: {len(preds)}")
        for idx, p in enumerate(preds):
            if isinstance(p, torch.Tensor):
                print(f"  preds[{idx}] shape: {p.shape}")
            else:
                print(f"  preds[{idx}] type: {type(p)}")
                
    # Let's inspect OBB detections using non_max_suppression
    from ultralytics.utils.nms import non_max_suppression
    is_end2end = getattr(model, "end2end", False)
    
    # Print max logit values in predictions to see if they are active
    print(f"Max class score logit: {preds[0][:, 4, :].max().item():.4f}")
    
    detections = non_max_suppression(
        preds,
        conf_thres=0.001,  # Try extremely low confidence first to see if there's any response
        iou_thres=0.45,
        nc=1,
        rotated=True,
        end2end=is_end2end
    )
    
    print("\nDetections (conf_thres=0.001):")
    for idx, det in enumerate(detections):
        print(f"Image {idx}: {len(det)} detections found.")
        if len(det) > 0:
            print(f"  det shape: {det.shape}")
            print(f"  det max confidence: {det[:, 4].max().item():.4f}")
            print(f"  det samples (first 5):\n{det[:5]}")

if __name__ == "__main__":
    diagnose()
