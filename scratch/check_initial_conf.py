import os
import torch
import torch.nn as nn

# Monkey-patch torch.load for PyTorch 2.6+ compatibility
original_torch_load = torch.load
def custom_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = custom_torch_load

# Monkey-patch Ultralytics to recognize .npy files BEFORE importing dataset modules
import ultralytics.data.utils as data_utils
if 'npy' not in data_utils.IMG_FORMATS:
    data_utils.IMG_FORMATS.add('npy')

from ultralytics import YOLO
from ultralytics.nn.tasks import OBBModel
from hsi_dataset import HSIDataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

def main():
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    best_rgb_weights = os.path.join(
        project_root,
        "baseline_official/runs_yolo26s_obb/yolo26s_fold2_obb/weights/best.pt"
    )
    
    # 1. Load OBB model
    yolo = YOLO(best_rgb_weights)
    model = yolo.model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 2. Attach Spectral Adaptor
    class Spectral3DCNN(nn.Module):
        def __init__(self, in_channels=30, out_channels=3):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            with torch.no_grad():
                self.conv.weight.zero_()
                self.conv.weight[0, 0, 0, 0] = 1.0
                self.conv.weight[1, 1, 0, 0] = 1.0
                self.conv.weight[2, 2, 0, 0] = 1.0

        def forward(self, x):
            x = self.conv(x)
            B, C, H, W = x.shape
            x_flat = x.view(B, -1)
            im_min = x_flat.min(dim=1, keepdim=True)[0]
            im_max = x_flat.max(dim=1, keepdim=True)[0]
            x = (x - im_min.view(B, 1, 1, 1)) / (im_max.view(B, 1, 1, 1) - im_min.view(B, 1, 1, 1) + 1e-8)
            return x

    model.spectral_adaptor = Spectral3DCNN(in_channels=30, out_channels=3).to(device)
    model.eval()

    # Monkey-patch forward
    if not hasattr(OBBModel, "_original_forward"):
        OBBModel._original_forward = OBBModel.forward
    
    def new_forward(self_model, x, *args, **kwargs):
        if isinstance(x, dict):
            img = x["img"]
        else:
            img = x
        is_hsi = img.size(1) == 30
        if hasattr(self_model, "spectral_adaptor") and is_hsi:
            img = self_model.spectral_adaptor(img)
        if isinstance(x, dict):
            x["img"] = img
        else:
            x = img
        return self_model._original_forward(x, *args, **kwargs)
    
    OBBModel.forward = new_forward

    # 3. Load some target training images
    target_data_yaml = os.path.join(project_root, "baseline_official/yolo26_fold2_hsi.yaml")
    from ultralytics.data.utils import check_det_dataset
    data_dict = check_det_dataset(target_data_yaml)
    
    # Load default configuration as hyper-parameters namespace
    hyp = get_cfg(DEFAULT_CFG)
    hyp.imgsz = 640
    
    dataset = HSIDataset(
        img_path=data_dict["train"],
        imgsz=640,
        batch_size=4,
        augment=False,
        hyp=hyp,
        stride=32,
        pad=0.5,
        data=data_dict
    )
    
    from ultralytics.utils.nms import non_max_suppression
    
    batch = dataset[0]
    img_val = batch['img']
    if not isinstance(img_val, torch.Tensor):
        img_tensor = torch.from_numpy(img_val).unsqueeze(0).to(device).float() / 255.0
    else:
        img_tensor = img_val.unsqueeze(0).to(device).float() / 255.0
    
    with torch.no_grad():
        preds = model(img_tensor)
        print(f"preds type: {type(preds)}")
        
        # Test NMS with end2end=True
        dets = non_max_suppression(
            preds,
            conf_thres=0.01,
            iou_thres=0.45,
            nc=1,
            rotated=True,
            end2end=True
        )
        
        print(f"dets len: {len(dets)}")
        if len(dets) > 0:
            print(f"dets[0] shape: {dets[0].shape}")
            if len(dets[0]) > 0:
                print(f"dets[0][0]: {dets[0][0]}")

if __name__ == "__main__":
    main()
