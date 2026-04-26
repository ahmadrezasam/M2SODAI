import torch
import cv2
import os
import numpy as np
from repro_uda.baseline_model import YOLOv5s
from repro_uda.dataset import reproUDADataset
from repro_uda.eval import xywh_to_xyxy
import torchvision

def visualize_debug(fold=1, domain='hsi_rgb'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    repro_data = os.path.join(project_root, "repro_uda", "data")
    val_txt = os.path.join(repro_data, f"fold_{fold}", "val.txt")
    
    # Load one image
    dataset = reproUDADataset(repro_data, val_txt, domain=domain, size=640, augment=False)
    img_tensor, targets, fname = dataset[0]
    
    # Model
    model = YOLOv5s(nc=1).to(device)
    ckpt_path = os.path.join(project_root, "repro_uda", "runs", f"baseline_fold_{fold}", "best.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        print(f"Loaded {ckpt_path}")
    
    model.eval()
    with torch.no_grad():
        preds = model(img_tensor.unsqueeze(0).to(device))
        if isinstance(preds, tuple):
            decoded = preds[0]
        else:
            decoded = preds
            
    # Post-process
    p = decoded[0]
    p[:, 5:] *= p[:, 4:5]
    conf, cls = p[:, 5:].max(1)
    
    # Use very low threshold to see "something"
    mask = p[:, 4] > 0.001
    p = p[mask]
    boxes = xywh_to_xyxy(p[:, :4])
    conf = conf[mask]
    
    nms_indices = torchvision.ops.nms(boxes, conf, 0.45)
    boxes = boxes[nms_indices].cpu().numpy()
    conf = conf[nms_indices].cpu().numpy()
    
    # Visualize
    img_cv = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Draw GT
    for t in targets:
        # t: [cls, x, y, w, h] normalized
        cx, cy, w, h = t[1:] * 640
        x1, y1 = int(cx - w/2), int(cy - h/2)
        x2, y2 = int(cx + w/2), int(cy + h/2)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    # Draw Preds (Top 5)
    for i in range(min(5, len(boxes))):
        box = boxes[i].astype(int)
        cv2.rectangle(img_cv, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.putText(img_cv, f"{conf[i]:.4f}", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
    save_path = os.path.join(project_root, "debug_pred.jpg")
    cv2.imwrite(save_path, img_cv)
    print(f"Saved debug visualization to {save_path}")

if __name__ == "__main__":
    visualize_debug()
