import os
import sys
import torch
import torch.nn as nn

# Define Spectral3DCNN in __main__ namespace to satisfy PyTorch load serialization
class Spectral3DCNN(nn.Module):
    """
    Optimized Spectral Adaptor using a 1x1 2D Convolution.
    Initializes to map PC1 -> R (channel 0), PC2 -> G (channel 1), PC3 -> B (channel 2)
    of PCA-compressed target HSI input, followed by image-level min-max normalization
    to project the 30-channel HSI into a premium pseudo-RGB colorful representation.
    """
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
        
        # Batch-vectorised min-max normalization per-image in the batch to [0.0, 1.0]
        B, C, H, W = x.shape
        x_flat = x.view(B, -1)
        im_min = x_flat.min(dim=1, keepdim=True)[0]
        im_max = x_flat.max(dim=1, keepdim=True)[0]
        
        # Normalize and reshape back
        x = (x - im_min.view(B, 1, 1, 1)) / (im_max.view(B, 1, 1, 1) - im_min.view(B, 1, 1, 1) + 1e-8)
        return x

# Inject Spectral3DCNN into sys.modules['__main__'] as well as train_dual_branch_mt_integrated
sys.modules['__main__'].Spectral3DCNN = Spectral3DCNN

# Add parent directory to path so we can import train_dual_branch_mt_integrated
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import train_dual_branch_mt_integrated to activate all monkey patches!
import train_dual_branch_mt_integrated
from ultralytics import YOLO

def evaluate_all():
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    weights_dir = os.path.join(project_root, "runs/mt_dual_branch/dual_branch_mt_fold2/weights")
    val_data = os.path.join(project_root, "baseline_official/yolo26_pca30_val.yaml")

    checkpoints = ["epoch0.pt", "epoch10.pt", "epoch20.pt", "epoch30.pt", "epoch40.pt", "epoch50.pt", "epoch60.pt", "best.pt", "last.pt"]
    
    print("\n" + "="*80)
    print("EVALUATING FOLD 2 MEAN TEACHER CHECKPOINTS ON HSI VAL SET")
    print("="*80 + "\n")

    results_table = []

    for cp in checkpoints:
        cp_path = os.path.join(weights_dir, cp)
        if not os.path.exists(cp_path):
            print(f"[-] Checkpoint {cp} not found, skipping...")
            continue
            
        print(f"\n[+] Loading checkpoint {cp}...")
        try:
            model = YOLO(cp_path)
            
            # Run validation
            results = model.val(
                data=val_data,
                split='val',
                imgsz=640,
                batch=8,
                device=0,
                single_cls=True,
                plots=False,
                save_json=False,
                verbose=False
            )
            
            metrics = results.results_dict
            precision = metrics.get('metrics/precision(B)', 0.0)
            recall = metrics.get('metrics/recall(B)', 0.0)
            map50 = metrics.get('metrics/mAP50(B)', 0.0)
            map50_95 = metrics.get('metrics/mAP50-95(B)', 0.0)
            
            results_table.append({
                "Checkpoint": cp,
                "Precision": precision,
                "Recall": recall,
                "mAP50": map50,
                "mAP50-95": map50_95
            })
            
            print(f"[*] {cp} metrics: Precision={precision:.4f}, Recall={recall:.4f}, mAP50={map50:.4f}, mAP50-95={map50_95:.4f}")
            
        except Exception as e:
            print(f"[!] Error evaluating {cp}: {e}")

    print("\n" + "="*80)
    print("SUMMARY OF FOLD 2 MEAN TEACHER CHECKPOINT METRICS ON HSI VAL SET")
    print("="*80)
    print(f"{'Checkpoint':<15} | {'Precision':<10} | {'Recall':<10} | {'mAP50':<10} | {'mAP50-95':<10}")
    print("-"*80)
    for r in results_table:
        print(f"{r['Checkpoint']:<15} | {r['Precision']:<10.4f} | {r['Recall']:<10.4f} | {r['mAP50']:<10.4f} | {r['mAP50-95']:<10.4f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    evaluate_all()
