import torch
from repro_uda.dataset import reproUDADataset, collate_fn
from torch.utils.data import DataLoader
import os

def verify():
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    repro_data = os.path.join(project_root, "repro_uda", "data")
    train_txt = os.path.join(repro_data, "fold_1", "train.txt")
    
    dataset = reproUDADataset(repro_data, train_txt, domain='rgb', size=640, augment=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    img, targets, fnames = next(iter(loader))
    
    print(f"Image shape: {img.shape}")
    print(f"Targets shape: {targets.shape}")
    if targets.shape[0] > 0:
        print(f"Target sample row 0: {targets[0].tolist()}")
        # Expected: [batch_idx, cls_id, x, y, w, h]
        # batch_idx should be 0..3, cls_id should be 0.0
        if targets.shape[1] == 6:
            print("SUCCESS: Target has 6 columns.")
            if targets[0, 1] == 0:
                print("SUCCESS: Class ID is preserved as 0.")
            else:
                print(f"FAILURE: Class ID is {targets[0,1]} (Expected 0).")
        else:
            print(f"FAILURE: Target has {targets.shape[1]} columns (Expected 6).")
    else:
        print("No targets in this batch, try again.")

if __name__ == "__main__":
    verify()
