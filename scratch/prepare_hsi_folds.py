import os
import glob

PROJECT_ROOT = "/home/ahmadreza/Downloads/Research/M2SODAI"
HSI_NPY_ROOT = os.path.join(PROJECT_ROOT, "data/hsi_npy")
BASELINE_ROOT = os.path.join(PROJECT_ROOT, "baseline_official")

def prepare_hsi_folds():
    for fold_num in [1, 2, 3]:
        print(f"Preparing Fold {fold_num} for HSI...")
        src_fold = os.path.join(BASELINE_ROOT, f"fold_{fold_num}_obb")
        dst_fold = os.path.join(BASELINE_ROOT, f"fold_{fold_num}_hsi")
        
        if not os.path.exists(src_fold):
            print(f"  Warning: Source folder {src_fold} not found. Skipping.")
            continue
            
        for subset in ["train", "val"]:
            # Create directories
            os.makedirs(os.path.join(dst_fold, subset, "images"), exist_ok=True)
            os.makedirs(os.path.join(dst_fold, subset, "labels"), exist_ok=True)
            
            # Symlink labels (same as RGB/PCA)
            src_lbl_dir = os.path.join(src_fold, subset, "labels")
            dst_lbl_dir = os.path.join(dst_fold, subset, "labels")
            for lbl in glob.glob(os.path.join(src_lbl_dir, "*.txt")):
                dst_lbl = os.path.join(dst_lbl_dir, os.path.basename(lbl))
                if not os.path.exists(dst_lbl):
                    os.symlink(os.path.abspath(lbl), dst_lbl)
            
            # Symlink HSI npy files
            # The npy files were saved in data/hsi_npy/train_pca/*.npy etc.
            # We need to match the filename from the labels
            pca_subdir = f"{subset}_pca"
            for lbl in os.listdir(dst_lbl_dir):
                base_name = lbl.replace(".txt", "")
                npy_src = os.path.join(HSI_NPY_ROOT, pca_subdir, f"{base_name}.npy")
                npy_dst = os.path.join(dst_fold, subset, "images", f"{base_name}.npy")
                
                if os.path.exists(npy_src):
                    if not os.path.exists(npy_dst):
                        os.symlink(os.path.abspath(npy_src), npy_dst)
                else:
                    print(f"  Warning: Missing HSI npy for {base_name} at {npy_src}")

        # Create data.yaml
        yaml_path = os.path.join(BASELINE_ROOT, f"yolo26_fold{fold_num}_hsi.yaml")
        with open(yaml_path, 'w') as f:
            f.write(f"path: {dst_fold}\n")
            f.write(f"train: train/images\n")
            f.write(f"val: val/images\n")
            f.write(f"names:\n  0: ship\n")
        print(f"  Created {yaml_path}")

if __name__ == "__main__":
    prepare_hsi_folds()
