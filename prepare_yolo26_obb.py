import os
import json
import numpy as np

PROJECT_ROOT = "/home/ahmadreza/Downloads/Research/M2SODAI"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "baseline_official")

def load_segmentation_map():
    """Load all segmentations from source JSONs for lookup."""
    seg_map = {}
    for split in ["train_coco", "val_coco"]:
        json_path = os.path.join(DATA_ROOT, split, "annotations.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            data = json.load(f)
        
        # Map basename -> list of segmentations
        img_id_to_name = {img['id']: os.path.basename(img['file_name']) for img in data['images']}
        for ann in data['annotations']:
            if ann['category_id'] == 0: # ship
                name = img_id_to_name[ann['image_id']]
                if name not in seg_map: seg_map[name] = []
                if 'segmentation' in ann and ann['segmentation']:
                    seg_map[name].append(ann['segmentation'][0])
    return seg_map

def prepare_all_folds_obb():
    seg_map = load_segmentation_map()
    
    for fold_num in [1, 2, 3]:
        print(f"Generating OBB labels for Fold {fold_num}...")
        source_fold = os.path.join(OUTPUT_BASE, f"fold_{fold_num}")
        output_fold = os.path.join(OUTPUT_BASE, f"fold_{fold_num}_obb")
        
        if not os.path.exists(source_fold):
            print(f"  Warning: Source folder {source_fold} not found. Skipping.")
            continue

        for subset in ["train", "val"]:
            src_img_dir = os.path.join(source_fold, subset, "images")
            if not os.path.exists(src_img_dir):
                continue
                
            dst_img_dir = os.path.join(output_fold, subset, "images")
            dst_lbl_dir = os.path.join(output_fold, subset, "labels")
            os.makedirs(dst_img_dir, exist_ok=True)
            os.makedirs(dst_lbl_dir, exist_ok=True)
            
            for img_name in os.listdir(src_img_dir):
                if not img_name.endswith(".jpg"): continue
                
                # 1. Symlink image
                src_path = os.path.join(src_img_dir, img_name)
                dst_path = os.path.join(dst_img_dir, img_name)
                if not os.path.exists(dst_path):
                    os.symlink(os.path.abspath(src_path), dst_path)
                
                # 2. Write OBB labels
                lbl_name = img_name.replace(".jpg", ".txt")
                with open(os.path.join(dst_lbl_dir, lbl_name), 'w') as f:
                    if img_name in seg_map:
                        for seg in seg_map[img_name]:
                            # Normalize corners (1600x1600 space)
                            poly = np.array(seg).reshape(-1, 2) / 1600.0
                            obb_str = " ".join([f"{v:.6f}" for v in poly.flatten()])
                            f.write(f"0 {obb_str}\n")

        # Create data.yaml
        yaml_path = os.path.join(OUTPUT_BASE, f"yolo26_fold{fold_num}_obb.yaml")
        with open(yaml_path, 'w') as f:
            f.write(f"path: {output_fold}\n")
            f.write(f"train: train/images\n")
            f.write(f"val: val/images\n")
            f.write(f"names:\n  0: ship\n")
        print(f"  Done! Created {yaml_path}")

if __name__ == "__main__":
    prepare_all_folds_obb()
