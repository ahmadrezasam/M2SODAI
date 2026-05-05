"""
Prepare data for official YOLOv5 lower bound baseline reproduction.

Paper methodology:
- Combine 988 train + 124 val optical images → 1112 total
- 3-fold cross-validation splits
- Test set: 125 HSI images (RGB and PCA compositions)
- Ship-only detection (single class)

This script creates:
1. 3-fold CV directories with RGB optical images + labels
3. HSI-PCA test images (first 3 principal components)
4. YAML configs for YOLOv5
"""

import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold
import shutil

PROJECT_ROOT = "/home/ahmadreza/Downloads/Research/M2SODAI"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "baseline_official")


def load_coco_annotations(json_path, category_id=0):
    """Load COCO annotations and filter for a specific category (default: ship=0)."""
    with open(json_path) as f:
        data = json.load(f)
    
    # Build image id -> info mapping
    img_lookup = {img['id']: img for img in data['images']}
    
    # Group annotations by image, filter for target category
    img_anns = {}
    for ann in data['annotations']:
        if ann['category_id'] == category_id:
            iid = ann['image_id']
            if iid not in img_anns:
                img_anns[iid] = []
            img_anns[iid].append(ann)
    
    return data['images'], img_anns, img_lookup


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """Convert COCO bbox [x_min, y_min, w, h] to YOLO [x_center, y_center, w, h] normalized."""
    x_min, y_min, bw, bh = bbox
    x_c = (x_min + bw / 2) / img_w
    y_c = (y_min + bh / 2) / img_h
    nw = bw / img_w
    nh = bh / img_h
    return x_c, y_c, nw, nh


def prepare_optical_data():
    """
    Step 1: Combine train + val optical data → create 3-fold CV splits.
    All 1112 optical images are used (including those without ships as background).
    """
    print("=" * 60)
    print("Step 1: Preparing 3-fold CV optical data")
    print("=" * 60)
    
    all_images = []
    all_anns = {}
    
    for split_name, coco_dir, jpeg_dir in [
        ("train", "train_coco", "train_coco"),
        ("val", "val_coco", "val_coco"),
    ]:
        json_path = os.path.join(DATA_ROOT, coco_dir, "annotations.json")
        images, img_anns, img_lookup = load_coco_annotations(json_path, category_id=0)
        
        for img_info in images:
            basename = os.path.basename(img_info['file_name'])  # e.g., "1186.jpg"
            img_id = img_info['id']
            
            # Full path to source RGB image
            rgb_path = os.path.join(DATA_ROOT, coco_dir, img_info['file_name'])
            
            if not os.path.exists(rgb_path):
                print(f"  Warning: RGB image not found: {rgb_path}")
                continue
            
            entry = {
                'basename': basename,
                'rgb_path': rgb_path,
                'width': img_info['width'],
                'height': img_info['height'],
                'anns': img_anns.get(img_id, []),  # ship annotations (may be empty)
                'split': split_name,
            }
            all_images.append(entry)
    
    print(f"  Total optical images: {len(all_images)}")
    imgs_with_ships = sum(1 for e in all_images if len(e['anns']) > 0)
    print(f"  Images with ship annotations: {imgs_with_ships}")
    print(f"  Background images (no ships): {len(all_images) - imgs_with_ships}")
    
    # Create 3-fold CV splits using StratifiedKFold to reduce variance
    # Stratify by number of ships per image (capped at 5 to ensure valid splits)
    basenames = [e['basename'] for e in all_images]
    ship_counts = [len(e['anns']) for e in all_images]
    strat_labels = [min(c, 5) for c in ship_counts]
    
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(basenames, strat_labels)):
        fold_num = fold_idx + 1
        print(f"\n  Fold {fold_num}: {len(train_indices)} train, {len(val_indices)} val")
        
        fold_dir = os.path.join(OUTPUT_ROOT, f"fold_{fold_num}")
        
        for subset, indices in [("train", train_indices), ("val", val_indices)]:
            img_out = os.path.join(fold_dir, subset, "images")
            lbl_out = os.path.join(fold_dir, subset, "labels")
            os.makedirs(img_out, exist_ok=True)
            os.makedirs(lbl_out, exist_ok=True)
            
            for i in indices:
                entry = all_images[i]
                # Copy/symlink RGB image
                dst_img = os.path.join(img_out, entry['basename'])
                if not os.path.exists(dst_img):
                    os.symlink(os.path.abspath(entry['rgb_path']), dst_img)
                
                # Write YOLO label (ship-only, class 0)
                lbl_path = os.path.join(lbl_out, entry['basename'].replace('.jpg', '.txt'))
                with open(lbl_path, 'w') as f:
                    for ann in entry['anns']:
                        xc, yc, nw, nh = coco_bbox_to_yolo(
                            ann['bbox'], entry['width'], entry['height']
                        )
                        f.write(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
                # Empty file for background images is correct for YOLOv5
        
        # Create YAML config for this fold
        yaml_path = os.path.join(OUTPUT_ROOT, f"fold_{fold_num}.yaml")
        with open(yaml_path, 'w') as f:
            f.write(f"# Fold {fold_num} - 3-fold CV for lower bound baseline\n")
            f.write(f"path: {fold_dir}\n")
            f.write(f"train: train/images\n")
            f.write(f"val: val/images\n")
            f.write(f"nc: 1\n")
            f.write(f"names: ['ship']\n")
    
    return all_images


def create_hyp_config():
    """
    Step 3: Create hyperparameter config matching the paper.
    Based on hyp.scratch-low.yaml with modifications:
    - lrf: 0.1 (final one-cycle learning rate)
    - scale: 0.9
    - mixup: 0.1
    """
    print("\n" + "=" * 60)
    print("Step 3: Creating hyperparameter config")
    print("=" * 60)
    
    # Start from hyp.scratch-low.yaml defaults, apply paper modifications
    hyp_content = """# Hyperparameters matching paper methodology
# Based on hyp.scratch-low.yaml with paper-specific modifications

lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.1  # final OneCycleLR learning rate (lr0 * lrf) — PAPER MODIFICATION (default: 0.01)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.5  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 1.0  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.1  # image translation (+/- fraction)
scale: 0.9  # image scale (+/- gain) — PAPER MODIFICATION (default: 0.5)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction)
flipud: 0.0  # image flip up-down (probability)
fliplr: 0.5  # image left-right flip (probability)
mosaic: 1.0  # image mosaic (probability)
mixup: 0.1  # image mixup (probability) — PAPER MODIFICATION (default: 0.0)
copy_paste: 0.0  # segment copy-paste (probability)
"""
    
    hyp_path = os.path.join(OUTPUT_ROOT, "hyp_paper.yaml")
    with open(hyp_path, 'w') as f:
        f.write(hyp_content)
    
    print(f"  Saved hyperparameters to {hyp_path}")


def verify_data():
    """Verify the prepared data."""
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    
    for fold in range(1, 4):
        fold_dir = os.path.join(OUTPUT_ROOT, f"fold_{fold}")
        for subset in ["train", "val"]:
            n_imgs = len(os.listdir(os.path.join(fold_dir, subset, "images")))
            n_lbls = len(os.listdir(os.path.join(fold_dir, subset, "labels")))
            print(f"  Fold {fold} {subset}: {n_imgs} images, {n_lbls} labels")
    
    for hsi_type in ['hsi_rgb', 'hsi_pca']:
        test_dir = os.path.join(OUTPUT_ROOT, f"test_{hsi_type}")
        n_imgs = len(os.listdir(os.path.join(test_dir, "images")))
        n_lbls = len(os.listdir(os.path.join(test_dir, "labels")))
        print(f"  Test {hsi_type}: {n_imgs} images, {n_lbls} labels")


if __name__ == "__main__":

    """
    NEVER run this function again!!!
    It deletes all my runs
    """
    # Clean start
    # if os.path.exists(OUTPUT_ROOT):
    #     print(f"Removing existing {OUTPUT_ROOT}")
    #     shutil.rmtree(OUTPUT_ROOT)
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    prepare_optical_data()
    prepare_hsi_test_data()
    create_hyp_config()
    verify_data()
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Output directory: {OUTPUT_ROOT}")
    print("=" * 60)
