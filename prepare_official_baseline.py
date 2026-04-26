"""
Prepare data for official YOLOv5 lower bound baseline reproduction.

Paper methodology:
- Combine 988 train + 124 val optical images → 1112 total
- 3-fold cross-validation splits
- Test set: 125 HSI images (RGB and PCA compositions)
- Ship-only detection (single class)

This script creates:
1. 3-fold CV directories with RGB optical images + labels
2. HSI-RGB test images (band selection mimicking RGB)
3. HSI-PCA test images (first 3 principal components)
4. YAML configs for YOLOv5
"""

import os
import json
import numpy as np
import scipy.io as sio
import cv2
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
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
    
    # Create 3-fold CV splits
    basenames = [e['basename'] for e in all_images]
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(basenames)):
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


def prepare_hsi_test_data():
    """
    Step 2: Prepare HSI test data (125 images) in both RGB and PCA compositions.
    Uses test_coco annotations (HSI coordinate space: 224x224).
    """
    print("\n" + "=" * 60)
    print("Step 2: Preparing HSI test data")
    print("=" * 60)
    
    # Load test annotations (from optical annotations - same bounding boxes)
    test_json = os.path.join(DATA_ROOT, "test_coco", "annotations.json")
    hsi_json = os.path.join(DATA_ROOT, "test_coco", "annotations_HSI.json")
    
    # Use optical annotations (1600x1600 coordinate space) for consistency
    # because that's what the paper uses - labels are in optical coordinates
    with open(test_json) as f:
        test_data = json.load(f)
    
    # Also load HSI annotations to get HSI coordinate mapping
    with open(hsi_json) as f:
        hsi_data = json.load(f)
    
    test_images = test_data['images']
    
    # Build per-image ship annotations (from optical annotations)
    img_anns = {}
    for ann in test_data['annotations']:
        if ann['category_id'] == 0:  # ship only
            iid = ann['image_id']
            if iid not in img_anns:
                img_anns[iid] = []
            img_anns[iid].append(ann)
    
    # HSI annotations (224x224 space)
    hsi_img_anns = {}
    for ann in hsi_data['annotations']:
        if ann['category_id'] == 0:
            iid = ann['image_id']
            if iid not in hsi_img_anns:
                hsi_img_anns[iid] = []
            hsi_img_anns[iid].append(ann)
    
    # Fit PCA on test HSI data (or better, on all available HSI data)
    print("  Fitting PCA on HSI data...")
    pca_samples = []
    mat_dir = os.path.join(DATA_ROOT, "test")
    
    for img_info in tqdm(test_images[:50], desc="  Sampling for PCA"):
        basename = os.path.basename(img_info['file_name']).replace('.jpg', '.mat')
        mat_path = os.path.join(mat_dir, basename)
        if os.path.exists(mat_path):
            hsi = sio.loadmat(mat_path)['data']  # (224, 224, 127)
            flat = hsi.reshape(-1, hsi.shape[2]).astype(np.float64)
            idx = np.random.choice(flat.shape[0], min(500, flat.shape[0]), replace=False)
            pca_samples.append(flat[idx])
    
    pca = PCA(n_components=3)
    pca.fit(np.concatenate(pca_samples, axis=0))
    print(f"  PCA explained variance: {pca.explained_variance_ratio_}")
    
    # Create HSI-RGB and HSI-PCA test directories
    for hsi_type in ['hsi_rgb', 'hsi_pca']:
        img_out = os.path.join(OUTPUT_ROOT, f"test_{hsi_type}", "images")
        lbl_out = os.path.join(OUTPUT_ROOT, f"test_{hsi_type}", "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
    
    # Process each test image
    print("  Processing test HSI images...")
    processed = 0
    for img_info in tqdm(test_images, desc="  Converting HSI"):
        img_id = img_info['id']
        basename = os.path.basename(img_info['file_name'])  # e.g., "993.jpg"
        mat_name = basename.replace('.jpg', '.mat')
        mat_path = os.path.join(mat_dir, mat_name)
        
        if not os.path.exists(mat_path):
            print(f"    Warning: {mat_path} not found")
            continue
        
        hsi = sio.loadmat(mat_path)['data']  # (224, 224, 127)
        h, w, c = hsi.shape
        
        # --- HSI-RGB: Select bands mimicking R, G, B ---
        # Common choices for 400-1000nm range in 4.5nm steps:
        # R ~ 650nm -> band ~(650-400)/4.5 = 55.6 -> band 56
        # G ~ 550nm -> band ~(550-400)/4.5 = 33.3 -> band 33  
        # B ~ 470nm -> band ~(470-400)/4.5 = 15.6 -> band 16
        # Using bands [53, 32, 11] as in existing code (close enough)
        hsi_rgb = hsi[:, :, [53, 32, 11]].astype(np.float32)
        for ch_i in range(3):
            c_min, c_max = hsi_rgb[:, :, ch_i].min(), hsi_rgb[:, :, ch_i].max()
            if c_max > c_min:
                hsi_rgb[:, :, ch_i] = (hsi_rgb[:, :, ch_i] - c_min) / (c_max - c_min) * 255.0
        hsi_rgb = np.clip(hsi_rgb, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_ROOT, "test_hsi_rgb", "images", basename), hsi_rgb)
        
        # --- HSI-PCA: First 3 principal components ---
        flat = hsi.reshape(-1, hsi.shape[2]).astype(np.float64)
        hsi_pca_img = pca.transform(flat).reshape(h, w, 3).astype(np.float32)
        for ch_i in range(3):
            c_min, c_max = hsi_pca_img[:, :, ch_i].min(), hsi_pca_img[:, :, ch_i].max()
            if c_max > c_min:
                hsi_pca_img[:, :, ch_i] = (hsi_pca_img[:, :, ch_i] - c_min) / (c_max - c_min) * 255.0
        hsi_pca_img = np.clip(hsi_pca_img, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_ROOT, "test_hsi_pca", "images", basename), hsi_pca_img)
        
        # --- Labels: Use HSI annotations (224x224 coordinate space) ---
        # Since HSI images are 224x224 and we DON'T resize them (YOLOv5 handles resizing),
        # we need labels in normalized coordinates relative to the HSI image.
        hsi_anns = hsi_img_anns.get(img_id, [])
        for hsi_type in ['hsi_rgb', 'hsi_pca']:
            lbl_path = os.path.join(OUTPUT_ROOT, f"test_{hsi_type}", "labels", basename.replace('.jpg', '.txt'))
            with open(lbl_path, 'w') as f:
                for ann in hsi_anns:
                    # HSI annotations bbox is in 224x224 space
                    xc, yc, nw, nh = coco_bbox_to_yolo(ann['bbox'], w, h)
                    f.write(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
        
        processed += 1
    
    print(f"  Processed {processed} test HSI images")
    
    # Create test YAML configs
    for hsi_type in ['hsi_rgb', 'hsi_pca']:
        yaml_path = os.path.join(OUTPUT_ROOT, f"test_{hsi_type}.yaml")
        test_dir = os.path.join(OUTPUT_ROOT, f"test_{hsi_type}")
        with open(yaml_path, 'w') as f:
            f.write(f"# Test config for {hsi_type}\n")
            f.write(f"path: {test_dir}\n")
            f.write(f"train: images  # not used for testing\n")
            f.write(f"val: images\n")  
            f.write(f"nc: 1\n")
            f.write(f"names: ['ship']\n")


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
