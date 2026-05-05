import os
import json
import numpy as np

PROJECT_ROOT = "/home/ahmadreza/Downloads/Research/M2SODAI"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
SOURCE_TEST_SET = os.path.join(PROJECT_ROOT, "baseline_official", "test_hsi_rgb_640_v3_bilinear")
OUTPUT_TEST_SET = os.path.join(PROJECT_ROOT, "baseline_official", "test_hsi_rgb_640_v3_bilinear_obb")

def prepare_test_obb():
    os.makedirs(os.path.join(OUTPUT_TEST_SET, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_TEST_SET, "labels"), exist_ok=True)
    
    # Load HSI annotations for segmentations
    with open(os.path.join(DATA_ROOT, "test_coco", "annotations_HSI.json")) as f:
        hsi_data = json.load(f)
        
    hsi_img_anns = {}
    for ann in hsi_data['annotations']:
        if ann['category_id'] == 0: # ship
            iid = ann['image_id']
            if iid not in hsi_img_anns:
                hsi_img_anns[iid] = []
            hsi_img_anns[iid].append(ann)

    img_id_to_name = {img['id']: os.path.basename(img['file_name']) for img in hsi_data['images']}

    print(f"Generating OBB labels for test set...")
    for img_id, anns in hsi_img_anns.items():
        if img_id not in img_id_to_name: continue
        basename = img_id_to_name[img_id].replace('.mat', '.jpg')
        
        # Symlink image from original set
        src_img = os.path.join(SOURCE_TEST_SET, "images", basename)
        dst_img = os.path.join(OUTPUT_TEST_SET, "images", basename)
        if os.path.exists(src_img) and not os.path.exists(dst_img):
            os.symlink(os.path.abspath(src_img), dst_img)
        
        # Write OBB labels
        lbl_path = os.path.join(OUTPUT_TEST_SET, "labels", basename.replace('.jpg', '.txt'))
        with open(lbl_path, 'w') as f:
            for ann in anns:
                if 'segmentation' in ann and ann['segmentation']:
                    # Normalize by 1600.0 to match training fold preparation
                    poly = np.array(ann['segmentation'][0]).reshape(-1, 2) / 1600.0
                    obb_str = " ".join([f"{v:.6f}" for v in poly.flatten()])
                    f.write(f"0 {obb_str}\n")

    # Create YAML
    yaml_path = os.path.join(PROJECT_ROOT, "baseline_official", "test_hsi_rgb_640_v3_bilinear_obb.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(OUTPUT_TEST_SET)}\ntrain: images\nval: images\nnc: 1\nnames: ['ship']\n")
    print(f"Created OBB test set at {OUTPUT_TEST_SET}")

if __name__ == "__main__":
    prepare_test_obb()
