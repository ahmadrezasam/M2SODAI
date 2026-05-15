import json
import os
import cv2
from tqdm import tqdm

def yolo_to_coco(data_dir, output_json):
    img_dir = os.path.join(data_dir, 'images')
    lbl_dir = os.path.join(data_dir, 'labels')
    
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "ship"}]
    }
    
    ann_id = 0
    for img_id, img_name in enumerate(tqdm(os.listdir(img_dir))):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        coco["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": w,
            "height": h
        })
        
        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        lbl_path = os.path.join(lbl_dir, lbl_name)
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    
                    # YOLO: class x_center y_center width height (normalized)
                    cls_id = int(parts[0])
                    x_c, y_c, bw, bh = map(float, parts[1:])
                    
                    # Convert to COCO: x_min, y_min, width, height (absolute)
                    abs_w = bw * w
                    abs_h = bh * h
                    abs_x = (x_c * w) - (abs_w / 2)
                    abs_y = (y_c * h) - (abs_h / 2)
                    
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 0, # Map all to ship for now
                        "bbox": [abs_x, abs_y, abs_w, abs_h],
                        "area": abs_w * abs_h,
                        "iscrowd": 0
                    })
                    ann_id += 1
                    
    with open(output_json, 'w') as f:
        json.dump(coco, f)
    print(f"Done! Saved {len(coco['images'])} images and {len(coco['annotations'])} annotations to {output_json}")

if __name__ == "__main__":
    yolo_to_coco('baseline_official/test_hsi_rgb_640_v3_bilinear', 'baseline_official/test_hsi_rgb_640_v3_bilinear/annotations.json')
