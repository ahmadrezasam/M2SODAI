
import json
import os
from pathlib import Path

def convert_coco_to_yolo(json_path, img_dir, out_images_dir, out_labels_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    # Image ID -> Info mapping
    images = {img['id']: img for img in data['images']}
    
    # Image ID -> Annotations mapping
    annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append(ann)

    print(f"Processing {len(images)} images from {json_path}...")
    
    for img_id, img_info in images.items():
        # Copy image (symlink preferred to save space/meta)
        src_img = os.path.join(img_dir, img_info['file_name'])
        dst_img = os.path.join(out_images_dir, os.path.basename(img_info['file_name']))
        
        if os.path.lexists(dst_img):
            os.remove(dst_img)
        os.symlink(os.path.abspath(src_img), dst_img)

        # Create label file
        label_file = os.path.join(out_labels_dir, Path(img_info['file_name']).stem + '.txt')
        with open(label_file, 'w') as f:
            if img_id in annotations:
                for ann in annotations[img_id]:
                    # COCO: [x_min, y_min, width, height]
                    # YOLO: [class_id, x_center, y_center, width, height] (normalized)
                    x_min, y_min, w, h = ann['bbox']
                    img_w, img_h = img_info['width'], img_info['height']
                    
                    x_center = (x_min + w/2) / img_w
                    y_center = (y_min + h/2) / img_h
                    norm_w = w / img_w
                    norm_h = h / img_h
                    
                    class_id = ann['category_id'] # 0 for ship, 1 for floatingmatter
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            else:
                # Background image - leave file empty
                pass

if __name__ == "__main__":
    base_data = "/home/ahmadreza/Downloads/Research/M2SODAI/data"
    output_base = "/home/ahmadreza/Downloads/Research/M2SODAI/data/coco_matching_setup"
    
    splits = [
        ('train_coco', 'train'),
        ('val_coco', 'val'),
        ('test_coco', 'test')
    ]
    
    for src_dir, dst_dir in splits:
        print(f"\n--- Converting {src_dir} to {dst_dir} ---")
        json_file = os.path.join(base_data, src_dir, "annotations.json")
        img_src = os.path.join(base_data, src_dir) 
             
        out_img = os.path.join(output_base, dst_dir, "images")
        out_lbl = os.path.join(output_base, dst_dir, "labels")
        
        convert_coco_to_yolo(json_file, img_src, out_img, out_lbl)

    print("\nAll conversions complete!")
