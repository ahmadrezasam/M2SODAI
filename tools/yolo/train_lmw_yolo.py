import os
import sys
import json
import yaml
from tqdm import tqdm

def convert_coco_to_yolo(json_path, img_base_dir, output_labels_dir, limit=None):
    """Converts COCO JSON annotations to YOLO txt format, with optional limit."""
    os.makedirs(output_labels_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = {img['id']: img for img in data['images']}
    
    # Sort IDs to ensure deterministic selection
    all_img_ids = sorted(images.keys())
    
    valid_image_ids = []
    count = 0
    for img_id in all_img_ids:
        img_info = images[img_id]
        file_name = img_info['file_name']
        
        # Filter for RGB images
        if 'JPEGImages' not in file_name:
            continue
            
        full_img_path = os.path.join(img_base_dir, file_name)
        if not os.path.exists(full_img_path):
            continue
            
        valid_image_ids.append(img_id)
        count += 1
        
        # Stop once we reach the desired number (e.g. 845 for Faster R-CNN parity)
        if limit and count >= limit:
            break
            
    print(f"Selecting {len(valid_image_ids)} images to match Faster R-CNN baseline.")
    
    # Initialize empty label files for selected images
    selected_ids_set = set(valid_image_ids)
    for img_id in valid_image_ids:
        img_info = images[img_id]
        txt_name = os.path.splitext(os.path.basename(img_info['file_name']))[0] + ".txt"
        txt_path = os.path.join(output_labels_dir, txt_name)
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        with open(txt_path, 'w') as f:
            pass

    # Process annotations for selected images only
    for ann in tqdm(data['annotations'], desc=f"Converting {os.path.basename(json_path)}"):
        image_id = ann['image_id']
        if image_id not in selected_ids_set:
            continue
            
        img_info = images[image_id]
        w, h = img_info['width'], img_info['height']
        
        x_min, y_min, bw, bh = ann['bbox']
        x_center = (x_min + bw / 2) / w
        y_center = (y_min + bh / 2) / h
        norm_bw = bw / w
        norm_bh = bh / h
        
        cls_id = ann['category_id']
        file_name = img_info['file_name']
        txt_name = os.path.splitext(os.path.basename(file_name))[0] + ".txt"
        txt_path = os.path.join(output_labels_dir, txt_name)
        
        with open(txt_path, 'a') as f:
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_bw:.6f} {norm_bh:.6f}\n")
            
    return valid_image_ids

def main():
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    lmw_yolo_dir = os.path.join(project_root, "mmdet/models/LMW-YOLO")
    data_dir = os.path.join(project_root, "data")
    yolo_data_dir = os.path.join(data_dir, "yolo_rgb")
    
    # 1. Dataset Conversion
    # Train set: EXACTLY 845 images to match Faster R-CNN
    # Val set: 125 images as per folder content
    splits = [('train', 845), ('val', None)]
    
    for split, limit in splits:
        json_path = os.path.join(data_dir, f"{split}_coco", "annotations.json")
        img_base_dir = os.path.join(data_dir, f"{split}_coco")
        labels_dir = os.path.join(yolo_data_dir, split, "labels")
        yolo_img_dir = os.path.join(yolo_data_dir, split, "images")
        
        if os.path.exists(labels_dir):
            import shutil
            shutil.rmtree(labels_dir)
        os.makedirs(labels_dir, exist_ok=True)
        
        if os.path.exists(yolo_img_dir):
            import shutil
            shutil.rmtree(yolo_img_dir)
        os.makedirs(yolo_img_dir, exist_ok=True)
        
        if os.path.exists(json_path):
            valid_ids = convert_coco_to_yolo(json_path, img_base_dir, labels_dir, limit=limit)
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            images = {img['id']: img for img in data['images']}
            
            selected_ids_set = set(valid_ids)
            linked_count = 0
            for img_info in data['images']:
                if img_info['id'] not in selected_ids_set:
                    continue
                
                file_name = img_info['file_name']
                src = os.path.join(img_base_dir, file_name)
                dst = os.path.join(yolo_img_dir, os.path.basename(file_name))
                if os.path.exists(src):
                    if not os.path.exists(dst):
                         os.symlink(src, dst)
                         linked_count += 1
            print(f"Created {linked_count} symlinks for {split} split.")

    # 2. Force fresh scan by removing old caches
    import glob
    for cache_file in glob.glob(os.path.join(yolo_data_dir, "**/*.cache"), recursive=True):
        os.remove(cache_file)
        print(f"Removed cache: {cache_file}")

    # 3. Create Dataset YAML
    dataset_cfg = {
        'path': yolo_data_dir,
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'ship', 1: 'floatingmatter'}
    }
    dataset_yaml_path = os.path.join(project_root, "configs/yolo/dataset_rgb.yaml")
    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False)

    # 4. Environment and YOLO setup
    if lmw_yolo_dir not in sys.path:
        sys.path.insert(0, lmw_yolo_dir)
    from ultralytics import YOLO
    import ultralytics.nn.tasks as tasks
    from ultralytics.modifiednn.modules.block import LKCA, MSDP
    setattr(tasks, 'LKCA', LKCA)
    setattr(tasks, 'MSDP', MSDP)

    weights_path = os.path.join(project_root, "yolo11n_lmw_init.pt")
    model = YOLO(weights_path)
    model.train(
        data=dataset_yaml_path,
        epochs=50,
        imgsz=640,
        batch=8,
        workers=2,
        device=0,
        project='runs/detect',
        name='lmw_yolo_rgb_fixed_845'
    )

if __name__ == "__main__":
    main()
