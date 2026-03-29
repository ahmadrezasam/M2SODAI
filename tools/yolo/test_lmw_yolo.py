import os
import sys
import json
import yaml
from tqdm import tqdm

def convert_coco_to_yolo(json_path, img_base_dir, output_labels_dir):
    """Converts COCO JSON annotations to YOLO txt format, filtering for RGB images."""
    os.makedirs(output_labels_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = {img['id']: img for img in data['images']}
    
    valid_image_ids = set()
    for img_id, img_info in images.items():
        file_name = img_info['file_name']
        if 'JPEGImages' not in file_name:
            continue
            
        full_img_path = os.path.join(img_base_dir, file_name)
        if not os.path.exists(full_img_path):
            continue
            
        valid_image_ids.add(img_id)
        
        txt_name = os.path.splitext(os.path.basename(file_name))[0] + ".txt"
        txt_path = os.path.join(output_labels_dir, txt_name)
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        with open(txt_path, 'w') as f:
            pass

    print(f"Initialized {len(valid_image_ids)} label files for test split.")
    
    for ann in tqdm(data['annotations'], desc=f"Converting {os.path.basename(json_path)}"):
        image_id = ann['image_id']
        if image_id not in valid_image_ids:
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
    split = 'test'
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
        valid_ids = convert_coco_to_yolo(json_path, img_base_dir, labels_dir)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
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
        print(f"Created {linked_count} symlinks for test split.")

    # 2. Update Dataset YAML
    dataset_yaml_path = os.path.join(project_root, "configs/yolo/dataset_rgb.yaml")
    with open(dataset_yaml_path, 'r') as f:
        dataset_cfg = yaml.safe_load(f)
    
    dataset_cfg['test'] = 'test/images'
    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False)

    # 3. Setup Environment and Import YOLO
    if lmw_yolo_dir not in sys.path:
        sys.path.insert(0, lmw_yolo_dir)
    from ultralytics import YOLO
    import ultralytics.nn.tasks as tasks
    from ultralytics.modifiednn.modules.block import LKCA, MSDP
    setattr(tasks, 'LKCA', LKCA)
    setattr(tasks, 'MSDP', MSDP)

    # 4. Find the best weights
    runs_dir = os.path.join(lmw_yolo_dir, "runs/detect/runs/detect")
    if not os.path.exists(runs_dir):
         runs_dir = os.path.join(project_root, "runs/detect")
    
    best_weights = None
    if os.path.exists(runs_dir):
        # Prefer the 845-matched run or the highres run if it finished
        all_runs = sorted([d for d in os.listdir(runs_dir) if 'lmw_yolo_rgb' in d], reverse=True)
        if all_runs:
            latest_run = all_runs[0]
            print(f"Using weights from run: {latest_run}")
            best_weights = os.path.join(runs_dir, latest_run, "weights/best.pt")
    
    if not best_weights or not os.path.exists(best_weights):
        print("Error: Could not find best.pt weights.")
        return

    print(f"Loading weights from {best_weights}...")
    model = YOLO(best_weights)
    
    # 5. Run Evaluation
    print("Running evaluation on test set...")
    metrics = model.val(data=dataset_yaml_path, split='test', batch=1, save_json=True)
    
    print("\nTest Results:")
    print(metrics.results_dict)

if __name__ == "__main__":
    main()
