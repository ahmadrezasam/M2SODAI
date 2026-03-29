import json
import os
import sys
import shutil
from tqdm import tqdm

def convert_coco_to_yolo_pca(json_path, images_dir, labels_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    os.makedirs(labels_dir, exist_ok=True)
    
    # Map image_id to file_name (basename)
    image_id_to_name = {img['id']: os.path.basename(img['file_name']) for img in data['images']}
    
    # Collect annotations by image_id
    img_annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    print(f"Converting annotations for {len(data['images'])} images...")
    for img_id, file_name in image_id_to_name.items():
        # Only process if image exists in PCA images dir
        if not os.path.exists(os.path.join(images_dir, file_name)):
            continue
            
        label_path = os.path.join(labels_dir, file_name.replace('.jpg', '.txt'))
        with open(label_path, 'w') as f:
            if img_id in img_annotations:
                for ann in img_annotations[img_id]:
                    # Image size is fixed 1600x1600 in our PCA projection
                    x, y, w, h = ann['bbox']
                    x_center = (x + w/2) / 1600.0
                    y_center = (y + h/2) / 1600.0
                    w_norm = w / 1600.0
                    h_norm = h / 1600.0
                    f.write(f"{ann['category_id']} {x_center} {y_center} {w_norm} {h_norm}\n")

def main():
    project_root = os.getcwd()
    yolo_pca_dir = os.path.join(project_root, "data/yolo_pca_eval/test")
    pca_images_dir = os.path.join(yolo_pca_dir, "images")
    pca_labels_dir = os.path.join(yolo_pca_dir, "labels")
    gt_json = os.path.join(project_root, "data/test_coco/annotations.json")
    
    # 1. Convert annotations
    convert_coco_to_yolo_pca(gt_json, pca_images_dir, pca_labels_dir)
    
    # 2. Create dataset.yaml
    dataset_yaml_path = os.path.join(yolo_pca_dir, "dataset.yaml")
    with open(dataset_yaml_path, 'w') as f:
        f.write(f"path: {yolo_pca_dir}\n")
        f.write(f"train: images\n")
        f.write(f"val: images\n")
        f.write(f"test: images\n")
        f.write("names:\n")
        f.write("  0: ship\n")
        f.write("  1: floatingmatter\n")

    # 3. Setup Environment and Inject Modules
    lmw_yolo_dir = os.path.join(project_root, "mmdet/models/LMW-YOLO")
    if lmw_yolo_dir not in sys.path:
        sys.path.insert(0, lmw_yolo_dir)
    
    from ultralytics import YOLO
    import ultralytics.nn.tasks as tasks
    from ultralytics.modifiednn.modules.block import LKCA, MSDP
    setattr(tasks, 'LKCA', LKCA)
    setattr(tasks, 'MSDP', MSDP)

    # 4. Load RGB Weights
    weights_path = "/home/ahmadreza/Downloads/Research/M2SODAI/mmdet/models/LMW-YOLO/runs/detect/runs/detect/lmw_yolo_rgb_fixed_845_BEST/weights/best.pt"
    model = YOLO(weights_path)
    
    # 5. Run Evaluation
    print("Running HSI PCA evaluation...")
    metrics = model.val(data=dataset_yaml_path, split='test', batch=1, save_json=True)
    
    print("\nHSI PCA Test Results:")
    print(metrics.results_dict)

if __name__ == "__main__":
    main()
