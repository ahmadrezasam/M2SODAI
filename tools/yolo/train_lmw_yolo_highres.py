import os
import sys
import yaml

def main():
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    lmw_yolo_dir = os.path.join(project_root, "mmdet/models/LMW-YOLO")
    dataset_yaml_path = os.path.join(project_root, "configs/yolo/dataset_rgb.yaml")
    
    # 1. Setup Environment and Import YOLO
    if lmw_yolo_dir not in sys.path:
        sys.path.insert(0, lmw_yolo_dir)
    
    from ultralytics import YOLO
    import ultralytics.nn.tasks as tasks
    from ultralytics.modifiednn.modules.block import LKCA, MSDP
    setattr(tasks, 'LKCA', LKCA)
    setattr(tasks, 'MSDP', MSDP)

    # 2. Use the Initialized Weights or Best Weights
    runs_dir = os.path.join(lmw_yolo_dir, "runs/detect/runs/detect")
    if not os.path.exists(runs_dir):
         runs_dir = os.path.join(project_root, "runs/detect")
    
    weights_path = os.path.join(project_root, "yolo11n_lmw_init.pt")
    
    if os.path.exists(runs_dir):
        # Prefer the best weights from previous runs if available
        all_runs = sorted([d for d in os.listdir(runs_dir) if d.startswith('lmw_yolo_rgb')], reverse=True)
        if all_runs:
            latest_run = all_runs[0]
            best_pt = os.path.join(runs_dir, latest_run, "weights/best.pt")
            if os.path.exists(best_pt):
                weights_path = best_pt

    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return

    print(f"Starting High-Res Training from: {weights_path}...")
    model = YOLO(weights_path)
    
    # 3. Start High-Resolution Training
    # imgsz=1600: Native resolution
    # batch=1: Essential for 16GB RAM systems at this resolution
    # workers=1: Minimize memory overhead from data loading
    model.train(
        data=dataset_yaml_path,
        epochs=100,
        imgsz=1600,
        batch=1,          # Force batch size 1 to avoid OOM
        workers=1,        # Force single worker to avoid RAM spike
        device=0,         # Use GPU
        project='runs/detect',
        name='lmw_yolo_rgb_1600'
    )

if __name__ == "__main__":
    main()
