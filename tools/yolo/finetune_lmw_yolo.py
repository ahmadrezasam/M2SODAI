import os
import sys

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

    # 2. Find the best weights to start from
    runs_dir = os.path.join(lmw_yolo_dir, "runs/detect/runs/detect")
    if not os.path.exists(runs_dir):
         runs_dir = os.path.join(project_root, "runs/detect")
    
    best_weights = None
    if os.path.exists(runs_dir):
        # Find the latest run folder
        all_runs = sorted([d for d in os.listdir(runs_dir) if d.startswith('lmw_yolo_rgb')], reverse=True)
        if all_runs:
            latest_run = all_runs[0]
            best_weights = os.path.join(runs_dir, latest_run, "weights/best.pt")
            
    if not best_weights or not os.path.exists(best_weights):
        print(f"Error: Could not find best.pt in {runs_dir}")
        return

    print(f"Fine-tuning from: {best_weights}")
    model = YOLO(best_weights)
    
    # 3. Start Fine-Tuning for 100 more epochs
    # Note: Using batch=8 and workers=2 as determined for stability
    model.train(
        data=dataset_yaml_path,
        epochs=100,
        imgsz=640,
        batch=8,
        workers=2,
        device=0,
        project='runs/detect',
        name='lmw_yolo_rgb_finetune'
    )

if __name__ == "__main__":
    main()
