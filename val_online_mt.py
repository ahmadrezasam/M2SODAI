import torch
from ultralytics import YOLO
import os

def validate_checkpoint(checkpoint_path, data_yaml, project_dir, run_name):
    print(f"\n{'='*60}")
    print(f"Validating: {os.path.basename(checkpoint_path)}")
    print(f"{'='*60}")
    
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    # Use a pre-trained OBB model as base to ensure all metadata/structure is correct
    base_model_path = os.path.join(project_root, "baseline_official/runs_yolo26s_obb/yolo26s_fold2_obb/weights/best.pt")
    
    if not os.path.exists(base_model_path):
        print(f"Error: Base model not found at {base_model_path}")
        return None

    # 1. Load base model
    model = YOLO(base_model_path)
    
    # 2. Load the Mean Teacher state dict
    print(f"Loading weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cuda')
    
    # Check if we need to strip 'model.' prefix (though from my check it seemed correct)
    # But let's be robust
    first_key = list(state_dict.keys())[0]
    if not first_key.startswith('model.'):
        # If weights were saved as student.state_dict() instead of student_model.state_dict()
        # but in our case they were saved as teacher.state_dict() where teacher is a Model instance.
        pass
        
    # Load into model.model
    msg = model.model.load_state_dict(state_dict, strict=True)
    print(f"Load result: {msg}")
    
    # 3. Run validation
    print(f"Starting validation on {data_yaml}")
    results = model.val(
        data=data_yaml,
        imgsz=640,
        batch=16,
        device=0,
        single_cls=True,
        save_json=True,
        project=project_dir,
        name=run_name,
        plots=True,
        conf=0.001, # Standard val threshold
        iou=0.6,    # Standard val threshold
    )
    
    return results

def main():
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    data_yaml = os.path.join(project_root, "baseline_official/test_hsi_rgb_640_v3_bilinear_obb.yaml")
    project_dir = "runs/online_mt_yolo26s_v3_fixed/evaluation"
    
    checkpoints = [
        os.path.join(project_root, "runs/online_mt_yolo26s_v3_fixed/mt_teacher_best.pt"),
        os.path.join(project_root, "runs/online_mt_yolo26s_v3_fixed/mt_teacher_epoch_50.pt"),
        os.path.join(project_root, "runs/online_mt_yolo26s_v3_fixed/mt_teacher_epoch_100.pt"),
    ]
    
    for ckpt in checkpoints:
        if not os.path.exists(ckpt):
            print(f"Skipping {ckpt} - file not found.")
            continue
            
        run_name = os.path.basename(ckpt).replace(".pt", "")
        results = validate_checkpoint(ckpt, data_yaml, project_dir, run_name)
        
        if results:
            print(f"\nResults for {os.path.basename(ckpt)}:")
            print(f"mAP@0.5: {results.results_dict['metrics/mAP50(B)']:.4f}")
            print(f"mAP@0.5:0.95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")

if __name__ == "__main__":
    main()
