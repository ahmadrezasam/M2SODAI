from ultralytics import YOLO
import os

def eval_pseudo2_model():
    print(f"\n{'='*60}")
    print(f"EVALUATING UDA ROUND 2 MODEL (YOLOv26s-OBB)")
    print(f"{'='*60}\n")
    
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    # Pointing to the Round 2 weights
    weights_path = os.path.join(project_root, 'baseline_official', 'runs_yolo26s_obb_uda', 'yolo26s_fold1_obb_pseudo2', 'weights', 'best.pt')
    # Pointing to the correct HSI OBB test YAML
    test_yaml = os.path.join(project_root, 'baseline_official', 'test_hsi_rgb_640_v3_bilinear_obb.yaml')
    # Output directory for evaluation results
    project_dir = os.path.join(project_root, 'baseline_official', 'eval_results_uda')
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}.")
        return
        
    model = YOLO(weights_path)
    
    print(f"Running validation on: {test_yaml}")
    results = model.val(
        data=test_yaml,
        imgsz=640,
        batch=16,
        device=0,
        single_cls=True,
        project=project_dir,
        name='eval_test_fold1_uda_obb'
    )
    
    metrics = results.results_dict
    map50 = metrics.get('metrics/mAP50(B)', 0)
    map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
    
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS (HSI DOMAIN)")
    print(f"{'='*60}")
    print(f"mAP50:     {map50:.4f}")
    print(f"mAP50-95:  {map50_95:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    eval_pseudo2_model()
