from ultralytics import YOLO
import os

def eval_uda_model_round2():
    print(f"\n{'='*40}")
    print(f"EVALUATING UDA ROUND 2 MODEL (OBB)")
    print(f"{'='*40}\n")
    
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    weights_path = os.path.join(project_root, 'baseline_official', 'runs_yolo26s_obb_uda', 'yolo26s_fold1_obb_pseudo_round2', 'weights', 'best.pt')
    test_yaml = os.path.join(project_root, 'baseline_official', 'test_hsi_rgb_640_v3_bilinear_obb.yaml')
    project_dir = os.path.join(project_root, 'baseline_official', 'eval_results_uda')
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}. Please run train_yolo26s_obb_pseudo_round2.py first.")
        return
        
    model = YOLO(weights_path)
    
    results = model.val(
        data=test_yaml,
        imgsz=640,
        batch=16,
        device=0,
        single_cls=True,
        project=project_dir,
        name='eval_test_fold1_uda_v3_bilinear_obb_round2'
    )
    
    metrics = results.results_dict
    map50 = metrics.get('metrics/mAP50(B)', metrics.get('metrics/mAP50(M)', 0))
    map50_95 = metrics.get('metrics/mAP50-95(B)', metrics.get('metrics/mAP50-95(M)', 0))
    
    print(f"\n{'='*40}")
    print("UDA ROUND 2 OBB RESULTS (TEST ON HSI IMAGES)")
    print(f"{'='*40}")
    print(f"mAP50:     {map50:.4f}")
    print(f"mAP50-95:  {map50_95:.4f}")
    print(f"{'='*40}")

if __name__ == "__main__":
    eval_uda_model_round2()
