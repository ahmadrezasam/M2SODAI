from ultralytics import YOLO
import os

def eval_fold1():
    print(f"\n{'='*70}")
    print(f"EVALUATING FOLD 1 UDA MODEL ON HSI TEST SET")
    print(f"{'='*70}\n")
    
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    weights_path = os.path.join(project_root, 'runs', 'mt_integrated', 'yolo26s_mt_fold1_adaptive', 'weights', 'last.pt')
    test_yaml = os.path.join(project_root, 'baseline_official', 'test_hsi_rgb_640_v3_bilinear_obb.yaml')
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}.")
        return
        
    model = YOLO(weights_path)
    
    results = model.val(
        data=test_yaml,
        imgsz=640,
        batch=16,
        device=0,
        single_cls=True,
        plots=True,
        save_json=True
    )
    
    metrics = results.results_dict
    precision = metrics.get('metrics/precision(B)', 0)
    recall = metrics.get('metrics/recall(B)', 0)
    map50 = metrics.get('metrics/mAP50(B)', 0)
    map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
    
    print(f"\n{'='*70}")
    print("FOLD 1 UDA OBB RESULTS (TEST ON HSI IMAGES)")
    print(f"{'='*70}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"mAP50:     {map50:.4f}")
    print(f"mAP50-95:  {map50_95:.4f}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    eval_fold1()
