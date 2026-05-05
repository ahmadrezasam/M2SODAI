from ultralytics import YOLO
import os
import pandas as pd

# Define paths
base_dir = '/home/ahmadreza/Downloads/Research/M2SODAI'
weights_dir = os.path.join(base_dir, 'baseline_official/runs_yolo26s_obb')
test_yaml = os.path.join(base_dir, 'baseline_official/test_hsi_rgb_640_v3_bilinear_obb.yaml')
project_dir = os.path.join(base_dir, 'baseline_official/eval_results')

# Folds to evaluate
folds = [1, 2, 3]
all_results = []

for fold in folds:
    print(f"\n{'='*40}")
    print(f"EVALUATING FOLD {fold} (OBB Model)")
    print(f"{'='*40}\n")
    
    weights_path = os.path.join(weights_dir, f'yolo26s_fold{fold}_obb/weights/best.pt')
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        continue
        
    model = YOLO(weights_path)
    
    results = model.val(
        data=test_yaml,
        imgsz=640,
        batch=16,
        device=0,
        single_cls=True,
        project=project_dir,
        name=f'eval_test_fold{fold}_v3_bilinear_obb_125'
    )
    
    # Extract metrics (Usually (B) even in some OBB versions, or (M))
    # We will try to detect the key
    metrics = results.results_dict
    map50 = metrics.get('metrics/mAP50(B)', metrics.get('metrics/mAP50(M)', 0))
    map50_95 = metrics.get('metrics/mAP50-95(B)', metrics.get('metrics/mAP50-95(M)', 0))
    
    all_results.append({
        'Fold': fold,
        'mAP50': map50,
        'mAP50-95': map50_95
    })

# Summary Table
if all_results:
    df = pd.DataFrame(all_results)
    print(f"\n{'='*40}")
    print("SUMMARY OBB RESULTS (EVALUATED ON 125 IMAGES)")
    print(f"{'='*40}")
    print(df.to_string(index=False))
    print(f"{'-'*40}")
    
    print(f"Mean mAP50:     {df['mAP50'].mean():.4f} ± {df['mAP50'].std():.4f}")
    print(f"Mean mAP50-95:  {df['mAP50-95'].mean():.4f} ± {df['mAP50-95'].std():.4f}")
    print(f"{'='*40}")