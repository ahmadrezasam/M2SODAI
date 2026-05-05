from ultralytics import RTDETR
import os
import pandas as pd
import numpy as np

# Define paths
base_dir = '/home/ahmadreza/Downloads/Research/M2SODAI'
weights_dir = os.path.join(base_dir, 'baseline_official/runs_rtdetr')
# Using the same test set as eval_yolo26 for fair comparison
test_yaml = os.path.join(base_dir, 'baseline_official/test_hsi_rgb_640_v3_bilinear.yaml')
project_dir = os.path.join(base_dir, 'baseline_official/eval_results_rtdetr')

# Folds to evaluate
folds = [1, 2, 3]

all_results = []

for fold in folds:
    print(f"\n{'='*40}")
    print(f"EVALUATING RT-DETR FOLD {fold}")
    print(f"{'='*40}\n")
    
    # Path to best weights for this fold
    weights_path = os.path.join(weights_dir, f'rtdetr_l_fold_{fold}/weights/best.pt')
    
    if not os.path.exists(weights_path):
        print(f"Warning: Weights not found at {weights_path}, checking alternative naming...")
        # Check if the folder name is different (e.g. if it auto-incremented)
        weights_path = os.path.join(weights_dir, f'rtdetr_l_fold_{fold}/weights/best.pt')
        
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        continue
        
    # Load the model
    model = RTDETR(weights_path)
    
    # Run validation on the test set
    results = model.val(
        data=test_yaml,
        imgsz=640,
        batch=16,
        device=0,
        project=project_dir,
        name=f'eval_test_fold{fold}'
    )
    
    # Extract metrics
    # Note: RT-DETR might use slightly different metric names in results_dict
    # but ultralytics usually standardizes them to mAP50(B) and mAP50-95(B)
    map50 = results.results_dict.get('metrics/mAP50(B)', 0)
    map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0)
    
    all_results.append({
        'Fold': fold,
        'mAP50': map50,
        'mAP50-95': map50_95
    })
    
    print(f"\nFold {fold} evaluation completed. mAP50: {map50:.4f}")

# Summary Table
if all_results:
    df = pd.DataFrame(all_results)
    print(f"\n{'='*40}")
    print("RT-DETR SUMMARY RESULTS")
    print(f"{'='*40}")
    print(df.to_string(index=False))
    print(f"{'-'*40}")
    
    mean_map50 = df['mAP50'].mean()
    std_map50 = df['mAP50'].std()
    mean_map50_95 = df['mAP50-95'].mean()
    std_map50_95 = df['mAP50-95'].std()
    
    print(f"Mean mAP50:     {mean_map50:.4f} ± {std_map50:.4f}")
    print(f"Mean mAP50-95:  {mean_map50_95:.4f} ± {std_map50_95:.4f}")
    print(f"{'='*40}")
